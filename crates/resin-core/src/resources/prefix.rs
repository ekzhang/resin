use lru::LruCache;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::HashMap;

// Frame arbitrary namespaces while retaining external storage's SHA-256 encoding.
pub(crate) type PrefixHash = (usize, [u8; 32]);

#[derive(Debug)]
struct PrefixHashes {
    alignment: usize,
    state: Sha256,
    values: Vec<[u8; 32]>,
}

impl PrefixHashes {
    fn new(namespace: &[u32], alignment: u32) -> Self {
        assert_ne!(alignment, 0);
        let mut state = Sha256::new();
        hash_tokens(&mut state, namespace);
        Self {
            alignment: alignment as usize,
            state,
            values: Vec::new(),
        }
    }

    /// Hash only complete blocks added since the previous update.
    fn update(&mut self, tokens: &[u32]) {
        for block in tokens[self.values.len() * self.alignment..].chunks_exact(self.alignment) {
            hash_tokens(&mut self.state, block);
            self.values.push(self.state.clone().finalize().into());
        }
    }
}

/// One contiguous identity per request, extended only with accepted output.
#[derive(Debug)]
pub struct PrefixIdentity {
    tokens: Vec<u32>,
    namespace: Vec<u32>,
    hashes: PrefixHashes,
}

impl PrefixIdentity {
    pub fn new(tokens: Vec<u32>, namespace: Vec<u32>, alignment: u32) -> Self {
        let mut hashes = PrefixHashes::new(&namespace, alignment);
        hashes.update(&tokens);
        Self {
            tokens,
            namespace,
            hashes,
        }
    }

    pub fn extend(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
        self.hashes.update(&self.tokens);
    }

    pub fn key(&self) -> PrefixKey<'_> {
        PrefixKey {
            hashes: Some(&self.hashes),
            ..PrefixKey::new(&self.tokens, &self.namespace)
        }
    }
}

fn hash_tokens(state: &mut Sha256, tokens: &[u32]) {
    let mut bytes = [0u8; 256];
    for chunk in tokens.chunks(bytes.len() / 4) {
        for (token, target) in chunk.iter().zip(bytes.as_chunks_mut::<4>().0) {
            *target = token.to_le_bytes();
        }
        state.update(&bytes[..chunk.len() * 4]);
    }
}

/// A logical token prefix and its opaque cache namespace.
#[derive(Clone, Debug, Default)]
pub struct PrefixKey<'a> {
    tokens: Cow<'a, [u32]>,
    namespace: Cow<'a, [u32]>,
    hashes: Option<&'a PrefixHashes>,
}

impl<'a> PrefixKey<'a> {
    pub fn new(tokens: &'a [u32], namespace: &'a [u32]) -> Self {
        Self {
            tokens: tokens.into(),
            namespace: namespace.into(),
            ..Self::default()
        }
    }

    pub(crate) fn namespace(salt: Option<&str>) -> Vec<u32> {
        salt.map(|salt| salt.bytes().map(u32::from).chain([u32::MAX]).collect())
            .unwrap_or_default()
    }

    pub fn parts(&self) -> (&[u32], &[u32]) {
        (&self.namespace, &self.tokens)
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn prefix(&self, length: usize) -> PrefixKey<'_> {
        PrefixKey {
            hashes: self.hashes,
            ..PrefixKey::new(&self.tokens[..length], &self.namespace)
        }
    }

    pub fn into_owned(self) -> PrefixKey<'static> {
        PrefixKey {
            tokens: self.tokens.into_owned().into(),
            namespace: self.namespace.into_owned().into(),
            ..PrefixKey::default()
        }
    }

    pub fn truncate(&mut self, length: usize) {
        match &mut self.tokens {
            Cow::Borrowed(tokens) => *tokens = &tokens[..length.min(tokens.len())],
            Cow::Owned(tokens) => tokens.truncate(length),
        }
    }

    pub(crate) fn hash(&self, alignment: usize) -> PrefixHash {
        let digest = if let Some(hashes) = self.hashes.filter(|hashes| {
            hashes.alignment == alignment && hashes.values.len() >= self.len() / alignment
        }) {
            hashes.values[self.len() / alignment - 1]
        } else {
            let mut state = Sha256::new();
            hash_tokens(&mut state, &self.namespace);
            hash_tokens(&mut state, &self.tokens);
            state.finalize().into()
        };
        (self.namespace.len(), digest)
    }

    fn block_hashes(&self, alignment: u32) -> Cow<'_, [[u8; 32]]> {
        let count = self.len() / alignment as usize;
        if let Some(hashes) = self
            .hashes
            .filter(|h| h.alignment == alignment as usize && h.values.len() >= count)
        {
            Cow::Borrowed(&hashes.values[..count])
        } else {
            let mut hashes = PrefixHashes::new(&self.namespace, alignment);
            hashes.update(&self.tokens);
            Cow::Owned(hashes.values)
        }
    }
}

#[derive(Debug)]
struct PrefixLink {
    parent: Option<PrefixHash>,
    references: usize,
    shared: bool,
}

#[derive(Debug)]
struct CachedPrefix<T> {
    value: T,
    retained: bool,
}

/// Flat prefix identities with least-recently-used immutable KV snapshots.
#[derive(Debug)]
pub(crate) struct PrefixCache<T> {
    alignment: u32,
    entries: LruCache<PrefixHash, CachedPrefix<T>>,
    // Only active request heads, cached endpoints, and their ancestors survive.
    links: HashMap<PrefixHash, PrefixLink>,
    requests: HashMap<u64, Option<PrefixHash>>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl<T> PrefixCache<T> {
    pub(crate) fn new(alignment: u32) -> Self {
        debug_assert_ne!(alignment, 0);
        Self {
            alignment,
            // The page pool drives eviction; the index must never drop owners itself.
            entries: LruCache::unbounded(),
            links: HashMap::new(),
            requests: HashMap::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub(crate) const fn alignment(&self) -> u32 {
        self.alignment
    }

    pub(crate) fn find(
        &mut self,
        key: &PrefixKey<'_>,
        max_prefix_tokens: usize,
    ) -> Option<(u32, &T)> {
        let alignment = self.alignment as usize;
        let usable = key.len().saturating_sub(1).min(max_prefix_tokens) / alignment * alignment;
        // Each snapshot owns all its required pages: an evicted shorter
        // checkpoint does not prevent a hit on a longer one.
        let prefix = key.prefix(usable);
        let hashes = prefix.block_hashes(self.alignment);
        for (index, digest) in hashes.iter().enumerate().rev() {
            let hash = (prefix.namespace.len(), *digest);
            if self.entries.contains(&hash) {
                self.hits = self.hits.saturating_add(1);
                return Some((
                    (index as u32 + 1) * self.alignment,
                    &self.entries.get(&hash).unwrap().value,
                ));
            }
        }
        self.misses = self.misses.saturating_add(1);
        None
    }

    #[cfg(test)]
    pub(crate) fn insert(&mut self, key: &PrefixKey<'_>, snapshot: T) -> Option<T> {
        self.insert_with_retention(key, snapshot, true)
    }

    pub(crate) fn insert_with_retention(
        &mut self,
        key: &PrefixKey<'_>,
        value: T,
        retained: bool,
    ) -> Option<T> {
        debug_assert!(!key.is_empty() && key.len().is_multiple_of(self.alignment as usize));
        let (head, _) = self.extend_links(key);
        let head = head.unwrap();
        self.links.get_mut(&head).unwrap().references += 1;
        let retained = retained || self.entries.peek(&head).is_some_and(|entry| entry.retained);
        let replaced = self.entries.put(head, CachedPrefix { value, retained });
        if replaced.is_some() {
            self.release_link(head);
        }
        replaced.map(|entry| entry.value)
    }

    /// Remember prompt ancestry without retaining its historical KV snapshots.
    pub(crate) fn observe(&mut self, request: u64, key: &PrefixKey<'_>) -> u32 {
        let (head, known) = self.extend_links(key);
        if !self.requests.contains_key(&request) && known != 0 {
            let shared = key.prefix(known as usize).hash(self.alignment as usize);
            self.links.get_mut(&shared).unwrap().shared = true;
        }
        if let Some(head) = head {
            self.links.get_mut(&head).unwrap().references += 1;
        }
        if let Some(previous) = self.requests.insert(request, head).flatten() {
            self.release_link(previous);
        }
        known
    }

    pub(crate) fn finish_request(&mut self, request: u64) {
        if let Some(head) = self.requests.remove(&request).flatten() {
            self.release_link(head);
        }
    }

    pub(crate) fn is_shared(&self, key: &PrefixKey<'_>) -> bool {
        !key.is_empty()
            && key.len().is_multiple_of(self.alignment as usize)
            && self
                .links
                .get(&key.hash(self.alignment as usize))
                .is_some_and(|link| link.shared)
    }

    pub(crate) fn retain(&mut self, hash: PrefixHash) -> bool {
        if let Some(entry) = self.entries.peek_mut(&hash) {
            return !std::mem::replace(&mut entry.retained, true);
        }
        false
    }

    pub(crate) fn peek(&self, hash: PrefixHash) -> Option<&T> {
        self.entries.peek(&hash).map(|entry| &entry.value)
    }

    pub(crate) fn remove_transient(&mut self, hash: PrefixHash) -> Option<T> {
        if self.entries.peek(&hash)?.retained || self.links[&hash].shared {
            return None;
        }
        let entry = self.entries.pop(&hash).unwrap();
        self.release_link(hash);
        Some(entry.value)
    }

    /// Add only the unknown suffix; each new child holds one parent reference.
    fn extend_links(&mut self, key: &PrefixKey<'_>) -> (Option<PrefixHash>, u32) {
        let hashes = key.block_hashes(self.alignment);
        let namespace = key.namespace.len();
        let known = hashes
            .iter()
            .rposition(|digest| self.links.contains_key(&(namespace, *digest)))
            .map_or(0, |index| index + 1);
        let mut head = known.checked_sub(1).map(|index| (namespace, hashes[index]));
        for &digest in &hashes[known..] {
            if let Some(parent) = head {
                self.links.get_mut(&parent).unwrap().references += 1;
            }
            let hash = (namespace, digest);
            self.links.insert(
                hash,
                PrefixLink {
                    parent: head,
                    references: 0,
                    shared: false,
                },
            );
            head = Some(hash);
        }
        (head, known as u32 * self.alignment)
    }

    fn release_link(&mut self, mut hash: PrefixHash) {
        loop {
            let link = self.links.get_mut(&hash).unwrap();
            link.references -= 1;
            if link.references != 0 {
                break;
            }
            let parent = self.links.remove(&hash).unwrap().parent;
            let Some(parent) = parent else { break };
            hash = parent;
        }
    }

    pub(crate) fn evict_lru(&mut self) -> Option<T> {
        let (hash, snapshot) = self.entries.pop_lru()?;
        self.release_link(hash);
        self.evictions = self.evictions.saturating_add(1);
        Some(snapshot.value)
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) const fn hits(&self) -> u64 {
        self.hits
    }

    pub(crate) const fn misses(&self) -> u64 {
        self.misses
    }

    pub(crate) const fn evictions(&self) -> u64 {
        self.evictions
    }
}

#[cfg(test)]
mod tests {
    use super::{PrefixCache, PrefixIdentity, PrefixKey};

    fn key(tokens: &[u32]) -> PrefixKey<'_> {
        PrefixKey::new(tokens, &[])
    }

    #[test]
    fn prefix_digests_match_external_storage_encoding_and_frame_namespaces() {
        for (namespace, expected) in [
            (
                vec![],
                "cf97adeedb59e05bfd73a2b4c2a8885708c4f4f70c84c64b27120e72ab733b72",
            ),
            (
                vec![97, u32::MAX],
                "0bb26e0f3090ec49fff665278d0bed6607bdbb68f7047cb43da95ea7db23033e",
            ),
        ] {
            let digest = PrefixKey::new(&[1, 2, 3, 4], &namespace).hash(2).1;
            assert_eq!(
                digest
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>(),
                expected
            );
        }
        // The same byte stream with a different namespace/token boundary must
        // remain distinct, even for arbitrary namespaces supplied by native users.
        assert_ne!(
            PrefixKey::new(&[2, 3], &[1]).hash(1),
            PrefixKey::new(&[3], &[1, 2]).hash(1)
        );
    }

    #[test]
    fn growing_identity_hashes_match_contiguous_prefixes() {
        let tokens = (1..=33).collect::<Vec<_>>();
        let namespace = PrefixKey::namespace(Some("media"));
        for prompt_length in 0..=tokens.len() {
            let mut identity =
                PrefixIdentity::new(tokens[..prompt_length].to_vec(), namespace.clone(), 4);
            for output in tokens[prompt_length..].chunks(3) {
                identity.extend(output);
                let key = identity.key();
                assert_eq!(key.tokens.as_ptr(), identity.tokens.as_ptr());
                assert_eq!(key.parts(), (namespace.as_slice(), &tokens[..key.len()]));
                for end in (4..=key.len()).step_by(4) {
                    assert_eq!(
                        key.prefix(end).hash(4),
                        PrefixKey::new(&tokens[..end], &namespace).hash(4)
                    );
                }
            }
            let key = identity.key();
            let mut cache = PrefixCache::new(4);
            cache.insert(&key.prefix(32), 7);
            assert_eq!(cache.find(&key, usize::MAX), Some((32, &7)));
        }
    }

    #[test]
    fn storage_keys_own_only_the_clipped_prefix_and_can_be_truncated() {
        let mut owned = {
            let mut identity =
                PrefixIdentity::new(vec![1, 2, 3], PrefixKey::namespace(Some("job")), 2);
            identity.extend(&[4, 5, 6, 7]);
            let key = identity.key();
            for length in 0..=8 {
                let mut truncated = key.clone();
                truncated.truncate(length);
                assert_eq!(truncated.tokens.as_ptr(), identity.tokens.as_ptr());
                assert_eq!(truncated.parts().1, &identity.tokens[..length.min(7)]);
            }
            key.prefix(6).into_owned()
        };
        assert_eq!(owned.parts().1, &[1, 2, 3, 4, 5, 6]);
        owned.truncate(4);
        assert_eq!(owned.parts().1, &[1, 2, 3, 4]);
        assert_eq!(
            owned.hash(2),
            PrefixKey::new(&[1, 2, 3, 4], &PrefixKey::namespace(Some("job"))).hash(2)
        );
    }

    #[test]
    fn namespace_frames_isolate_first_blocks_without_changing_lengths() {
        let mut cache = PrefixCache::new(2);
        for (value, salt) in [None, Some(""), Some("a"), Some("abc")]
            .into_iter()
            .enumerate()
        {
            let namespace = PrefixKey::namespace(salt);
            let key = PrefixKey::new(&[1, 2, 3], &namespace);
            assert_eq!(cache.find(&key.prefix(3), usize::MAX), None);
            cache.insert(&key.prefix(2), value);
            assert_eq!(cache.find(&key, usize::MAX), Some((2, &value)));
        }
        for value in 0..4 {
            assert_eq!(cache.evict_lru(), Some(value));
        }
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn hits_and_replacements_update_lru_order() {
        let mut cache = PrefixCache::new(1);
        cache.insert(&key(&[1]), "one");
        cache.insert(&key(&[2]), "two");
        cache.insert(&key(&[3]), "three");
        assert_eq!(cache.find(&key(&[1, 99]), usize::MAX), Some((1, &"one")));
        assert_eq!(cache.insert(&key(&[2]), "new two"), Some("two"));
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.evict_lru(), Some("three"));
        assert_eq!(cache.evict_lru(), Some("one"));
        assert_eq!(cache.evict_lru(), Some("new two"));
        assert_eq!(cache.evict_lru(), None);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.evictions(), 3);
    }

    #[test]
    fn replay_limit_is_aligned_without_excluding_the_boundary_again() {
        let mut cache = PrefixCache::new(4);
        let tokens = (0..16).collect::<Vec<_>>();
        for end in [4, 8, 12] {
            cache.insert(&key(&tokens[..end]), end);
        }
        assert_eq!(cache.find(&key(&tokens), 8), Some((8, &8)));
        assert_eq!(cache.find(&key(&tokens), 7), Some((4, &4)));
        assert_eq!(cache.find(&key(&tokens), 0), None);
    }

    #[test]
    fn evicting_shorter_checkpoints_preserves_longer_ones() {
        let mut cache = PrefixCache::new(1);
        cache.insert(&key(&[1]), 1);
        cache.insert(&key(&[1, 2]), 2);
        cache.insert(&key(&[1, 2, 3]), 3);
        assert_eq!(cache.evict_lru(), Some(1));
        assert_eq!(cache.find(&key(&[1, 2, 99]), usize::MAX), Some((2, &2)));
        assert_eq!(cache.evict_lru(), Some(3));
        assert_eq!(cache.evict_lru(), Some(2));
        assert!(cache.entries.is_empty());

        cache.insert(&key(&[4, 5]), 4);
        assert_eq!(cache.find(&key(&[4, 5, 99]), usize::MAX), Some((2, &4)));
        assert_eq!(cache.evict_lru(), Some(4));
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn wide_cache_eviction_removes_hash_entries() {
        let mut cache = PrefixCache::new(2);
        for id in 0..1024 {
            cache.insert(&key(&[id, id + 1]), id);
        }
        for id in 0..1024 {
            assert_eq!(cache.evict_lru(), Some(id));
            assert_eq!(cache.entries.len(), 1023 - id as usize);
        }
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn shared_boundaries_are_detected_without_historical_snapshots() {
        let mut cache = PrefixCache::new(2);
        let original = key(&[1, 2, 3, 4, 5, 6]);
        let branch = key(&[1, 2, 3, 4, 9, 10]);
        assert_eq!(cache.observe(1, &original), 0);
        cache.insert_with_retention(&original, "original end", false);
        cache.finish_request(1);

        // The endpoint retains ancestry, but no KV exists at the fork yet.
        assert_eq!(cache.observe(2, &branch), 4);
        assert!(cache.is_shared(&original.prefix(4)));
        assert_eq!(cache.find(&branch, usize::MAX), None);
        cache.insert_with_retention(&branch.prefix(4), "recomputed fork", false);
        assert_eq!(
            cache.find(&branch, usize::MAX),
            Some((4, &"recomputed fork"))
        );
        assert_eq!(cache.remove_transient(branch.prefix(4).hash(2)), None);

        // Dropping either branch leaves the other's ancestry intact.
        assert_eq!(
            cache.remove_transient(original.hash(2)),
            Some("original end")
        );
        cache.finish_request(2);
        assert_eq!(cache.links.len(), 2);
        assert_eq!(cache.evict_lru(), Some("recomputed fork"));
        assert!(cache.links.is_empty());
        assert!(cache.requests.is_empty());
    }

    #[test]
    fn rolling_checkpoint_retention_does_not_accumulate_snapshots() {
        let mut cache = PrefixCache::new(2);
        let mut identity = PrefixIdentity::new(vec![1], vec![], 2);
        cache.observe(1, &identity.key());
        let mut previous = None;
        for token in 2..=128 {
            identity.extend(&[token]);
            cache.observe(1, &identity.key());
            if token % 2 != 0 {
                continue;
            }
            let key = identity.key();
            assert!(!cache.is_shared(&key));
            cache.insert_with_retention(&key, token, false);
            if let Some(previous) = previous {
                assert_eq!(cache.remove_transient(previous), Some(token - 2));
            }
            previous = Some(key.hash(2));
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.links.len(), token as usize / 2);
        }
        let endpoint = previous.unwrap();
        cache.retain(endpoint);
        cache.finish_request(1);
        assert_eq!(cache.peek(endpoint), Some(&128));
        assert_eq!(cache.remove_transient(endpoint), None);
        // Another materialization cannot downgrade a preserved endpoint.
        assert_eq!(
            cache.insert_with_retention(&identity.key(), 129, false),
            Some(128)
        );
        assert_eq!(cache.remove_transient(endpoint), None);
        assert_eq!(cache.evict_lru(), Some(129));
        assert!(cache.links.is_empty());
    }

    #[test]
    fn observed_prefixes_are_namespace_isolated_and_release_short_requests() {
        let mut cache = PrefixCache::<()>::new(2);
        let tokens = [1, 2, 3, 4];
        let first = PrefixKey::new(&tokens, &[1]);
        let second = PrefixKey::new(&tokens, &[2]);
        assert_eq!(cache.observe(1, &first), 0);
        assert_eq!(cache.observe(2, &second), 0);
        assert_eq!(cache.observe(3, &second), 4);
        assert!(!cache.is_shared(&first));
        assert!(cache.is_shared(&second));
        cache.finish_request(2);
        assert_eq!(cache.links.len(), 4);
        cache.finish_request(3);
        assert_eq!(cache.links.len(), 2);
        cache.observe(1, &first.prefix(1));
        assert!(cache.links.is_empty());
        cache.finish_request(1);
        assert!(cache.requests.is_empty());
    }
}
