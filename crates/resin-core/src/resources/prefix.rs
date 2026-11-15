use lru::LruCache;
use sha2::{Digest, Sha256};
use std::borrow::Cow;

// Frame arbitrary namespaces while retaining external storage's SHA-256 encoding.
type PrefixHash = (usize, [u8; 32]);

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

    fn hash(&self, alignment: usize) -> PrefixHash {
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
}

/// Flat prefix identities with least-recently-used immutable KV snapshots.
#[derive(Debug)]
pub(crate) struct PrefixCache<T> {
    alignment: u32,
    entries: LruCache<PrefixHash, T>,
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
        let mut computed;
        let hashes = match prefix
            .hashes
            .filter(|h| h.alignment == alignment && h.values.len() >= usable / alignment)
        {
            Some(hashes) => hashes,
            None => {
                computed = PrefixHashes::new(&prefix.namespace, self.alignment);
                computed.update(&prefix.tokens);
                &computed
            }
        };
        for (index, digest) in hashes.values[..usable / alignment].iter().enumerate().rev() {
            let hash = (prefix.namespace.len(), *digest);
            if self.entries.contains(&hash) {
                self.hits = self.hits.saturating_add(1);
                return Some((
                    (index as u32 + 1) * self.alignment,
                    self.entries.get(&hash).unwrap(),
                ));
            }
        }
        self.misses = self.misses.saturating_add(1);
        None
    }

    pub(crate) fn insert(&mut self, key: &PrefixKey<'_>, snapshot: T) -> Option<T> {
        debug_assert!(!key.is_empty() && key.len().is_multiple_of(self.alignment as usize));
        self.entries
            .put(key.hash(self.alignment as usize), snapshot)
    }

    pub(crate) fn evict_lru(&mut self) -> Option<T> {
        let (_, snapshot) = self.entries.pop_lru()?;
        self.evictions = self.evictions.saturating_add(1);
        Some(snapshot)
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
}
