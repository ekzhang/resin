use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug)]
struct Entry<T> {
    snapshot: T,
    previous: Option<usize>,
    next: Option<usize>,
}

#[derive(Debug)]
struct Node<T> {
    parent: Option<usize>,
    block: Option<Arc<[u32]>>,
    children: HashMap<Arc<[u32]>, usize>,
    entry: Option<Entry<T>>,
}

impl<T> Default for Node<T> {
    fn default() -> Self {
        Self {
            parent: None,
            block: None,
            children: HashMap::new(),
            entry: None,
        }
    }
}

/// Exact-token block trie with least-recently-used immutable KV snapshots.
/// Each distinct token block is stored once per parent, avoiding the quadratic
/// metadata of retaining a complete token vector at every checkpoint.
#[derive(Debug)]
pub(crate) struct PrefixCache<T> {
    alignment: u32,
    oldest: Option<usize>,
    newest: Option<usize>,
    nodes: Vec<Node<T>>,
    free_nodes: Vec<usize>,
    entries: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl<T> PrefixCache<T> {
    pub(crate) fn new(alignment: u32) -> Self {
        debug_assert_ne!(alignment, 0);
        Self {
            alignment,
            oldest: None,
            newest: None,
            nodes: vec![Node::default()],
            free_nodes: Vec::new(),
            entries: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub(crate) const fn alignment(&self) -> u32 {
        self.alignment
    }

    /// Inspect an exact checkpoint without changing lookup counters or LRU age.
    pub(crate) fn get_exact(&self, tokens: &[u32]) -> Option<&T> {
        if tokens.is_empty() || !tokens.len().is_multiple_of(self.alignment as usize) {
            return None;
        }
        let mut node = 0;
        for block in tokens.chunks_exact(self.alignment as usize) {
            node = *self.nodes[node].children.get(block)?;
        }
        self.nodes[node].entry.as_ref().map(|entry| &entry.snapshot)
    }

    pub(crate) fn find(&mut self, tokens: &[u32], max_prefix_tokens: usize) -> Option<(u32, &T)> {
        let alignment = self.alignment as usize;
        let usable = if tokens.len().is_multiple_of(alignment) {
            tokens.len().saturating_sub(alignment)
        } else {
            tokens.len() / alignment * alignment
        }
        .min(max_prefix_tokens / alignment * alignment);
        let mut node = 0usize;
        let mut matched = None;
        for (block_index, block) in tokens[..usable].chunks_exact(alignment).enumerate() {
            let Some(&child) = self.nodes[node].children.get(block) else {
                break;
            };
            node = child;
            if self.nodes[node].entry.is_some() {
                matched = Some((node, (block_index + 1) * alignment));
            }
        }
        let Some((node, length)) = matched else {
            self.misses = self.misses.saturating_add(1);
            return None;
        };
        self.remove_from_lru(node);
        self.push_recent(node);
        let entry = self.nodes[node]
            .entry
            .as_ref()
            .expect("matched prefix node keeps its snapshot");
        self.hits = self.hits.saturating_add(1);
        Some((length as u32, &entry.snapshot))
    }

    pub(crate) fn insert(&mut self, tokens: &[u32], snapshot: T) -> Option<T> {
        debug_assert!(!tokens.is_empty() && tokens.len().is_multiple_of(self.alignment as usize));
        let mut node = 0usize;
        for block in tokens.chunks_exact(self.alignment as usize) {
            node = match self.nodes[node].children.get(block).copied() {
                Some(child) => child,
                None => {
                    let block: Arc<[u32]> = block.into();
                    let child = self.allocate_node(node, Arc::clone(&block));
                    self.nodes[node].children.insert(block, child);
                    child
                }
            };
        }
        if self.nodes[node].entry.is_some() {
            self.remove_from_lru(node);
        }
        let previous = self.nodes[node].entry.replace(Entry {
            snapshot,
            previous: None,
            next: None,
        });
        self.push_recent(node);
        if previous.is_none() {
            self.entries += 1;
        }
        previous.map(|entry| entry.snapshot)
    }

    pub(crate) fn evict_lru(&mut self) -> Option<T> {
        let node = self.oldest?;
        self.remove_from_lru(node);
        let snapshot = self.nodes[node]
            .entry
            .take()
            .expect("selected LRU node keeps its snapshot")
            .snapshot;
        self.entries -= 1;
        self.evictions = self.evictions.saturating_add(1);
        self.prune(node);
        Some(snapshot)
    }

    fn remove_from_lru(&mut self, node: usize) {
        let entry = self.nodes[node].entry.as_ref().expect("LRU entry exists");
        let (previous, next) = (entry.previous, entry.next);
        if let Some(previous) = previous {
            self.nodes[previous]
                .entry
                .as_mut()
                .expect("previous LRU entry exists")
                .next = next;
        } else {
            self.oldest = next;
        }
        if let Some(next) = next {
            self.nodes[next]
                .entry
                .as_mut()
                .expect("next LRU entry exists")
                .previous = previous;
        } else {
            self.newest = previous;
        }
    }

    fn push_recent(&mut self, node: usize) {
        let entry = self.nodes[node].entry.as_mut().expect("LRU entry exists");
        entry.previous = self.newest;
        entry.next = None;
        if let Some(previous) = self.newest {
            self.nodes[previous]
                .entry
                .as_mut()
                .expect("previous LRU entry exists")
                .next = Some(node);
        } else {
            self.oldest = Some(node);
        }
        self.newest = Some(node);
    }

    fn allocate_node(&mut self, parent: usize, block: Arc<[u32]>) -> usize {
        if let Some(index) = self.free_nodes.pop() {
            self.nodes[index] = Node {
                parent: Some(parent),
                block: Some(block),
                ..Node::default()
            };
            index
        } else {
            let index = self.nodes.len();
            self.nodes.push(Node {
                parent: Some(parent),
                block: Some(block),
                ..Node::default()
            });
            index
        }
    }

    fn prune(&mut self, mut node: usize) {
        while node != 0 && self.nodes[node].entry.is_none() && self.nodes[node].children.is_empty()
        {
            let parent = self.nodes[node]
                .parent
                .expect("non-root prefix nodes keep their parent");
            let block = self.nodes[node]
                .block
                .take()
                .expect("non-root prefix node keeps its block");
            self.nodes[parent].children.remove(block.as_ref());
            self.nodes[node] = Node::default();
            self.free_nodes.push(node);
            node = parent;
        }
    }

    pub(crate) const fn len(&self) -> usize {
        self.entries
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
    use super::PrefixCache;

    #[test]
    fn hits_and_replacements_update_lru_order() {
        let mut cache = PrefixCache::new(1);
        cache.insert(&[1], "one");
        cache.insert(&[2], "two");
        cache.insert(&[3], "three");
        assert_eq!(cache.find(&[1, 99], usize::MAX), Some((1, &"one")));
        assert_eq!(cache.insert(&[2], "new two"), Some("two"));
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
            cache.insert(&tokens[..end], end);
        }
        assert_eq!(cache.find(&tokens, 8), Some((8, &8)));
        assert_eq!(cache.find(&tokens, 7), Some((4, &4)));
        assert_eq!(cache.find(&tokens, 0), None);
    }

    #[test]
    fn evicting_ancestors_preserves_descendants_and_reuses_pruned_nodes() {
        let mut cache = PrefixCache::new(1);
        cache.insert(&[1], 1);
        cache.insert(&[1, 2], 2);
        cache.insert(&[1, 2, 3], 3);
        assert_eq!(cache.evict_lru(), Some(1));
        assert_eq!(cache.find(&[1, 2, 99], usize::MAX), Some((2, &2)));
        assert_eq!(cache.evict_lru(), Some(3));
        assert_eq!(cache.evict_lru(), Some(2));
        assert!(cache.nodes[0].children.is_empty());
        assert_eq!(cache.free_nodes.len(), 3);

        cache.insert(&[4, 5], 4);
        assert_eq!(cache.nodes.len(), 4);
        assert_eq!(cache.find(&[4, 5, 99], usize::MAX), Some((2, &4)));
        assert_eq!(cache.evict_lru(), Some(4));
        assert_eq!(cache.oldest, None);
        assert_eq!(cache.newest, None);
    }

    #[test]
    fn wide_cache_eviction_removes_exact_parent_edges() {
        let mut cache = PrefixCache::new(2);
        for id in 0..1024 {
            cache.insert(&[id, id + 1], id);
        }
        for id in 0..1024 {
            assert_eq!(cache.evict_lru(), Some(id));
            assert_eq!(cache.nodes[0].children.len(), 1023 - id as usize);
        }
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.free_nodes.len(), 1024);
    }
}
