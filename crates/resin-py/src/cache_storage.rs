//! Synchronous callbacks borrow existing checkpoint owners until return.
use pyo3::prelude::*;
use resin_core::resources::paged_kv::{PrefixStorage, PrefixStorageGroup};

#[derive(Debug)]
pub(crate) struct PythonPrefixStorage(pub Py<PyAny>);

impl PrefixStorage for PythonPrefixStorage {
    fn lookup(&mut self, tokens: &[u32], maximum: u32, alignment: u32) -> u32 {
        Python::attach(|py| {
            self.0
                .bind(py)
                .call_method1("lookup", (tokens, maximum, alignment))?
                .extract()
        })
        .unwrap_or(0)
    }

    fn transfer(&mut self, tokens: &[u32], groups: &[PrefixStorageGroup], load: bool) -> bool {
        Python::attach(|py| {
            self.0
                .bind(py)
                .call_method1("transfer", (tokens, groups, load))?
                .extract()
        })
        .unwrap_or(false)
    }
}
