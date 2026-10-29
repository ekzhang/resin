use pyo3::prelude::*;
use resin_core::resources::paged_kv::{
    PrefixStorage, PrefixStorageGroup, PrefixStorageOperation, StorageOp,
};

#[derive(Debug)]
pub(crate) struct PythonPrefixStorage(pub Py<PyAny>, pub Py<PyAny>);
#[derive(Debug)]
struct Operation {
    future: Option<Py<PyAny>>,
    _owner: PythonPrefixStorage,
    lookup: bool,
}
impl PrefixStorageOperation for Operation {
    fn poll(&mut self, wait: bool) -> Option<u32> {
        let Some(operation) = &self.future else {
            return self.lookup.then_some(0);
        };
        Python::try_attach(|py| operation.bind(py).call_method1("poll", (wait,))?.extract())
            .and_then(|result: PyResult<Option<u32>>| result.unwrap_or(self.lookup.then_some(0)))
    }
}
impl PythonPrefixStorage {
    fn wrap(&self, py: Python<'_>, result: PyResult<Bound<'_, PyAny>>, lookup: bool) -> StorageOp {
        Box::new(Operation {
            future: result.ok().map(Bound::unbind),
            _owner: Self(self.0.clone_ref(py), self.1.clone_ref(py)),
            lookup,
        })
    }
}
impl PrefixStorage for PythonPrefixStorage {
    fn lookup(&mut self, tokens: &[u32], maximum: u32, alignment: u32) -> StorageOp {
        Python::attach(|py| {
            self.wrap(
                py,
                self.0
                    .bind(py)
                    .call_method1("lookup", (tokens, maximum, alignment)),
                true,
            )
        })
    }
    fn transfer(&mut self, tokens: &[u32], groups: &[PrefixStorageGroup], load: bool) -> StorageOp {
        Python::attach(|py| {
            let result = self.1.bind(py).call1((groups,)).and_then(|buffers| {
                self.0.bind(py).call_method1(
                    if load { "load" } else { "store" },
                    (tokens, groups, buffers),
                )
            });
            self.wrap(py, result, false)
        })
    }
}
