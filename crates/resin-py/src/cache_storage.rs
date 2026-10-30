use pyo3::prelude::*;
use resin_core::resources::paged_kv::{
    PrefixStorage, PrefixStorageGroup, PrefixStorageOperation, StorageOp,
};

#[derive(Debug)]
pub(crate) struct PythonPrefixStorage(pub Py<PyAny>);
#[derive(Debug)]
struct Operation(Py<PyAny>, bool);
impl PrefixStorageOperation for Operation {
    fn poll(&mut self, wait: bool) -> Option<u32> {
        Python::try_attach(|py| self.0.bind(py).call_method1("poll", (wait,))?.extract())
            .and_then(|r: PyResult<Option<u32>>| r.unwrap_or(self.1.then_some(0)))
    }
}
impl PythonPrefixStorage {
    fn wrap(&self, py: Python<'_>, result: PyResult<Bound<'_, PyAny>>, lookup: bool) -> StorageOp {
        Box::new(Operation(
            result
                .map(Bound::unbind)
                .unwrap_or_else(|_| self.0.clone_ref(py)),
            lookup,
        ))
    }
}
impl PrefixStorage for PythonPrefixStorage {
    fn lookup(
        &mut self,
        tokens: &[u32],
        maximum: u32,
        alignment: u32,
        salt: Option<&str>,
    ) -> StorageOp {
        Python::attach(|py| {
            self.wrap(
                py,
                self.0
                    .bind(py)
                    .call_method1("lookup", (tokens, maximum, alignment, salt)),
                true,
            )
        })
    }
    fn transfer(
        &mut self,
        tokens: &[u32],
        groups: &[PrefixStorageGroup],
        load: bool,
        salt: Option<&str>,
    ) -> StorageOp {
        Python::attach(|py| {
            self.wrap(
                py,
                self.0
                    .bind(py)
                    .call_method1("transfer", (tokens, groups, load, salt)),
                false,
            )
        })
    }
}
