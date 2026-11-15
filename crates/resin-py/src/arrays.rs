//! Read-only NumPy views over immutable native plan metadata.

use numpy::ndarray::ArrayView1;
use numpy::{Element, PyArray1, PyArrayLike1, PyArrayMethods};
use pyo3::prelude::*;

/// Borrow native storage without copying and keep its Python owner alive.
pub(crate) fn readonly_view<'py, T: Element>(
    owner: Bound<'py, PyAny>,
    values: &[T],
) -> Bound<'py, PyArray1<T>> {
    let view = ArrayView1::from(values);
    // SAFETY: every caller passes a frozen Python owner whose backing vector is
    // never mutated or reallocated. NumPy retains `owner` as the base object.
    let array = unsafe { PyArray1::borrow_from_array(&view, owner) };
    drop(array.readwrite().make_nonwriteable());
    array
}

pub(crate) fn owned_vec<T: Copy + Element>(values: PyArrayLike1<'_, T>) -> Vec<T> {
    values.as_array().iter().copied().collect()
}

// Declare the element type and backing field once for each immutable array.
// Custom properties and methods remain ordinary Rust in the same impl.
macro_rules! array_methods {
    (impl $class:ident {
        arrays { $($(#[$doc:meta])* $name:ident: $element:ty => $($field:ident).+;)* }
        $($methods:item)*
    }) => {
        #[pyo3::pymethods]
        impl $class {
            $($(#[$doc])* #[getter]
            fn $name<'py>(this: pyo3::Bound<'py, Self>) -> pyo3::Bound<'py, numpy::PyArray1<$element>> {
                $crate::arrays::readonly_view(this.clone().into_any(), &this.get().$($field).+)
            })*
            $($methods)*
        }
    };
}
pub(crate) use array_methods;
