//! NumPy SoA transport: acquire array borrows once, retain them through commit.

use numpy::{Element, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use resin_core::logprobs::{LogprobView, TopLogprobView};
use resin_core::output::TokenRows;

use crate::core_error;

// This is the public array schema. Generate only Python ownership, properties,
// constructor plumbing and borrow guards; validation and score views stay below.
macro_rules! forward_output {
    ($(#[$doc:meta])* required { $($required:ident: $required_type:ty => $required_hint:literal),* $(,)? }
     scores { $($score:ident: $score_type:ty => $score_hint:literal),* $(,)? }) => {
        $(#[$doc])*
        #[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
        pub struct ForwardOutput {
            $(#[pyo3(get)] $required: Py<PyArray1<$required_type>>,)*
            $(#[pyo3(get)] $score: Option<Py<PyArray1<$score_type>>>,)*
            state_checkpoints: Vec<u64>,
        }

        #[pymethods]
        impl ForwardOutput {
            #[new]
            #[pyo3(signature = (*, $($required: $required_hint,)* $($score: $score_hint = None,)* state_checkpoints = Vec::new()))]
            #[allow(clippy::too_many_arguments)]
            fn new(
                py: Python<'_>,
                $($required: Py<PyArray1<$required_type>>,)*
                $($score: Option<Py<PyArray1<$score_type>>>,)*
                state_checkpoints: Vec<u64>,
            ) -> PyResult<Self> {
                Self { $($required,)* $($score,)* state_checkpoints }.checked(py)
            }

            #[getter]
            fn state_checkpoints(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
                Ok(pyo3::types::PyTuple::new(py, self.state_checkpoints.iter().copied())?.unbind())
            }

            fn with_state_checkpoints(&self, py: Python<'_>, records: Vec<u64>) -> Self {
                Self {
                    $($required: self.$required.clone_ref(py),)*
                    $($score: self.$score.as_ref().map(|array| array.clone_ref(py)),)*
                    state_checkpoints: records,
                }
            }
        }

        pub struct BorrowedScores<'py> {
            $($score: Option<PyReadonlyArray1<'py, $score_type>>,)*
        }

        impl<'py> BorrowedScores<'py> {
            fn borrow(output: &ForwardOutput, py: Python<'py>) -> PyResult<Self> {
                Ok(Self {
                    $($score: output.$score.as_ref().map(|array| borrow(array, py)).transpose()?,)*
                })
            }
        }
    };
}

forward_output! {
    /// Borrowed NumPy SoA output, valid until the synchronous step returns.
    ///
    /// All arrays must have their exact declared dtype, one dimension, and contiguous
    /// storage. Construction retains the arrays without copying. Do not mutate or
    /// recycle their storage until step() returns. Token rows use token_indptr;
    /// score rows use logprob_indptr (defaults to token_indptr). An empty score row
    /// opts out; every other score row must cover the entire raw output token row.
    /// top_indptr partitions top_token_ids/top_logprobs by scored token. All three
    /// top_* arrays are optional together.
    required {
        token_ids: u32 => "npt.NDArray[np.uint32]",
        token_indptr: u32 => "npt.NDArray[np.uint32]",
    }
    scores {
        logprobs: f32 => "npt.NDArray[np.float32] | None",
        logprob_indptr: u32 => "npt.NDArray[np.uint32] | None",
        top_indptr: u32 => "npt.NDArray[np.uint32] | None",
        top_token_ids: u32 => "npt.NDArray[np.uint32] | None",
        top_logprobs: f32 => "npt.NDArray[np.float32] | None",
        prompt_logprobs: f32 => "npt.NDArray[np.float32] | None",
        prompt_logprob_indptr: u32 => "npt.NDArray[np.uint32] | None",
        prompt_top_indptr: u32 => "npt.NDArray[np.uint32] | None",
        prompt_top_token_ids: u32 => "npt.NDArray[np.uint32] | None",
        prompt_top_logprobs: f32 => "npt.NDArray[np.float32] | None",
        prompt_target_indptr: u32 => "npt.NDArray[np.uint32] | None",
        prompt_target_token_ids: u32 => "npt.NDArray[np.uint32] | None",
        prompt_target_logprobs: f32 => "npt.NDArray[np.float32] | None",
    }
}

impl ForwardOutput {
    fn checked(mut self, py: Python<'_>) -> PyResult<Self> {
        if self.prompt_logprobs.is_some() != self.prompt_logprob_indptr.is_some() {
            return Err(PyValueError::new_err(
                "prompt scores require both prompt_logprobs and prompt_logprob_indptr",
            ));
        }
        for (name, columns) in [
            (
                "prompt_top",
                [
                    self.prompt_top_indptr.is_some(),
                    self.prompt_top_token_ids.is_some(),
                    self.prompt_top_logprobs.is_some(),
                ],
            ),
            (
                "prompt_target",
                [
                    self.prompt_target_indptr.is_some(),
                    self.prompt_target_token_ids.is_some(),
                    self.prompt_target_logprobs.is_some(),
                ],
            ),
        ] {
            let count = columns.into_iter().filter(|present| *present).count();
            if count != 0 && (count != 3 || self.prompt_logprobs.is_none()) {
                return Err(PyValueError::new_err(format!(
                    "{name} scores require all three arrays and prompt_logprobs"
                )));
            }
        }
        let top_count = usize::from(self.top_indptr.is_some())
            + usize::from(self.top_token_ids.is_some())
            + usize::from(self.top_logprobs.is_some());
        if top_count != 0 && top_count != 3 {
            return Err(PyValueError::new_err(
                "top logprobs require all three top_* arrays",
            ));
        }
        if self.logprobs.is_none() && (self.logprob_indptr.is_some() || top_count != 0) {
            return Err(PyValueError::new_err(
                "score offsets and alternatives require logprobs",
            ));
        }
        self.logprob_indptr = self.logprobs.as_ref().map(|_| {
            self.logprob_indptr
                .take()
                .unwrap_or_else(|| self.token_indptr.clone_ref(py))
        });
        Ok(self)
    }
}

fn borrow<'py, T: Element>(
    array: &Py<PyArray1<T>>,
    py: Python<'py>,
) -> PyResult<PyReadonlyArray1<'py, T>> {
    let guard = array
        .bind(py)
        .try_readonly()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    guard.as_slice().map_err(|_| {
        PyValueError::new_err(
            "ForwardOutput arrays must be contiguous; no implicit copy is performed",
        )
    })?;
    Ok(guard)
}

fn top_view<'a>(
    indptr: &'a Option<PyReadonlyArray1<'_, u32>>,
    token_ids: &'a Option<PyReadonlyArray1<'_, u32>>,
    values: &'a Option<PyReadonlyArray1<'_, f32>>,
) -> Option<TopLogprobView<'a>> {
    Some(TopLogprobView {
        indptr: indptr.as_ref()?.as_slice().unwrap(),
        token_ids: token_ids.as_ref()?.as_slice().unwrap(),
        values: values.as_ref()?.as_slice().unwrap(),
    })
}

pub struct BorrowedOutput<'py> {
    pub tokens: BorrowedTokens<'py>,
    pub scores: BorrowedScores<'py>,
    pub state_checkpoints: Vec<u64>,
}

pub struct BorrowedTokens<'py> {
    token_ids: PyReadonlyArray1<'py, u32>,
    indptr: PyReadonlyArray1<'py, u32>,
}

impl BorrowedTokens<'_> {
    pub fn view(&self) -> PyResult<TokenRows<'_>> {
        TokenRows::new(
            self.token_ids.as_slice().unwrap(),
            self.indptr.as_slice().unwrap(),
        )
        .map_err(core_error)
    }
}

impl BorrowedScores<'_> {
    // Contiguity is checked when guards are acquired. Guards retain the arrays
    // and their owners throughout synchronous completion.
    pub fn logprobs(&self) -> Option<LogprobView<'_>> {
        Some(LogprobView {
            indptr: self.logprob_indptr.as_ref()?.as_slice().unwrap(),
            values: self.logprobs.as_ref()?.as_slice().unwrap(),
            top: top_view(&self.top_indptr, &self.top_token_ids, &self.top_logprobs),
            targets: None,
        })
    }

    pub fn prompt_logprobs(&self) -> Option<LogprobView<'_>> {
        Some(LogprobView {
            indptr: self.prompt_logprob_indptr.as_ref()?.as_slice().unwrap(),
            values: self.prompt_logprobs.as_ref()?.as_slice().unwrap(),
            top: top_view(
                &self.prompt_top_indptr,
                &self.prompt_top_token_ids,
                &self.prompt_top_logprobs,
            ),
            targets: top_view(
                &self.prompt_target_indptr,
                &self.prompt_target_token_ids,
                &self.prompt_target_logprobs,
            ),
        })
    }
}

impl ForwardOutput {
    pub fn borrow<'py>(&self, py: Python<'py>) -> PyResult<BorrowedOutput<'py>> {
        Ok(BorrowedOutput {
            tokens: BorrowedTokens {
                token_ids: borrow(&self.token_ids, py)?,
                indptr: borrow(&self.token_indptr, py)?,
            },
            scores: BorrowedScores::borrow(self, py)?,
            state_checkpoints: self.state_checkpoints.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_schema_preserves_array_aliases_and_guards_every_column() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = pyo3::types::PyDict::new(py);
            let mut ids = Vec::new();
            for name in [
                "token_ids",
                "token_indptr",
                "logprob_indptr",
                "top_indptr",
                "top_token_ids",
                "prompt_logprob_indptr",
                "prompt_top_indptr",
                "prompt_top_token_ids",
                "prompt_target_indptr",
                "prompt_target_token_ids",
            ] {
                let array = PyArray1::from_vec(py, vec![0u32, 1]);
                kwargs.set_item(name, &array).unwrap();
                ids.push(array);
            }
            let mut values = Vec::new();
            for name in [
                "logprobs",
                "top_logprobs",
                "prompt_logprobs",
                "prompt_top_logprobs",
                "prompt_target_logprobs",
            ] {
                let array = PyArray1::from_vec(py, vec![-0.5f32]);
                kwargs.set_item(name, &array).unwrap();
                values.push(array);
            }
            let output = py
                .get_type::<ForwardOutput>()
                .call((), Some(&kwargs))
                .unwrap()
                .cast_into::<ForwardOutput>()
                .unwrap();
            for (name, array) in kwargs.iter() {
                let name = name.cast::<pyo3::types::PyString>().unwrap();
                assert!(output.getattr(name).unwrap().is(&array));
            }
            let copy = output.borrow().with_state_checkpoints(py, vec![7]);
            assert!(copy.token_ids.bind(py).is(&ids[0]));
            assert!(
                copy.prompt_logprobs
                    .as_ref()
                    .unwrap()
                    .bind(py)
                    .is(&values[2])
            );
            assert!(output.borrow().state_checkpoints.is_empty());
            let borrowed = copy.borrow(py).unwrap();
            assert_eq!(borrowed.state_checkpoints, [7]);
            assert!(borrowed.scores.logprobs().unwrap().top.is_some());
            assert!(borrowed.scores.prompt_logprobs().unwrap().targets.is_some());
            // Separate allocations ensure a missing column guard cannot be
            // masked by a guard on another column that aliases the same array.
            assert!(ids.iter().all(|array| array.try_readwrite().is_err()));
            assert!(values.iter().all(|array| array.try_readwrite().is_err()));
            drop(borrowed);
            assert!(ids.iter().all(|array| array.try_readwrite().is_ok()));
            assert!(values.iter().all(|array| array.try_readwrite().is_ok()));
        });
    }

    #[test]
    fn token_views_keep_numpy_storage_borrowed_until_completion_finishes() {
        Python::attach(|py| {
            let ids = PyArray1::from_vec(py, vec![7, u32::MAX, 9]).unbind();
            let offsets = PyArray1::from_vec(py, vec![0, 1, 3]).unbind();
            let borrowed = BorrowedTokens {
                token_ids: borrow(&ids, py).unwrap(),
                indptr: borrow(&offsets, py).unwrap(),
            };
            let rows = borrowed.view().unwrap();
            assert_eq!(rows.row(1), [u32::MAX, 9]);
            assert_eq!(
                rows.row(0).as_ptr(),
                borrowed.token_ids.as_slice().unwrap().as_ptr()
            );
            assert!(ids.bind(py).try_readwrite().is_err());
            assert!(offsets.bind(py).try_readwrite().is_err());
            drop(borrowed);
            assert!(ids.bind(py).try_readwrite().is_ok());
            assert!(offsets.bind(py).try_readwrite().is_ok());
        });
    }
}
