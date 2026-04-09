use std::fmt;

use ndarray_linalg::Solve;
use numpy::{
    ndarray::{Array1, Array2},
    PyArray2, PyArrayMethods,
};
use pyo3::{
    exceptions::{PyException, PyRuntimeError, PyValueError},
    prelude::*,
    IntoPyObjectExt, PyErrArguments,
};

use crate::{common_types::HydroForcing, hydro::HydrologicZone};

const FIND_ROOT_TOL: f64 = 1e-8;
const FIND_ROOT_MAXITER: usize = 100;
const MULTI_MAXITER: usize = 100;
const MULTI_TOL: f64 = 1e-12;
const APPROX_FPRIME_DX: f64 = 1e-8;
const APPROX_FPRIME_REL_DX: f64 = 1e-3;
const F_PRIME_MIN_VAL: f64 = 1e-22;
const TIKHONOV_LAMBDA: f64 = 1e-12;

pub trait ObjectiveFunctionScalar {
    fn evaluate(&self, x: f64) -> f64;
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum ScalarRootFindingError {
    IterationError(),
    NanError(),
}

#[pyclass(extends=PyException)]
#[derive(Debug, Clone)]
pub struct IterationError;

#[pyclass(extends=PyException, subclass)]
#[derive(Clone)] // Clone is useful for the conversion layer
pub struct OptimizationError {
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub final_err: f64,
    #[pyo3(get)]
    pub last_x: Vec<f64>,
    #[pyo3(get)]
    pub last_f_x: Vec<f64>,
    #[pyo3(get)]
    pub initial_x: Vec<f64>,
    #[pyo3(get)]
    pub jacobian: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub message: String,
}

impl OptimizationError {
    pub fn from_state(state: OptimizerState, message: String) -> Self {
        let jac = arr_to_vec(state.jacobian.clone());

        Self {
            iterations: state.iteration,
            final_err: state.error,
            last_x: state.final_x.to_vec(),
            last_f_x: state.last_f_x.to_vec(),
            initial_x: state.initial_x.to_vec(),
            jacobian: jac,
            message,
        }
    }
}

#[pymethods]
impl OptimizationError {
    #[new]
    pub fn new(
        iterations: usize,
        final_err: f64,
        last_x: Vec<f64>,
        last_f_x: Vec<f64>,
        initial_x: Vec<f64>,
        jacobian: Vec<Vec<f64>>,
        message: String,
    ) -> Self {
        Self {
            iterations,
            final_err,
            last_x,
            last_f_x,
            initial_x,
            jacobian,
            message,
        }
    }

    pub fn __str__(&self) -> String {
        format!(
            "{} (iters: {}, last_x: {:?})",
            self.message, self.iterations, self.last_x
        )
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct OptimizerState {
    iteration: usize,
    final_x: Array1<f64>,
    last_f_x: Array1<f64>,
    initial_x: Array1<f64>,
    jacobian: Array2<f64>,
    error: f64,
}

#[pyclass(extends=PyException)]
#[derive(Debug, Clone)]
pub struct OtherError;

#[derive(Debug, Clone)]
pub enum RootFindingError {
    IterationError(OptimizerState),
    LinearSystemError(OptimizerState),
    Other(OptimizerState),
}

fn arr_to_vec(x: Array2<f64>) -> Vec<Vec<f64>> {
    let arrs: Vec<Vec<f64>> = x.rows().into_iter().map(|x| x.to_vec()).collect();
    arrs
}

impl PyErrArguments for RootFindingError {
    fn arguments(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            RootFindingError::IterationError(x) => {
                let args = (
                    x.iteration,
                    x.error,
                    x.final_x.to_vec(),
                    x.initial_x.to_vec(),
                    arr_to_vec(x.jacobian),
                    "Exceeded maximum iterations without finding root".to_string(),
                );

                args.into_py_any(py).unwrap()
            }
            RootFindingError::LinearSystemError(x) => {
                let args = (
                    x.iteration,
                    x.error,
                    x.final_x.to_vec(),
                    x.initial_x.to_vec(),
                    arr_to_vec(x.jacobian),
                    "Linear algebra when solving system".to_string(),
                );

                args.into_py_any(py).unwrap()
            }
            RootFindingError::Other(x) => {
                let args = (
                    x.iteration,
                    x.error,
                    x.final_x.to_vec(),
                    x.initial_x.to_vec(),
                    arr_to_vec(x.jacobian),
                    "Other error when solving optimizer".to_string(),
                );

                args.into_py_any(py).unwrap()
            }
        }
    }
}

impl pyo3::PyErrArguments for OptimizationError {
    fn arguments(self, py: Python<'_>) -> Py<PyAny> {
        (
            self.iterations,
            self.final_err,
            self.last_x,
            self.last_f_x,
            self.initial_x,
            self.jacobian,
            self.message,
        )
            .into_py_any(py)
            .unwrap()
    }
}

impl From<RootFindingError> for PyErr {
    fn from(err: RootFindingError) -> Self {
        match &err {
            RootFindingError::IterationError(x) => PyErr::new::<OptimizationError, _>(err),
            RootFindingError::LinearSystemError(x) => PyErr::new::<OptimizationError, _>(err),
            RootFindingError::Other(x) => PyErr::new::<OptimizationError, _>(err),
        }
    }
}

// #[pyclass(extends=PyException)]
#[derive(Debug, Clone)]
pub struct MatMulError;

impl fmt::Display for ScalarRootFindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Failed to find root")
    }
}

impl fmt::Display for RootFindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg: String = match self {
            Self::IterationError(x) => {
                format!("Maximum iterations reached")
            }
            Self::LinearSystemError(s) => {
                format!(
                    "Failed to solve linear system on iteration {} with:\njac={}\nx={}\nx0={}",
                    s.iteration, s.jacobian, s.final_x, s.initial_x
                )
            }
            Self::Other(x) => {
                format!("Other error")
            }
        };
        write!(f, "{}", msg)
    }
}

impl fmt::Display for MatMulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Failed to find multiply matrix and vector")
    }
}

pub fn find_root_rust<F>(f: F, x_init: f64) -> Result<f64, ScalarRootFindingError>
where
    F: Fn(f64) -> f64,
{
    let mut x_0 = x_init;
    let mut x_1 = x_0 + 0.1;
    let mut fx_0 = f(x_0);
    let mut fx_1 = f(x_1);
    let mut err = fx_1.abs();
    let mut counter = 0;

    while err > FIND_ROOT_TOL {
        if (fx_0 - fx_1).abs() < 1e-12 {
            break;
        }

        let x_n = x_1 - fx_1 * (x_1 - x_0) / (fx_1 - fx_0);
        x_0 = x_1;
        fx_0 = fx_1;

        x_1 = x_n;
        fx_1 = f(x_n);

        err = fx_1.abs();
        counter += 1;
        if counter >= FIND_ROOT_MAXITER {
            return Err(ScalarRootFindingError::IterationError());
        }

        if x_1.is_nan() {
            return Err(ScalarRootFindingError::NanError());
        }
    }

    Ok(x_1)
}

pub fn bisect_rust<F>(f: F, x_max: f64) -> Result<f64, ScalarRootFindingError>
where
    F: Fn(f64) -> f64,
{
    let mut x_l = 0.0;
    let mut x_r = x_max;
    let mut x_m = 0.5 * (x_l + x_r);
    let mut fx_l = f(x_l);
    let mut fx_r = f(x_r);
    let mut err = fx_r.abs();
    let mut counter = 0;

    if fx_l.signum() == fx_r.signum() {
        return Err(ScalarRootFindingError::NanError());
    }

    while err > FIND_ROOT_TOL {
        counter += 1;
        if counter >= FIND_ROOT_MAXITER {
            return Err(ScalarRootFindingError::IterationError());
        }

        if (fx_l - fx_r).abs() < 1e-12 {
            break;
        }
        let fx_m: f64 = f(x_m);

        err = fx_m.abs();
        if err <= FIND_ROOT_TOL {
            return Ok(x_m);
        }

        if fx_l.signum() == fx_m.signum() {
            // Root is to the right of the midpoint
            x_l = x_m;
            fx_l = fx_m;
            x_m = 0.5 * (x_l + x_r);
            continue;
        } else {
            // Root is to the left of the midpoint
            x_r = x_m;
            fx_r = fx_m;
            x_m = 0.5 * (x_l + x_r);
            continue;
        }
    }

    Ok(x_m)
}

#[pyfunction]
pub fn find_root(
    py: Python<'_>,
    func: Bound<'_, PyAny>,
    s_0: f64,
    d: Bound<'_, PyAny>,
    dt: f64,
) -> PyResult<f64> {
    // 1. Try to see if 'func' is actually one of our Rust Zone classes
    // (This is the most performant way)
    if let Ok(zone) = func.cast::<HydrologicZone>() {
        let zone_ref = zone.borrow();
        let forcing = d.extract::<HydroForcing>()?; // Extract Rust struct from Python object

        let residual = |s: f64| zone_ref.__implicit_eulers_func(s, s_0, &forcing, dt);
        match find_root_rust(residual, s_0) {
            Ok(v) => return Ok(v),
            Err(_) => {
                return Err(PyValueError::new_err(
                    "Failed to find scalar root in my own function",
                ))
            }
        }
    }

    // 2. Fallback: It's a pure Python function
    let py_residual = move |s: f64| {
        func.call1((s, s_0, &d, dt))
            .and_then(|res| res.extract::<f64>())
            .expect("Python callback failed")
    };

    match find_root_rust(py_residual, s_0) {
        Ok(v) => return Ok(v),
        Err(_) => {
            return Err(PyValueError::new_err(
                "Failed to find scalar root in my own function",
            ))
        }
    }
}

/// Calculate the null space using Scipy
pub fn null_space_scipy(mat: &Array2<f64>) -> PyResult<Array2<f64>> {
    let res: PyResult<Array2<f64>> = Python::attach(|py| {
        let linalg =
            PyModule::import(py, "scipy.linalg").expect("Failed to get scipy.linalg module");
        let py_mat = PyArray2::from_array(py, &mat);
        let null_space_res = linalg
            .getattr("null_space")?
            .call1((&py_mat,))?
            .cast::<PyArray2<f64>>()
            .map_err(|e| {
                let msg = format!(
                    "Failed to convert Scipy result to Rust type: {}",
                    e.to_string()
                );
                return PyRuntimeError::new_err(msg);
            })?
            .to_owned_array();
        return Ok(null_space_res);
    });
    res
}

/// Calculate the null space using Scipy
pub fn pinv_scipy(mat: &Array2<f64>) -> PyResult<Array2<f64>> {
    let res: PyResult<Array2<f64>> = Python::attach(|py| {
        let linalg =
            PyModule::import(py, "scipy.linalg").expect("Failed to get scipy.linalg module");
        let py_mat = PyArray2::from_array(py, &mat);

        let pinv_arr: Array2<f64> = linalg
            .getattr("pinv")?
            .call1((&py_mat,))?
            .cast::<PyArray2<f64>>()
            .map_err(|e| {
                let msg = format!(
                    "Failed to convert Scipy result to Rust type: {}",
                    e.to_string()
                );
                return PyRuntimeError::new_err(msg);
            })?
            .to_owned_array();

        Ok(pinv_arr)
    });

    res
}

pub fn approx_fprime<F>(f: F, x: &Array1<f64>, verbose: bool) -> Array2<f64>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let n: usize = x.len();
    let mut jac_x: Array2<f64> = Array2::zeros((n, n));

    for (i, x_i) in x.iter().enumerate() {
        let dx = match x_i.abs() < F_PRIME_MIN_VAL {
            true => APPROX_FPRIME_DX,
            false => APPROX_FPRIME_REL_DX * x_i.abs(),
        };

        let mut x_up: Array1<f64> = x.clone();
        let mut x_dn: Array1<f64> = x.clone();
        x_up[i] = x_i + dx;
        x_dn[i] = x_i - dx;

        let fx_up: Array1<f64> = f(&x_up);
        let fx_dn: Array1<f64> = f(&x_dn);

        let jac_x_i: Array1<f64> = (&fx_up - &fx_dn) / (2.0 * dx);

        if verbose {
            Python::attach(|py| {
                py.detach(|| {
                    eprintln!("i={}, x_i={:e}, dx={:e}", i, x_i, dx);
                    eprintln!("x_up={}", &x_up);
                    eprintln!("x_dn={}", &x_dn);
                    eprintln!("fx_up={}", &fx_up);
                    eprintln!("fx_dn={}", &fx_dn);
                    eprintln!("jac_x_i={}", &jac_x_i);
                })
            });
        }

        jac_x.column_mut(i).assign(&jac_x_i);
    }

    jac_x
}

/// Find the root of the linear problem using Damped Least squares with a constant regularization term
pub fn find_root_multi<'a, F>(f: &F, x_0: Array1<f64>, verbose: bool) -> PyResult<Array1<f64>>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut x: Array1<f64> = x_0.clone();
    let mut f_x: Array1<f64> = f(&x);
    let mut err: f64 = f_x.abs().mean().unwrap();
    let mut jac_x: Array2<f64> = Array2::zeros((1, 1));

    for i in 0..MULTI_MAXITER {
        if err <= MULTI_TOL {
            return Ok(x);
        }

        jac_x = approx_fprime(f, &x, verbose);
        let jac_x_t = jac_x.t();
        let mut a_mat: Array2<f64> = jac_x_t.dot(&jac_x);
        for i in 0..a_mat.nrows() {
            a_mat[(i, i)] += TIKHONOV_LAMBDA;
        }
        let b: Array1<f64> = jac_x_t.dot(&f_x);

        if x.is_any_nan() || f_x.is_any_nan() || jac_x.is_any_nan() {
            let final_state = OptimizerState {
                iteration: i,
                final_x: x.clone(),
                last_f_x: f_x.clone(),
                initial_x: x_0.clone(),
                jacobian: jac_x.clone(),
                error: err,
            };
            let err = OptimizationError::from_state(
                final_state,
                "Got NaN values in an array".to_string(),
            );
            return Err(PyErr::new::<OptimizationError, _>(err));
        }

        let step: Array1<f64> = match a_mat.solve(&b) {
            Ok(v) => v,
            Err(e) => {
                return {
                    let final_state = OptimizerState {
                        iteration: i,
                        final_x: x.clone(),
                        last_f_x: f_x.clone(),
                        initial_x: x_0.clone(),
                        jacobian: jac_x.clone(),
                        error: err,
                    };
                    let err = OptimizationError::from_state(
                        final_state,
                        format!("Linear algebra error: {}", e.to_string()),
                    );
                    Err(PyErr::new::<OptimizationError, _>(err))
                }
            }
        };
        let x_new: Array1<f64> = &x - &step;
        x = x_new;
        f_x = f(&x);
        err = (f_x.pow2()).mean().unwrap();

        if verbose {
            Python::attach(|py| {
                py.detach(|| {
                    eprintln!("x after i={}: {}", i, &x);
                    eprintln!("err: {}", err);
                    eprintln!("\n\n");
                })
            });
        }
    }

    let final_state = OptimizerState {
        iteration: MULTI_MAXITER,
        final_x: x.clone(),
        last_f_x: f_x.clone(),
        initial_x: x_0.clone(),
        jacobian: jac_x.clone(),
        error: err,
    };

    let err = OptimizationError::from_state(final_state, "Exceeded maximum".to_string());
    Err(PyErr::new::<OptimizationError, _>(err))
}

pub fn matmul(a: &Array2<f64>, x: &Array1<f64>) -> Result<Array1<f64>, MatMulError> {
    let n_out = a.shape()[0];
    let mut output: Array1<f64> = Array1::zeros(n_out);

    if a.shape()[1] != x.shape()[0] {
        return Err(MatMulError);
    }

    for i in 0..n_out {
        let mut row_sum = 0.0;
        for (j, x_j) in x.iter().enumerate() {
            row_sum += a[(i, j)] * x_j;
        }

        output[i] = row_sum;
    }

    Ok(output)
}
