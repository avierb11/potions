use std::fmt;

use ndarray_linalg::Solve;
use numpy::{
    ndarray::{Array1, Array2, ShapeError},
    PyArray2, PyArrayMethods,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

use crate::{common_types::HydroForcing, hydro::HydrologicZone};

const FIND_ROOT_TOL: f64 = 1e-6;
const FIND_ROOT_MAXITER: usize = 50;
const MULTI_MAXITER: usize = 100;
const MULTI_TOL: f64 = 1e-6;
const APPROX_FPRIME_DX: f64 = 1e-3;

pub trait ObjectiveFunctionScalar {
    fn evaluate(&self, x: f64) -> f64;
}

#[derive(Debug, Clone)]
pub struct ScalarRootFindingError;

#[derive(Debug, Clone)]
pub struct IterationError;

#[derive(Debug, Clone)]
pub struct LinearSystemError {
    pub iter: usize,
    pub jac: Array2<f64>,
    pub x: Array1<f64>,
    pub x_0: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct OtherError;

#[derive(Debug, Clone)]
pub enum RootFindingError {
    IterationError(IterationError),
    LinearSystemError(LinearSystemError),
    Other(OtherError),
}

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
            Self::IterationError(s) => {
                format!("Maximum iterations reached")
            }
            Self::LinearSystemError(s) => {
                format!(
                    "Failed to solve linear system on iteration {} with:\njac={}\nx={}\nx0={}",
                    s.iter, s.jac, s.x, s.x_0
                )
            }
            Self::Other(_) => {
                format!("Other error")
            }
        };
        write!(f, "{}", msg)
    }
}

impl fmt::Display for LinearSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Failed to solve linear system")
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
            return Err(ScalarRootFindingError);
        }
    }

    Ok(x_1)
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
    if let Ok(zone) = func.downcast::<HydrologicZone>() {
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

pub fn approx_fprime<F>(f: F, x: &Array1<f64>) -> Array2<f64>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let n: usize = x.len();
    let mut jac_x: Array2<f64> = Array2::zeros((n, n));

    for (i, x_i) in x.iter().enumerate() {
        let mut x_up: Array1<f64> = x.clone();
        let mut x_dn: Array1<f64> = x.clone();
        x_up[i] = x_i + APPROX_FPRIME_DX;
        x_dn[i] = x_i - APPROX_FPRIME_DX;

        let fx_up: Array1<f64> = f(&x_up);
        let fx_dn: Array1<f64> = f(&x_dn);

        let jac_x_i: Array1<f64> = (fx_up - fx_dn) / (2.0 * APPROX_FPRIME_DX);
        jac_x.column_mut(i).assign(&jac_x_i);
    }

    jac_x
}

pub fn find_root_multi<'a, F>(f: &F, x_0: Array1<f64>) -> Result<Array1<f64>, RootFindingError>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut x: Array1<f64> = x_0.clone();
    let mut f_x: Array1<f64> = f(&x);
    let mut err: f64 = (f_x.pow2()).mean().unwrap();

    for i in 0..MULTI_MAXITER {
        if err <= MULTI_TOL {
            // eprintln!("Found solution...");
            return Ok(x);
        }

        let jac_x: Array2<f64> = approx_fprime(f, &x);
        let step: Array1<f64> = match jac_x.solve(&f_x) {
            Ok(v) => v,
            Err(_) => {
                return {
                    let err = LinearSystemError {
                        iter: i,
                        jac: jac_x,
                        x: x,
                        x_0: x_0,
                    };
                    Err(RootFindingError::LinearSystemError(err))
                }
            }
        };
        let x_new: Array1<f64> = &x - &step;
        x = x_new;
        f_x = f(&x);
        err = (f_x.pow2()).mean().unwrap();
    }

    dbg!(&x_0);
    dbg!(&x);
    dbg!(&err);

    Err(RootFindingError::IterationError(IterationError))
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
