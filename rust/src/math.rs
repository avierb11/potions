use std::fmt;

use ndarray_linalg::Solve;
use numpy::{
    ndarray::{Array1, Array2},
    PyArray2, PyArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*};

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
pub struct RootFindingError;

impl fmt::Display for ScalarRootFindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Failed to find root")
    }
}

impl fmt::Display for RootFindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Failed to find root")
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
pub fn null_space_scipy(mat: &Array2<f64>) -> Array2<f64> {
    Python::attach(|py| {
        let linalg =
            PyModule::import(py, "scipy.linalg").expect("Failed to get scipy.linalg module");
        let py_mat = PyArray2::from_array(py, &mat);
        let null_space_res: Array2<f64> = linalg
            .getattr("null_space")
            .expect("Failed to get `null_space` from scipy.linalg")
            .call1((&py_mat,))
            .expect("Failed to call `null_space`")
            .cast::<PyArray2<f64>>()
            .expect("Failed to cast Python null space result to rust type")
            .to_owned_array();

        return null_space_res;
    });

    panic!("Failed to find result")
}

/// Calculate the null space using Scipy
pub fn pinv_scipy(mat: &Array2<f64>) -> Array2<f64> {
    Python::attach(|py| {
        let linalg =
            PyModule::import(py, "scipy.linalg").expect("Failed to get scipy.linalg module");
        let py_mat = PyArray2::from_array(py, &mat);
        let null_space_res: Array2<f64> = linalg
            .getattr("pinv")
            .expect("Failed to get `null_space` from scipy.linalg")
            .call1((&py_mat,))
            .expect("Failed to call `null_space`")
            .cast::<PyArray2<f64>>()
            .expect("Failed to cast Python null space result to rust type")
            .to_owned_array();

        return null_space_res;
    });

    panic!("Failed to find result")
}

pub fn approx_fprime<F>(f: F, x: &Array1<f64>) -> Array2<f64>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    unimplemented!()
}

pub fn find_root_multi<'a, F>(f: &F, x_0: Array1<f64>) -> Result<Array1<f64>, RootFindingError>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut x: Array1<f64> = x_0.clone();
    let mut f_x: Array1<f64> = f(&x);
    let mut err: f64 = (f_x.pow2()).mean().unwrap();

    for _ in 0..MULTI_MAXITER {
        if err <= MULTI_TOL {
            return Ok(x);
        }

        let jac_x: Array2<f64> = approx_fprime(f, &x);
        let step: Array1<f64> = match jac_x.solve(&f_x) {
            Ok(v) => v,
            Err(_) => return Err(RootFindingError),
        };
        let x_new: Array1<f64> = &x - &step;
        x = x_new;
        f_x = f(&x);
        err = (f_x.pow2()).mean().unwrap();
    }
    Err(RootFindingError)
}
