use pyo3::prelude::*;

use crate::{common_types::HydroForcing, hydro::HydrologicZone};

const FIND_ROOT_TOL: f64 = 1e-6;
const FIND_ROOT_MAXITER: usize = 50;

pub trait ObjectiveFunctionScalar {
    fn evaluate(&self, x: f64) -> f64;
}

// pub struct PyScalarObjective<'py>(pub Bound<'py, PyAny>);

// impl ObjectiveFunctionScalar for PyScalarObjective {
//     fn evaluate(&self, x: f64) -> f64 {
//         self.0
//             .call1(x)
//             .expect("Python function call failed")
//             .extract()
//             .expect("Could not extract f64 from Python return")
//     }
// }

pub fn find_root_rust<F>(f: F, x_init: f64) -> f64
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
            panic!("Failed to find root");
        }
    }

    return x_1;
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
        return Ok(find_root_rust(residual, s_0));
    }

    // 2. Fallback: It's a pure Python function
    let py_residual = move |s: f64| {
        func.call1((s, s_0, &d, dt))
            .and_then(|res| res.extract::<f64>())
            .expect("Python callback failed")
    };

    Ok(find_root_rust(py_residual, s_0))
}
