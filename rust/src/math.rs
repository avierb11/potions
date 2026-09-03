use faer::prelude::*;
// Solver types need explicit imports
use faer::linalg::solvers::{PartialPivLu, Qr};
use std::fmt;

/// Solve `a · x = b`, where `a` is a square `Array2` and `b` a vector. Returns
/// the solution as a fresh `Array1`.
///
/// The system matrix here is always a dense array built by elementwise
/// arithmetic on fresh ndarrays, so it is contiguous (standard / row-major, C
/// order). We coerce it to a standard-layout COW view and hand faer a
/// **zero-copy** `MatRef::from_row_major_slice` instead of an element-by-element
/// `O(n²)` copy — the old path re-copied every entry on *every* Newton /
/// Levenberg–Marquardt iteration. The rhs is taken by a cheap `to_vec()` so the
/// solver can write the solution in place.
fn solve_f64(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
    let (nrows, ncols) = a.dim();
    // The system matrix is always contiguous. `as_standard_layout` returns a
    // zero-copy COW view when the array is already row-major (the common case)
    // and clones only if it ever arrives in column-major order, after which
    // `as_slice` is the flat **row-major** buffer that `from_row_major_slice`
    // expects. (Feeding a row-major buffer to the column-major constructor is
    // the classic silent-misread bug: it scrambles the matrix, so Newton takes
    // near-but-wrong steps and the residual stalls at ~1e-8 instead of ~1e-13.)
    let a_std = a.as_standard_layout();
    let a_data = a_std
        .as_slice()
        .ok_or_else(|| "solver system matrix must be contiguous".to_string())?;
    let ma = MatRef::from_row_major_slice(a_data, nrows, ncols);
    let mut xs: Vec<f64> = b.to_vec();
    let rhs = ColMut::from_slice_mut(xs.as_mut_slice()).as_mat_mut();
    PartialPivLu::<f64>::new(ma).solve_in_place(rhs);
    Ok(Array1::from_vec(xs))
}

/// Least-squares solve `argmin_x ||a x - b||`, used in-place as a fallback when
/// `a` is singular / rectangular. Same zero-copy view treatment as [`solve_f64`].
fn least_squares_f64(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
    let (nrows, ncols) = a.dim();
    let a_std = a.as_standard_layout();
    let a_data = a_std
        .as_slice()
        .ok_or_else(|| "solver system matrix must be contiguous".to_string())?;
    let ma = MatRef::from_row_major_slice(a_data, nrows, ncols);
    let mut xs: Vec<f64> = b.to_vec();
    let rhs = ColMut::from_slice_mut(xs.as_mut_slice()).as_mat_mut();
    Qr::new(ma).solve_lstsq_in_place(rhs);
    Ok(Array1::from_vec(xs))
}

fn solve_with_fallback_f64(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
    // Guard against a dimension mismatch between the system matrix and the right-hand
    // side. Historically this surfaced as an opaque ndarray panic ("ndarray: inputs
    // 2 x 2 and 0 x 1 are not compatible ..."), which is unactionable. Surface a
    // descriptive message instead, plus a debug log to help trace the offending
    // caller when the invariant is violated.
    if a.shape()[1] != b.len() {
        eprintln!(
            "[root-finder] shape mismatch: matrix {:?} vs rhs len {} ({} x {})",
            a.shape(),
            a.shape()[1],
            b.len(),
            b.len()
        );
        return Err(format!(
            "incompatible system size: matrix {:?} (expected {} columns) but rhs has length {}",
            a.shape(),
            a.shape()[1],
            b.len()
        ));
    }
    match solve_f64(a, b) {
        Ok(v) => Ok(v),
        Err(e) => least_squares_f64(a, b).map_err(|e2| e2),
    }
}
use numpy::{
    ndarray::{Array1, Array2},
    PyArray2, PyArrayMethods,
};
use pyo3::{
    exceptions::{PyException, PyRuntimeError, PyValueError},
    prelude::*,
    IntoPyObjectExt,
};

use crate::{common_types::HydroForcing, hydro::HydrologicZone};

const FIND_ROOT_TOL: f64 = 1e-6;
const FIND_ROOT_MAXITER: usize = 100;
const MULTI_MAXITER: usize = 100;
const MULTI_TOL: f64 = 1e-12;
const MULTI_TOL_STEP: f64 = 1e-12;
const APPROX_FPRIME_DX: f64 = 1e-8;
const APPROX_FPRIME_REL_DX: f64 = 1e-3;
const F_PRIME_MIN_VAL: f64 = 1e-22;

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
    pub errors: Vec<f64>,
    #[pyo3(get)]
    pub xs: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub fxs: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub jacobians: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub lambdas: Vec<f64>,
    #[pyo3(get)]
    pub message: String,
}

impl OptimizationError {
    pub fn from_state(state: OptimizerState, message: String) -> Self {
        let xs: Vec<Vec<f64>> = state.xs.iter().map(|x| x.to_vec()).collect();
        let fxs: Vec<Vec<f64>> = state.fxs.iter().map(|x| x.to_vec()).collect();
        let jacobians: Vec<Vec<Vec<f64>>> = state
            .jacobians
            .iter()
            .map(|x| arr_to_vec(x.clone()))
            .collect();

        Self {
            iterations: state.iteration,
            final_err: state.error,
            last_x: state.final_x.to_vec(),
            last_f_x: state.last_f_x.to_vec(),
            initial_x: state.initial_x.to_vec(),
            jacobian: arr_to_vec(state.jacobian),
            errors: state.errors,
            xs,
            fxs,
            jacobians,
            lambdas: state.lambdas,
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
        errors: Vec<f64>,
        last_x: Vec<f64>,
        last_f_x: Vec<f64>,
        initial_x: Vec<f64>,
        jacobian: Vec<Vec<f64>>,
        xs: Vec<Vec<f64>>,
        fxs: Vec<Vec<f64>>,
        jacobians: Vec<Vec<Vec<f64>>>,
        lambdas: Vec<f64>,
        message: String,
    ) -> Self {
        Self {
            iterations,
            final_err,
            last_x,
            last_f_x,
            initial_x,
            jacobian,
            errors,
            xs,
            fxs,
            jacobians,
            lambdas,
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
    pub iteration: usize,
    pub final_x: Array1<f64>,
    pub last_f_x: Array1<f64>,
    pub initial_x: Array1<f64>,
    pub jacobian: Array2<f64>,
    pub error: f64,
    pub errors: Vec<f64>,
    pub xs: Vec<Array1<f64>>,
    pub fxs: Vec<Array1<f64>>,
    pub jacobians: Vec<Array2<f64>>,
    pub lambdas: Vec<f64>,
}

#[pyclass(extends=PyException)]
#[derive(Debug, Clone)]
pub struct OtherError;

fn arr_to_vec(x: Array2<f64>) -> Vec<Vec<f64>> {
    let arrs: Vec<Vec<f64>> = x.rows().into_iter().map(|x| x.to_vec()).collect();
    arrs
}

/// Compact, human-readable magnitude summary of a vector: mean and max `|v|`,
/// plus — when present — the indexes holding NaN / Inf. Used to enrich solver
/// failure messages so they name *where* the numerics broke rather than only
/// that they did. Truncated to the first 8 problematic indexes when there are
/// many, so a pathological vector can never blow up the message.
fn describe_vector(v: &Array1<f64>) -> String {
    let mut sum = 0.0f64;
    let mut max: f64 = 0.0;
    let mut nan_idx: Vec<usize> = Vec::new();
    let mut inf_idx: Vec<usize> = Vec::new();
    for (i, e) in v.iter().enumerate() {
        if e.is_nan() {
            nan_idx.push(i);
            continue;
        }
        if e.is_infinite() {
            inf_idx.push(i);
            continue;
        }
        sum += e;
        let ae = e.abs();
        if ae > max {
            max = ae;
        }
    }
    let mean = if v.len() == 0 { 0.0 } else { sum / v.len() as f64 };
    let mut s = format!("mean={:.4e} max|.|={:.4e}", mean, max);
    if !nan_idx.is_empty() {
        let shown: String = if nan_idx.len() > 8 {
            format!("{:?},… ({} total)", &nan_idx[..8], nan_idx.len())
        } else {
            format!("{:?}", nan_idx)
        };
        s.push_str(&format!(", NaN@{}", shown));
    }
    if !inf_idx.is_empty() {
        let shown: String = if inf_idx.len() > 8 {
            format!("{:?},… ({} total)", &inf_idx[..8], inf_idx.len())
        } else {
            format!("{:?}", inf_idx)
        };
        s.push_str(&format!(", Inf@{}", shown));
    }
    s
}

/// Build the human-readable failure line for a solver `OptimizerState` (used by
/// the Levenberg-Marquardt path, which constructs states directly rather than
/// through [`make_error_from_parts`]).
fn enrich_state_message(state: &OptimizerState, subsystem: &str, reason: &str) -> String {
    let diag = format!(
        "iteration={}/{} | residual mean|f|={} | x=[{}] | J: {}",
        state.iteration,
        MULTI_MAXITER,
        state.error,
        describe_vector(&state.final_x),
        describe_matrix(&state.jacobian)
    );
    format!(
        "Levenberg-Marquardt ({}): {} | {}",
        subsystem, reason, diag
    )
}

/// Summarize a Jacobian matrix for diagnostics: shape, mean, max `|·|`, the
/// total count of NaN entries, and (if any) the `[row][col]` of the first one.
fn describe_matrix(m: &Array2<f64>) -> String {
    let n = m.shape()[0];
    let p = m.shape()[1];
    let mut maxv: f64 = 0.0;
    let mut n_nan = 0usize;
    let mut first_nan: Option<[usize; 2]> = None;
    let mut sum = 0.0f64;
    let mut n_entries = 0usize;
    for i in 0..n {
        for j in 0..p {
            let e = m[(i, j)];
            n_entries += 1;
            if e.is_nan() {
                n_nan += 1;
                if first_nan.is_none() {
                    first_nan = Some([i, j]);
                }
                continue;
            }
            let ae = e.abs();
            if ae > maxv {
                maxv = ae;
            }
            sum += e;
        }
    }
    let mean = if n_entries == 0 { 0.0 } else { sum / n_entries as f64 };
    let mut s = format!(
        "shape=[{}x{}] mean={:.4e} max|.|={:.4e} nan_count={}",
        n, p, mean, maxv, n_nan
    );
    if let Some([i, j]) = first_nan {
        s.push_str(&format!(", first NaN at [{}][{}]", i, j));
    }
    s
}

impl pyo3::PyErrArguments for OptimizationError {
    fn arguments(self, py: Python<'_>) -> Py<PyAny> {
        (
            self.iterations,
            self.final_err,
            self.errors,
            self.last_x,
            self.last_f_x,
            self.initial_x,
            self.jacobian,
            self.xs,
            self.fxs,
            self.jacobians,
            self.lambdas,
            self.message,
        )
            .into_py_any(py)
            .unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct MatMulError;

impl fmt::Display for ScalarRootFindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Failed to find root")
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

pub fn approx_fprime<F>(f: F, x: &Array1<f64>, verbose: bool) -> Array2<f64>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let n: usize = x.len();
    let mut jac_x: Array2<f64> = Array2::zeros((n, n));
    const EPSILON: f64 = f64::EPSILON;
    let rel_step = EPSILON.sqrt();

    let f_x: Array1<f64> = f(x);

    for (i, x_i) in x.iter().enumerate() {
        // let dx = match x_i.abs() < F_PRIME_MIN_VAL {
        //     true => APPROX_FPRIME_DX,
        //     false => APPROX_FPRIME_REL_DX * x_i.abs(),
        // };
        let dx = rel_step * x_i.abs().max(1.0);
        let inv_dx = 1.0 / dx;

        let mut x_up: Array1<f64> = x.clone();
        // let mut x_dn: Array1<f64> = x.clone();
        x_up[i] = x_i + dx;
        // x_dn[i] = x_i - dx;

        let fx_up: Array1<f64> = f(&x_up);
        // let fx_dn: Array1<f64> = f(&x_dn);

        let jac_x_i: Array1<f64> = inv_dx * (&fx_up - &f_x);

        jac_x.column_mut(i).assign(&jac_x_i);
    }

    jac_x
}

struct MultiFailureParts {
    iteration: usize,
    final_x: Array1<f64>,
    last_f_x: Array1<f64>,
    initial_x: Array1<f64>,
    jacobian: Array2<f64>,
    error: f64,
    errors: Vec<f64>,
    xs: Vec<Array1<f64>>,
    fxs: Vec<Array1<f64>>,
    jacobians: Vec<Array2<f64>>,
}

fn make_error_from_parts(parts: MultiFailureParts, reason: String) -> PyErr {
    // Build a message that names the *cause*, the *subsystem*, and the key
    // numerics (residual norm, offending vector, Jacobian) at the failure point,
    // instead of the old terse reasons. The structured iterate history is still
    // available via `OptimizationError`'s getters; this is the human-readable
    // line that survives into logs and `str(exception)`.
    let diag = format!(
        "iteration={}/{} | residual mean|f|={} | residual f=[{}] | x=[{}] | J: {}",
        parts.iteration,
        MULTI_MAXITER,
        parts.error,
        describe_vector(&parts.last_f_x),
        describe_vector(&parts.final_x),
        describe_matrix(&parts.jacobian)
    );
    let message = format!("{} | {}", reason, diag);

    let final_state = OptimizerState {
        iteration: parts.iteration,
        final_x: parts.final_x.clone(),
        last_f_x: parts.last_f_x.clone(),
        initial_x: parts.initial_x.clone(),
        jacobian: parts.jacobian.clone(),
        error: parts.error,
        errors: parts.errors,
        xs: parts.xs,
        fxs: parts.fxs,
        jacobians: parts.jacobians,
        lambdas: Vec::new(),
    };
    let opt = OptimizationError::from_state(final_state, message);
    PyException::new_err(opt.message)
}

fn newton_failure(parts: MultiFailureParts) -> PyErr {
    make_error_from_parts(parts, "Newton did not converge".to_string())
}

fn newton_nan_error(parts: MultiFailureParts) -> PyErr {
    make_error_from_parts(
        parts,
        "NaN/Inf encountered in Newton iteration".to_string(),
    )
}

fn newton_linear_error(parts: MultiFailureParts, e: String) -> PyErr {
    make_error_from_parts(
        parts,
        format!("Newton linear solve failed: {}", e),
    )
}

fn max_abs(v: &Array1<f64>) -> f64 {
    v.iter().map(|a| a.abs()).fold(0.0f64, f64::max)
}

fn newton_relative_step_step(x: &Array1<f64>, step: &Array1<f64>) -> bool {
    let x_scale: f64 = f64::max(max_abs(x), 1.0);
    let step_norm: f64 = max_abs(step);
    step_norm <= MULTI_TOL_STEP * x_scale
}

/// Fraction of the *initial* objective that a stalled iterate must reach before
/// it is accepted as a converged plateau (on top of an absolute floor chosen by
/// the caller). The Newton primaries keep a strict floor of `MULTI_TOL` (1e-12):
/// when one of them stalls just above that, it must *fail* so the caller can
/// defer to the Levenberg-Marquardt fallback — loosening the primary here would
/// accept a transient plateau, skip the fallback, and propagate a coarser state
/// downstream.
const STAGNATION_REL_TOL: f64 = 1e-9;

/// Physical-residual floor, in `mean|f|` units, used *only* by the
/// Levenberg-Marquardt path as its last-chance plateau bar. An ill-conditioned
/// exponential `10^(N x)` speciation step — especially when warm-started from the
/// previous time step, which collapses the `STAGNATION_REL_TOL` term toward
/// zero — legitimately stalls a few decades above `MULTI_TOL`. That is already a
/// physically-converged state; LM is the fallback of last resort, so accepting it
/// here cannot suppress a later, better solve. The floor sits ~3 orders of
/// magnitude below any genuine mis-solve (whose residual stays O(0.1)-O(1)), so
/// a real failure is still reported.
const STAGNATION_LM_FLOOR: f64 = 1e-4;

/// The objective `err` must fall below the *larger* of `abs_floor` and
/// `STAGNATION_REL_TOL` times the objective at the solver's start, for a stalled
/// iterate to be accepted. Passing `MULTI_TOL` as `abs_floor` reproduces the
/// strict (Newton-primary) bar; passing `STAGNATION_LM_FLOOR` relaxes the bar
/// for the LM fallback so a warm-started converged plateau is not over-rejected.
fn stagnation_objective_bar(abs_floor: f64, initial_err: f64) -> f64 {
    abs_floor.max(STAGNATION_REL_TOL * initial_err)
}

/// Whether a solve should be accepted as a converged plateau: the relative step
/// `step_converged` is true (the iterate has stopped moving, e.g. LM damping at
/// its cap) and the best objective found is at or below the stagnation bar for
/// this solve. `abs_floor` selects the strict (Newton) or relaxed (LM) bar.
/// Callers pass the best objective seen so the plateau, not the
/// (possibly-worse) current iterate, decides.
fn should_accept_stagnation(
    step_converged: bool,
    abs_floor: f64,
    initial_err: f64,
    best_err: f64,
) -> bool {
    step_converged && best_err <= stagnation_objective_bar(abs_floor, initial_err)
}

/// Find the root of the linear problem using Newton's method with a
/// finite-difference Jacobian
pub fn find_root_multi<'a, F>(f: &F, x_0: Array1<f64>, verbose: bool) -> PyResult<Array1<f64>>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut x: Array1<f64> = x_0.clone();
    let mut f_x: Array1<f64> = f(&x);
    let mut err: f64 = f_x.abs().mean().unwrap();
    let initial_err: f64 = err;
    let mut best_x: Array1<f64> = x.clone();
    let mut best_err: f64 = err;
    let mut jac_x: Array2<f64> = Array2::zeros((1, 1));
    let mut errors: Vec<f64> = Vec::with_capacity(MULTI_MAXITER + 1);
    let mut xs: Vec<Array1<f64>> = vec![x.clone()];
    let mut fxs: Vec<Array1<f64>> = vec![f_x.clone()];
    let mut jacobians: Vec<Array2<f64>> = Vec::new();
    errors.push(err);

    for i in 0..MULTI_MAXITER {
        if err <= MULTI_TOL {
            return Ok(x);
        }

        jac_x = approx_fprime(f, &x, verbose);
        jacobians.push(jac_x.clone());

        if x.is_any_nan() || f_x.is_any_nan() || jac_x.is_any_nan() {
            return Err(newton_nan_error(MultiFailureParts {
                iteration: i,
                final_x: x.clone(),
                last_f_x: f_x.clone(),
                initial_x: x_0.clone(),
                jacobian: jac_x.clone(),
                error: err,
                errors: std::mem::take(&mut errors),
                xs: std::mem::take(&mut xs),
                fxs: std::mem::take(&mut fxs),
                jacobians: std::mem::take(&mut jacobians),
            }));
        }

        let step: Array1<f64> = match solve_with_fallback_f64(&jac_x, &f_x) {
            Ok(v) => v,
            Err(e) => {
                return Err(newton_linear_error(
                    MultiFailureParts {
                        iteration: i,
                        final_x: x.clone(),
                        last_f_x: f_x.clone(),
                        initial_x: x_0.clone(),
                        jacobian: jac_x.clone(),
                        error: err,
                        errors: std::mem::take(&mut errors),
                        xs: std::mem::take(&mut xs),
                        fxs: std::mem::take(&mut fxs),
                        jacobians: std::mem::take(&mut jacobians),
                    },
                    e,
                ))
            }
        };

        let x_new: Array1<f64> = &x - &step;
        let converged = newton_relative_step_step(&x, &step);
        x = x_new;
        f_x = f(&x);
        err = f_x.abs().mean().unwrap();
        if err < best_err && !err.is_nan() {
            best_err = err;
            best_x = x.clone();
        }
        errors.push(err);

        xs.push(x.clone());
        fxs.push(f_x.clone());

        if verbose {
            Python::attach(|py| {
                py.detach(|| {
                    eprintln!("x after i={}: {}", i, &x);
                    eprintln!("err: {}", err);
                    eprintln!("\n\n");
                })
            });
        }

        if converged && err <= MULTI_TOL {
            return Ok(x);
        }
        // The iterate has stopped moving; accept the best iterate if the residual
        // is a physically-converged plateau rather than a genuine failure.
        if should_accept_stagnation(converged, MULTI_TOL, initial_err, best_err) {
            return Ok(best_x);
        }
    }

    Err(newton_failure(MultiFailureParts {
        iteration: MULTI_MAXITER,
        final_x: x,
        last_f_x: f_x,
        initial_x: x_0,
        jacobian: jac_x,
        error: err,
        errors,
        xs,
        fxs,
        jacobians,
    }))
}

/// Find the root of a nonlinear system using Newton's method with an
/// *analytical* Jacobian.
///
/// Each iteration evaluates the residual once and the Jacobian once, which is
/// dramatically cheaper than the `n + 1` residual evaluations required by the
/// finite-difference version ([`find_root_multi`]).
pub fn find_root_multi_analytic<'a, F, G>(
    f: &F,
    jf: &G,
    x_0: Array1<f64>,
    verbose: bool,
) -> PyResult<Array1<f64>>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
    G: Fn(&Array1<f64>) -> Array2<f64>,
{
    let mut x: Array1<f64> = x_0.clone();
    let mut f_x: Array1<f64> = f(&x);
    let mut err: f64 = f_x.abs().mean().unwrap_or(f64::MAX);
    let initial_err: f64 = err;
    let mut best_x: Array1<f64> = x.clone();
    let mut best_err: f64 = err;

    // A system with no degrees of freedom (e.g. a speciation problem for a
    // network that has no independent conservation/charge rows) is vacuously
    // satisfied: return the initial guess. Without this, applying/
    // broadcasting a degenerate Jacobian would surface as an opaque ndarray
    // panic ("inputs N x M and 0 x 1 are not compatible for matrix
    // multiplication").
    if x.len() == 0 || f_x.len() == 0 {
        return Ok(x);
    }
    let mut jac_x: Array2<f64> = Array2::zeros((1, 1));
    let mut errors: Vec<f64> = Vec::with_capacity(MULTI_MAXITER + 1);
    let mut xs: Vec<Array1<f64>> = vec![x.clone()];
    let mut fxs: Vec<Array1<f64>> = vec![f_x.clone()];
    let mut jacobians: Vec<Array2<f64>> = Vec::new();
    errors.push(err);

    for i in 0..MULTI_MAXITER {
        if err <= MULTI_TOL {
            return Ok(x);
        }

        jac_x = jf(&x);
        jacobians.push(jac_x.clone());

        if x.is_any_nan() || f_x.is_any_nan() || jac_x.is_any_nan() {
            return Err(newton_nan_error(MultiFailureParts {
                iteration: i,
                final_x: x.clone(),
                last_f_x: f_x.clone(),
                initial_x: x_0.clone(),
                jacobian: jac_x.clone(),
                error: err,
                errors: std::mem::take(&mut errors),
                xs: std::mem::take(&mut xs),
                fxs: std::mem::take(&mut fxs),
                jacobians: std::mem::take(&mut jacobians),
            }));
        }

        let step: Array1<f64> = match solve_with_fallback_f64(&jac_x, &f_x) {
            Ok(v) => v,
            Err(e) => {
                return Err(newton_linear_error(
                    MultiFailureParts {
                        iteration: i,
                        final_x: x.clone(),
                        last_f_x: f_x.clone(),
                        initial_x: x_0.clone(),
                        jacobian: jac_x.clone(),
                        error: err,
                        errors: std::mem::take(&mut errors),
                        xs: std::mem::take(&mut xs),
                        fxs: std::mem::take(&mut fxs),
                        jacobians: std::mem::take(&mut jacobians),
                    },
                    e,
                ))
            }
        };

        let x_new: Array1<f64> = &x - &step;
        let converged = newton_relative_step_step(&x, &step);
        x = x_new;
        f_x = f(&x);
        err = f_x.abs().mean().unwrap();
        if err < best_err && !err.is_nan() {
            best_err = err;
            best_x = x.clone();
        }
        errors.push(err);

        xs.push(x.clone());
        fxs.push(f_x.clone());

        if verbose {
            Python::attach(|py| {
                py.detach(|| {
                    eprintln!("x after i={}: {}", i, &x);
                    eprintln!("err: {}", err);
                    eprintln!("\n\n");
                })
            });
        }

        if converged && err <= MULTI_TOL {
            return Ok(x);
        }
        // Stalled iterate: accept the best iterate if the residual is a
        // physically-converged plateau rather than a genuine failure.
        if should_accept_stagnation(converged, MULTI_TOL, initial_err, best_err) {
            return Ok(best_x);
        }
    }

    Err(newton_failure(MultiFailureParts {
        iteration: MULTI_MAXITER,
        final_x: x,
        last_f_x: f_x,
        initial_x: x_0,
        jacobian: jac_x,
        error: err,
        errors,
        xs,
        fxs,
        jacobians,
    }))
}

/// Find the root of a nonlinear system using Newton's method with a *fused*
/// residual-and-Jacobian evaluation.
///
/// The callback `ej` returns the residual and the analytic Jacobian at a state
/// in a single call (the kinetic terms — the dominant cost — are then computed
/// once per iteration instead of once per evaluation). All other behaviour
/// (convergence test, NaN guards, error records) matches
/// [`find_root_multi_analytic`], so results are identical; only the split into
/// two separate residual and Jacobian closures is collapsed into one.
pub fn find_root_multi_analytic_fused<'a, E>(
    ej: E,
    x_0: Array1<f64>,
    verbose: bool,
) -> PyResult<Array1<f64>>
where
    E: Fn(&Array1<f64>) -> (Array1<f64>, Array2<f64>),
{
    let mut x: Array1<f64> = x_0.clone();
    let (f_x, jac_x_0) = ej(&x);
    let mut f_x: Array1<f64> = f_x;
    let mut jac_x: Array2<f64> = jac_x_0;
    let mut err: f64 = f_x.abs().mean().unwrap_or(f64::MAX);
    let initial_err: f64 = err;
    let mut best_x: Array1<f64> = x.clone();
    let mut best_err: f64 = err;

    // A system with no degrees of freedom (e.g. a speciation problem for a
    // network that has no independent conservation/charge rows) is vacuously
    // satisfied: return the initial guess. Without this, applying/
    // broadcasting a degenerate Jacobian would surface as an opaque ndarray
    // panic ("inputs N x M and 0 x 1 are not compatible for matrix
    // multiplication").
    if x.len() == 0 || f_x.len() == 0 {
        return Ok(x);
    }
    let mut errors: Vec<f64> = Vec::with_capacity(MULTI_MAXITER + 1);
    let mut xs: Vec<Array1<f64>> = vec![x.clone()];
    let mut fxs: Vec<Array1<f64>> = vec![f_x.clone()];
    let mut jacobians: Vec<Array2<f64>> = Vec::new();
    errors.push(err);

    for i in 0..MULTI_MAXITER {
        if err <= MULTI_TOL {
            return Ok(x);
        }

        jacobians.push(jac_x.clone());

        if x.is_any_nan() || f_x.is_any_nan() || jac_x.is_any_nan() {
            return Err(newton_nan_error(MultiFailureParts {
                iteration: i,
                final_x: x.clone(),
                last_f_x: f_x.clone(),
                initial_x: x_0.clone(),
                jacobian: jac_x.clone(),
                error: err,
                errors: std::mem::take(&mut errors),
                xs: std::mem::take(&mut xs),
                fxs: std::mem::take(&mut fxs),
                jacobians: std::mem::take(&mut jacobians),
            }));
        }

        let step: Array1<f64> = match solve_with_fallback_f64(&jac_x, &f_x) {
            Ok(v) => v,
            Err(e) => {
                return Err(newton_linear_error(
                    MultiFailureParts {
                        iteration: i,
                        final_x: x.clone(),
                        last_f_x: f_x.clone(),
                        initial_x: x_0.clone(),
                        jacobian: jac_x.clone(),
                        error: err,
                        errors: std::mem::take(&mut errors),
                        xs: std::mem::take(&mut xs),
                        fxs: std::mem::take(&mut fxs),
                        jacobians: std::mem::take(&mut jacobians),
                    },
                    e,
                ))
            }
        };

        let x_new: Array1<f64> = &x - &step;
        let converged = newton_relative_step_step(&x, &step);
        x = x_new;
        let (f_new, jac_new) = ej(&x);
        f_x = f_new;
        jac_x = jac_new;
        err = f_x.abs().mean().unwrap();
        if err < best_err && !err.is_nan() {
            best_err = err;
            best_x = x.clone();
        }
        errors.push(err);

        xs.push(x.clone());
        fxs.push(f_x.clone());

        if verbose {
            Python::attach(|py| {
                py.detach(|| {
                    eprintln!("x after i={}: {}", i, &x);
                    eprintln!("err: {}", err);
                    eprintln!("\n\n");
                })
            });
        }

        if converged && err <= MULTI_TOL {
            return Ok(x);
        }
        // Stalled iterate: accept the best iterate if the residual is a
        // physically-converged plateau rather than a genuine failure.
        if should_accept_stagnation(converged, MULTI_TOL, initial_err, best_err) {
            return Ok(best_x);
        }
    }

    Err(newton_failure(MultiFailureParts {
        iteration: MULTI_MAXITER,
        final_x: x,
        last_f_x: f_x,
        initial_x: x_0,
        jacobian: jac_x,
        error: err,
        errors,
        xs,
        fxs,
        jacobians,
    }))
}

pub fn levenberg_marquardt<'a, F>(f: &F, x_0: Array1<f64>, verbose: bool) -> PyResult<Array1<f64>>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut x: Array1<f64> = x_0.clone();
    let mut f_x: Array1<f64> = f(&x);
    // Levenberg-Marquardt's canonical objective is the summed squared residual.
    // (Note: the diagnostic message reports `mean|f|`, not this, so the printed
    // number understates this objective by orders of magnitude for a well-solved
    // step — the objective, not the message, governs convergence.)
    let mut err: f64 = 0.5 * f_x.dot(&f_x);
    let initial_err: f64 = err;
    let mut best_x: Array1<f64> = x.clone();
    let mut best_err: f64 = err;
    let mut jac_x: Array2<f64> = Array2::zeros((1, 1));
    let mut errors: Vec<f64> = Vec::with_capacity(MULTI_MAXITER + 1);
    let mut xs: Vec<Array1<f64>> = vec![x.clone()];
    let mut fxs: Vec<Array1<f64>> = vec![f_x.clone()];
    let mut jacobians: Vec<Array2<f64>> = Vec::new();
    let mut lambda = 1e-6;
    let mut lambdas: Vec<f64> = vec![lambda];
    errors.push(err);

    for i in 0..MULTI_MAXITER {
        if err <= MULTI_TOL {
            return Ok(x);
        }

        jac_x = approx_fprime(f, &x, verbose);

        if x.is_any_nan() || f_x.is_any_nan() || jac_x.is_any_nan() {
            let final_state = OptimizerState {
                iteration: i,
                final_x: x.clone(),
                last_f_x: f_x.clone(),
                initial_x: x_0.clone(),
                jacobian: jac_x.clone(),
                error: err,
                errors,
                xs,
                fxs,
                jacobians,
                lambdas,
            };

            let err = OptimizationError::from_state(
                final_state.clone(),
                enrich_state_message(&final_state, "reactive-transport", "NaN/Inf encountered in iteration"),
            );
            return Err(PyException::new_err(err.message));
        }

        let jac_x_t = jac_x.t();
        let a_mat_base: Array2<f64> = jac_x_t.dot(&jac_x);
        let diag_a = a_mat_base.diag().to_owned();
        let mut a_mat_damped = a_mat_base.clone();

        for i in 0..a_mat_damped.nrows() {
            a_mat_damped[(i, i)] += lambda * diag_a[i];
        }

        let b: Array1<f64> = jac_x_t.dot(&f_x);

        let step: Array1<f64> = match solve_with_fallback_f64(&a_mat_damped, &b) {
            Ok(v) => v,
            Err(e) => {
                let final_state = OptimizerState {
                    iteration: i,
                    final_x: x.clone(),
                    last_f_x: f_x.clone(),
                    initial_x: x_0.clone(),
                    jacobian: jac_x.clone(),
                    error: err,
                    errors,
                    xs,
                    fxs,
                    jacobians,
                    lambdas,
                };
                let opt_err = OptimizationError::from_state(
                    final_state.clone(),
                    enrich_state_message(
                        &final_state,
                        "reactive-transport",
                        &format!("linear solve failed: {}", e),
                    ),
                );
                return Err(PyException::new_err(opt_err.message));
            }
        };

        let step_converged = newton_relative_step_step(&x, &step);
        let x_test: Array1<f64> = &x - &step;
        let f_x_test: Array1<f64> = f(&x_test);
        let err_test: f64 = 0.5 * f_x_test.dot(&f_x_test);
        // Track the best accepted iterate so a stalled (damping-limited) run can
        // still return the most reduced residual it ever reached.
        if err_test < best_err && !err_test.is_nan() {
            best_err = err_test;
            best_x = x_test.clone();
        }

        // Calculate gain ratio
        let actual_reduction = err - err_test;
        let predicted_reduction = step.dot(&b) - 0.5 * step.dot(&a_mat_base.dot(&step));
        let rho = if predicted_reduction.abs() < 1e-12 {
            0.0
        } else {
            actual_reduction / predicted_reduction
        };

        if rho > 0.0 {
            // The step was good, accept it
            x = x_test;
            f_x = f_x_test;
            err = err_test;

            // Update lambda based on how good the prediction was
            lambda *= ((1.0 / 3.0) as f64).max(1.0 - (2.0 * rho - 1.0).powi(3));
            lambda = lambda.max(1e-16); // Lower bound for lambda

            // Only record state on successful steps
            errors.push(err);
            xs.push(x.clone());
            fxs.push(f_x.clone());
            lambdas.push(lambda);
            jacobians.push(jac_x.clone());

            if step_converged
                && f_x.abs().mean().unwrap_or(f64::MAX) <= MULTI_TOL
            {
                return Ok(x);
            }
        } else {
            // The step was bad, reject it and become more cautious
            lambda *= 2.0;
            lambda = lambda.min(1e16); // Upper bound for lambda

            // Do not record the state, as we haven't moved
        }

        // The iterate has stopped moving (relative step converged, e.g. damping at
        // its cap): accept the best iterate if the objective is a
        // physically-converged plateau rather than a genuine non-convergence.
        if should_accept_stagnation(step_converged, MULTI_TOL, initial_err, best_err) {
            return Ok(best_x);
        }

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
        errors,
        xs,
        fxs,
        jacobians,
        lambdas,
    };

    let err = OptimizationError::from_state(
        final_state.clone(),
        enrich_state_message(&final_state, "reactive-transport", "did not converge"),
    );
    Err(PyException::new_err(err.message))
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
