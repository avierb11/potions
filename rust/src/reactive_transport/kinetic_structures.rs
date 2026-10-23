use numpy::{
    array,
    ndarray::{s, Array1, Array2},
    PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, ToPyArray,
};
use polars::{frame::DataFrame, prelude::Float64Type};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_polars::{PyDataFrame, PySeries};

use crate::{
    common_types::RtForcing,
    math::{approx_fprime, find_root_multi_analytic_then_lm, null_space_scipy, pinv_scipy},
    molar, molar_per_time,
    units::{mole_per_liter, MolarConcentration},
};
const PARAMETERS_PER_MINERAL: usize = 4;

/// Floor (mol/L) applied before `log10` in the TST activity expressions.
/// Root finders (Newton, LM, fsolve) can probe trial states with zero or
/// negative concentrations; clamping keeps each log-contribution large but
/// finite instead of NaN, which the solvers' sufficiency checks would reject.
const LOG10_CONC_FLOOR: f64 = 1e-300;

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct MonodParameters {
    #[pyo3(get)]
    pub monod_mat: PyDataFrame,
    #[pyo3(get)]
    pub inhib_mat: PyDataFrame,
    pub monod_np: Array2<f64>,
    pub inhib_np: Array2<f64>,
}

impl MonodParameters {
    pub fn monod_np(&self) -> &Array2<f64> {
        &self.monod_np
    }

    pub fn inhib_np(&self) -> &Array2<f64> {
        &self.inhib_np
    }

    pub fn rate_rust(&self, chms: &Array1<molar>) -> Array1<molar_per_time> {
        let n_minerals: usize = self.monod_np.shape()[0];
        let n_species: usize = self.monod_np.shape()[1];
        let mut monod: Array1<molar_per_time> = Array1::zeros(n_minerals);
        let mut inhib: Array1<molar_per_time> = Array1::zeros(n_minerals);

        // Monod / inhibition terms are same-dimension concentration ratios
        // (mol/L over mol/L) -> `Ratio`; `.value` is the raw dimensionless
        // number (M0 spike test 8). Boundary values stay raw f64.
        for i in 0..n_minerals {
            let mut prod = 1.0;
            for j in 0..n_species {
                let monod_ij = self.monod_np[(i, j)];
                if monod_ij.is_finite() {
                    let k = MolarConcentration::new::<mole_per_liter>(monod_ij);
                    let c = MolarConcentration::new::<mole_per_liter>(chms[j]);
                    prod *= (c / (k + c)).value;
                }
            }
            monod[i] = prod;
        }

        for i in 0..n_minerals {
            let mut prod = 1.0;
            for j in 0..n_species {
                let inhib_ij = self.inhib_np[(i, j)];
                if inhib_ij.is_finite() {
                    let ki = MolarConcentration::new::<mole_per_liter>(inhib_ij);
                    let c = MolarConcentration::new::<mole_per_liter>(chms[j]);
                    prod *= (ki / (ki + c)).value;
                }
            }
            inhib[i] = prod;
        }

        Array1::from_iter(monod.iter().zip(inhib).map(|(x, y)| x * y))
    }
}

#[pymethods]
impl MonodParameters {
    #[new]
    pub fn new(monod_mat: PyDataFrame, inhib_mat: PyDataFrame) -> PyResult<Self> {
        let monod_arr_pre = monod_mat.0.to_ndarray::<Float64Type>(Default::default());
        let inhib_arr_pre = inhib_mat.0.to_ndarray::<Float64Type>(Default::default());

        let monod_arr: Array2<f64> = match monod_arr_pre {
            Ok(v) => v,
            Err(_) => return Err(PyValueError::new_err("Failed to get Monod matrix")),
        };

        let inhib_arr: Array2<f64> = match inhib_arr_pre {
            Ok(v) => v,
            Err(_) => return Err(PyValueError::new_err("Failed to get inhibition matrix")),
        };

        Ok(Self {
            monod_mat,
            inhib_mat,
            monod_np: monod_arr,
            inhib_np: inhib_arr,
        })
    }

    /// Compact summary reporting the shape of the Monod (and inhibition)
    /// matrices: one row per mineral, one column per species.
    pub fn __repr__(&self) -> String {
        let shape = self.monod_np.shape();
        format!(
            "MonodParameters(num_minerals={},num_species={})",
            shape[0], shape[1]
        )
    }

    pub fn rate<'py>(
        &self,
        py: Python<'py>,
        chms: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let chms_arr: Array1<f64> = chms.to_owned_array();

        self.rate_rust(&chms_arr).to_pyarray(py)
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct TstParameters {
    #[pyo3(get)]
    pub stoich: PyDataFrame,
    #[pyo3(get)]
    pub dep: PyDataFrame,
    #[pyo3(get)]
    pub min_eq_const: PySeries,
    stoich_np: Array2<f64>,
    dep_np: Array2<f64>,
    min_eq_const_np: Array1<f64>,
}

impl TstParameters {
    // Matrix-vector product against the (possibly empty) stoichiometry matrix.
    // For matrix `m` with shape (a, b) and vector `x` with shape (b,):
    // returns a length-`a` vector `y` where `y[i] = sum_j m[i, j] * x[j]`.
    fn matvec(m: &Array2<f64>, x: &Array1<f64>) -> Array1<f64> {
        let a = m.shape()[0];
        let b = m.shape()[1];
        if b == 0 {
            return Array1::zeros(a);
        }
        m.dot(x)
    }

    // Calculate the solubility product for each of the minerals.
    // Returns (linear Q, exponent e) with Q == 10^e, so callers never need to
    // take `log10` of a concentration they have already reduced to `10^e`.
    pub fn calculate_solubility_product(
        &self,
        chms: &Array1<molar>,
    ) -> (Array1<molar_per_time>, Array1<f64>) {
        // log10 / 10^x over the log-linear activity exponents — outside uom's
        // scope (no log/exp in 0.38), so kept raw f64.  Concentrations are
        // clamped to `LOG10_CONC_FLOOR` first: a zero or negative trial value
        // (transient during a root find) yields a large but finite
        // contribution instead of NaN.
        let log_conc: Array1<f64> = chms.map(|x| x.max(LOG10_CONC_FLOOR).log10());
        let e: Array1<f64> = Self::matvec(&self.stoich_np, &log_conc);
        let q: Array1<f64> = e.map(|x| (10_f64).powf(*x)); // Convert back to linear scale
        (q, e)
    }

    // Same as `calculate_solubility_product` but for the "dependence" (forward-rate)
    // term
    pub fn calculate_dependence_term(&self, chms: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        // log10 / 10^x over the log-linear activity exponents (forward-rate
        // dependence) — outside uom's scope (no log/exp), so kept raw f64.
        // Concentrations are clamped to `LOG10_CONC_FLOOR` first (see
        // `calculate_solubility_product`).
        let log_conc: Array1<f64> = chms.map(|x| x.max(LOG10_CONC_FLOOR).log10());
        let e: Array1<f64> = Self::matvec(&self.dep_np, &log_conc);
        let q: Array1<f64> = e.map(|x| (10_f64).powf(*x)); // Convert back to linear scale
        (q, e)
    }

    // TST rate for each mineral as a function of the (log-linear) exponents of the
    // solubility product and the dependence term:
    //     rate_i = 10^(f_i) * (1 - 10^(e_i) / K_i)
    // where K_i is the mineral equilibrium constant.
    pub fn rates_from_exponents(&self, e_dep: &Array1<f64>, e_q: &Array1<f64>) -> Array1<f64> {
        // TST rate d * (1 - Q/K): log-linear exponents (powf) and the dimensionless
        // reaction quotient — kept raw f64 (uom has no log/powf).
        let d = e_dep.map(|x| (10_f64).powf(*x));
        let q = e_q.map(|x| (10_f64).powf(*x));
        d * (1.0 - q / &self.min_eq_const_np)
    }

    pub fn rate_rust(&self, chms: &Array1<f64>) -> Array1<f64> {
        let (solubility_product, _) = self.calculate_solubility_product(chms);
        let (dependence, _) = self.calculate_dependence_term(chms);

        let tst_rates: Array1<f64> =
            dependence * (1.0 - solubility_product / &self.min_eq_const_np);

        tst_rates
    }

    pub fn stoich_np(&self) -> &Array2<f64> {
        &self.stoich_np
    }

    pub fn dep_np(&self) -> &Array2<f64> {
        &self.dep_np
    }

    pub fn eq_const_np(&self) -> &Array1<f64> {
        &self.min_eq_const_np
    }
}

#[pymethods]
impl TstParameters {
    #[new]
    pub fn new(stoich: PyDataFrame, dep: PyDataFrame, min_eq_const: PySeries) -> PyResult<Self> {
        let stoich_arr: Array2<f64> = match stoich.0.to_ndarray::<Float64Type>(Default::default()) {
            Ok(v) => v,
            Err(_) => {
                return Err(PyValueError::new_err(
                    "Failed to get mineral stoichiometry array",
                ))
            }
        };
        let dep_arr: Array2<f64> = match dep.0.to_ndarray::<Float64Type>(Default::default()) {
            Ok(v) => v,
            Err(_) => {
                return Err(PyValueError::new_err(
                    "Failed to get mineral stoichiometry array",
                ))
            }
        };

        let eq_const_arr: Array1<f64> = match min_eq_const.0.f64() {
            Ok(v) => v.to_ndarray().unwrap().iter().cloned().collect(),
            Err(_) => {
                return Err(PyValueError::new_err(
                    "Failed to get mineral equilibrium constants",
                ))
            }
        };

        Ok(Self {
            stoich,
            dep,
            min_eq_const,
            stoich_np: stoich_arr,
            dep_np: dep_arr,
            min_eq_const_np: eq_const_arr,
        })
    }

    /// Compact summary reporting the shape of the stoichiometry (and
    /// dependence) matrices: one row per TST mineral, one column per species.
    pub fn __repr__(&self) -> String {
        let shape = self.stoich_np.shape();
        format!(
            "TstParameters(num_minerals={},num_species={})",
            shape[0], shape[1]
        )
    }

    pub fn rate<'py>(
        &self,
        py: Python<'py>,
        chms: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let chms_arr: Array1<f64> = chms.to_owned_array();
        let rate_out: Array1<f64> = self.rate_rust(&chms_arr);
        rate_out.to_pyarray(py)
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct EquilibriumParameters {
    #[pyo3(get)]
    stoich: PyDataFrame,
    #[pyo3(get)]
    log_eq_consts: PySeries,
    #[pyo3(get)]
    total: PyDataFrame,
    total_mat: Array2<f64>,
    stoich_null_space: Array2<f64>,
    log10_k_w: Array1<f64>,
    x_particular: Array1<f64>,
}

impl EquilibriumParameters {
    const LNL10: f64 = 2.302585092994046;

    pub fn conc_func_rust(&self, x_free: &Array1<f64>) -> Array1<f64> {
        let exp: Array1<f64> = self.stoich_null_space.dot(x_free) + &self.x_particular;
        exp.map(|x| (10.0f64).powf(*x))
    }

    pub fn stoich_null_space(&self) -> &Array2<f64> {
        &self.stoich_null_space
    }

    /// Dimension of the free-variable vector `x_free`: the number of columns in the
    /// stoichiometry null space, i.e. the number of independent conservation /
    /// charge-balance constraints (equivalently, the number of rows in `total_mat`).
    /// This is NOT the number of species (the number of *rows* of the null space).
    pub fn num_free(&self) -> usize {
        self.stoich_null_space.shape()[1]
    }

    pub fn residual_rust(&self, x_free: &Array1<f64>, c_tot: &Array1<f64>) -> Array1<f64> {
        c_tot - &self.total_mat.dot(&self.conc_func_rust(x_free))
    }

    // Analytical Jacobian of the speciation residual.
    //
    // The residual is `f(x) = c_tot - A . g(x)` where
    //   `A = total_mat` (num_total x num_species, constant),
    //   `g(x) = 10^(N x + b)`, `N = stoich_null_space` (num_species x n_free),
    //   `b = x_particular` (num_species).
    // By the chain rule (with d(10^y)/dy = ln(10) * 10^y)
    //   J[i, j] = -ln(10) * sum_k A[i, k] * g[k] * N[k, j]
    // i.e. `J = -ln(10) * (A * g[:, None]) @ N`. A naive finite-difference
    // Jacobian would need `n_free + 1` residual evaluations (each a
    // matrix-vector product plus a `10^` over all species); this touches each
    // element a constant number of times.
    pub fn jacobian_residual_rust(&self, x_free: &Array1<f64>) -> Array2<f64> {
        let g: Array1<f64> = self.conc_func_rust(x_free);
        let a = self.total_mat.shape(); // (num_total, num_species)
        let n = self.stoich_null_space.shape(); // (num_species, n_free)
        let num_total: usize = a[0];
        let num_species: usize = a[1];
        let n_free: usize = n[1];
        let mut jac: Array2<f64> = Array2::zeros((num_total, n_free));
        for i in 0..num_total {
            for j in 0..n_free {
                let mut acc: f64 = 0.0;
                for k in 0..num_species {
                    acc += self.total_mat[(i, k)] * g[k] * self.stoich_null_space[(k, j)];
                }
                jac[(i, j)] = -Self::LNL10 * acc;
            }
        }
        jac
    }

    // Solve the speciation problem for the *free-variables* `x_free`, i.e. the
    // values in the null-space basis such that `10^(N x_free + b)` reproduces the
    // equilibrium concentrations.
    //
    // `initial_x` is a warm start for the Newton iteration (typically the
    // solution from the previous time step, which is usually very close to the new
    // solution), or zeros for a cold start.
    pub fn x_free_solve_rust(
        &self,
        chms: &Array1<f64>,
        initial_x: &Array1<f64>,
        verbose: bool,
    ) -> PyResult<Array1<f64>> {
        let num_species = chms.len();
        let spec_free = self.num_free(); // columns of the null space (free variables)
        let spec_rows = self.stoich_null_space.shape()[0]; // rows of the null space (species dimension)

        // A network with no equilibrium reactions has an empty null space and
        // therefore no free variables: there is nothing to solve for. Return the
        // (empty) initial guess instead of attempting a degenerate / non-square
        // Newton or Levenberg-Marquardt step, which would panic with an ndarray
        // shape error (e.g. "2 x 2 and 0 x 1").
        if spec_free == 0 {
            return Ok(initial_x.clone());
        }

        // Structural invariant for a non-degenerate speciation system: the
        // free-variable vector, the null-space basis, the conservation matrix and
        // the concentration vector must all agree on the number of species. When
        // they disagree, fail loudly with a message that names every offending
        // dimension, instead of deferring to an opaque ndarray shape panic
        // ("2 x 2 and 0 x 1") raised from inside the solver where it is impossible
        // to tell which input is inconsistent.
        if spec_rows != num_species
            || self.total_mat.shape()[1] != num_species
            || initial_x.len() != spec_free
        {
            let msg = format!(
                "Inconsistent speciation system for a network with {num_species} species: \
                 stoich_null_space is {spec_rows}x{spec_free} (expected {num_species}x{spec_free}), \
                 total_mat is {}x{} (expected ?x{num_species}), and x_free has length {} (expected {spec_free}). \
                 A network with *no* equilibrium reactions yields an empty null space and is \
                 handled by skipping speciation entirely; if this network is supposed to carry \
                 equilibrium reactions, the equilibrium stoichiometry/conservation matrices are \
                 mis-specified.",
                self.total_mat.shape()[0],
                self.total_mat.shape()[1],
                initial_x.len(),
            );
            return Err(PyValueError::new_err(msg));
        }

        let chms_arr: Array1<f64> = chms.clone();
        let mut c_tot: Array1<f64> = self.total_mat.dot(&chms_arr);

        // Get the charge balance location
        let s: &DataFrame = &self.total.0;

        let charge_ind = s
            .get_column_names()
            .into_iter()
            .position(|x| x.as_str() == "H+");

        match charge_ind {
            Some(i) => c_tot[i] = 0.0,
            None => (),
        };

        // Create the callable functions for solving the root
        let f_to_solve = |x: &Array1<f64>| self.residual_rust(x, &c_tot);
        let j_to_solve = |x: &Array1<f64>| self.jacobian_residual_rust(x);

        // Solve with Newton's method using the analytical Jacobian, falling back
        // to Levenberg-Marquardt (itself using the analytic Jacobian) if that fails to
        // converge. LM resumes from the best iterate Newton reached.
        find_root_multi_analytic_then_lm(&f_to_solve, &j_to_solve, initial_x.clone(), verbose)
    }

    // Public speciation entry point: returns the equilibrium *concentrations*.
    // This keeps the existing behavior (cold start from zeros) for callers that do
    // not warm-start.
    pub fn solve_equilibrium_rust(
        &self,
        chms: &Array1<f64>,
        verbose: bool,
    ) -> PyResult<Array1<f64>> {
        // With no free variables there is nothing to re-speciate; the
        // (de-)hydrated / kinetic state is already the equilibrium state.
        if self.num_free() == 0 {
            return Ok(chms.clone());
        }
        let cold: Array1<f64> = Array1::zeros(self.num_free());
        let v = self.x_free_solve_rust(chms, &cold, verbose)?;
        Ok(self.conc_func_rust(&v))
    }
}

#[pymethods]
impl EquilibriumParameters {
    #[new]
    pub fn new(stoich: PyDataFrame, log_eq_consts: PySeries, total: PyDataFrame) -> PyResult<Self> {
        let stoich_mat: Array2<f64> = stoich
            .0
            .to_ndarray::<Float64Type>(polars::prelude::IndexOrder::C)
            .expect("Failed to get stoichiometry matrix from dataframe");

        let total_mat: Array2<f64> = total
            .0
            .to_ndarray::<Float64Type>(polars::prelude::IndexOrder::C)
            .expect("Failed to get stoichiometry matrix from dataframe");

        let eq_const_arr: Array1<f64> = match log_eq_consts.0.f64() {
            Ok(v) => v.to_ndarray().unwrap().iter().cloned().collect(),
            Err(_) => {
                return Err(PyValueError::new_err(
                    "Failed to get mineral equilibrium constants",
                ))
            }
        };

        let stoich_null_space: Array2<f64> = null_space_scipy(&stoich_mat.clone())?;
        let eq_const_arr: Array1<f64> = match log_eq_consts.0.f64() {
            Ok(v) => v.to_ndarray().unwrap().iter().cloned().collect(),
            Err(_) => {
                return Err(PyValueError::new_err(
                    "Failed to get mineral equilibrium constants",
                ))
            }
        };

        let stoich_pinv: Array2<f64> = pinv_scipy(&stoich_mat.clone())?;
        let x_particular: Array1<f64> = stoich_pinv.dot(&eq_const_arr);

        Ok(Self {
            log10_k_w: eq_const_arr,
            stoich: stoich,
            log_eq_consts: log_eq_consts,
            stoich_null_space: stoich_null_space,
            total: total,
            total_mat: total_mat,
            x_particular: x_particular,
        })
    }

    /// Compact summary reporting the speciation system dimensions: the number
    /// of species (columns of the conservation matrix), the number of
    /// conserved quantities (rows, including charge balance), and the number
    /// of free (independent) log-concentration variables.
    pub fn __repr__(&self) -> String {
        let total = self.total_mat.shape();
        let null_space = self.stoich_null_space.shape();
        format!(
            "EquilibriumParameters(num_species={},num_total={},num_free={})",
            total[1], total[0], null_space[1]
        )
    }

    #[pyo3(signature = (chms, verbose=false, x0=None))]
    pub fn solve_equilibrium<'py>(
        &self,
        py: Python<'py>,
        chms: PyReadonlyArray1<f64>,
        verbose: bool,
        x0: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let chms_arr: Array1<f64> = chms.to_owned_array();

        // No free variables => nothing to re-speciate; return the input
        // concentrations (identity) rather than an empty vector.
        if self.num_free() == 0 {
            return Ok(chms_arr.to_pyarray(py));
        }

        // Optional warm start. If not provided, fall back to a cold start from
        // zeros so that the public method's behaviour is unchanged.
        let initial_x: Array1<f64> = match x0 {
            Some(arr) => arr.to_owned_array(),
            None => Array1::zeros(self.num_free()),
        };

        let v = self.x_free_solve_rust(&chms_arr, &initial_x, verbose)?;
        let concs = self.conc_func_rust(&v);
        Ok(concs.to_pyarray(py))
    }

    pub fn conc_func<'py>(
        &self,
        py: Python<'py>,
        x_free: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let x_free_arr: Array1<f64> = x_free.to_owned_array();

        self.conc_func_rust(&x_free_arr).to_pyarray(py)
    }

    pub fn residual<'py>(
        &self,
        py: Python<'py>,
        x_free: PyReadonlyArray1<f64>,
        c_tot: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let x_free_arr: Array1<f64> = x_free.to_owned_array();
        let c_tot_arr: Array1<f64> = c_tot.to_owned_array();
        self.residual_rust(&x_free_arr, &c_tot_arr).to_pyarray(py)
    }

    /// return the jacobian matrix of this problem
    pub fn residual_jacobian<'py>(
        &self,
        py: Python<'py>,
        x_free: PyReadonlyArray1<f64>,
        c_tot: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let x_free_arr = x_free.to_owned_array();
        let c_tot_arr = c_tot.to_owned_array();

        let f = |x: &Array1<f64>| self.residual_rust(x, &c_tot_arr);

        let jac = approx_fprime(&f, &x_free_arr, false);

        jac.to_pyarray(py)
    }

    pub fn total_mat_shape(&self) -> () {
        let shape = self.total_mat.shape();
        println!("{:?}", shape);
    }

    pub fn get_total_mat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.total_mat.to_pyarray(py)
    }

    pub fn get_stoich_null_space<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.stoich_null_space.to_pyarray(py)
    }

    pub fn get_x_particular<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.x_particular.to_pyarray(py)
    }

    pub fn stoich_null_space_shape(&self) -> () {
        let shape = self.stoich_null_space.shape();
        println!("{:?}", shape);
    }

    pub fn log10_k_w_shape(&self) -> () {
        let shape = self.log10_k_w.shape();
        println!("{:?}", shape);
    }

    pub fn x_particular_shape(&self) -> () {
        let shape = self.x_particular.shape();
        println!("{:?}", shape);
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct MineralAuxParams {
    #[pyo3(get)]
    sw_threshold: f64,
    #[pyo3(get)]
    sw_exp: f64,
    // #[pyo3(get)]
    // n_alpha: f64,
    #[pyo3(get)]
    q_10: f64,
    #[pyo3(get)]
    reactivity: f64,
}

#[pymethods]
impl MineralAuxParams {
    #[new]
    pub fn new(sw_threshold: f64, sw_exp: f64, q_10: f64, reactivity: f64) -> Self {
        Self {
            sw_threshold,
            sw_exp,
            // n_alpha,
            q_10,
            reactivity,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "MineralAuxParams(sw_threshold={},sw_exp={},q_10={},reactivity={})",
            self.sw_threshold, self.sw_exp, self.q_10, self.reactivity
        )
    }

    pub fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let param_vec = vec![
            self.sw_threshold,
            self.sw_exp,
            // self.n_alpha,
            self.q_10,
            self.reactivity,
        ];
        Array1::from_vec(param_vec).to_pyarray(py)
    }

    #[staticmethod]
    pub fn from_array(arr: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let nd_arr: Array1<f64> = arr.to_owned_array();
        if nd_arr.len() != PARAMETERS_PER_MINERAL {
            let msg = format!(
                "Passed incorrect number of parameters to MineralAuxParams::from_array: {}",
                nd_arr
            );
            return Err(PyValueError::new_err(msg));
        } else {
            Ok(Self {
                sw_threshold: nd_arr[0],
                sw_exp: nd_arr[1],
                // n_alpha: nd_arr[2],
                q_10: nd_arr[2],
                reactivity: nd_arr[3],
            })
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct ZoneDimensions {
    #[pyo3(get)]
    pub max_water: f64,
    #[pyo3(get)]
    pub passive_water_storage: f64,
}

#[pymethods]
impl ZoneDimensions {
    #[new]
    pub fn new(max_water: f64, passive_water_storage: f64) -> Self {
        Self {
            max_water,
            passive_water_storage,
        }
    }

    pub fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_array(py, &array![self.max_water, self.passive_water_storage])
    }

    #[staticmethod]
    pub fn from_array(arr: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let x: Array1<f64> = arr.to_owned_array();

        match x.len() {
            2 => Ok(ZoneDimensions {
                max_water: x[0],
                passive_water_storage: x[1],
            }),
            _ => Err(PyValueError::new_err(
                "Incorrect number of arguments passed to `ZoneDimensions::from_array`",
            )),
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "ZoneDimensions(max_water={},passive_water_storage={})",
            self.max_water, self.passive_water_storage
        )
    }

    pub fn to_string(&self) -> String {
        self.__repr__()
    }
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct MineralParameters {
    pub sw_threshold: Array1<f64>,
    pub sw_exp: Array1<f64>,
    // pub n_alpha: Array1<f64>,
    pub q_10: Array1<f64>,
    pub reactivity: Array1<f64>,
}

impl MineralParameters {
    pub fn soil_water_factor_rust(&self, forc: &RtForcing) -> Array1<f64> {
        // Soil-water activity factor: dimensionless s_w ratios with a runtime
        // exponent (powf, sw_exp) — a rate parameter, kept raw f64.
        let num_minerals = self.sw_threshold.shape()[0];
        let mut arr: Array1<f64> = Array1::zeros(num_minerals);

        for i in 0..num_minerals {
            if forc.s_w >= self.sw_threshold[i] {
                arr[i] = ((1.0 - forc.s_w) / (1.0 - self.sw_threshold[i])).powf(self.sw_exp[i])
            } else {
                arr[i] = (forc.s_w / self.sw_threshold[i]).powf(self.sw_exp[i])
            }
        }
        arr
    }

    pub fn temperature_factor_rust(&self, forc: &RtForcing) -> Array1<f64> {
        // Q10 temperature factor q10^((T - 20)/10): absolute °C base and a
        // runtime exponent (powf) — kept raw f64 (uom 0.38: kelvin-only TT,
        // integer-only powers).
        let num_minerals = self.sw_threshold.shape()[0];
        let mut arr: Array1<f64> = Array1::zeros(num_minerals);

        for i in 0..num_minerals {
            arr[i] = self.q_10[i].powf((forc.hydro_forc.temp - 20.0) / 10.0)
        }
        arr
    }

    // pub fn water_table_factor_rust(&self, forc: &RtForcing) -> Array1<f64> {
    //     let num_minerals = self.sw_threshold.shape()[0];
    //     let mut arr: Array1<f64> = Array1::zeros(num_minerals);

    //     for i in 0..num_minerals {
    //         let n_alpha_i = self.n_alpha[i];
    //         let gw_val = match n_alpha_i.abs() >= 1e-12 {
    //             true => 1.0,
    //             false => (-n_alpha_i.abs() * forc.z_w.powf(n_alpha_i.signum())).exp(),
    //         };
    //         arr[i] = gw_val;
    //     }

    //     arr
    // }

    pub fn factor_rust(&self, forc: &RtForcing) -> Array1<f64> {
        let sw_factor: Array1<f64> = self.soil_water_factor_rust(&forc);
        let temp_factor: Array1<f64> = self.temperature_factor_rust(&forc);
        // let gw_factor: Array1<f64> = self.water_table_factor_rust(&forc);
        let fact: Array1<f64> = sw_factor * temp_factor;
        fact
    }
}

#[pymethods]
impl MineralParameters {
    #[new]
    pub fn new<'py>(
        py: Python<'py>,
        sw_threshold: PyReadonlyArray1<f64>,
        sw_exp: PyReadonlyArray1<f64>,
        // n_alpha: PyReadonlyArray1<f64>,
        q_10: PyReadonlyArray1<f64>,
        reactivity: PyReadonlyArray1<f64>,
    ) -> Self {
        Self {
            sw_threshold: sw_threshold.to_owned_array(),
            sw_exp: sw_exp.to_owned_array(),
            // n_alpha: n_alpha.to_owned_array(),
            q_10: q_10.to_owned_array(),
            reactivity: reactivity.to_owned_array(),
        }
    }

    /// Compact summary reporting how many minerals these per-mineral
    /// parameter arrays describe.
    pub fn __repr__(&self) -> String {
        format!(
            "MineralParameters(num_minerals={})",
            self.sw_threshold.len()
        )
    }

    pub fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let num_minerals = self.sw_threshold.len();
        let mut arr: Array1<f64> = Array1::zeros(PARAMETERS_PER_MINERAL * num_minerals);
        let ppm = PARAMETERS_PER_MINERAL;
        for i in 0..num_minerals {
            arr[i * ppm + 0] = self.sw_threshold[i];
            arr[i * ppm + 1] = self.sw_exp[i];
            // arr[i * ppm + 2] = self.n_alpha[i];
            arr[i * ppm + 2] = self.q_10[i];
            arr[i * ppm + 3] = self.reactivity[i];
        }

        arr.to_pyarray(py)
    }

    #[staticmethod]
    #[pyo3(signature=(arr, natural_scales=true))]
    pub fn from_array(arr: PyReadonlyArray1<f64>, natural_scales: bool) -> PyResult<Self> {
        let x: Array1<f64> = arr.to_owned_array();
        // dbg!(&x);
        if x.len() % PARAMETERS_PER_MINERAL != 0 {
            let msg = format!("Failed to create new MineralParameters from array because the shape is wrong, expected multiple of {} but got {}", PARAMETERS_PER_MINERAL, x.len());
            return Err(PyValueError::new_err(msg));
        }
        let num_minerals: usize = x.len() / PARAMETERS_PER_MINERAL;

        let mut sw_threshold: Array1<f64> = Array1::zeros(num_minerals);
        let mut sw_exp: Array1<f64> = Array1::zeros(num_minerals);
        // let mut n_alpha: Array1<f64> = Array1::zeros(num_minerals);
        let mut q_10: Array1<f64> = Array1::zeros(num_minerals);
        let mut reactivity: Array1<f64> = Array1::zeros(num_minerals);

        for i in 0..num_minerals {
            sw_threshold[i] = x[i * PARAMETERS_PER_MINERAL + 0];
            sw_exp[i] = x[i * PARAMETERS_PER_MINERAL + 1];
            // n_alpha[i] = x[i * PARAMETERS_PER_MINERAL + 2];
            q_10[i] = x[i * PARAMETERS_PER_MINERAL + 2];
            reactivity[i] = if natural_scales {
                (10_f64).powf(x[i * PARAMETERS_PER_MINERAL + 3])
            } else {
                x[i * PARAMETERS_PER_MINERAL + 3]
            };
        }

        Ok(Self {
            sw_threshold,
            sw_exp,
            // n_alpha,
            q_10,
            reactivity,
        })
    }

    pub fn soil_water_factor<'py>(
        &self,
        py: Python<'py>,
        forc: &RtForcing,
    ) -> Bound<'py, PyArray1<f64>> {
        self.soil_water_factor_rust(forc).to_pyarray(py)
    }

    pub fn temperature_factor<'py>(
        &self,
        py: Python<'py>,
        forc: &RtForcing,
    ) -> Bound<'py, PyArray1<f64>> {
        self.temperature_factor_rust(forc).to_pyarray(py)
    }

    // pub fn water_table_factor<'py>(
    //     &self,
    //     py: Python<'py>,
    //     forc: &RtForcing,
    // ) -> Bound<'py, PyArray1<f64>> {
    //     self.water_table_factor_rust(forc).to_pyarray(py)
    // }

    pub fn factor<'py>(&self, py: Python<'py>, forc: RtForcing) -> Bound<'py, PyArray1<f64>> {
        self.factor_rust(&forc).to_pyarray(py)
    }

    #[staticmethod]
    pub fn from_mineral_parameters(minerals: Vec<MineralAuxParams>) -> PyResult<Self> {
        let mut sw_thrs: Vec<f64> = Vec::new();
        let mut sw_exps: Vec<f64> = Vec::new();
        // let mut n_alphas: Vec<f64> = Vec::new();
        let mut q_10s: Vec<f64> = Vec::new();
        let mut reactivitys: Vec<f64> = Vec::new();

        for m in minerals {
            sw_thrs.push(m.sw_threshold);
            sw_exps.push(m.sw_exp);
            // n_alphas.push(m.n_alpha);
            q_10s.push(m.q_10);
            reactivitys.push(m.reactivity);
        }

        Ok(Self {
            sw_threshold: Array1::from_vec(sw_thrs),
            sw_exp: Array1::from_vec(sw_exps),
            // n_alpha: Array1::from_vec(n_alphas),
            q_10: Array1::from_vec(q_10s),
            reactivity: Array1::from_vec(reactivitys),
        })
    }

    pub fn get_reactivity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        return self.reactivity.to_pyarray(py);
    }

    pub fn get_sw_threshold<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        return self.sw_threshold.to_pyarray(py);
    }

    pub fn get_sw_exp<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        return self.sw_exp.to_pyarray(py);
    }

    pub fn get_q_10<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        return self.q_10.to_pyarray(py);
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RtParameters {
    #[pyo3(get)]
    pub dimensions: ZoneDimensions,
    #[pyo3(get)]
    pub mineral_params: Option<MineralParameters>,
}

#[pymethods]
impl RtParameters {
    #[new]
    pub fn new(
        dimensions: ZoneDimensions,
        mineral_params: Option<MineralParameters>,
    ) -> RtParameters {
        Self {
            dimensions,
            mineral_params,
        }
    }

    /// Nest the zone-dimensions repr and a compact summary of the (optional)
    /// mineral parameters.
    pub fn __repr__(&self) -> String {
        let mineral_summary: String = match &self.mineral_params {
            Some(v) => format!("MineralParameters(num_minerals={})", v.sw_threshold.len()),
            None => "None".to_owned(),
        };
        format!(
            "RtParameters(dimensions={},mineral_params={})",
            self.dimensions.__repr__(),
            mineral_summary
        )
    }

    pub fn to_array<'py>(
        &self,
        py: Python<'py>,
        include_minerals: bool,
    ) -> Bound<'py, PyArray1<f64>> {
        let dim_arr: Vec<f64> = self.dimensions.to_array(py).to_vec().unwrap();
        let min_arr: Vec<f64> = match &self.mineral_params {
            Some(v) => v.to_array(py).to_vec().unwrap(),
            None => Vec::new(),
        };

        let comps: Vec<Vec<f64>> = vec![dim_arr, min_arr];
        let params: Vec<f64> = comps.concat();

        PyArray1::from_vec(py, params)
    }

    #[staticmethod]
    #[pyo3(signature=(arr, natural_scales=true))]
    pub fn from_array<'py>(
        py: Python<'py>,
        arr: PyReadonlyArray1<f64>,
        natural_scales: bool,
    ) -> PyResult<Self> {
        let x: Array1<f64> = arr.to_owned_array();
        let size_params: Array1<f64> = x.slice(s![0..2]).to_owned();
        let size_params_py: Bound<'_, PyArray1<f64>> = PyArray1::from_array(py, &size_params);
        let dimensions = match ZoneDimensions::from_array(size_params_py.readonly()) {
            Ok(v) => v,
            Err(_) => return Err(PyValueError::new_err("Failed to construct soil parameters")),
        };

        if x.len() == 2 {
            // There are no reactions in this zone
            return Ok(Self {
                dimensions,
                mineral_params: None,
            });
        }

        let min_params: Array1<f64> = x.slice(s![2..]).to_owned();

        let min_params_py: Bound<'_, PyArray1<f64>> = PyArray1::from_array(py, &min_params);
        let mineral_params =
            match MineralParameters::from_array(min_params_py.readonly(), natural_scales) {
                Ok(v) => v,
                Err(e) => {
                    let msg = format!("Failed to construct mineral parameters: {}", e.to_string());
                    return Err(PyValueError::new_err(msg));
                }
            };

        Ok(Self {
            dimensions,
            mineral_params: Some(mineral_params),
        })
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RiverDimensions {
    #[pyo3(get, set)]
    pub bed_depth: f64,
    #[pyo3(get, set)]
    pub passive_water_storage: f64,
}

#[pymethods]
impl RiverDimensions {
    #[new]
    pub fn new(bed_depth: f64, passive_water_storage: f64) -> Self {
        Self {
            bed_depth,
            passive_water_storage,
        }
    }

    #[staticmethod]
    pub fn num_parameters() -> usize {
        2
    }

    pub fn __repr__(&self) -> String {
        format!(
            "RiverDimensions(bed_depth={},passive_water_storage={})",
            self.bed_depth, self.passive_water_storage
        )
    }

    pub fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray::from_vec(py, vec![self.bed_depth, self.passive_water_storage])
    }

    #[staticmethod]
    pub fn from_array(arr: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let x = arr.to_owned_array();

        match x.len() {
            2 => Ok(Self {
                bed_depth: x[0],
                passive_water_storage: x[1],
            }),
            _ => Err(PyValueError::new_err(
                "Incorrect array size passed to RiverDimensions::from_array",
            )),
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RiverParameters {
    #[pyo3(get, set)]
    pub dimensions: RiverDimensions,
    #[pyo3(get, set)]
    pub mineral_params: Option<MineralParameters>,
}

#[pymethods]
impl RiverParameters {
    #[new]
    pub fn new(dimensions: RiverDimensions, mineral_params: Option<MineralParameters>) -> Self {
        Self {
            dimensions,
            mineral_params,
        }
    }

    /// Nest the river-dimensions repr and a compact summary of the (optional)
    /// mineral parameters.
    pub fn __repr__(&self) -> String {
        let mineral_summary: String = match &self.mineral_params {
            Some(v) => format!("MineralParameters(num_minerals={})", v.sw_threshold.len()),
            None => "None".to_owned(),
        };
        format!(
            "RiverParameters(dimensions={},mineral_params={})",
            self.dimensions.__repr__(),
            mineral_summary
        )
    }

    pub fn to_array<'py>(
        &self,
        py: Python<'py>,
        include_minerals: bool,
    ) -> Bound<'py, PyArray1<f64>> {
        let dim_arr: Vec<f64> = self.dimensions.to_array(py).to_vec().unwrap();
        let min_arr: Vec<f64> = match &self.mineral_params {
            Some(v) => v.to_array(py).to_vec().unwrap(),
            None => Vec::new(),
        };

        let comps: Vec<Vec<f64>> = vec![dim_arr, min_arr];
        let params: Vec<f64> = comps.concat();

        PyArray1::from_vec(py, params)
    }

    #[staticmethod]
    #[pyo3(signature=(arr, natural_scales=true))]
    pub fn from_array<'py>(
        py: Python<'py>,
        arr: PyReadonlyArray1<f64>,
        natural_scales: bool,
    ) -> PyResult<Self> {
        let x: Array1<f64> = arr.to_owned_array();
        let size_params: Array1<f64> = x.slice(s![0..RiverDimensions::num_parameters()]).to_owned();
        let size_params_py: Bound<'_, PyArray1<f64>> = PyArray1::from_array(py, &size_params);
        let dimensions = match RiverDimensions::from_array(size_params_py.readonly()) {
            Ok(v) => v,
            Err(_) => return Err(PyValueError::new_err("Failed to construct soil parameters")),
        };

        if x.len() == RiverDimensions::num_parameters() {
            // There are no reactions in this zone
            return Ok(Self {
                dimensions,
                mineral_params: None,
            });
        }

        let min_params: Array1<f64> = x.slice(s![RiverDimensions::num_parameters()..]).to_owned();

        let min_params_py: Bound<'_, PyArray1<f64>> = PyArray1::from_array(py, &min_params);
        let mineral_params =
            match MineralParameters::from_array(min_params_py.readonly(), natural_scales) {
                Ok(v) => v,
                Err(e) => {
                    let msg = format!("Failed to construct mineral parameters: {}", e.to_string());
                    return Err(PyValueError::new_err(msg));
                }
            };

        Ok(Self {
            dimensions,
            mineral_params: Some(mineral_params),
        })
    }
}
