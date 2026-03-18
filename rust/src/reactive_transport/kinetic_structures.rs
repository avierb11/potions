use numpy::{
    array,
    ndarray::{s, Array1, Array2},
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, ToPyArray,
};
use polars::{frame::DataFrame, prelude::Float64Type};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_polars::{PyDataFrame, PySeries};

use crate::{
    common_types::RtForcing,
    math::{find_root_multi, matmul, null_space_scipy, pinv_scipy, OtherError, RootFindingError},
};
const PARAMETERS_PER_MINERAL: usize = 5;

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
    pub fn rate_rust(&self, chms: &Array1<f64>) -> Array1<f64> {
        let n_minerals: usize = self.monod_np.shape()[0];
        let n_species: usize = self.monod_np.shape()[1];
        let mut monod: Array1<f64> = Array1::zeros(n_minerals);
        let mut inhib: Array1<f64> = Array1::zeros(n_minerals);

        for i in 0..n_minerals {
            let mut prod = 1.0;
            for j in 0..n_species {
                let monod_ij = self.monod_np[(i, j)];
                if monod_ij.is_finite() {
                    prod *= chms[j] / (monod_ij + chms[j]);
                }
            }
            monod[i] = prod;
        }

        for i in 0..n_minerals {
            let mut prod = 1.0;
            for j in 0..n_species {
                let inhib_ij = self.inhib_np[(i, j)];
                if inhib_ij.is_finite() {
                    prod *= inhib_ij / (inhib_ij + chms[j]);
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
    // Calculate the solubility product for each of the minerals
    pub fn calculate_solubility_product(&self, chms: &Array1<f64>) -> Array1<f64> {
        let log_conc: Array1<f64> = chms.map(|x| x.log10());
        let n_minerals: usize = self.stoich_np.shape()[0];
        let n_species: usize = self.stoich_np.shape()[1];
        let mut log_qs: Array1<f64> = Array1::zeros(n_minerals);

        for i in 0..n_minerals {
            let mut q_i = 0.0;
            for (j, log_c_j) in log_conc.iter().enumerate() {
                q_i += *log_c_j * self.stoich_np[(i, j)];
            }
            log_qs[i] = q_i;
        }

        log_qs.map(|x| (10_f64).powf(*x)) // Convert back to linear scale
    }

    pub fn calculate_dependence_term(&self, chms: &Array1<f64>) -> Array1<f64> {
        let log_conc: Array1<f64> = chms.map(|x| x.log10());
        let n_minerals: usize = self.stoich_np.shape()[0];
        let n_species: usize = self.stoich_np.shape()[1];
        let mut log_deps: Array1<f64> = Array1::zeros(n_minerals);

        for i in 0..n_minerals {
            let mut q_i = 0.0;
            for (j, log_c_j) in log_conc.iter().enumerate() {
                q_i += *log_c_j * self.dep_np[(i, j)];
            }
            log_deps[i] = q_i;
        }

        log_deps.map(|x| (10_f64).powf(*x)) // Convert back to linear scale
    }

    pub fn rate_rust(&self, chms: &Array1<f64>) -> Array1<f64> {
        let solubility_product: Array1<f64> = self.calculate_solubility_product(chms);
        let dependence: Array1<f64> = self.calculate_dependence_term(chms);

        let tst_rates: Array1<f64> =
            dependence * (1.0 - solubility_product / &self.min_eq_const_np);

        tst_rates
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
    pub fn conc_func_rust(&self, x_free: &Array1<f64>) -> Array1<f64> {
        let exp: Array1<f64> = self.stoich_null_space.dot(x_free) + &self.x_particular;
        exp.map(|x| (10.0f64).powf(*x))
    }

    pub fn residual_rust(&self, x_free: &Array1<f64>, c_tot: &Array1<f64>) -> Array1<f64> {
        c_tot - &self.total_mat.dot(&self.conc_func_rust(x_free))
    }

    pub fn solve_equilibrium_rust(
        &self,
        chms: &Array1<f64>,
    ) -> Result<Array1<f64>, RootFindingError> {
        let chms_arr: Array1<f64> = chms.clone();
        // let mut c_tot: Array1<f64> = self.total_mat.t().dot(&chms_arr);
        let mut c_tot: Array1<f64> = match matmul(&self.total_mat, &chms_arr) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Matrix multiplication error. Shapes were wrong");
                return Err(RootFindingError::Other(OtherError));
            }
        };

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

        // Create the callable function for solving the root
        let f_to_solve = |x: &Array1<f64>| self.residual_rust(x, &c_tot);

        let initial_guess: Array1<f64> = Array1::zeros(c_tot.shape()[0]);

        // Solve the problem
        match find_root_multi(&f_to_solve, initial_guess) {
            Ok(v) => Ok(self.conc_func_rust(&v)),
            Err(e) => Err(e),
        }
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

    pub fn solve_equilibrium<'py>(
        &self,
        py: Python<'py>,
        chms: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let chms_arr: Array1<f64> = chms.to_owned_array();

        match self.solve_equilibrium_rust(&chms_arr) {
            Ok(v) => Ok(v.to_pyarray(py)),
            Err(_) => Err(PyValueError::new_err("Failed to solve speciation")),
        }
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
    #[pyo3(get)]
    n_alpha: f64,
    #[pyo3(get)]
    q_10: f64,
    #[pyo3(get)]
    ssa: f64,
}

#[pymethods]
impl MineralAuxParams {
    #[new]
    pub fn new(sw_threshold: f64, sw_exp: f64, n_alpha: f64, q_10: f64, ssa: f64) -> Self {
        Self {
            sw_threshold,
            sw_exp,
            n_alpha,
            q_10,
            ssa,
        }
    }

    pub fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let param_vec = vec![
            self.sw_threshold,
            self.sw_exp,
            self.n_alpha,
            self.q_10,
            self.ssa,
        ];
        Array1::from_vec(param_vec).to_pyarray(py)
    }

    #[staticmethod]
    pub fn from_array(arr: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let nd_arr: Array1<f64> = arr.to_owned_array();
        if nd_arr.len() != 5 {
            let msg = format!(
                "Passed incorrect number of parameters to MineralAuxParams::from_array: {}",
                nd_arr
            );
            return Err(PyValueError::new_err(msg));
        } else {
            Ok(Self {
                sw_threshold: nd_arr[0],
                sw_exp: nd_arr[1],
                n_alpha: nd_arr[2],
                q_10: nd_arr[3],
                ssa: nd_arr[4],
            })
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct ZoneDimensions {
    #[pyo3(get)]
    porosity: f64,
    #[pyo3(get)]
    depth: f64,
    #[pyo3(get)]
    passive_water_storage: f64,
}

#[pymethods]
impl ZoneDimensions {
    #[new]
    pub fn new(porosity: f64, depth: f64, passive_water_storage: f64) -> Self {
        Self {
            porosity,
            depth,
            passive_water_storage,
        }
    }

    pub fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_array(
            py,
            &array![self.porosity, self.depth, self.passive_water_storage],
        )
    }

    #[staticmethod]
    pub fn from_array(arr: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let x: Array1<f64> = arr.to_owned_array();

        match x.len() {
            3 => Ok(ZoneDimensions {
                porosity: x[0],
                depth: x[1],
                passive_water_storage: x[2],
            }),
            _ => Err(PyValueError::new_err(
                "Incorrect number of arguments passed to `ZoneDimensions::from_array`",
            )),
        }
    }

    #[getter]
    pub fn max_water_volume(&self) -> f64 {
        self.porosity * self.depth - self.passive_water_storage
    }
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct MineralParameters {
    pub sw_threshold: Array1<f64>,
    pub sw_exp: Array1<f64>,
    pub n_alpha: Array1<f64>,
    pub q_10: Array1<f64>,
    pub ssa: Array1<f64>,
}

impl MineralParameters {
    pub fn soil_water_factor_rust(&self, forc: &RtForcing) -> Array1<f64> {
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
        let num_minerals = self.sw_threshold.shape()[0];
        let mut arr: Array1<f64> = Array1::zeros(num_minerals);

        for i in 0..num_minerals {
            arr[i] = self.q_10[i].powf((forc.hydro_forc.temp - 20.0) / 10.0)
        }
        arr
    }

    pub fn water_table_factor_rust(&self, forc: &RtForcing) -> Array1<f64> {
        let num_minerals = self.sw_threshold.shape()[0];
        let mut arr: Array1<f64> = Array1::zeros(num_minerals);

        for i in 0..num_minerals {
            let n_alpha_i = self.n_alpha[i];
            let gw_val = match n_alpha_i.abs() >= 1e-12 {
                true => 1.0,
                false => (-n_alpha_i.abs() * forc.z_w.powf(n_alpha_i.signum())).exp(),
            };
            arr[i] = gw_val;
        }

        arr
    }

    pub fn factor_rust(&self, forc: &RtForcing) -> Array1<f64> {
        let sw_factor: Array1<f64> = self.soil_water_factor_rust(&forc);
        let temp_factor: Array1<f64> = self.temperature_factor_rust(&forc);
        let gw_factor: Array1<f64> = self.water_table_factor_rust(&forc);
        let fact: Array1<f64> = sw_factor * temp_factor * gw_factor;
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
        n_alpha: PyReadonlyArray1<f64>,
        q_10: PyReadonlyArray1<f64>,
        ssa: PyReadonlyArray1<f64>,
    ) -> Self {
        Self {
            sw_threshold: sw_threshold.to_owned_array(),
            sw_exp: sw_exp.to_owned_array(),
            n_alpha: n_alpha.to_owned_array(),
            q_10: q_10.to_owned_array(),
            ssa: ssa.to_owned_array(),
        }
    }

    pub fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let num_minerals = self.sw_threshold.len();
        let mut arr: Array1<f64> = Array1::zeros(PARAMETERS_PER_MINERAL * num_minerals);
        let ppm = PARAMETERS_PER_MINERAL;
        for i in 0..num_minerals {
            arr[i * ppm + 0] = self.sw_threshold[i];
            arr[i * ppm + 1] = self.sw_exp[i];
            arr[i * ppm + 2] = self.n_alpha[i];
            arr[i * ppm + 3] = self.q_10[i];
            arr[i * ppm + 4] = self.ssa[i];
        }

        arr.to_pyarray(py)
    }

    #[staticmethod]
    pub fn from_array(arr: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let x: Array1<f64> = arr.to_owned_array();
        if x.len() % PARAMETERS_PER_MINERAL != 0 {
            return Err(PyValueError::new_err(
                "Failed to create new MineralParameters from array",
            ));
        }
        let num_minerals: usize = x.len() / PARAMETERS_PER_MINERAL;

        let mut sw_threshold: Array1<f64> = Array1::zeros(num_minerals);
        let mut sw_exp: Array1<f64> = Array1::zeros(num_minerals);
        let mut n_alpha: Array1<f64> = Array1::zeros(num_minerals);
        let mut q_10: Array1<f64> = Array1::zeros(num_minerals);
        let mut ssa: Array1<f64> = Array1::zeros(num_minerals);

        for i in 0..num_minerals {
            sw_threshold[i] = x[i * PARAMETERS_PER_MINERAL + 0];
            sw_exp[i] = x[i * PARAMETERS_PER_MINERAL + 1];
            n_alpha[i] = x[i * PARAMETERS_PER_MINERAL + 2];
            q_10[i] = x[i * PARAMETERS_PER_MINERAL + 3];
            ssa[i] = x[i * PARAMETERS_PER_MINERAL + 4];
        }

        Ok(Self {
            sw_threshold,
            sw_exp,
            n_alpha,
            q_10,
            ssa,
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

    pub fn water_table_factor<'py>(
        &self,
        py: Python<'py>,
        forc: &RtForcing,
    ) -> Bound<'py, PyArray1<f64>> {
        self.water_table_factor_rust(forc).to_pyarray(py)
    }

    pub fn factor<'py>(&self, py: Python<'py>, forc: RtForcing) -> Bound<'py, PyArray1<f64>> {
        self.factor_rust(&forc).to_pyarray(py)
    }

    #[staticmethod]
    pub fn from_mineral_parameters(minerals: Vec<MineralAuxParams>) -> PyResult<Self> {
        let mut sw_thrs: Vec<f64> = Vec::new();
        let mut sw_exps: Vec<f64> = Vec::new();
        let mut n_alphas: Vec<f64> = Vec::new();
        let mut q_10s: Vec<f64> = Vec::new();
        let mut ssas: Vec<f64> = Vec::new();

        for m in minerals {
            sw_thrs.push(m.sw_threshold);
            sw_exps.push(m.sw_exp);
            n_alphas.push(m.n_alpha);
            q_10s.push(m.q_10);
            ssas.push(m.ssa);
        }

        Ok(Self {
            sw_threshold: Array1::from_vec(sw_thrs),
            sw_exp: Array1::from_vec(sw_exps),
            n_alpha: Array1::from_vec(n_alphas),
            q_10: Array1::from_vec(q_10s),
            ssa: Array1::from_vec(ssas),
        })
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RtParameters {
    #[pyo3(get)]
    pub dimensions: ZoneDimensions,
    #[pyo3(get)]
    pub mineral_params: MineralParameters,
}

#[pymethods]
impl RtParameters {
    #[new]
    pub fn new(dimensions: ZoneDimensions, mineral_params: MineralParameters) -> RtParameters {
        Self {
            dimensions,
            mineral_params,
        }
    }

    pub fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let dim_arr = self.dimensions.to_array(py);
        let min_arr = self.mineral_params.to_array(py);

        let comps = vec![dim_arr.to_vec().unwrap(), min_arr.to_vec().unwrap()];
        let params = comps.concat();

        PyArray1::from_vec(py, params)
    }

    #[staticmethod]
    pub fn from_array<'py>(py: Python<'py>, arr: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let x: Array1<f64> = arr.to_owned_array();
        let size_params: Array1<f64> = x.slice(s![0..3]).to_owned();
        let min_params: Array1<f64> = x.slice(s![3..]).to_owned();

        let size_params_py: Bound<'_, PyArray1<f64>> = PyArray1::from_array(py, &size_params);
        let dimensions = match ZoneDimensions::from_array(size_params_py.readonly()) {
            Ok(v) => v,
            Err(_) => return Err(PyValueError::new_err("Failed to construct soil parameters")),
        };

        let min_params_py: Bound<'_, PyArray1<f64>> = PyArray1::from_array(py, &min_params);
        let mineral_params = match MineralParameters::from_array(min_params_py.readonly()) {
            Ok(v) => v,
            Err(_) => {
                return Err(PyValueError::new_err(
                    "Failed to construct mineral parameters",
                ))
            }
        };

        Ok(Self {
            dimensions,
            mineral_params: mineral_params,
        })
    }
}
