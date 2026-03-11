use std::collections::HashMap;

use numpy::{PyArrayMethods, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

use crate::{
    common_types::{HydroForcing, HydroStep},
    math::find_root_rust,
};

#[pyclass(subclass)]
pub struct HydrologicZone {
    #[pyo3(get, set)]
    __name: String,
}

#[pymethods]
impl HydrologicZone {
    #[new]
    #[pyo3(signature = (name="unnamed".to_owned()))]
    pub fn new(name: String) -> PyClassInitializer<Self> {
        let new_val = Self {
            __name: name.clone(),
        };

        PyClassInitializer::from(new_val)
    }

    pub fn __implicit_eulers_func(&self, s: f64, s_0: f64, d: &HydroForcing, dt: f64) -> f64 {
        (s_0 - s) + dt * self.mass_balance(s, d)
    }

    pub fn step(&self, s_0: f64, d: HydroForcing, dt: f64) -> HydroStep {
        let f = |s| self.__implicit_eulers_func(s, s_0, &d, dt);
        let s_new = find_root_rust(f, s_0).max(0.0);

        HydroStep {
            state: s_new,
            forc_flux: self.forc_flux(s_new, &d),
            lat_flux: self.lat_flux(s_new, &d),
            vert_flux: self.vert_flux(s_new, &d),
            vap_flux: self.vap_flux(s_new, &d),
            q_in: d.q_in,
            lat_flux_ext: self.lat_flux_ext(s_new, &d),
            vert_flux_ext: self.vert_flux_ext(s_new, &d),
        }
    }

    pub fn mass_balance(&self, s: f64, d: &HydroForcing) -> f64 {
        d.q_in + self.forc_flux(s, d)
            - self.vap_flux(s, d)
            - self.lat_flux(s, d)
            - self.vert_flux(s, d)
    }

    pub fn forc_flux(&self, s: f64, d: &HydroForcing) -> f64 {
        0.0
    }

    pub fn vap_flux(&self, s: f64, d: &HydroForcing) -> f64 {
        0.0
    }

    pub fn lat_flux(&self, s: f64, d: &HydroForcing) -> f64 {
        0.0
    }

    pub fn vert_flux(&self, s: f64, d: &HydroForcing) -> f64 {
        0.0
    }

    pub fn lat_flux_ext(&self, s: f64, d: &HydroForcing) -> f64 {
        0.0
    }

    pub fn vert_flux_ext(&self, s: f64, d: &HydroForcing) -> f64 {
        0.0
    }

    pub fn param_list(&self) -> Vec<f64> {
        Vec::new()
    }

    #[getter]
    pub fn name(&self) -> String {
        self.__name.to_owned()
    }

    pub fn columns(&self, _zone_id: usize) -> Vec<String> {
        vec![
            format!("s_{}", self.name()),
            format!("q_forc_{}", self.name()),
            format!("q_vap_{}", self.name()),
            format!("q_lat_{}", self.name()),
            format!("q_vert_{}", self.name()),
            format!("q_lat_ext_{}", self.name()),
            format!("q_vert_ext_{}", self.name()),
        ]
    }

    #[staticmethod]
    pub fn default(py: Python<'_>) -> PyResult<Bound<'_, Self>> {
        // 1. Create the initializer (the blueprint)
        let initializer = PyClassInitializer::from(HydrologicZone::new(Self::base_name()));

        // 2. Use Bound::new to allocate the object on the Python heap
        // This is the "magic" step that consumes the initializer
        let bound_instance = Bound::new(py, initializer)?;

        Ok(bound_instance)
    }

    #[staticmethod]
    pub fn num_parameters() -> usize {
        0
    }

    #[staticmethod]
    pub fn default_parameter_range() -> HashMap<String, (f64, f64)> {
        HashMap::new()
    }

    #[staticmethod]
    pub fn base_name() -> String {
        "unnamed".to_string()
    }

    #[staticmethod]
    pub fn from_array<'py>(arr: PyReadonlyArray1<f64>) -> PyResult<Self> {
        if arr
            .len()
            .expect("Failed to get length of array in `from_array`")
            != Self::num_parameters()
        {
            return Err(PyValueError::new_err("Wrong number of arguments"));
        } else {
            Ok(Self {
                __name: Self::base_name(),
            })
        }
    }

    #[staticmethod]
    pub fn parameter_names() -> Vec<String> {
        Vec::new()
    }

    #[staticmethod]
    pub fn default_init_state() -> f64 {
        0.0
    }

    #[staticmethod]
    pub fn from_dict(params: HashMap<String, f64>) -> Self {
        unimplemented!()
    }
}

#[pyclass(from_py_object, extends=HydrologicZone)]
#[derive(Clone, Debug)]
pub struct SnowZone {
    #[pyo3(get, set)]
    tt: f64,
    #[pyo3(get, set)]
    fmax: f64,
}

#[pymethods]
impl SnowZone {
    #[new]
    #[pyo3(signature=(tt=0.0, fmax=1.0, name="snow".to_owned()))]
    fn new(tt: f64, fmax: f64, name: String) -> PyClassInitializer<Self> {
        let parent = HydrologicZone::new(name);
        let child = Self { tt, fmax };
        PyClassInitializer::from(parent).add_subclass(child)
    }

    pub fn mass_balance(&self, s: f64, d: &HydroForcing) -> f64 {
        d.q_in + self.forc_flux(s, d) - self.vert_flux(s, d)
    }

    pub fn __implicit_eulers_func(&self, s: f64, s_0: f64, d: &HydroForcing, dt: f64) -> f64 {
        (s_0 - s) + dt * self.mass_balance(s, d)
    }

    pub fn step(&self, s_0: f64, d: HydroForcing, dt: f64) -> HydroStep {
        let f = |s| self.__implicit_eulers_func(s, s_0, &d, dt);
        let s_new = find_root_rust(f, s_0).max(0.0);

        HydroStep {
            state: s_new,
            forc_flux: self.forc_flux(s_new, &d),
            lat_flux: 0.0,
            vert_flux: self.vert_flux(s_new, &d),
            vap_flux: 0.0,
            q_in: d.q_in,
            lat_flux_ext: 0.0,
            vert_flux_ext: self.vert_flux_ext(s_new, &d),
        }
    }

    fn forc_flux(&self, s: f64, d: &HydroForcing) -> f64 {
        if d.temp <= self.tt {
            d.precip
        } else {
            0.0
        }
    }

    fn vert_flux(&self, s: f64, d: &HydroForcing) -> f64 {
        if d.temp > self.tt {
            let melt: f64 = self.fmax * (d.temp - self.tt);
            s.min(melt)
        } else {
            0.0
        }
    }

    fn vert_flux_ext(&self, s: f64, d: &HydroForcing) -> f64 {
        if d.temp > self.tt {
            self.vert_flux(s, d) + d.precip
        } else {
            return self.vert_flux(s, d);
        }
    }

    fn param_list(&self) -> Vec<f64> {
        vec![self.tt, self.fmax]
    }

    #[staticmethod]
    fn default(py: Python<'_>) -> PyResult<Bound<'_, Self>> {
        // 1. Create the initializer (the blueprint)
        let initializer = PyClassInitializer::from(HydrologicZone::new(Self::base_name()))
            .add_subclass(Self { tt: 0.0, fmax: 1.0 });

        // 2. Use Bound::new to allocate the object on the Python heap
        // This is the "magic" step that consumes the initializer
        let bound_instance = Bound::new(py, initializer)?;

        Ok(bound_instance)
    }

    #[staticmethod]
    fn parameter_names() -> Vec<String> {
        vec!["tt".to_owned(), "fmax".to_owned()]
    }

    #[staticmethod]
    fn num_parameters() -> usize {
        2
    }

    #[staticmethod]
    fn base_name() -> String {
        "snow".to_owned()
    }

    #[staticmethod]
    fn default_parameter_range() -> HashMap<String, (f64, f64)> {
        HashMap::from([
            ("tt".to_owned(), (-1.0, 1.0)),
            ("fmax".to_owned(), (0.5, 5.0)),
        ])
    }

    #[staticmethod]
    fn from_array<'py>(py: Python<'py>, arr: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, Self>> {
        let vals: Vec<f64> = arr.to_vec().expect("Failed to get parameters");
        if vals.len() != Self::num_parameters() {
            return Err(PyValueError::new_err("Wrong number of arguments"));
        } else {
            let child: PyClassInitializer<SnowZone> =
                Self::new(vals[0], vals[1], Self::base_name());

            return Bound::new(py, child);
        }
    }
}

#[pyclass(from_py_object, extends=HydrologicZone)]
#[derive(Clone, Debug)]
pub struct SurfaceZone {
    #[pyo3(get, set)]
    fc: f64,
    #[pyo3(get, set)]
    lp: f64,
    #[pyo3(get, set)]
    beta: f64,
    #[pyo3(get, set)]
    k0: f64,
    #[pyo3(get, set)]
    thr: f64,
}

#[pymethods]
impl SurfaceZone {
    #[new]
    #[pyo3(signature=(fc=100.0, lp=0.5, beta=1.0, k0=0.1, thr=10.0, name="surface".to_owned()))]
    fn new(
        fc: f64,
        lp: f64,
        beta: f64,
        k0: f64,
        thr: f64,
        name: String,
    ) -> PyClassInitializer<Self> {
        let parent = HydrologicZone::new(name);
        let child = Self {
            fc,
            lp,
            beta,
            k0,
            thr,
        };
        PyClassInitializer::from(parent).add_subclass(child)
    }

    #[staticmethod]
    fn default(py: Python<'_>) -> PyResult<Bound<'_, Self>> {
        // 1. Create the initializer (the blueprint)
        let initializer = PyClassInitializer::from(HydrologicZone::new(Self::base_name()))
            .add_subclass(Self {
                fc: 100.0,
                lp: 0.5,
                beta: 1.0,
                k0: 0.1,
                thr: 10.0,
            });

        // 2. Use Bound::new to allocate the object on the Python heap
        // This is the "magic" step that consumes the initializer
        let bound_instance = Bound::new(py, initializer)?;

        Ok(bound_instance)
    }

    pub fn __implicit_eulers_func(&self, s: f64, s_0: f64, d: &HydroForcing, dt: f64) -> f64 {
        (s_0 - s) + dt * self.mass_balance(s, d)
    }

    pub fn mass_balance(&self, s: f64, d: &HydroForcing) -> f64 {
        d.q_in + self.forc_flux(s, d)
            - self.vert_flux(s, d)
            - self.lat_flux(s, d)
            - self.vap_flux(s, d)
    }

    pub fn step(&self, s_0: f64, d: HydroForcing, dt: f64) -> HydroStep {
        let f = |s| self.__implicit_eulers_func(s, s_0, &d, dt);
        let s_new = find_root_rust(f, s_0).max(0.0);

        HydroStep {
            state: s_new,
            forc_flux: self.forc_flux(s_new, &d),
            lat_flux: self.lat_flux(s_new, &d),
            vert_flux: self.vert_flux(s_new, &d),
            vap_flux: self.vap_flux(s_new, &d),
            q_in: d.q_in,
            lat_flux_ext: self.lat_flux_ext(s_new, &d),
            vert_flux_ext: self.vert_flux_ext(s_new, &d),
        }
    }

    fn forc_flux(&self, s: f64, d: &HydroForcing) -> f64 {
        0.0
    }

    fn vert_flux(&self, s: f64, d: &HydroForcing) -> f64 {
        d.q_in * (s / self.fc).powf(self.beta)
    }

    fn vap_flux(&self, s: f64, d: &HydroForcing) -> f64 {
        d.pet * (s / (self.fc * self.lp)).min(1.0)
    }

    fn lat_flux(&self, s: f64, d: &HydroForcing) -> f64 {
        (self.k0 * (s - self.thr)).max(0.0)
    }

    fn lat_flux_ext(&self, s: f64, d: &HydroForcing) -> f64 {
        0.0
    }

    fn vert_flux_ext(&self, s: f64, d: &HydroForcing) -> f64 {
        0.0
    }

    fn param_list(&self) -> Vec<f64> {
        vec![self.fc, self.lp, self.beta, self.k0, self.thr]
    }

    #[staticmethod]
    fn parameter_names() -> Vec<String> {
        vec![
            "fc".to_owned(),
            "lp".to_owned(),
            "beta".to_owned(),
            "k0".to_owned(),
            "thr".to_owned(),
        ]
    }

    #[staticmethod]
    fn num_parameters() -> usize {
        5
    }

    #[staticmethod]
    fn base_name() -> String {
        "surface".to_owned()
    }

    #[staticmethod]
    fn default_parameter_range() -> HashMap<String, (f64, f64)> {
        HashMap::from([
            ("fc".to_owned(), (50.0, 1_000.0)),
            ("lp".to_owned(), (0.05, 1.0)),
            ("beta".to_owned(), (0.05, 5.0)),
            ("k0".to_owned(), (0.0, 1.0)),
            ("thr".to_owned(), (0.0, 1_000.0)),
        ])
    }

    #[staticmethod]
    fn default_init_state() -> f64 {
        25.0
    }

    #[staticmethod]
    fn from_array<'py>(py: Python<'py>, arr: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, Self>> {
        let vals: Vec<f64> = arr.to_vec().expect("Failed to get parameters");
        if vals.len() != Self::num_parameters() {
            return Err(PyValueError::new_err("Wrong number of arguments"));
        } else {
            let child: PyClassInitializer<Self> = Self::new(
                vals[0],
                vals[1],
                vals[2],
                vals[3],
                vals[4],
                Self::base_name(),
            );

            return Bound::new(py, child);
        }
    }
}

#[pyclass(from_py_object, extends=HydrologicZone)]
#[derive(Clone, Debug)]
pub struct GroundZone {
    #[pyo3(get, set)]
    k: f64,
    #[pyo3(get, set)]
    alpha: f64,
    #[pyo3(get, set)]
    perc: f64,
}

#[pymethods]
impl GroundZone {
    #[new]
    #[pyo3(signature=(k=1e-2, alpha=1.0, perc=1.0, name="ground".to_owned()))]
    fn new(k: f64, alpha: f64, perc: f64, name: String) -> PyClassInitializer<Self> {
        let parent = HydrologicZone::new(name);
        let child = Self { k, alpha, perc };
        PyClassInitializer::from(parent).add_subclass(child)
    }

    #[staticmethod]
    fn default(py: Python<'_>) -> PyResult<Bound<'_, Self>> {
        // 1. Create the initializer (the blueprint)
        let initializer = PyClassInitializer::from(HydrologicZone::new(Self::base_name()))
            .add_subclass(Self {
                k: 1e-2,
                alpha: 1.0,
                perc: 1.0,
            });

        // 2. Use Bound::new to allocate the object on the Python heap
        // This is the "magic" step that consumes the initializer
        let bound_instance = Bound::new(py, initializer)?;

        Ok(bound_instance)
    }

    pub fn mass_balance(&self, s: f64, d: &HydroForcing) -> f64 {
        d.q_in - self.vert_flux(s, d) - self.lat_flux(s, d)
    }

    pub fn __implicit_eulers_func(&self, s: f64, s_0: f64, d: &HydroForcing, dt: f64) -> f64 {
        (s_0 - s) + dt * self.mass_balance(s, d)
    }

    pub fn step(&self, s_0: f64, d: HydroForcing, dt: f64) -> HydroStep {
        let f = |s| self.__implicit_eulers_func(s, s_0, &d, dt);
        let s_new = find_root_rust(f, s_0).max(0.0);

        HydroStep {
            state: s_new,
            forc_flux: 0.0,
            lat_flux: self.lat_flux(s_new, &d),
            vert_flux: self.vert_flux(s_new, &d),
            vap_flux: 0.0,
            q_in: d.q_in,
            lat_flux_ext: 0.0,
            vert_flux_ext: 0.0,
        }
    }

    fn vert_flux(&self, s: f64, d: &HydroForcing) -> f64 {
        s.min(self.perc)
    }

    fn lat_flux(&self, s: f64, d: &HydroForcing) -> f64 {
        if s < 1e-12 {
            0.0
        } else {
            self.k * s.max(0.0).powf(self.alpha)
        }
    }

    fn param_list(&self) -> Vec<f64> {
        vec![self.k, self.alpha, self.perc]
    }

    #[staticmethod]
    fn parameter_names() -> Vec<String> {
        vec!["k".to_owned(), "alpha".to_owned(), "perc".to_owned()]
    }

    #[staticmethod]
    fn num_parameters() -> usize {
        3
    }

    #[staticmethod]
    fn base_name() -> String {
        "ground".to_owned()
    }

    #[staticmethod]
    fn default_parameter_range() -> HashMap<String, (f64, f64)> {
        HashMap::from([
            ("k".to_owned(), (1e-5, 0.1)),
            ("alpha".to_owned(), (0.5, 3.0)),
            ("perc".to_owned(), (0.0, 5.0)),
        ])
    }

    #[staticmethod]
    fn default_init_state() -> f64 {
        10.0
    }

    #[staticmethod]
    fn from_array<'py>(py: Python<'py>, arr: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, Self>> {
        let vals: Vec<f64> = arr.to_vec().expect("Failed to get parameters");
        if vals.len() != Self::num_parameters() {
            return Err(PyValueError::new_err("Wrong number of arguments"));
        } else {
            let child: PyClassInitializer<Self> =
                Self::new(vals[0], vals[1], vals[2], Self::base_name());

            return Bound::new(py, child);
        }
    }
}

#[pyclass(from_py_object, extends=HydrologicZone)]
#[derive(Clone, Debug)]
pub struct GroundZoneB {
    #[pyo3(get, set)]
    k: f64,
    #[pyo3(get, set)]
    alpha: f64,
}

#[pymethods]
impl GroundZoneB {
    #[new]
    #[pyo3(signature=(k=1e-2, alpha=1.0, name="ground_bottom".to_owned()))]
    fn new(k: f64, alpha: f64, name: String) -> PyClassInitializer<Self> {
        let parent = HydrologicZone::new(name);
        let child = Self { k, alpha };
        PyClassInitializer::from(parent).add_subclass(child)
    }

    #[staticmethod]
    fn default(py: Python<'_>) -> PyResult<Bound<'_, Self>> {
        // 1. Create the initializer (the blueprint)
        let initializer = PyClassInitializer::from(HydrologicZone::new(Self::base_name()))
            .add_subclass(Self {
                k: 1e-2,
                alpha: 1.0,
            });

        // 2. Use Bound::new to allocate the object on the Python heap
        // This is the "magic" step that consumes the initializer
        let bound_instance = Bound::new(py, initializer)?;

        Ok(bound_instance)
    }

    pub fn __implicit_eulers_func(&self, s: f64, s_0: f64, d: &HydroForcing, dt: f64) -> f64 {
        (s_0 - s) + dt * self.mass_balance(s, d)
    }

    pub fn mass_balance(&self, s: f64, d: &HydroForcing) -> f64 {
        d.q_in - self.lat_flux(s, d)
    }

    pub fn step(&self, s_0: f64, d: HydroForcing, dt: f64) -> HydroStep {
        let f = |s| self.__implicit_eulers_func(s, s_0, &d, dt);
        let s_new = find_root_rust(f, s_0).max(0.0);

        HydroStep {
            state: s_new,
            forc_flux: 0.0,
            lat_flux: self.lat_flux(s_new, &d),
            vert_flux: 0.0,
            vap_flux: 0.0,
            q_in: d.q_in,
            lat_flux_ext: 0.0,
            vert_flux_ext: 0.0,
        }
    }

    fn lat_flux(&self, s: f64, d: &HydroForcing) -> f64 {
        if s < 1e-12 {
            0.0
        } else {
            self.k * s.max(0.0).powf(self.alpha)
        }
    }

    fn param_list(&self) -> Vec<f64> {
        vec![self.k, self.alpha]
    }

    #[staticmethod]
    fn parameter_names() -> Vec<String> {
        vec!["k".to_owned(), "alpha".to_owned()]
    }

    #[staticmethod]
    fn num_parameters() -> usize {
        2
    }

    #[staticmethod]
    fn base_name() -> String {
        "ground_bottom".to_owned()
    }

    #[staticmethod]
    fn default_parameter_range() -> HashMap<String, (f64, f64)> {
        HashMap::from([
            ("k".to_owned(), (1e-5, 0.1)),
            ("alpha".to_owned(), (0.5, 3.0)),
        ])
    }

    #[staticmethod]
    fn default_init_state() -> f64 {
        10.0
    }

    #[staticmethod]
    fn from_array<'py>(py: Python<'py>, arr: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, Self>> {
        let vals: Vec<f64> = arr.to_vec().expect("Failed to get parameters");
        if vals.len() != Self::num_parameters() {
            return Err(PyValueError::new_err("Wrong number of arguments"));
        } else {
            let child: PyClassInitializer<Self> = Self::new(vals[0], vals[1], Self::base_name());

            return Bound::new(py, child);
        }
    }
}
