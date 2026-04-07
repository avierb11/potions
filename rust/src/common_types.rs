use std::collections::HashMap;

use numpy::{
    ndarray::{Array1, Array2},
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
use pyo3::prelude::*;
use pyo3_polars::PySeries;

use crate::{celsius, mm_water, molar, moles};

pub const ZERO_CONC: f64 = 1e-20;

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct LapseRateParameters {
    #[pyo3(get, set)]
    pub temp_factor: f64,
    #[pyo3(get, set)]
    pub precip_factor: f64,
}

#[pymethods]
impl LapseRateParameters {
    #[new]
    pub fn new(temp_factor: f64, precip_factor: f64) -> PyResult<Self> {
        Ok(Self {
            temp_factor,
            precip_factor,
        })
    }

    pub fn scale_temperature(&self, gauge_elevation: f64, temp_series: PySeries) -> PySeries {
        unimplemented!()
    }

    pub fn scale_precipitation(&self, gauge_elevation: f64, precip_series: PySeries) -> PySeries {
        unimplemented!()
    }

    pub fn scale_forcing_data(
        &self,
        gauge_elevation: f64,
        elev: f64,
        forcing_data: &HydroForcing,
    ) -> HydroForcing {
        unimplemented!()
    }

    #[staticmethod]
    pub fn default_parameter_range() -> HashMap<String, (f64, f64)> {
        unimplemented!()
    }

    #[staticmethod]
    pub fn from_dict(data: HashMap<String, f64>) -> LapseRateParameters {
        unimplemented!()
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct HydroForcing {
    #[pyo3(get, set)]
    pub precip: mm_water,
    #[pyo3(get, set)]
    pub temp: celsius,
    #[pyo3(get, set)]
    pub pet: mm_water,
    #[pyo3(get, set)]
    pub q_in: mm_water,
}

#[pymethods]
impl HydroForcing {
    #[new]
    pub fn new(precip: f64, temp: f64, pet: f64, q_in: f64) -> PyResult<Self> {
        Ok(Self {
            precip,
            temp,
            pet,
            q_in,
        })
    }

    pub fn __repr__(&self) -> String {
        format!(
            "HydroForcing(precip={:.2},temp={:.2},pet={:.2},q_in={:.2})",
            self.precip, self.temp, self.pet, self.q_in
        )
    }

    pub fn to_string(&self) -> String {
        self.__repr__()
    }

    pub fn __eq__(&self, other: HydroForcing) -> bool {
        let diffs: Vec<f64> = vec![
            self.precip - other.precip,
            self.temp - other.temp,
            self.pet - other.pet,
            self.q_in - other.q_in,
        ];
        diffs.iter().map(|x| x.abs()).all(|x| x <= 1e-12)
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct HydroStep {
    #[pyo3(get, set)]
    pub state: mm_water,
    #[pyo3(get, set)]
    pub forc_flux: mm_water,
    #[pyo3(get, set)]
    pub lat_flux: mm_water,
    #[pyo3(get, set)]
    pub vert_flux: mm_water,
    #[pyo3(get, set)]
    pub vap_flux: mm_water,
    #[pyo3(get, set)]
    pub q_in: mm_water,
    #[pyo3(get, set)]
    pub lat_flux_ext: mm_water,
    #[pyo3(get, set)]
    pub vert_flux_ext: mm_water,
}

#[pymethods]
impl HydroStep {
    #[new]
    fn new(
        state: f64,
        forc_flux: f64,
        lat_flux: f64,
        vert_flux: f64,
        vap_flux: f64,
        q_in: f64,
        lat_flux_ext: f64,
        vert_flux_ext: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            state,
            forc_flux,
            lat_flux,
            vert_flux,
            vap_flux,
            q_in,
            lat_flux_ext,
            vert_flux_ext,
        })
    }

    pub fn __repr__(&self) -> String {
        format!(
            "HydroStep(\n\tstate={:.2},\n\tforc_flux={:.2},\n\tlat_flux={:.2},\n\tvert_flux={:.2},\n\tvap_flux={:.2},\n\tq_in={:.2},\n\tlat_flux_ext={:.2},\n\tvert_flux_ext={:.2})",
            self.state, self.forc_flux, self.lat_flux, self.vert_flux, self.vap_flux, self.q_in, self.lat_flux_ext, self.vert_flux_ext
        )
    }

    pub fn to_string(&self) -> String {
        self.__repr__()
    }

    // Total amount of water exiting the zone and carrying dissolved solutes
    pub fn q_internal(&self) -> f64 {
        self.lat_flux + self.vert_flux
    }

    // Total amount of water that does not interact with the zone
    pub fn q_external(&self) -> f64 {
        self.lat_flux_ext + self.vert_flux_ext - self.q_internal()
    }

    // Total rate of water exiting the zone
    pub fn total_water_out(&self) -> f64 {
        self.lat_flux + self.vert_flux + self.vap_flux
    }

    pub fn __eq__(&self, other: HydroStep) -> bool {
        let diffs: Vec<f64> = vec![
            self.state - other.state,
            self.forc_flux - other.forc_flux,
            self.lat_flux - other.lat_flux,
            self.vert_flux - other.vert_flux,
            self.vap_flux - other.vap_flux,
            self.q_in - other.q_in,
            self.lat_flux_ext - other.lat_flux_ext,
            self.vert_flux_ext - other.vert_flux_ext,
        ];
        diffs.iter().map(|x| x.abs()).all(|x| x <= 1e-12)
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RtForcing {
    pub _conc_in: Array1<molar>,
    #[pyo3(get, set)]
    pub hydro_step: HydroStep,
    #[pyo3(get, set)]
    pub hydro_forc: HydroForcing,
    #[pyo3(get, set)]
    pub s_w: f64,
    #[pyo3(get, set)]
    pub z_w: f64,
}

#[pymethods]
impl RtForcing {
    #[new]
    fn new(
        conc_in: PyReadonlyArray1<f64>,
        hydro_step: HydroStep,
        hydro_forc: HydroForcing,
        s_w: f64,
        z_w: f64,
    ) -> PyResult<Self> {
        let conc_arr: Array1<f64> = conc_in.to_owned_array();
        Ok(Self {
            _conc_in: conc_arr,
            hydro_step,
            hydro_forc,
            s_w,
            z_w,
        })
    }

    #[getter]
    fn conc_in<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self._conc_in.to_pyarray(py)
    }

    #[setter(conc_in)]
    fn set_conc_in<'py>(&mut self, py: Python<'py>, arr: PyReadonlyArray1<f64>) -> PyResult<()> {
        self._conc_in = arr.to_owned_array();
        Ok(())
    }

    #[getter]
    fn q_out(&self) -> f64 {
        self.hydro_step.lat_flux_ext + self.hydro_step.vert_flux_ext
    }

    pub fn print_forc(&self) -> () {
        dbg!(self);
    }

    fn __repr__(&self) -> String {
        format!(
            "RtForcing(\n\tconc_in=array({}),\nhydro_step={},\nhydro_forc={},\ns_w={:.2},\nz_w={}\n)",
            self._conc_in.to_string(),
            self.hydro_step.to_string(),
            self.hydro_forc.to_string(),
            self.s_w,
            self.z_w
        )
    }

    pub fn __eq__(&self, other: &RtForcing) -> bool {
        let conc_eq: bool = (&self._conc_in - &other._conc_in)
            .map(|x| x.abs())
            .iter()
            .all(|x| *x < 1e-12);

        let comps: Vec<bool> = vec![
            conc_eq,
            self.hydro_step.__eq__(other.hydro_step.clone()),
            self.hydro_forc.__eq__(other.hydro_forc.clone()),
            (self.s_w - other.s_w).abs() < 1e-12,
            (self.z_w - other.z_w).abs() < 1e-12,
        ];

        comps.iter().all(|x| *x)
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }
}

#[pyclass]
#[derive(Debug)]
pub struct RtStep {
    #[pyo3(get, set)]
    pub state: Py<PyArray1<molar>>,
    #[pyo3(get, set)]
    pub total_moles: Py<PyArray1<moles>>,
    #[pyo3(get, set)]
    pub conc_in: Py<PyArray1<molar>>,
    #[pyo3(get, set)]
    pub mass_in: Py<PyArray1<moles>>,
    #[pyo3(get, set)]
    pub lat_conc: Py<PyArray1<molar>>,
    #[pyo3(get, set)]
    pub vert_conc: Py<PyArray1<molar>>,
    #[pyo3(get, set)]
    pub lat_mass: Py<PyArray1<moles>>,
    #[pyo3(get, set)]
    pub vert_mass: Py<PyArray1<moles>>,
    #[pyo3(get, set)]
    pub mineral_rates: Py<PyArray1<f64>>,
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct MiscData {
    pub mineral_stoichiometry: Array2<f64>,
    pub species_mobility: Array1<bool>,
    pub mineral_molar_mass: Array1<f64>,
    pub rate_const: Array1<f64>,
}

#[pymethods]
impl MiscData {
    #[new]
    pub fn new(
        mineral_stoichiometry: PyReadonlyArray2<f64>,
        species_mobility: PyReadonlyArray1<bool>,
        mineral_molar_mass: PyReadonlyArray1<f64>,
        rate_const: PyReadonlyArray1<f64>,
    ) -> Self {
        Self {
            mineral_stoichiometry: mineral_stoichiometry.to_owned_array(),
            species_mobility: species_mobility.to_owned_array(),
            mineral_molar_mass: mineral_molar_mass.to_owned_array(),
            rate_const: rate_const.to_owned_array(),
        }
    }

    pub fn get_mineral_stoichiometry<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.mineral_stoichiometry.to_pyarray(py)
    }

    pub fn get_species_mobility<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.species_mobility.to_pyarray(py)
    }

    pub fn get_mineral_molar_mass<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.mineral_molar_mass.to_pyarray(py)
    }

    pub fn get_rate_const<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.rate_const.to_pyarray(py)
    }
}
