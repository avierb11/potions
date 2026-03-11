use std::collections::HashMap;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PySeries;

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct ForcingData {
    #[pyo3(get, set)]
    pub precip: f64,
    #[pyo3(get, set)]
    pub temp: f64,
    #[pyo3(get, set)]
    pub pet: f64,
}

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
        forcing_data: &ForcingData,
    ) -> ForcingData {
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
    pub precip: f64,
    #[pyo3(get, set)]
    pub temp: f64,
    #[pyo3(get, set)]
    pub pet: f64,
    #[pyo3(get, set)]
    pub q_in: f64,
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
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct HydroStep {
    #[pyo3(get, set)]
    pub state: f64,
    #[pyo3(get, set)]
    pub forc_flux: f64,
    #[pyo3(get, set)]
    pub lat_flux: f64,
    #[pyo3(get, set)]
    pub vert_flux: f64,
    #[pyo3(get, set)]
    pub vap_flux: f64,
    #[pyo3(get, set)]
    pub q_in: f64,
    #[pyo3(get, set)]
    pub lat_flux_ext: f64,
    #[pyo3(get, set)]
    pub vert_flux_ext: f64,
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
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RtForcing {
    #[pyo3(get, set)]
    conc_in: Vec<f64>,
    #[pyo3(get, set)]
    hydro_step: HydroStep,
    #[pyo3(get, set)]
    hydro_forc: HydroForcing,
    #[pyo3(get, set)]
    s_w: f64,
    #[pyo3(get, set)]
    z_w: f64,
}

#[pymethods]
impl RtForcing {
    #[new]
    fn new(
        conc_in: Vec<f64>,
        hydro_step: HydroStep,
        hydro_forc: HydroForcing,
        s_w: f64,
        z_w: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            conc_in,
            hydro_step,
            hydro_forc,
            s_w,
            z_w,
        })
    }

    #[getter]
    fn q_out(&self) -> f64 {
        self.hydro_step.lat_flux_ext + self.hydro_step.vert_flux_ext
    }
}
