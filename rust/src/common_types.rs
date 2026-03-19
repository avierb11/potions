use std::collections::HashMap;

use numpy::{ndarray::Array1, PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray};
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

    pub fn __repr__(&self) -> String {
        format!(
            "HydroForcing(precip={:.2},temp={:.2},pet={:.2},q_in={:.2})",
            self.precip, self.temp, self.pet, self.q_in
        )
    }

    pub fn to_string(&self) -> String {
        self.__repr__()
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

    pub fn __repr__(&self) -> String {
        format!(
            "HydroStep(state={:.2},forc_flux={:.2},lat_flux={:.2},vert_flux={:.2},vap_flux={:.2},q_in={:.2},lat_flux_ext={:.2},vert_flux_ext={:.2})",
            self.state, self.forc_flux, self.lat_flux, self.vert_flux, self.vap_flux, self.q_in, self.lat_flux_ext, self.vert_flux_ext
        )
    }

    pub fn to_string(&self) -> String {
        self.__repr__()
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RtForcing {
    pub _conc_in: Array1<f64>,
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
            "RtForcing(\n\tconc_in=array({}),\n\thydro_step={},\n\thydro_forc={},\n\ts_w={:.2},\n\tz_w={}\n)",
            self._conc_in.to_string(),
            self.hydro_step.to_string(),
            self.hydro_forc.to_string(),
            self.s_w,
            self.z_w
        )
    }
}
