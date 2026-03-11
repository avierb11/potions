#![allow(dead_code)]
#![allow(unused_variables)]

use pyo3::prelude::*;

use crate::{
    common_types::{ForcingData, HydroForcing, HydroStep, LapseRateParameters, RtForcing},
    hydro::{GroundZone, GroundZoneB, HydrologicZone, SnowZone, SurfaceZone},
};
pub mod common_types;
pub mod hydro;
pub mod math;

#[pyfunction]
fn hello_rust() -> PyResult<String> {
    Ok("Hello from Rust!".into())
}

// #[pymodule]
// fn core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(hello_rust, m)?)?;
//     #[pymodule_export]
//     use common_types::ForcingData;
//     Ok(())
// }

#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ForcingData>()?;
    m.add_class::<HydroForcing>()?;
    m.add_class::<HydroStep>()?;
    m.add_class::<LapseRateParameters>()?;
    m.add_class::<RtForcing>()?;

    // #[pymodule_export]
    // use crate::hydro::{GroundZone, GroundZoneB, SnowZone, SurfaceZone};
    m.add_class::<GroundZone>()?;
    m.add_class::<GroundZoneB>()?;
    m.add_class::<SnowZone>()?;
    m.add_class::<SurfaceZone>()?;
    m.add_class::<HydrologicZone>()?;

    Ok(())
}
