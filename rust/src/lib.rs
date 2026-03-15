#![allow(dead_code)]
#![allow(unused_variables)]

use pyo3::prelude::*;

use crate::{
    common_types::{ForcingData, HydroForcing, HydroStep, LapseRateParameters, RtForcing},
    hydro::{GroundZone, GroundZoneB, HydrologicZone, SnowZone, SurfaceZone},
    reactive_transport::{
        kinetic_structures::{
            EquilibriumParameters, MineralAuxParams, MineralParameters, MonodParameters,
            RtParameters, TstParameters, ZoneDimensions,
        },
        rt_zone::RtZone,
    },
};
pub mod common_types;
pub mod hydro;
pub mod math;
pub mod reactive_transport;

#[pyfunction]
fn hello_rust() -> PyResult<String> {
    Ok("Hello from Rust!".into())
}

#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ForcingData>()?;
    m.add_class::<HydroForcing>()?;
    m.add_class::<HydroStep>()?;
    m.add_class::<LapseRateParameters>()?;
    m.add_class::<RtForcing>()?;

    // Hydrology
    m.add_class::<GroundZone>()?;
    m.add_class::<GroundZoneB>()?;
    m.add_class::<SnowZone>()?;
    m.add_class::<SurfaceZone>()?;
    m.add_class::<HydrologicZone>()?;

    // Kinetic structures
    m.add_class::<MonodParameters>()?;
    m.add_class::<TstParameters>()?;
    m.add_class::<EquilibriumParameters>()?;
    m.add_class::<MineralAuxParams>()?;
    m.add_class::<ZoneDimensions>()?;
    m.add_class::<MineralParameters>()?;
    m.add_class::<RtParameters>()?;

    // RtZone
    m.add_class::<RtZone>()?;

    Ok(())
}
