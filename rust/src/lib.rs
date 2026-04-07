#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_camel_case_types)]

#[macro_use]
extern crate uom;
use pyo3::{create_exception, exceptions::PyException, prelude::*};

unit! {
    system: uom::si;
    quantity: uom::si::length;
    @millimeter_water: 1_000.0; "mm", "millimeter", "millimeters";
}

type molar = f64;
type moles = f64;
type mm_water = f64;
type celsius = f64;
type moles_per_time = f64;
type molar_per_time = f64;

use crate::{
    common_types::{HydroForcing, HydroStep, LapseRateParameters, RtForcing, RtStep},
    hydro::{GroundZone, GroundZoneB, HydrologicZone, SnowZone, SurfaceZone},
    math::OptimizationError,
    reactive_transport::{
        database::{
            ChemicalDatabase, ExchangeReaction, MineralKineticData, MineralSpecies, MonodReaction,
            PrimaryAqueousSpecies, SecondarySpecies, TstReaction,
        },
        kinetic_structures::{
            EquilibriumParameters, MineralAuxParams, MineralParameters, MonodParameters,
            RiverDimensions, RiverParameters, RtParameters, TstParameters, ZoneDimensions,
        },
        reaction_network::ReactionNetwork,
        river_zone::RiverZone,
        rt_zone::RtZone,
    },
};
pub mod common_types;
pub mod hydro;
pub mod math;
pub mod reactive_transport;

create_exception!(math, ScalarRootFindingError, PyException);
create_exception!(math, RootFindingError, PyException);
create_exception!(math, IterationError, RootFindingError);
create_exception!(math, LinearSystemError, RootFindingError);
create_exception!(math, OtherError, RootFindingError);
create_exception!(math, MatMulError, PyException);

#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_class::<MineralParameters>()?;
    m.add_class::<EquilibriumParameters>()?;
    m.add_class::<MonodParameters>()?;
    m.add_class::<TstParameters>()?;
    m.add_class::<MineralAuxParams>()?;
    m.add_class::<MineralAuxParams>()?;
    m.add_class::<RtParameters>()?;
    m.add_class::<ZoneDimensions>()?;
    m.add_class::<RiverDimensions>()?;
    m.add_class::<RiverParameters>()?;

    // RtZone
    m.add_class::<RtStep>()?;
    m.add_class::<RtZone>()?;

    // River zone
    m.add_class::<RiverZone>()?;

    // Reaction network
    m.add_class::<ReactionNetwork>()?;

    // Database things
    m.add_class::<PrimaryAqueousSpecies>()?;
    m.add_class::<SecondarySpecies>()?;
    m.add_class::<MineralSpecies>()?;
    m.add_class::<TstReaction>()?;
    m.add_class::<MonodReaction>()?;
    m.add_class::<MineralKineticData>()?;
    m.add_class::<ChemicalDatabase>()?;
    m.add_class::<ExchangeReaction>()?;

    // Math things
    m.add(
        "ScalarRootFindingError",
        m.py().get_type::<ScalarRootFindingError>(),
    )?;
    m.add("MatMulError", m.py().get_type::<MatMulError>())?;
    m.add("IterationError", m.py().get_type::<IterationError>())?;
    m.add("LinearSystemError", m.py().get_type::<LinearSystemError>())?;
    m.add("OtherError", m.py().get_type::<OtherError>())?;
    m.add_class::<OptimizationError>()?;
    Ok(())
}
