#![allow(dead_code)]
#![allow(unused_variables)]

use pyo3::{create_exception, exceptions::PyException, prelude::*};

use crate::{
    common_types::{ForcingData, HydroForcing, HydroStep, LapseRateParameters, RtForcing},
    hydro::{GroundZone, GroundZoneB, HydrologicZone, SnowZone, SurfaceZone},
    reactive_transport::{
        database::{
            ChemicalDatabase, MineralKineticData, MineralSpecies, MonodReaction,
            PrimaryAqueousSpecies, SecondarySpecies, TstReaction,
        },
        kinetic_structures::{
            EquilibriumParameters, MineralAuxParams, MineralParameters, MonodParameters,
            RtParameters, TstParameters, ZoneDimensions,
        },
        reaction_network::ReactionNetwork,
        rt_zone::{RtStep, RtZone},
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
    m.add_class::<MineralParameters>()?;
    m.add_class::<EquilibriumParameters>()?;
    m.add_class::<MonodParameters>()?;
    m.add_class::<TstParameters>()?;
    m.add_class::<MineralAuxParams>()?;
    m.add_class::<MineralAuxParams>()?;
    m.add_class::<RtParameters>()?;
    m.add_class::<ZoneDimensions>()?;

    // RtZone
    m.add_class::<RtStep>()?;
    m.add_class::<RtZone>()?;

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

    // Math things
    m.add(
        "ScalarRootFindingError",
        m.py().get_type::<ScalarRootFindingError>(),
    )?;
    m.add("MatMulError", m.py().get_type::<MatMulError>())?;
    m.add("IterationError", m.py().get_type::<IterationError>())?;
    m.add("LinearSystemError", m.py().get_type::<LinearSystemError>())?;
    m.add("OtherError", m.py().get_type::<OtherError>())?;
    Ok(())
}
