use numpy::{PyArray1, ndarray::Array1};
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySeries};

use crate::reactive_transport::{database::{MineralKineticData, MineralSpecies, PrimaryAqueousSpecies, SecondarySpecies}, kinetic_structures::{EquilibriumParameters, MonodParameters, TstParameters}};

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct ReactionNetwork;

#[pymethods]
impl ReactionNetwork {
    #[new]
    pub fn new(
        primary_aqueous: Vec<PrimaryAqueousSpecies>, 
        mineral: Vec<MineralSpecies>,
        secondary: Vec<SecondarySpecies>,
        mineral_kinetics: MineralKineticData
    ) -> Self {
        unimplemented!()
    }

    #[getter]
    pub fn species(&self) -> PyDataFrame {
        unimplemented!()
    }

    #[getter]
    pub fn species_order(&self) -> Vec<String> {
        unimplemented!()
    }

    #[getter]
    pub fn has_exchange(&self) -> bool {
        unimplemented!()
    }

    #[getter]
    pub fn charges(&self) -> PySeries {
        unimplemented!()
    }

    #[getter]
    pub fn equilibrium_species(&self) -> PyDataFrame {
        unimplemented!()
    }

    #[getter]
    pub fn kinetic_species(&self) -> PyDataFrame {
        unimplemented!()
    }

    #[getter]
    pub fn equilibrium_parameters(&self) -> PyResult<EquilibriumParameters> {
        unimplemented!()
    }

    #[getter]
    pub fn tst_params(&self) -> PyResult<TstParameters> {
        unimplemented!()
    }

    #[getter]
    pub fn monod_params(&self) -> PyResult<MonodParameters> {
        unimplemented!()
    }

    #[getter]
    pub fn species_names(&self) -> Vec<String> {
        unimplemented!()
    }

    #[getter]
    pub fn mineral_species_names(&self) -> Vec<String> {
        unimplemented!()
    }

    #[getter]
    pub fn mineral_stoichiometry(&self) -> PyDataFrame {
        unimplemented!()
    }

    #[getter]
    pub fn transport_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        unimplemented!()
    }

    #[getter]
    pub fn mineral_molar_masses<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        unimplemented!()
    }

    #[getter]
    pub fn rate_consts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        unimplemented!()
    }

    #[getter]
    pub fn num_minerals(&self) -> usize {
        unimplemented!()
    }

    #[getter]
    pub fn num_mineral_parameters(&self) -> usize {
        unimplemented!()
    }

    #[getter]
    pub fn num_aqueous_species(&self) -> usize {
        unimplemented!()
    }

    pub fn get_default_aqueous_initial_state<'py>(&self, py: Python<'py>, init_conc: f64) -> Bound<'py, PyArray1<f64>> {
        unimplemented!()
    }
}
