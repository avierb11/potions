use std::collections::HashMap;

use pyo3::prelude::*;

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct PrimaryAqueousSpecies {
    name: String,
    molar_mass: f64,
    charge: f64,
    dh_size_param: f64,
}

#[pymethods]
impl PrimaryAqueousSpecies {
    #[new]
    pub fn new(name: String, molar_mass: f64, charge: f64, dh_size_param: f64) -> Self {
        Self {
            name, molar_mass, charge, dh_size_param
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct SecondarySpecies {
    name: String,
    stoichiometry: HashMap<String, f64>,
    eq_consts: Vec<f64>,
    dh_size_param: f64,
    charge: f64,
    molar_mass: f64
}

#[pymethods]
impl SecondarySpecies {
    #[new]
    pub fn new(name: String, stoichiometry: HashMap<String, f64>, eq_consts: Vec<f64>, molar_mass: f64, charge: f64, dh_size_param: f64) -> Self {
        Self {
            name, molar_mass, charge, dh_size_param, stoichiometry, eq_consts
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct MineralSpecies {
    name: String,
    molar_mass: f64,
    stoichiometry: HashMap<String, f64>,
    eq_consts: Vec<f64>,
    molar_volume: f64,
}

pub enum PrimarySpecies {
    Aqueous(PrimaryAqueousSpecies),
    Mineral(MineralSpecies)
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct TstReaction {
    mineral_name: String,
    label: String,
    rate_constant: f64, // Base-10 logarithm of the rate constant at 25 C
    dependence: HashMap<String, f64>
}


#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct MonodReaction {
    mineral_name: String,
    label: String,
    rate_constant: f64, // Base-10 logarithm of the rate constant at 25 C
    monod_terms: HashMap<String, f64>,
    inhib_terms: HashMap<String, f64>
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct MineralKineticData {
    pub tst_reactions: HashMap<String, TstReaction>,
    pub monod_reactions: HashMap<String, MonodReaction>
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct ExchangeReaction {
    pub name: String,
    pub stoichiometry: HashMap<String, f64>,
    pub log10_k_eq: f64,
    pub charge: f64,
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct ChemicalDatabase {
    pub primary_species: HashMap<String, PrimaryAqueousSpecies>,
    pub secondary_species: HashMap<String, SecondarySpecies>,
    pub mineral_species: HashMap<String, MineralSpecies>,
    pub exchange_reactions: HashMap<String, ExchangeReaction>,
    pub tst_reactions: HashMap<String, HashMap<String, TstReaction>>,
    pub monod_reactions: HashMap<String, HashMap<String, MonodReaction>>,
}


#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub enum SpeciesRequestType {
    Single(String),
    Multiple(Vec<String>)
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub enum MineralReactionType {
    Name(String),
    Mineral(MineralSpecies)
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub enum MineralKineticReaction {
    Tst(TstReaction),
    Monod(MonodReaction)
}


#[pymethods]
impl ChemicalDatabase {
    #[new]
    pub fn new(
        primary_species: HashMap<String, PrimaryAqueousSpecies>,
        secondary_species: HashMap<String, SecondarySpecies>,
        mineral_species: HashMap<String, MineralSpecies>,
        exchange_reactions: HashMap<String, ExchangeReaction>,
        tst_reactions: HashMap<String, HashMap<String, TstReaction>>,
        monod_reactions: HashMap<String, HashMap<String, MonodReaction>>,
    ) -> Self {
        unimplemented!()
    }

    #[staticmethod]
    pub fn load_default() -> PyResult<Self> {
        unimplemented!()
    }

    pub fn to_file(&self, file_path: String) -> () {
        unimplemented!()
    }

    #[staticmethod]
    pub fn from_file(file_path: String) -> PyResult<Self> {
        unimplemented!()
    }

    pub fn get_primary_aqueous_species(&self, s: SpeciesRequestType) -> PyResult<Vec<PrimaryAqueousSpecies>> {
        unimplemented!()
    }

    pub fn get_mineral_species(&self, s: SpeciesRequestType) -> PyResult<Vec<MineralSpecies>> {
        unimplemented!()
    }

    pub fn get_secondary_species(&self, s: SpeciesRequestType) -> PyResult<Vec<SecondarySpecies>> {
        unimplemented!()
    }

    pub fn get_single_mineral_reaction(&self, mineral: MineralReactionType, label: String) -> PyResult<(String, MineralKineticReaction)> {
        unimplemented!()
    }

    pub fn get_mineral_reactions(&self, mineral: Vec<MineralReactionType>, labels: Vec<String>) -> MineralKineticData {
        unimplemented!()
    }

    pub fn get_exchange_reactions(&self, species_name: SpeciesRequestType) -> Vec<ExchangeReaction> {
        unimplemented!()
    }
}