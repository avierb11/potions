use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::Read, path::Path};
// use uom::si::f64::*;
// use uom::si::{f64::MolarMass, molar_mass::gram_per_mole};

use pyo3::{
    exceptions::{PyFileNotFoundError, PyKeyError, PyValueError},
    prelude::*,
};

const DEFAULT_DATABASE_STR: &str = include_str!("default_database.json");

#[pyclass(from_py_object, module = "potions.core")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrimaryAqueousSpecies {
    pub name: String,
    pub molar_mass: f64,
    pub charge: f64,
    pub dh_size_param: f64,
}

#[pymethods]
impl PrimaryAqueousSpecies {
    #[new]
    pub fn new(name: String, molar_mass: f64, charge: f64, dh_size_param: f64) -> Self {
        Self {
            name,
            molar_mass,
            charge,
            dh_size_param,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PrimaryAqueousSpecies(name={},molar_mass={},charge={},dh_size_param={})",
            self.name, self.molar_mass, self.charge, self.dh_size_param
        )
    }

    pub fn __getstate__(&self) -> (String, f64, f64, f64) {
        (
            self.name.clone(),
            self.molar_mass,
            self.charge,
            self.dh_size_param,
        )
    }

    pub fn __setstate__(&mut self, state: (String, f64, f64, f64)) {
        let (name, molar_mass, charge, dh_size_param) = state;
        self.name = name;
        self.molar_mass = molar_mass;
        self.charge = charge;
        self.dh_size_param = dh_size_param;
    }

    pub fn __getnewargs__(&self) -> (String, f64, f64, f64) {
        (
            self.name.clone(),
            self.molar_mass,
            self.charge,
            self.dh_size_param,
        )
    }
}

#[pyclass(from_py_object, module = "potions.core")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SecondarySpecies {
    pub name: String,
    pub stoichiometry: HashMap<String, f64>,
    pub eq_consts: Vec<f64>,
    pub dh_size_param: f64,
    pub charge: f64,
    pub molar_mass: f64,
}

#[pymethods]
impl SecondarySpecies {
    #[new]
    pub fn new(
        name: String,
        stoichiometry: HashMap<String, f64>,
        eq_consts: Vec<f64>,
        molar_mass: f64,
        charge: f64,
        dh_size_param: f64,
    ) -> Self {
        Self {
            name,
            molar_mass,
            charge,
            dh_size_param,
            stoichiometry,
            eq_consts,
        }
    }

    pub fn __getstate__(&self) -> (String, HashMap<String, f64>, Vec<f64>, f64, f64, f64) {
        (
            self.name.clone(),
            self.stoichiometry.clone(),
            self.eq_consts.clone(),
            self.dh_size_param,
            self.charge,
            self.molar_mass,
        )
    }

    pub fn __setstate__(
        &mut self,
        state: (
            String,
            std::collections::HashMap<String, f64>,
            Vec<f64>,
            f64,
            f64,
            f64,
        ),
    ) {
        let (name, stoichiometry, eq_consts, dh_size_param, charge, molar_mass) = state;
        self.name = name;
        self.stoichiometry = stoichiometry;
        self.eq_consts = eq_consts;
        self.dh_size_param = dh_size_param;
        self.charge = charge;
        self.molar_mass = molar_mass;
    }

    pub fn __getnewargs__(
        &self,
    ) -> (
        String,
        std::collections::HashMap<String, f64>,
        Vec<f64>,
        f64,
        f64,
        f64,
    ) {
        (
            self.name.clone(),
            self.stoichiometry.clone(),
            self.eq_consts.clone(),
            self.dh_size_param,
            self.charge,
            self.molar_mass,
        )
    }
}

#[pyclass(from_py_object, module = "potions.core")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MineralSpecies {
    pub name: String,
    pub molar_mass: f64,
    pub stoichiometry: HashMap<String, f64>,
    pub eq_consts: Vec<f64>,
    pub molar_volume: f64,
}

#[pymethods]
impl MineralSpecies {
    #[new]
    pub fn new(
        name: String,
        molar_mass: f64,
        stoichiometry: HashMap<String, f64>,
        eq_consts: Vec<f64>,
        molar_volume: f64,
    ) -> Self {
        Self {
            name,
            molar_mass,
            stoichiometry,
            eq_consts,
            molar_volume,
        }
    }

    pub fn __getstate__(
        &self,
    ) -> (
        String,
        f64,
        std::collections::HashMap<String, f64>,
        Vec<f64>,
        f64,
    ) {
        (
            self.name.clone(),
            self.molar_mass,
            self.stoichiometry.clone(),
            self.eq_consts.clone(),
            self.molar_volume,
        )
    }

    pub fn __setstate__(
        &mut self,
        state: (
            String,
            f64,
            std::collections::HashMap<String, f64>,
            Vec<f64>,
            f64,
        ),
    ) {
        let (name, molar_mass, stoichiometry, eq_consts, molar_volume) = state;
        self.name = name;
        self.molar_mass = molar_mass;
        self.stoichiometry = stoichiometry;
        self.eq_consts = eq_consts;
        self.molar_volume = molar_volume;
    }

    pub fn __getnewargs__(
        &self,
    ) -> (
        String,
        f64,
        std::collections::HashMap<String, f64>,
        Vec<f64>,
        f64,
    ) {
        (
            self.name.clone(),
            self.molar_mass,
            self.stoichiometry.clone(),
            self.eq_consts.clone(),
            self.molar_volume,
        )
    }
}

pub enum PrimarySpecies {
    Aqueous(PrimaryAqueousSpecies),
    Mineral(MineralSpecies),
}

#[pyclass(from_py_object, module = "potions.core")]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TstReaction {
    pub mineral_name: String,
    pub label: String,
    pub rate_constant: f64, // Base-10 logarithm of the rate constant at 25 C
    pub dependence: HashMap<String, f64>,
}

#[pymethods]
impl TstReaction {
    #[new]
    pub fn new(
        mineral_name: String,
        label: String,
        rate_constant: f64, // Base-10 logarithm of the rate constant at 25 C
        dependence: HashMap<String, f64>,
    ) -> Self {
        Self {
            mineral_name,
            label,
            rate_constant,
            dependence,
        }
    }

    pub fn __getstate__(&self) -> (String, String, f64, std::collections::HashMap<String, f64>) {
        (
            self.mineral_name.clone(),
            self.label.clone(),
            self.rate_constant,
            self.dependence.clone(),
        )
    }

    pub fn __setstate__(
        &mut self,
        state: (String, String, f64, std::collections::HashMap<String, f64>),
    ) {
        let (mineral_name, label, rate_constant, dependence) = state;
        self.mineral_name = mineral_name;
        self.label = label;
        self.rate_constant = rate_constant;
        self.dependence = dependence;
    }

    pub fn __getnewargs__(&self) -> (String, String, f64, std::collections::HashMap<String, f64>) {
        (
            self.mineral_name.clone(),
            self.label.clone(),
            self.rate_constant,
            self.dependence.clone(),
        )
    }
}

#[pyclass(from_py_object, module = "potions.core")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MonodReaction {
    pub mineral_name: String,
    pub label: String,
    pub rate_constant: f64, // Base-10 logarithm of the rate constant at 25 C
    pub monod_terms: HashMap<String, f64>,
    pub inhib_terms: HashMap<String, f64>,
}

#[pymethods]
impl MonodReaction {
    #[new]
    pub fn new(
        mineral_name: String,
        label: String,
        rate_constant: f64, // Base-10 logarithm of the rate constant at 25 C
        monod_terms: HashMap<String, f64>,
        inhib_terms: HashMap<String, f64>,
    ) -> Self {
        Self {
            mineral_name,
            label,
            rate_constant,
            monod_terms,
            inhib_terms,
        }
    }

    pub fn __getstate__(
        &self,
    ) -> (
        String,
        String,
        f64,
        std::collections::HashMap<String, f64>,
        std::collections::HashMap<String, f64>,
    ) {
        (
            self.mineral_name.clone(),
            self.label.clone(),
            self.rate_constant,
            self.monod_terms.clone(),
            self.inhib_terms.clone(),
        )
    }

    pub fn __setstate__(
        &mut self,
        state: (
            String,
            String,
            f64,
            std::collections::HashMap<String, f64>,
            std::collections::HashMap<String, f64>,
        ),
    ) {
        let (mineral_name, label, rate_constant, monod_terms, inhib_terms) = state;
        self.mineral_name = mineral_name;
        self.label = label;
        self.rate_constant = rate_constant;
        self.monod_terms = monod_terms;
        self.inhib_terms = inhib_terms;
    }

    pub fn __getnewargs__(
        &self,
    ) -> (
        String,
        String,
        f64,
        std::collections::HashMap<String, f64>,
        std::collections::HashMap<String, f64>,
    ) {
        (
            self.mineral_name.clone(),
            self.label.clone(),
            self.rate_constant,
            self.monod_terms.clone(),
            self.inhib_terms.clone(),
        )
    }
}

#[pyclass(from_py_object, module = "potions.core")]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MineralKineticData {
    pub tst_reactions: HashMap<String, TstReaction>,
    pub monod_reactions: HashMap<String, MonodReaction>,
}

#[pymethods]
impl MineralKineticData {
    #[new]
    pub fn new(
        tst_reactions: HashMap<String, TstReaction>,
        monod_reactions: HashMap<String, MonodReaction>,
    ) -> Self {
        Self {
            tst_reactions,
            monod_reactions,
        }
    }

    pub fn __getstate__(
        &self,
    ) -> (
        std::collections::HashMap<String, TstReaction>,
        std::collections::HashMap<String, MonodReaction>,
    ) {
        (self.tst_reactions.clone(), self.monod_reactions.clone())
    }

    pub fn __setstate__(
        &mut self,
        state: (
            std::collections::HashMap<String, TstReaction>,
            std::collections::HashMap<String, MonodReaction>,
        ),
    ) {
        let (tst_reactions, monod_reactions) = state;
        self.tst_reactions = tst_reactions;
        self.monod_reactions = monod_reactions;
    }

    pub fn __getnewargs__(
        &self,
    ) -> (
        std::collections::HashMap<String, TstReaction>,
        std::collections::HashMap<String, MonodReaction>,
    ) {
        (self.tst_reactions.clone(), self.monod_reactions.clone())
    }
}

#[pyclass(from_py_object, module = "potions.core")]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExchangeReaction {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub stoichiometry: HashMap<String, f64>,
    #[pyo3(get, set)]
    pub log10_k_eq: f64,
    #[pyo3(get, set)]
    pub charge: f64,
}

#[pyclass(from_py_object, module = "potions.core")]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChemicalDatabase {
    pub primary_species: HashMap<String, PrimaryAqueousSpecies>,
    pub secondary_species: HashMap<String, SecondarySpecies>,
    pub mineral_species: HashMap<String, MineralSpecies>,
    pub exchange_reactions: HashMap<String, ExchangeReaction>,
    pub tst_reactions: HashMap<String, HashMap<String, TstReaction>>,
    pub monod_reactions: HashMap<String, HashMap<String, MonodReaction>>,
}

#[pyclass(from_py_object, module = "potions.core")]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum MineralReactionType {
    Name(String),
    Mineral(MineralSpecies),
}

#[pyclass(from_py_object, module = "potions.core")]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum MineralKineticReaction {
    Tst(TstReaction),
    Monod(MonodReaction),
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
        Self {
            primary_species,
            secondary_species,
            mineral_species,
            exchange_reactions,
            tst_reactions,
            monod_reactions,
        }
    }

    #[staticmethod]
    pub fn load_default() -> PyResult<Self> {
        match serde_json::from_str(DEFAULT_DATABASE_STR) {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err("Failed to deconstruct database")),
        }
    }

    #[staticmethod]
    pub fn from_file(file_path: String) -> PyResult<Self> {
        let path = Path::new(file_path.as_str());

        if !path.exists() {
            let msg: String = format!("Database file at '{}' does not exist", file_path);
            return Err(PyFileNotFoundError::new_err(msg));
        }

        let mut file = File::open(path).expect("Failed to open database file");
        let mut contents = String::new();
        let _num_bytes = file
            .read_to_string(&mut contents)
            .expect("Failed to read file to string");

        match serde_json::from_str(contents.as_ref()) {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err("Failed to deconstruct database")),
        }
    }

    pub fn get_primary_aqueous_species(
        &self,
        primary_names: Vec<String>,
    ) -> PyResult<Vec<PrimaryAqueousSpecies>> {
        let mut spec: Vec<PrimaryAqueousSpecies> = Vec::new();

        for s in primary_names {
            if !self.primary_species.contains_key(&s) {
                let msg: String = format!("Database does not contain primary species '{}'", s);
                return Err(PyValueError::new_err(msg));
            } else {
                spec.push(self.primary_species.get(&s).unwrap().clone());
            }
        }

        return Ok(spec);
    }

    pub fn get_mineral_species(&self, mineral_names: Vec<String>) -> PyResult<Vec<MineralSpecies>> {
        let mut spec: Vec<MineralSpecies> = Vec::new();

        for s in mineral_names {
            if !self.mineral_species.contains_key(&s) {
                let msg: String = format!("Database does not contain mineral species '{}'", s);
                return Err(PyValueError::new_err(msg));
            } else {
                spec.push(self.mineral_species.get(&s).unwrap().clone());
            }
        }

        return Ok(spec);
    }

    pub fn get_secondary_species(
        &self,
        secondary_names: Vec<String>,
    ) -> PyResult<Vec<SecondarySpecies>> {
        let mut spec: Vec<SecondarySpecies> = Vec::new();

        for s in secondary_names {
            if !self.secondary_species.contains_key(&s) {
                let msg: String = format!("Database does not contain mineral species '{}'", s);
                return Err(PyValueError::new_err(msg));
            } else {
                spec.push(self.secondary_species.get(&s).unwrap().clone());
            }
        }
        Ok(spec)
    }

    pub fn get_single_mineral_reaction(
        &self,
        mineral: String,
        label: String,
    ) -> PyResult<(String, MineralKineticReaction)> {
        let mineral_name = mineral.clone();

        if self.tst_reactions.contains_key(&mineral_name) {
            let lbls = self.tst_reactions.get(&mineral_name).unwrap();

            if lbls.contains_key(&label) {
                let kin_rxn = lbls.get(&label).unwrap().clone();

                return Ok(("tst".to_string(), MineralKineticReaction::Tst(kin_rxn)));
            } else {
                let msg: String = format!(
                    "Mineral '{}' has no reaction labeled '{}'",
                    mineral_name, label
                );
                return Err(PyKeyError::new_err(msg));
            }
        } else if self.monod_reactions.contains_key(&mineral_name) {
            let lbls = self.monod_reactions.get(&mineral_name).unwrap();

            if lbls.contains_key(&label) {
                let kin_rxn = lbls.get(&label).unwrap().clone();

                return Ok(("tst".to_string(), MineralKineticReaction::Monod(kin_rxn)));
            } else {
                let msg: String = format!(
                    "Mineral '{}' has no reaction labeled '{}'",
                    mineral_name, label
                );
                return Err(PyKeyError::new_err(msg));
            }
        } else {
            let msg: String = format!(
                "Chemical database does not contain kinetic data for '{}'",
                mineral_name
            );
            return Err(PyKeyError::new_err(msg));
        }
    }

    pub fn get_mineral_reactions(
        &self,
        mineral_names: Vec<String>,
        labels: Vec<String>,
    ) -> PyResult<MineralKineticData> {
        if mineral_names.len() != labels.len() {
            return Err(PyValueError::new_err(
                "Must pass the same number of minerals and labels",
            ));
        }
        let mut tst_rxns: HashMap<String, TstReaction> = HashMap::new();
        let mut monod_rxns: HashMap<String, MonodReaction> = HashMap::new();

        for (min, lbl) in mineral_names.iter().zip(labels) {
            let (_, rxn) = self.get_single_mineral_reaction(min.clone(), lbl)?;

            match rxn {
                MineralKineticReaction::Tst(v) => {
                    tst_rxns.insert(min.clone(), v);
                }
                MineralKineticReaction::Monod(v) => {
                    monod_rxns.insert(min.clone(), v);
                }
            }
        }

        Ok(MineralKineticData {
            tst_reactions: tst_rxns,
            monod_reactions: monod_rxns,
        })
    }

    pub fn get_exchange_reactions(
        &self,
        species_names: Vec<String>,
    ) -> PyResult<Vec<ExchangeReaction>> {
        let mut spec: Vec<ExchangeReaction> = Vec::new();

        for s in species_names {
            if !self.exchange_reactions.contains_key(&s) {
                let msg: String = format!("Database does not contain mineral species '{}'", s);
                return Err(PyValueError::new_err(msg));
            } else {
                spec.push(self.exchange_reactions.get(&s).unwrap().clone());
            }
        }
        Ok(spec)
    }
}
