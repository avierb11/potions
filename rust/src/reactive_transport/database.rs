use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, io::Read, path::Path};
// use uom::si::f64::*;
// use uom::si::{f64::MolarMass, molar_mass::gram_per_mole};

use pyo3::{
    exceptions::{PyFileNotFoundError, PyKeyError, PyValueError},
    prelude::*,
};

const DEFAULT_DATABASE_STR: &str = include_str!("default_database.json");

/// Number of names previewed per section in `ChemicalDatabase::summary()`.
const SUMMARY_PREVIEW_LEN: usize = 5;

/// Format a `String`-keyed `f64` map as a Python-style dict literal with
/// single-quoted, sorted keys, e.g. `{'H+': -1.0, 'HCO3-': 1.0}`.
///
/// Keys are sorted so the output is deterministic even though `HashMap`
/// iteration order is not.
fn format_string_map(map: &HashMap<String, f64>) -> String {
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    let items: Vec<String> = keys
        .iter()
        .map(|k| format!("'{}': {}", k, map[*k]))
        .collect();
    format!("{{{}}}", items.join(", "))
}

/// Format a map's keys as a Python-style list literal of single-quoted,
/// sorted names, e.g. `['calcite', 'gibbsite']`.
fn format_key_list<V>(map: &HashMap<String, V>) -> String {
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    let items: Vec<String> = keys.iter().map(|k| format!("'{}'", k)).collect();
    format!("[{}]", items.join(", "))
}

/// Return a `String` map's keys as a new, sorted `Vec<String>`.
///
/// `HashMap` iteration order is nondeterministic; sorting gives a stable
/// listing for display and test purposes.
fn sorted_keys<V>(map: &HashMap<String, V>) -> Vec<String> {
    let mut keys: Vec<String> = map.keys().cloned().collect();
    keys.sort();
    keys
}

/// Format one section of `ChemicalDatabase::summary()`: the label, the total
/// count, and a preview of up to `SUMMARY_PREVIEW_LEN` sorted names
/// (with `...` when truncated).
fn summary_section(label: &str, names: &[String]) -> String {
    let mut sorted: Vec<&String> = names.iter().collect();
    sorted.sort();
    let preview: Vec<&str> = sorted
        .iter()
        .take(SUMMARY_PREVIEW_LEN)
        .map(|s| s.as_str())
        .collect();
    let preview = preview.join(", ");
    if names.len() > SUMMARY_PREVIEW_LEN {
        format!(
            "  {:<16} {:>5}  ({} shown: {}...)",
            label,
            names.len(),
            SUMMARY_PREVIEW_LEN,
            preview
        )
    } else {
        format!("  {:<16} {:>5}  ({})", label, names.len(), preview)
    }
}

#[pyclass(from_py_object, module = "potions.core")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrimaryAqueousSpecies {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub molar_mass: f64,
    #[pyo3(get)]
    pub charge: f64,
    #[pyo3(get)]
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
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub stoichiometry: HashMap<String, f64>,
    #[pyo3(get)]
    pub eq_consts: Vec<f64>,
    #[pyo3(get)]
    pub dh_size_param: f64,
    #[pyo3(get)]
    pub charge: f64,
    #[pyo3(get)]
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

    pub fn __repr__(&self) -> String {
        format!(
            "SecondarySpecies(name={},stoichiometry={},eq_consts={:?},dh_size_param={},charge={},molar_mass={})",
            self.name,
            format_string_map(&self.stoichiometry),
            self.eq_consts,
            self.dh_size_param,
            self.charge,
            self.molar_mass
        )
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
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub molar_mass: f64,
    #[pyo3(get)]
    pub stoichiometry: HashMap<String, f64>,
    #[pyo3(get)]
    pub eq_consts: Vec<f64>,
    #[pyo3(get)]
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

    pub fn __repr__(&self) -> String {
        format!(
            "MineralSpecies(name={},molar_mass={},stoichiometry={},eq_consts={:?},molar_volume={})",
            self.name,
            self.molar_mass,
            format_string_map(&self.stoichiometry),
            self.eq_consts,
            self.molar_volume
        )
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
    #[pyo3(get)]
    pub mineral_name: String,
    #[pyo3(get)]
    pub label: String,
    #[pyo3(get)]
    pub rate_constant: f64, // Base-10 logarithm of the rate constant at 25 C
    #[pyo3(get)]
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

    pub fn __repr__(&self) -> String {
        format!(
            "TstReaction(mineral_name={},label={},rate_constant={},dependence={})",
            self.mineral_name,
            self.label,
            self.rate_constant,
            format_string_map(&self.dependence)
        )
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
    #[pyo3(get)]
    pub mineral_name: String,
    #[pyo3(get)]
    pub label: String,
    #[pyo3(get)]
    pub rate_constant: f64, // Base-10 logarithm of the rate constant at 25 C
    #[pyo3(get)]
    pub monod_terms: HashMap<String, f64>,
    #[pyo3(get)]
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

    pub fn __repr__(&self) -> String {
        format!(
            "MonodReaction(mineral_name={},label={},rate_constant={},monod_terms={},inhib_terms={})",
            self.mineral_name,
            self.label,
            self.rate_constant,
            format_string_map(&self.monod_terms),
            format_string_map(&self.inhib_terms)
        )
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
    #[pyo3(get)]
    pub tst_reactions: HashMap<String, TstReaction>,
    #[pyo3(get)]
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

    /// Show the mineral names of the reactions held, one list per reaction
    /// type. Full reaction details are available through the individual
    /// `TstReaction` / `MonodReaction` entries.
    pub fn __repr__(&self) -> String {
        format!(
            "MineralKineticData(tst_reactions={},monod_reactions={})",
            format_key_list(&self.tst_reactions),
            format_key_list(&self.monod_reactions)
        )
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

#[pymethods]
impl ExchangeReaction {
    #[new]
    pub fn new(
        name: String,
        stoichiometry: HashMap<String, f64>,
        log10_k_eq: f64,
        charge: f64,
    ) -> Self {
        Self {
            name,
            stoichiometry,
            log10_k_eq,
            charge,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "ExchangeReaction(name={},stoichiometry={},log10_k_eq={},charge={})",
            self.name,
            format_string_map(&self.stoichiometry),
            self.log10_k_eq,
            self.charge
        )
    }

    pub fn __getstate__(&self) -> (String, HashMap<String, f64>, f64, f64) {
        (
            self.name.clone(),
            self.stoichiometry.clone(),
            self.log10_k_eq,
            self.charge,
        )
    }

    pub fn __setstate__(&mut self, state: (String, HashMap<String, f64>, f64, f64)) {
        let (name, stoichiometry, log10_k_eq, charge) = state;
        self.name = name;
        self.stoichiometry = stoichiometry;
        self.log10_k_eq = log10_k_eq;
        self.charge = charge
    }

    pub fn __getnewargs__(&self) -> (String, HashMap<String, f64>, f64, f64) {
        (
            self.name.clone(),
            self.stoichiometry.clone(),
            self.log10_k_eq,
            self.charge,
        )
    }
}

#[pyclass(from_py_object, module = "potions.core")]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChemicalDatabase {
    #[pyo3(get)]
    pub primary_species: HashMap<String, PrimaryAqueousSpecies>,
    #[pyo3(get)]
    pub secondary_species: HashMap<String, SecondarySpecies>,
    #[pyo3(get)]
    pub mineral_species: HashMap<String, MineralSpecies>,
    #[pyo3(get)]
    pub exchange_reactions: HashMap<String, ExchangeReaction>,
    #[pyo3(get)]
    pub tst_reactions: HashMap<String, HashMap<String, TstReaction>>,
    #[pyo3(get)]
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

    /// A compact one-line summary reporting the number of entries of each
    /// species and reaction type. Use `summary()` for a detailed,
    /// multi-line listing of the individual names.
    pub fn __repr__(&self) -> String {
        format!(
            "ChemicalDatabase(primary_species={},secondary_species={},mineral_species={},exchange_reactions={},tst_reactions={},monod_reactions={})",
            self.primary_species.len(),
            self.secondary_species.len(),
            self.mineral_species.len(),
            self.exchange_reactions.len(),
            self.tst_reactions.len(),
            self.monod_reactions.len()
        )
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

    /// Names of all primary aqueous species in the database, sorted
    /// alphabetically so the listing is stable and easy to scan.
    pub fn get_primary_aqueous_species_names(&self) -> Vec<String> {
        sorted_keys(&self.primary_species)
    }

    /// Names of all secondary (aqueous equilibrium) species in the database,
    /// sorted alphabetically.
    pub fn get_secondary_species_names(&self) -> Vec<String> {
        sorted_keys(&self.secondary_species)
    }

    /// Names of all mineral species in the database, sorted alphabetically.
    pub fn get_mineral_species_names(&self) -> Vec<String> {
        sorted_keys(&self.mineral_species)
    }

    /// Names of all exchange (ion-exchange) species in the database, sorted
    /// alphabetically.
    pub fn get_exchange_species_names(&self) -> Vec<String> {
        sorted_keys(&self.exchange_reactions)
    }

    /// Mineral names that carry TST (transition-state) kinetic reactions,
    /// sorted alphabetically.
    pub fn get_tst_mineral_names(&self) -> Vec<String> {
        sorted_keys(&self.tst_reactions)
    }

    /// Mineral names that carry Monod kinetic reactions, sorted alphabetically.
    pub fn get_monod_mineral_names(&self) -> Vec<String> {
        sorted_keys(&self.monod_reactions)
    }

    /// A multi-line, human-readable listing of the database contents,
    /// intended for `print()`.
    ///
    /// Each section reports the total number of entries followed by a short
    /// preview of the first few (sorted) names, with `...` when truncated.
    /// Use the per-type `get_*_names()` methods to obtain complete name lists.
    pub fn summary(&self) -> String {
        let primary_names = sorted_keys(&self.primary_species);
        let secondary_names = sorted_keys(&self.secondary_species);
        let mineral_names = sorted_keys(&self.mineral_species);
        let exchange_names = sorted_keys(&self.exchange_reactions);
        let tst_names = sorted_keys(&self.tst_reactions);
        let monod_names = sorted_keys(&self.monod_reactions);

        let mut lines: Vec<String> = Vec::new();
        lines.push(format!(
            "ChemicalDatabase summary: {} primary aqueous, {} secondary, {} mineral, {} exchange, {} TST minerals, {} Monod minerals",
            primary_names.len(),
            secondary_names.len(),
            mineral_names.len(),
            exchange_names.len(),
            tst_names.len(),
            monod_names.len()
        ));
        lines.push(summary_section("primary aqueous", &primary_names));
        lines.push(summary_section("secondary", &secondary_names));
        lines.push(summary_section("mineral", &mineral_names));
        lines.push(summary_section("exchange", &exchange_names));
        lines.push(summary_section("TST minerals", &tst_names));
        lines.push(summary_section("Monod minerals", &monod_names));
        lines.join("\n")
    }
}
