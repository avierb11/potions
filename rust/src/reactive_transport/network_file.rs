//! The self-contained network file format (`network.yaml`).
//!
//! A network file is a YAML document with six sections, in this
//! order: `primary_species`, `secondary_species`, `mineral_species`,
//! `exchange_reactions`, `tst_reactions`, `monod_reactions`.  Every
//! map is a `BTreeMap`, so two serializations of equal content are
//! byte-identical — the fingerprint stability the GUI relies on.  The
//! kinetics sections are flattened relative to the database: one
//! entry per mineral, the label living inside the entry.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};

use super::database::{
    ExchangeReaction, MineralKineticData, MineralSpecies, MonodReaction, PrimaryAqueousSpecies,
    SecondarySpecies, TstReaction,
};
use super::reaction_network::ReactionNetwork;

/// A single validation finding: the rule code (`V1`…`V9` block,
/// `W1`/`W2` warn) and a human-readable message naming the offender.
#[pyclass(from_py_object, module = "potions.core")]
#[derive(Clone, Debug, PartialEq)]
pub struct NetworkValidationIssue {
    #[pyo3(get)]
    pub code: String,
    #[pyo3(get)]
    pub message: String,
}

#[pymethods]
impl NetworkValidationIssue {
    pub fn __repr__(&self) -> String {
        format!(
            "NetworkValidationIssue(code='{}', message='{}')",
            self.code, self.message
        )
    }

    pub fn __getstate__(&self) -> (String, String) {
        (self.code.clone(), self.message.clone())
    }

    pub fn __setstate__(&mut self, state: (String, String)) {
        let (code, message) = state;
        self.code = code;
        self.message = message;
    }

    pub fn __getnewargs__(&self) -> (String, String) {
        (self.code.clone(), self.message.clone())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FilePrimaryAqueousSpecies {
    pub name: String,
    pub molar_mass: f64,
    pub charge: f64,
    pub dh_size_param: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileSecondarySpecies {
    pub name: String,
    #[serde(default)]
    pub stoichiometry: BTreeMap<String, f64>,
    #[serde(default)]
    pub eq_consts: Vec<f64>,
    pub dh_size_param: f64,
    pub charge: f64,
    pub molar_mass: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileMineralSpecies {
    pub name: String,
    pub molar_mass: f64,
    #[serde(default)]
    pub stoichiometry: BTreeMap<String, f64>,
    #[serde(default)]
    pub eq_consts: Vec<f64>,
    pub molar_volume: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileExchangeReaction {
    pub name: String,
    #[serde(default)]
    pub stoichiometry: BTreeMap<String, f64>,
    pub log10_k_eq: f64,
    pub charge: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileTstReaction {
    pub mineral_name: String,
    pub label: String,
    pub rate_constant: f64,
    #[serde(default)]
    pub dependence: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileMonodReaction {
    pub mineral_name: String,
    pub label: String,
    pub rate_constant: f64,
    #[serde(default)]
    pub monod_terms: BTreeMap<String, f64>,
    #[serde(default)]
    pub inhib_terms: BTreeMap<String, f64>,
}

#[pyclass(module = "potions.core")]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NetworkFile {
    #[serde(default)]
    pub primary_species: BTreeMap<String, FilePrimaryAqueousSpecies>,
    #[serde(default)]
    pub secondary_species: BTreeMap<String, FileSecondarySpecies>,
    #[serde(default)]
    pub mineral_species: BTreeMap<String, FileMineralSpecies>,
    #[serde(default)]
    pub exchange_reactions: BTreeMap<String, FileExchangeReaction>,
    #[serde(default)]
    pub tst_reactions: BTreeMap<String, FileTstReaction>,
    #[serde(default)]
    pub monod_reactions: BTreeMap<String, FileMonodReaction>,
}

/// The species a stoichiometry may reference (the N1 / V3 set).
fn carried_names(network: &ReactionNetwork) -> BTreeSet<String> {
    let mut names: BTreeSet<String> = network.species_names().into_iter().collect();
    if network.has_exchange() {
        names.insert("X-".to_string());
    }
    names
}

impl NetworkFile {
    /// Build the file view of `network`, applying N1: stoichiometry
    /// terms whose key is not a species the network carries are
    /// dropped (the runtime already reads absent keys as 0.0 — the
    /// canonical case is the database's CO2(aq) `H2O` term).
    pub fn build_from_network(network: &ReactionNetwork) -> Self {
        let names = carried_names(network);
        let keep = |stoich: &HashMap<String, f64>| -> BTreeMap<String, f64> {
            stoich
                .iter()
                .filter(|(key, _)| names.contains(key.as_str()))
                .map(|(key, value)| (key.clone(), *value))
                .collect()
        };

        Self {
            primary_species: network
                .primary_aqueous
                .iter()
                .map(|s| {
                    (
                        s.name.clone(),
                        FilePrimaryAqueousSpecies {
                            name: s.name.clone(),
                            molar_mass: s.molar_mass,
                            charge: s.charge,
                            dh_size_param: s.dh_size_param,
                        },
                    )
                })
                .collect(),
            secondary_species: network
                .secondary
                .iter()
                .map(|s| {
                    (
                        s.name.clone(),
                        FileSecondarySpecies {
                            name: s.name.clone(),
                            stoichiometry: keep(&s.stoichiometry),
                            eq_consts: s.eq_consts.clone(),
                            dh_size_param: s.dh_size_param,
                            charge: s.charge,
                            molar_mass: s.molar_mass,
                        },
                    )
                })
                .collect(),
            mineral_species: network
                .mineral
                .iter()
                .map(|s| {
                    (
                        s.name.clone(),
                        FileMineralSpecies {
                            name: s.name.clone(),
                            molar_mass: s.molar_mass,
                            stoichiometry: keep(&s.stoichiometry),
                            eq_consts: s.eq_consts.clone(),
                            molar_volume: s.molar_volume,
                        },
                    )
                })
                .collect(),
            exchange_reactions: network
                .exchange_species
                .iter()
                .map(|e| {
                    (
                        e.name.clone(),
                        FileExchangeReaction {
                            name: e.name.clone(),
                            stoichiometry: keep(&e.stoichiometry),
                            log10_k_eq: e.log10_k_eq,
                            charge: e.charge,
                        },
                    )
                })
                .collect(),
            tst_reactions: network
                .mineral_kinetics
                .tst_reactions
                .iter()
                .map(|(key, r)| {
                    (
                        key.clone(),
                        FileTstReaction {
                            mineral_name: r.mineral_name.clone(),
                            label: r.label.clone(),
                            rate_constant: r.rate_constant,
                            dependence: r
                                .dependence
                                .iter()
                                .map(|(dk, dv)| (dk.clone(), *dv))
                                .collect(),
                        },
                    )
                })
                .collect(),
            monod_reactions: network
                .mineral_kinetics
                .monod_reactions
                .iter()
                .map(|(key, r)| {
                    (
                        key.clone(),
                        FileMonodReaction {
                            mineral_name: r.mineral_name.clone(),
                            label: r.label.clone(),
                            rate_constant: r.rate_constant,
                            monod_terms: r
                                .monod_terms
                                .iter()
                                .map(|(dk, dv)| (dk.clone(), *dv))
                                .collect(),
                            inhib_terms: r
                                .inhib_terms
                                .iter()
                                .map(|(dk, dv)| (dk.clone(), *dv))
                                .collect(),
                        },
                    )
                })
                .collect(),
        }
    }

    /// Rebuild a `ReactionNetwork` from the file, using the existing
    /// (lenient) constructor.  Validate separately (`validate()`); the
    /// constructor does not enforce the file rules.
    pub fn build_to_network(&self) -> PyResult<ReactionNetwork> {
        let primary: Vec<PrimaryAqueousSpecies> = self
            .primary_species
            .values()
            .map(|e| {
                PrimaryAqueousSpecies::new(e.name.clone(), e.molar_mass, e.charge, e.dh_size_param)
            })
            .collect();
        let mineral: Vec<MineralSpecies> = self
            .mineral_species
            .values()
            .map(|e| {
                MineralSpecies::new(
                    e.name.clone(),
                    e.molar_mass,
                    e.stoichiometry.clone().into_iter().collect(),
                    e.eq_consts.clone(),
                    e.molar_volume,
                )
            })
            .collect();
        let secondary: Vec<SecondarySpecies> = self
            .secondary_species
            .values()
            .map(|e| {
                SecondarySpecies::new(
                    e.name.clone(),
                    e.stoichiometry.clone().into_iter().collect(),
                    e.eq_consts.clone(),
                    e.molar_mass,
                    e.charge,
                    e.dh_size_param,
                )
            })
            .collect();
        let exchange: Vec<ExchangeReaction> = self
            .exchange_reactions
            .values()
            .map(|e| {
                ExchangeReaction::new(
                    e.name.clone(),
                    e.stoichiometry.clone().into_iter().collect(),
                    e.log10_k_eq,
                    e.charge,
                )
            })
            .collect();
        let tst: HashMap<String, TstReaction> = self
            .tst_reactions
            .values()
            .map(|e| {
                (
                    e.mineral_name.clone(),
                    TstReaction::new(
                        e.mineral_name.clone(),
                        e.label.clone(),
                        e.rate_constant,
                        e.dependence.clone().into_iter().collect(),
                    ),
                )
            })
            .collect();
        let monod: HashMap<String, MonodReaction> = self
            .monod_reactions
            .values()
            .map(|e| {
                (
                    e.mineral_name.clone(),
                    MonodReaction::new(
                        e.mineral_name.clone(),
                        e.label.clone(),
                        e.rate_constant,
                        e.monod_terms.clone().into_iter().collect(),
                        e.inhib_terms.clone().into_iter().collect(),
                    ),
                )
            })
            .collect();

        let kinetics = MineralKineticData::new(tst, monod);
        ReactionNetwork::new(primary, mineral, secondary, kinetics, exchange)
    }

    /// The canonical YAML bytes: section order = field declaration
    /// order, keys in `BTreeMap` order at every level, 2-space
    /// indentation, the serializer's scalar rendering as-is.  Two
    /// calls on equal content return equal bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_yaml::to_string(self)
            .expect("serializing an in-memory NetworkFile to YAML cannot fail")
            .into_bytes()
    }

    /// Parse YAML bytes into a file; a syntax error surfaces as
    /// `ValueError` carrying the parser's message.
    pub fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        serde_yaml::from_slice(bytes)
            .map_err(|err| PyValueError::new_err(format!("Invalid network YAML: {err}")))
    }
}

#[pymethods]
impl NetworkFile {
    #[staticmethod]
    pub fn from_file(path: &str) -> PyResult<NetworkFile> {
        load_network_file(path)
    }

    #[staticmethod]
    pub fn from_network(py: Python, network: &ReactionNetwork) -> PyResult<Py<NetworkFile>> {
        Py::new(py, NetworkFile::build_from_network(network))
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Py<PyBytes> {
        PyBytes::new(py, &self.canonical_bytes()).into()
    }

    pub fn to_file(&self, path: &str) -> PyResult<()> {
        std::fs::write(path, self.canonical_bytes()).map_err(|err| {
            PyOSError::new_err(format!("could not write network file '{path}': {err}"))
        })
    }

    pub fn to_network(&self) -> PyResult<ReactionNetwork> {
        self.build_to_network()
    }

    pub fn validate(&self) -> Vec<NetworkValidationIssue> {
        validate_file(self)
    }

    /// A compact one-line summary: the entry count of each section.
    pub fn __repr__(&self) -> String {
        format!(
            "NetworkFile(primary={}, secondary={}, mineral={}, exchange={}, tst={}, monod={})",
            self.primary_species.len(),
            self.secondary_species.len(),
            self.mineral_species.len(),
            self.exchange_reactions.len(),
            self.tst_reactions.len(),
            self.monod_reactions.len(),
        )
    }
}

/// The species names a file's stoichiometries may reference — the
/// V3/V7 reference set: the name field of every primary, secondary,
/// mineral, and exchange entry, plus the implicit exchange site `X-`
/// when the exchange section is non-empty.
fn file_carried_names(file: &NetworkFile) -> BTreeSet<String> {
    let mut names: BTreeSet<String> = file
        .primary_species
        .values()
        .map(|s| s.name.clone())
        .chain(file.secondary_species.values().map(|s| s.name.clone()))
        .chain(file.mineral_species.values().map(|s| s.name.clone()))
        .chain(file.exchange_reactions.values().map(|e| e.name.clone()))
        .collect();
    if !file.exchange_reactions.is_empty() {
        names.insert("X-".to_string());
    }
    names
}

/// The file's stoichiometries in deterministic order — secondary,
/// mineral, then exchange, each in `BTreeMap` key order — tagged with
/// the section label the rule messages use.
fn file_stoichiometries<'a>(
    file: &'a NetworkFile,
) -> Vec<(&'static str, &'a str, &'a BTreeMap<String, f64>)> {
    file.secondary_species
        .values()
        .map(|s| ("secondary species", s.name.as_str(), &s.stoichiometry))
        .chain(
            file.mineral_species
                .values()
                .map(|s| ("mineral", s.name.as_str(), &s.stoichiometry)),
        )
        .chain(
            file.exchange_reactions
                .values()
                .map(|e| ("exchange species", e.name.as_str(), &e.stoichiometry)),
        )
        .collect()
}

/// One validation finding.
fn issue(code: &str, message: impl Into<String>) -> NetworkValidationIssue {
    NetworkValidationIssue {
        code: code.to_string(),
        message: message.into(),
    }
}

/// The validation findings of a network file, in stable order:
/// V1..V9 (blocking) then W1, W2 (warnings).  Empty means valid.
/// Rules follow the spec's "Validation rules" section:
///
/// - V1 at least one primary aqueous species.
/// - V2 species names unique across all five species sections
///   (primary, secondary, mineral, exchange, implicit X-).
/// - V3 every stoichiometry key (secondary/mineral/exchange) resolves
///   to a carried species (V3 set = primary ∪ secondary ∪ mineral ∪
///   exchange names ∪ {X-} when the exchange section is non-empty).
/// - V4 every secondary species appears in its own stoichiometry with
///   a negative coefficient (the database's self-term convention), and
///   every exchange product appears in its own stoichiometry with a
///   negative coefficient (the exchange-init residual needs c_i).
/// - V5 every mineral has exactly one kinetics entry (present in
///   exactly one of tst_reactions / monod_reactions, keyed by name).
/// - V6 every kinetics entry is self-consistent: mineral_name equals
///   its map key, label is non-empty.
/// - V7 every kinetics dependence / monod_terms / inhib_terms key
///   resolves to a network species (the V3 set).
/// - V8 every secondary and mineral carries at least two eq_consts
///   (the solver reads index 1).
/// - V9 H+ referenced in any stoichiometry exists as a primary.
/// - W1 the network has no H+ (no charge-balance row; warn only).
/// - W2 an exchange section is present but no stoichiometry
///   references X- (the exchange is disconnected; warn only).
pub fn validate_file(file: &NetworkFile) -> Vec<NetworkValidationIssue> {
    let mut issues: Vec<NetworkValidationIssue> = Vec::new();
    let carried = file_carried_names(file);
    let stoichiometries = file_stoichiometries(file);
    let hplus_primary = file
        .primary_species
        .iter()
        .any(|(key, s)| key == "H+" || s.name == "H+");
    let hplus_referenced = stoichiometries
        .iter()
        .any(|(_, _, stoich)| stoich.contains_key("H+"));

    // V1: the speciation solve needs at least one primary.
    if file.primary_species.is_empty() {
        issues.push(issue("V1", "no primary aqueous species"));
    }

    // V2: species names are unique across all five sections (the four
    // explicit ones plus the implicit `X-`, counted once).
    let mut name_counts: BTreeMap<String, usize> = BTreeMap::new();
    for name in file
        .primary_species
        .values()
        .map(|s| s.name.as_str())
        .chain(file.secondary_species.values().map(|s| s.name.as_str()))
        .chain(file.mineral_species.values().map(|s| s.name.as_str()))
        .chain(file.exchange_reactions.values().map(|e| e.name.as_str()))
    {
        *name_counts.entry(name.to_string()).or_insert(0) += 1;
    }
    if !file.exchange_reactions.is_empty() {
        *name_counts.entry("X-".to_string()).or_insert(0) += 1;
    }
    for (name, count) in &name_counts {
        if *count > 1 {
            issues.push(issue(
                "V2",
                format!("species name '{name}' appears in multiple species sections"),
            ));
        }
    }

    // V3: every stoichiometry key resolves to a carried species.
    for (label, name, stoich) in &stoichiometries {
        for key in stoich.keys() {
            if !carried.contains(key) {
                issues.push(issue(
                    "V3",
                    format!("stoichiometry of {label} '{name}' references unknown species '{key}'"),
                ));
            }
        }
    }

    // V4: secondary and exchange entries carry a negative self-term.
    for (label, name, stoich) in &stoichiometries {
        if *label == "mineral" {
            continue;
        }
        if !stoich
            .get(*name)
            .is_some_and(|coefficient| *coefficient < 0.0)
        {
            issues.push(issue(
                "V4",
                format!("{label} '{name}' has no negative self-term in its stoichiometry"),
            ));
        }
    }

    // V5: every mineral has exactly one kinetics entry.
    for (name, _) in &file.mineral_species {
        let has_tst = file.tst_reactions.contains_key(name);
        let has_monod = file.monod_reactions.contains_key(name);
        if has_tst && has_monod {
            issues.push(issue(
                "V5",
                format!("mineral '{name}' has both TST and Monod kinetics entries"),
            ));
        } else if !has_tst && !has_monod {
            issues.push(issue(
                "V5",
                format!("mineral '{name}' has no kinetics entry"),
            ));
        }
    }

    // V6: every kinetics entry is self-consistent (TST entries first,
    // then Monod, each in `BTreeMap` key order; the name mismatch
    // takes precedence over the empty label, at most one finding
    // per entry).
    for (key, mineral_name, label) in file
        .tst_reactions
        .iter()
        .map(|(key, e)| (key.as_str(), &e.mineral_name, &e.label))
        .chain(
            file.monod_reactions
                .iter()
                .map(|(key, e)| (key.as_str(), &e.mineral_name, &e.label)),
        )
    {
        if mineral_name.as_str() != key {
            issues.push(issue(
                "V6",
                format!(
                    "kinetics entry '{key}' has mineral_name '{mineral_name}' differing from its key"
                ),
            ));
        } else if label.is_empty() {
            issues.push(issue(
                "V6",
                format!("kinetics entry '{key}' has an empty label"),
            ));
        }
    }

    // V7: every kinetics dependence / monod / inhib key resolves to a
    // carried species.
    for (key, entry) in &file.tst_reactions {
        for term in entry.dependence.keys() {
            if !carried.contains(term) {
                issues.push(issue(
                    "V7",
                    format!("kinetics entry '{key}' references unknown species '{term}'"),
                ));
            }
        }
    }
    for (key, entry) in &file.monod_reactions {
        for term in entry.monod_terms.keys().chain(entry.inhib_terms.keys()) {
            if !carried.contains(term) {
                issues.push(issue(
                    "V7",
                    format!("kinetics entry '{key}' references unknown species '{term}'"),
                ));
            }
        }
    }

    // V8: secondary and mineral species carry at least two eq_consts.
    for (name, entry) in &file.secondary_species {
        if entry.eq_consts.len() < 2 {
            issues.push(issue(
                "V8",
                format!("secondary species '{name}' carries fewer than two eq_consts"),
            ));
        }
    }
    for (name, entry) in &file.mineral_species {
        if entry.eq_consts.len() < 2 {
            issues.push(issue(
                "V8",
                format!("mineral '{name}' carries fewer than two eq_consts"),
            ));
        }
    }

    // V9: a referenced H+ must exist as a primary.
    if hplus_referenced && !hplus_primary {
        issues.push(issue(
            "V9",
            "H+ is referenced in stoichiometries but is not a primary species",
        ));
    }

    // W1: no H+ primary (no charge-balance row) — warn only.
    if !hplus_primary {
        issues.push(issue(
            "W1",
            "no H+ primary species: the speciation solve has no charge-balance row",
        ));
    }

    // W2: an exchange section no stoichiometry reaches — warn only.
    if !file.exchange_reactions.is_empty()
        && !stoichiometries
            .iter()
            .any(|(_, _, stoich)| stoich.contains_key("X-"))
    {
        issues.push(issue(
            "W2",
            "no stoichiometry references X-: the exchange site pool is disconnected",
        ));
    }

    issues
}

/// Read, parse, and V-gate a network YAML file.  A missing path is
/// `FileNotFoundError`; a syntax error is `ValueError` (the parser's
/// message); any V-issue is `ValueError` joining every V-issue as
/// `"<code>: <message>"` separated by "; ".  W-issues never block.
pub(crate) fn load_network_file(path: &str) -> PyResult<NetworkFile> {
    let path = std::path::Path::new(path);
    if !path.exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "network file at '{}' does not exist",
            path.display()
        )));
    }
    let contents = std::fs::read(path)
        .map_err(|err| PyOSError::new_err(format!("could not read '{}': {err}", path.display())))?;
    let file = NetworkFile::from_bytes(&contents)?;
    let all_issues: Vec<NetworkValidationIssue> = validate_file(&file);
    let errors: Vec<&NetworkValidationIssue> = all_issues
        .iter()
        .filter(|issue| issue.code.starts_with('V'))
        .collect();
    if !errors.is_empty() {
        let joined: Vec<String> = errors
            .iter()
            .map(|i| format!("{}: {}", i.code, i.message))
            .collect();
        return Err(PyValueError::new_err(format!(
            "network file '{}' failed validation: {}",
            path.display(),
            joined.join("; ")
        )));
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet, HashMap};

    use pyo3::exceptions::PyFileNotFoundError;
    use pyo3::Python;

    use super::{
        load_network_file, validate_file, FileMonodReaction, NetworkFile, NetworkValidationIssue,
    };
    use crate::reactive_transport::database::{
        ChemicalDatabase, ExchangeReaction, MineralKineticData, MineralKineticReaction,
        PrimaryAqueousSpecies, SecondarySpecies, TstReaction,
    };
    use crate::reactive_transport::reaction_network::ReactionNetwork;

    /// The default East River network built from the compiled-in database —
    /// the same selector path as `common_networks.py::get_soc_calcite_network`.
    fn east_river_network() -> ReactionNetwork {
        let db = ChemicalDatabase::load_default().unwrap();
        let primary = db
            .get_primary_aqueous_species(vec![
                "DOC".into(),
                "Ca++".into(),
                "HCO3-".into(),
                "H+".into(),
            ])
            .unwrap();
        let mineral = db
            .get_mineral_species(vec![
                "SOC(s)".into(),
                "SOC-lz(s)".into(),
                "Calcite(s)".into(),
            ])
            .unwrap();
        let secondary = db
            .get_secondary_species(vec!["CO2(aq)".into(), "CO3--".into()])
            .unwrap();
        let mut tst: HashMap<String, TstReaction> = HashMap::new();
        let mut monod: HashMap<String, crate::reactive_transport::database::MonodReaction> =
            HashMap::new();
        for (mineral_name, label) in [
            ("Calcite(s)", "default"),
            ("SOC(s)", "test"),
            ("SOC-lz(s)", "test"),
        ] {
            match db
                .get_single_mineral_reaction(mineral_name.into(), label.into())
                .unwrap()
                .1
            {
                MineralKineticReaction::Tst(r) => {
                    tst.insert(mineral_name.into(), r);
                }
                MineralKineticReaction::Monod(r) => {
                    monod.insert(mineral_name.into(), r);
                }
            }
        }
        let exchange = vec![db
            .exchange_reactions
            .get("XDOC")
            .cloned()
            .expect("the default database carries the XDOC exchange reaction")];
        ReactionNetwork::new(
            primary,
            mineral,
            secondary,
            MineralKineticData::new(tst, monod),
            exchange,
        )
        .unwrap()
    }

    /// The East River file view — the fixture for the validation tests.
    fn east_river_file() -> NetworkFile {
        NetworkFile::build_from_network(&east_river_network())
    }

    /// The rule codes of a finding list, in order.
    fn issue_codes(issues: &[NetworkValidationIssue]) -> Vec<&str> {
        issues.iter().map(|i| i.code.as_str()).collect()
    }

    /// The first finding with the given rule code, if any.
    fn find_issue<'a>(
        issues: &'a [NetworkValidationIssue],
        code: &str,
    ) -> Option<&'a NetworkValidationIssue> {
        issues.iter().find(|i| i.code == code)
    }

    /// A unique per-process path in the system temp dir for file tests.
    fn temp_network_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "network_file_test_{tag}_{}.yaml",
            std::process::id()
        ))
    }

    fn section_key_names<V>(map: &BTreeMap<String, V>) -> Vec<&str> {
        map.keys().map(String::as_str).collect()
    }

    #[test]
    fn test_from_network_golden_values() {
        let db = ChemicalDatabase::load_default().unwrap();
        let file = east_river_file();

        assert_eq!(
            section_key_names(&file.primary_species),
            ["Ca++", "DOC", "H+", "HCO3-"]
        );
        assert_eq!(
            section_key_names(&file.secondary_species),
            ["CO2(aq)", "CO3--"]
        );
        assert_eq!(
            section_key_names(&file.mineral_species),
            ["Calcite(s)", "SOC(s)", "SOC-lz(s)"]
        );
        assert_eq!(section_key_names(&file.exchange_reactions), ["XDOC"]);
        assert_eq!(section_key_names(&file.tst_reactions), ["Calcite(s)"]);
        assert_eq!(
            section_key_names(&file.monod_reactions),
            ["SOC(s)", "SOC-lz(s)"]
        );

        // Self-consistent value checks against the database objects.
        let db_doc = &db.primary_species["DOC"];
        assert_eq!(file.primary_species["DOC"].molar_mass, db_doc.molar_mass);
        assert_eq!(file.primary_species["DOC"].charge, db_doc.charge);
        assert_eq!(
            file.primary_species["DOC"].dh_size_param,
            db_doc.dh_size_param
        );

        let db_co2 = &db.secondary_species["CO2(aq)"];
        let co2 = &file.secondary_species["CO2(aq)"];
        assert_eq!(co2.molar_mass, db_co2.molar_mass);
        assert_eq!(co2.charge, db_co2.charge);
        assert_eq!(co2.eq_consts, db_co2.eq_consts);

        let db_xdoc = &db.exchange_reactions["XDOC"];
        assert_eq!(
            file.exchange_reactions["XDOC"].log10_k_eq,
            db_xdoc.log10_k_eq
        );

        let db_calcite = db
            .tst_reactions
            .get("Calcite(s)")
            .unwrap()
            .get("default")
            .unwrap();
        assert_eq!(
            file.tst_reactions["Calcite(s)"].rate_constant,
            db_calcite.rate_constant
        );

        // N1 proof: the database's CO2(aq) stoichiometry carries an `H2O`
        // key that is not a species the network carries; it must be gone,
        // and no entry in any section may carry an `H2O` key.
        assert_eq!(
            section_key_names(&co2.stoichiometry),
            ["CO2(aq)", "H+", "HCO3-"]
        );
        for stoich in file
            .secondary_species
            .values()
            .map(|s| &s.stoichiometry)
            .chain(file.mineral_species.values().map(|s| &s.stoichiometry))
            .chain(file.exchange_reactions.values().map(|e| &e.stoichiometry))
        {
            assert!(!stoich.contains_key("H2O"));
        }
    }

    #[test]
    fn test_section_and_key_order_in_bytes() {
        let file = east_river_file();
        let text = String::from_utf8(file.canonical_bytes()).expect("canonical YAML is UTF-8");

        let sections = [
            "primary_species:",
            "secondary_species:",
            "mineral_species:",
            "exchange_reactions:",
            "tst_reactions:",
            "monod_reactions:",
        ];
        let positions: Vec<usize> = sections
            .iter()
            .map(|s| {
                text.find(*s)
                    .unwrap_or_else(|| panic!("section header {:?} missing from:\n{}", s, text))
            })
            .collect();
        for window in positions.windows(2) {
            assert!(
                window[0] < window[1],
                "section headers out of order:\n{text}"
            );
        }

        // The inner `name:` lines of the primary section, in byte order.
        // (The `name:` field repeats per entry, so find the first
        // occurrence of each value after the section header; the
        // optional double quotes keep the check robust to scalar
        // quoting.)
        let primary_at = text.find("primary_species:").unwrap();
        let primary_names = ["Ca++", "DOC", "H+", "HCO3-"];
        let mut prev: usize = primary_at;
        for name in primary_names {
            let plain = format!("name: {name}");
            let quoted = format!("name: \"{name}\"");
            let pos = text[primary_at..]
                .find(&plain)
                .or_else(|| text[primary_at..].find(&quoted))
                .map(|p| p + primary_at)
                .unwrap_or_else(|| panic!("inner name line {name:?} missing"));
            assert!(pos > prev, "primary inner name lines out of order:\n{text}");
            prev = pos;
        }

        let secondary_at = text.find("secondary_species:").unwrap();
        let co2_pos = text[secondary_at..]
            .find("name: CO2(aq)")
            .or_else(|| text[secondary_at..].find("name: \"CO2(aq)\""))
            .map(|p| p + secondary_at)
            .unwrap_or_else(|| panic!("inner name line \"CO2(aq)\" missing"));
        let co3_pos = text[secondary_at..]
            .find("name: CO3--")
            .or_else(|| text[secondary_at..].find("name: \"CO3--\""))
            .map(|p| p + secondary_at)
            .unwrap_or_else(|| panic!("inner name line \"CO3--\" missing"));
        assert!(
            co2_pos < co3_pos,
            "secondary inner name lines out of order:\n{text}"
        );
    }

    #[test]
    fn test_round_trip_values() {
        let file = east_river_file();
        let parsed = NetworkFile::from_bytes(&file.canonical_bytes()).expect("round-trip parse");
        assert_eq!(parsed, file);
        assert_eq!(file.canonical_bytes(), file.canonical_bytes());
    }

    #[test]
    fn test_n1_drops_unknown_keeps_xminus() {
        let primary = vec![PrimaryAqueousSpecies::new("A".into(), 1.0, 0.0, 1.0)];
        let mut stoich: HashMap<String, f64> = HashMap::new();
        stoich.insert("A".into(), 1.0);
        stoich.insert("H2O".into(), 1.0);
        stoich.insert("S".into(), -1.0);
        let secondary = vec![SecondarySpecies::new(
            "S".into(),
            stoich,
            Vec::new(),
            1.0,
            0.0,
            1.0,
        )];
        let mut exchange_stoich: HashMap<String, f64> = HashMap::new();
        exchange_stoich.insert("X-".into(), 1.0);
        exchange_stoich.insert("A".into(), 1.0);
        exchange_stoich.insert("E".into(), -1.0);
        let exchange = vec![ExchangeReaction::new("E".into(), exchange_stoich, 0.0, 0.0)];
        let network = ReactionNetwork::new(
            primary,
            Vec::new(),
            secondary,
            MineralKineticData::new(HashMap::new(), HashMap::new()),
            exchange,
        )
        .unwrap();

        let file = NetworkFile::build_from_network(&network);

        let s_stoich = &file.secondary_species["S"].stoichiometry;
        assert_eq!(s_stoich.len(), 2);
        assert_eq!(s_stoich.get("A"), Some(&1.0));
        assert_eq!(s_stoich.get("S"), Some(&-1.0));
        assert!(!s_stoich.contains_key("H2O"));

        // The implicit exchange site `X-` is a carried species, so the
        // exchange stoichiometry keeps it (as well as `A` and `E`).
        let e_stoich = &file.exchange_reactions["E"].stoichiometry;
        assert_eq!(e_stoich.get("X-"), Some(&1.0));
        assert_eq!(e_stoich.get("A"), Some(&1.0));
        assert_eq!(e_stoich.get("E"), Some(&-1.0));
    }

    #[test]
    fn test_to_network_round_trip() {
        let network = east_river_network();
        let file = NetworkFile::build_from_network(&network);
        let rebuilt = file.build_to_network().expect("file to network");

        // The file's canonical key order sorts the species, so compare
        // the species sets and counts rather than construction order.
        assert_eq!(
            rebuilt.num_species(),
            network.num_species(),
            "species count must survive the round trip"
        );
        let original_names: Vec<String> = network.species_names();
        let rebuilt_names: Vec<String> = rebuilt.species_names();
        let original_set: BTreeSet<&str> = original_names.iter().map(String::as_str).collect();
        let rebuilt_set: BTreeSet<&str> = rebuilt_names.iter().map(String::as_str).collect();
        assert_eq!(rebuilt_set, original_set);

        // The file is a fixed point of file -> network -> file.
        assert_eq!(NetworkFile::build_from_network(&rebuilt), file);
    }

    #[test]
    fn test_validate_east_river_clean() {
        let issues = validate_file(&east_river_file());
        assert!(
            issues.is_empty(),
            "East River file must be clean: {issues:?}"
        );
    }

    #[test]
    fn test_validate_v1_no_primary() {
        let mut file = east_river_file();
        file.primary_species.clear();

        let issues = validate_file(&file);
        let issue = find_issue(&issues, "V1").expect("V1 must fire");
        assert_eq!(issue.message, "no primary aqueous species");
        // Clearing every primary also strands their stoichiometry
        // references (V3) and the charge-balance row (W1); only the
        // V1 finding is pinned by this test.
        assert!(issue_codes(&issues).contains(&"V3"));
        assert!(issue_codes(&issues).contains(&"W1"));
    }

    #[test]
    fn test_validate_v2_duplicate_name() {
        let mut file = east_river_file();
        // Rename the CO2(aq) entry to the primary name DOC and re-point
        // its self-term.  The rename fires V2; the stranded `CO2(aq)`
        // references left in the SOC minerals' stoichiometries fire V3
        // (a rename that drops references is exactly what V3 catches).
        let mut entry = file
            .secondary_species
            .remove("CO2(aq)")
            .expect("East River carries CO2(aq)");
        entry.name = "DOC".to_string();
        entry.stoichiometry.remove("CO2(aq)");
        entry.stoichiometry.insert("DOC".to_string(), -1.0);
        file.secondary_species.insert("DOC".to_string(), entry);

        let issues = validate_file(&file);
        let issue = find_issue(&issues, "V2").expect("V2 must fire");
        assert_eq!(
            issue.message,
            "species name 'DOC' appears in multiple species sections"
        );
        assert!(
            issue_codes(&issues).contains(&"V3"),
            "the stranded CO2(aq) references must fire V3"
        );
    }

    #[test]
    fn test_validate_v3_unknown_stoich_key() {
        let mut file = east_river_file();
        // The N1 canonical case, re-introduced by a "hand-edited file".
        file.secondary_species
            .get_mut("CO2(aq)")
            .expect("East River carries CO2(aq)")
            .stoichiometry
            .insert("H2O".to_string(), 1.0);

        let issues = validate_file(&file);
        assert_eq!(issue_codes(&issues), ["V3"]);
        assert_eq!(
            issues[0].message,
            "stoichiometry of secondary species 'CO2(aq)' references unknown species 'H2O'"
        );
    }

    #[test]
    fn test_validate_v4a_missing_secondary_self_term() {
        let mut file = east_river_file();
        file.secondary_species
            .get_mut("CO3--")
            .expect("East River carries CO3--")
            .stoichiometry
            .remove("CO3--");

        let issues = validate_file(&file);
        assert_eq!(issue_codes(&issues), ["V4"]);
        assert_eq!(
            issues[0].message,
            "secondary species 'CO3--' has no negative self-term in its stoichiometry"
        );
    }

    #[test]
    fn test_validate_v4b_missing_exchange_self_term() {
        let mut file = east_river_file();
        file.exchange_reactions
            .get_mut("XDOC")
            .expect("East River carries XDOC")
            .stoichiometry
            .remove("XDOC");

        let issues = validate_file(&file);
        assert_eq!(issue_codes(&issues), ["V4"]);
        assert_eq!(
            issues[0].message,
            "exchange species 'XDOC' has no negative self-term in its stoichiometry"
        );
    }

    #[test]
    fn test_validate_v5a_both_kinetics_entries() {
        let mut file = east_river_file();
        // Give Calcite(s) a TST entry and a copied Monod entry.
        let calcite_tst = file.tst_reactions["Calcite(s)"].clone();
        file.monod_reactions.insert(
            "Calcite(s)".to_string(),
            FileMonodReaction {
                mineral_name: calcite_tst.mineral_name,
                label: calcite_tst.label,
                rate_constant: calcite_tst.rate_constant,
                monod_terms: BTreeMap::new(),
                inhib_terms: BTreeMap::new(),
            },
        );

        let issues = validate_file(&file);
        assert_eq!(issue_codes(&issues), ["V5"]);
        assert_eq!(
            issues[0].message,
            "mineral 'Calcite(s)' has both TST and Monod kinetics entries"
        );
    }

    #[test]
    fn test_validate_v5b_no_kinetics_entry() {
        let mut file = east_river_file();
        file.monod_reactions.remove("SOC-lz(s)");

        let issues = validate_file(&file);
        assert_eq!(issue_codes(&issues), ["V5"]);
        assert_eq!(
            issues[0].message,
            "mineral 'SOC-lz(s)' has no kinetics entry"
        );
    }

    #[test]
    fn test_validate_v6a_empty_label() {
        let mut file = east_river_file();
        file.tst_reactions
            .get_mut("Calcite(s)")
            .expect("East River carries Calcite(s)")
            .label
            .clear();

        let issues = validate_file(&file);
        assert_eq!(issue_codes(&issues), ["V6"]);
        assert_eq!(
            issues[0].message,
            "kinetics entry 'Calcite(s)' has an empty label"
        );
    }

    #[test]
    fn test_validate_v6b_mineral_name_mismatch() {
        let mut file = east_river_file();
        file.monod_reactions
            .get_mut("SOC(s)")
            .expect("East River carries SOC(s)")
            .mineral_name = "Other".to_string();

        let issues = validate_file(&file);
        assert_eq!(issue_codes(&issues), ["V6"]);
        assert_eq!(
            issues[0].message,
            "kinetics entry 'SOC(s)' has mineral_name 'Other' differing from its key"
        );
    }

    #[test]
    fn test_validate_v7_unknown_kinetics_key() {
        let mut file = east_river_file();
        file.monod_reactions
            .get_mut("SOC-lz(s)")
            .expect("East River carries SOC-lz(s)")
            .monod_terms
            .insert("H2O".to_string(), 1.0);

        let issues = validate_file(&file);
        assert_eq!(issue_codes(&issues), ["V7"]);
        assert_eq!(
            issues[0].message,
            "kinetics entry 'SOC-lz(s)' references unknown species 'H2O'"
        );
    }

    #[test]
    fn test_validate_v8_fewer_than_two_eq_consts() {
        let mut file = east_river_file();
        file.secondary_species
            .get_mut("CO2(aq)")
            .expect("East River carries CO2(aq)")
            .eq_consts
            .truncate(1);

        let issues = validate_file(&file);
        assert_eq!(issue_codes(&issues), ["V8"]);
        assert_eq!(
            issues[0].message,
            "secondary species 'CO2(aq)' carries fewer than two eq_consts"
        );
    }

    #[test]
    fn test_validate_v9_hplus_referenced_but_not_primary() {
        let mut file = east_river_file();
        file.primary_species.remove("H+");

        let issues = validate_file(&file);
        // V9 is one global finding; the stranded H+ references (CO2(aq)
        // and CO3--) also fire two V3s, and W1 fires (no H+ primary).
        assert_eq!(issue_codes(&issues), ["V3", "V3", "V9", "W1"]);
        let v9 = find_issue(&issues, "V9").expect("V9 must fire");
        assert_eq!(
            v9.message,
            "H+ is referenced in stoichiometries but is not a primary species"
        );
    }

    #[test]
    fn test_validate_w1_no_hplus() {
        let mut file = east_river_file();
        file.primary_species.remove("H+");
        for stoich in file
            .secondary_species
            .values_mut()
            .map(|s| &mut s.stoichiometry)
            .chain(
                file.mineral_species
                    .values_mut()
                    .map(|s| &mut s.stoichiometry),
            )
            .chain(
                file.exchange_reactions
                    .values_mut()
                    .map(|e| &mut e.stoichiometry),
            )
        {
            stoich.remove("H+");
        }

        let issues = validate_file(&file);
        assert_eq!(issue_codes(&issues), ["W1"]);
        assert_eq!(
            issues[0].message,
            "no H+ primary species: the speciation solve has no charge-balance row"
        );
    }

    #[test]
    fn test_validate_w2_exchange_disconnected() {
        let mut file = east_river_file();
        file.exchange_reactions
            .get_mut("XDOC")
            .expect("East River carries XDOC")
            .stoichiometry
            .remove("X-");

        let issues = validate_file(&file);
        assert_eq!(issue_codes(&issues), ["W2"]);
        assert_eq!(
            issues[0].message,
            "no stoichiometry references X-: the exchange site pool is disconnected"
        );
    }

    #[test]
    fn test_from_file_error_paths() {
        // A missing path is FileNotFoundError.
        let err = load_network_file("/definitely/not/a/network.yaml").unwrap_err();
        Python::attach(|py| {
            assert!(
                err.is_instance_of::<PyFileNotFoundError>(py),
                "missing path must raise FileNotFoundError"
            );
        });

        // A YAML syntax error is ValueError carrying the parser's message.
        let syntax_path = temp_network_path("syntax");
        std::fs::write(&syntax_path, b": [unclosed").expect("write the temp file");
        let err = load_network_file(syntax_path.to_str().unwrap()).unwrap_err();
        let message = Python::attach(|py| err.value(py).to_string());
        assert!(
            message.contains("Invalid network YAML"),
            "syntax error message was: {message}"
        );
        let _ = std::fs::remove_file(&syntax_path);

        // The V-gate: a hand-written file violating V1 and V8 fails
        // with both findings joined into one ValueError.  (The YAML is
        // written line-verbatim — a `\n\` continuation would strip the
        // indentation and flatten the structure.)
        let vgate_yaml = "\
primary_species: {}
secondary_species:
  S:
    name: S
    stoichiometry:
      S: -1.0
    eq_consts:
    - 1.0
    dh_size_param: 1.0
    charge: 0.0
    molar_mass: 100.0
";
        let vgate_path = temp_network_path("vgate");
        std::fs::write(&vgate_path, vgate_yaml).expect("write the temp file");
        let err = load_network_file(vgate_path.to_str().unwrap()).unwrap_err();
        let message = Python::attach(|py| err.value(py).to_string());
        assert!(message.contains("V1:"), "expected V1 in: {message}");
        assert!(message.contains("V8:"), "expected V8 in: {message}");
        let _ = std::fs::remove_file(&vgate_path);
    }

    #[test]
    fn test_to_file_round_trip_bytes() {
        let file = east_river_file();
        let path = temp_network_path("round_trip");
        let path_str = path.to_str().expect("temp path is UTF-8");

        file.to_file(path_str).expect("write the file");
        let written = std::fs::read(&path).expect("read the bytes back");
        assert_eq!(written, file.canonical_bytes());

        let reread = NetworkFile::from_file(path_str).expect("the clean file passes the gate");
        assert_eq!(reread, file);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_reaction_network_from_file_to_file() {
        let network = east_river_network();
        let path = temp_network_path("network_round_trip");
        let path_str = path.to_str().expect("temp path is UTF-8");

        network.to_file(path_str).expect("write the file");
        let rebuilt = ReactionNetwork::from_file(path_str).expect("the clean file passes the gate");

        let original_names = network.species_names();
        let original_names: BTreeSet<&str> = original_names.iter().map(String::as_str).collect();
        let rebuilt_names = rebuilt.species_names();
        let rebuilt_names: BTreeSet<&str> = rebuilt_names.iter().map(String::as_str).collect();
        assert_eq!(rebuilt_names, original_names);
        assert!(
            rebuilt.validate().is_empty(),
            "the rebuilt network must be clean"
        );

        let _ = std::fs::remove_file(&path);
    }
}
