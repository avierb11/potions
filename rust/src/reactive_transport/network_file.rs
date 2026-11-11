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

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use super::database::{
    ExchangeReaction, MineralKineticData, MineralSpecies, MonodReaction, PrimaryAqueousSpecies,
    SecondarySpecies, TstReaction,
};
use super::reaction_network::ReactionNetwork;

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
    pub fn from_network(network: &ReactionNetwork) -> Self {
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
    pub fn to_network(&self) -> PyResult<ReactionNetwork> {
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

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet, HashMap};

    use super::NetworkFile;
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

    fn section_key_names<V>(map: &BTreeMap<String, V>) -> Vec<&str> {
        map.keys().map(String::as_str).collect()
    }

    #[test]
    fn test_from_network_golden_values() {
        let db = ChemicalDatabase::load_default().unwrap();
        let file = NetworkFile::from_network(&east_river_network());

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
        let file = NetworkFile::from_network(&east_river_network());
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
        let file = NetworkFile::from_network(&east_river_network());
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

        let file = NetworkFile::from_network(&network);

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
        let file = NetworkFile::from_network(&network);
        let rebuilt = file.to_network().expect("file to network");

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
        assert_eq!(NetworkFile::from_network(&rebuilt), file);
    }
}
