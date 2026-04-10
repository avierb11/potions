use std::f64;
use std::{collections::HashMap, ops::Not};

use numpy::{ndarray::Array1, PyArray1, ToPyArray};
use polars::prelude::*;
use polars::{df, prelude::NamedFrom, series::Series};
use pyo3::exceptions::PyRuntimeError;
use pyo3::{exceptions::PyKeyError, prelude::*};
use pyo3_polars::{PyDataFrame, PySeries};

use crate::reactive_transport::database::ExchangeReaction;
use crate::reactive_transport::{
    database::{MineralKineticData, MineralSpecies, PrimaryAqueousSpecies, SecondarySpecies},
    kinetic_structures::{EquilibriumParameters, MonodParameters, TstParameters},
};

#[pyclass(from_py_object, module = "potions.core")]
#[derive(Clone, Debug)]
pub struct ReactionNetwork {
    #[pyo3(get)]
    pub primary_aqueous: Vec<PrimaryAqueousSpecies>,
    #[pyo3(get)]
    pub mineral: Vec<MineralSpecies>,
    #[pyo3(get)]
    pub secondary: Vec<SecondarySpecies>,
    #[pyo3(get)]
    pub mineral_kinetics: MineralKineticData,
    #[pyo3(get)]
    pub exchange_species: Vec<ExchangeReaction>,
    #[pyo3(get)]
    pub species: PyDataFrame,
}

#[pymethods]
impl ReactionNetwork {
    #[new]
    pub fn new(
        primary_aqueous: Vec<PrimaryAqueousSpecies>,
        mineral: Vec<MineralSpecies>,
        secondary: Vec<SecondarySpecies>,
        mineral_kinetics: MineralKineticData,
        exchange_species: Vec<ExchangeReaction>,
    ) -> PyResult<Self> {
        let mut species_types: Vec<String> = Vec::new();
        let mut names: Vec<String> = Vec::new();

        species_types.append(&mut vec!["primary".to_owned(); primary_aqueous.len()]);
        species_types.append(&mut vec!["secondary".to_owned(); secondary.len()]);
        species_types.append(&mut vec!["mineral".to_owned(); mineral.len()]);

        for x in &primary_aqueous {
            names.push(x.name.clone());
        }

        for x in &secondary {
            names.push(x.name.clone());
        }

        for x in &mineral {
            names.push(x.name.clone());
        }

        if exchange_species.len() > 0 {
            species_types.append(&mut vec!["exchange".to_owned(); exchange_species.len() + 1]);

            names.push("X-".to_string());

            for x in &exchange_species {
                names.push(x.name.clone());
            }
        }

        let species_df = df!(
            "name" => names,
            "type" => species_types
        )
        .expect("Failed to create dataframe");

        let py_df = PyDataFrame(species_df);

        Ok(Self {
            primary_aqueous,
            mineral,
            secondary,
            mineral_kinetics,
            exchange_species,
            species: py_df,
        })
    }

    #[getter]
    pub fn species_order(&self) -> Vec<String> {
        let spec = self
            .species
            .0
            .column("name")
            .expect("Failed to get column 'names'")
            .clone();
        let spec_vec: Vec<String> = spec
            .str()
            .expect("Failed to convert name column to string type")
            .into_iter()
            .map(|x| x.unwrap().to_string())
            .collect();
        spec_vec
    }

    #[getter]
    pub fn has_exchange(&self) -> bool {
        self.exchange_species.len() > 0
    }

    #[getter]
    pub fn charges(&self) -> PySeries {
        let mut charge_vec: Vec<f64> = Vec::new();
        for x in &self.primary_aqueous {
            charge_vec.push(x.charge);
        }

        for x in &self.secondary {
            charge_vec.push(x.charge);
        }

        for x in &self.mineral {
            charge_vec.push(0.0);
        }

        if self.has_exchange() {
            charge_vec.push(-1.0);
            for exch in self.exchange_species.iter() {
                charge_vec.push(exch.charge);
            }
        }

        let ser = Series::new("charge".into(), charge_vec);

        PySeries(ser)
    }

    #[getter]
    pub fn equilibrium_species(&self) -> PyDataFrame {
        let mineral_mask = self
            .species
            .0
            .column("type")
            .expect("Failed to get 'type' column")
            .str()
            .expect("Failed to convert column to string")
            .equal("mineral")
            .not();
        let exchange_mask = self
            .species
            .0
            .column("type")
            .expect("Failed to get 'type' column")
            .str()
            .expect("Failed to convert column to string")
            .equal("exchange")
            .not();

        let mask: ChunkedArray<BooleanType> = mineral_mask
            .into_iter()
            .zip(exchange_mask.into_iter())
            .map(|(a, b)| match (a, b) {
                (Some(a_i), Some(b_i)) => a_i & b_i,
                _ => false,
            })
            .collect();

        let eq_spec = self
            .species
            .clone()
            .0
            .filter(&mask)
            .expect("Failed to filter dataframe with mask");
        PyDataFrame(eq_spec)
    }

    #[getter]
    pub fn kinetic_species(&self) -> PyDataFrame {
        self.species.clone()
    }

    #[getter]
    pub fn equilibrium_parameters(&self) -> PyResult<EquilibriumParameters> {
        let mut total_columns: Vec<Column> = Vec::new();
        let mut total_colnames: Vec<String> = Vec::new();
        let mut sec_stoich_columns: Vec<Column> = Vec::new();
        let mut sec_stoich_colnames: Vec<String> = Vec::new();
        let mut log_eq_consts: Vec<f64> = Vec::new();

        // ==== Construct the equilibrium matrix ==== //

        for spec in self.secondary.iter() {
            let mut stoich_i: Vec<f64> = Vec::new();
            log_eq_consts.push(spec.eq_consts[1]);

            for s_i in self.species_names().iter() {
                if spec.stoichiometry.contains_key(s_i) {
                    stoich_i.push(spec.stoichiometry[s_i]);
                } else {
                    stoich_i.push(0.0);
                }
            }

            sec_stoich_columns.push(Column::new(spec.name.clone().into(), stoich_i));
            sec_stoich_colnames.push(spec.name.clone().into())
        }

        for exch in self.exchange_species.iter() {
            log_eq_consts.push(exch.log10_k_eq);
            let mut stoich_i: Vec<f64> = Vec::new();

            for s_i in self.species_names().iter() {
                if exch.stoichiometry.contains_key(s_i) {
                    stoich_i.push(exch.stoichiometry[s_i]);
                } else {
                    stoich_i.push(0.0);
                }
            }

            sec_stoich_columns.push(Column::new(exch.name.clone().into(), stoich_i));
        }

        // ========================================== //

        // ==== Mass Conservation matrix ==== //
        let species_ids = self.species_indices();
        for (i, tot_spec_i) in self.total_species().iter().enumerate() {
            total_colnames.push(tot_spec_i.clone().into());
            let tot_id: usize = species_ids[tot_spec_i];
            // Check on each of the equiliubrium reactions or charge balance
            if tot_spec_i == "H+" {
                // Do charge balance on H+
                let charges = self.charges().0;
                total_columns.push(charges.into());
            } else {
                let mut tot_i: Array1<f64> = Array1::zeros(self.num_species());

                tot_i[tot_id] = 1.0; // Make sure that the base primary species is tracked

                for sec in self.secondary.iter() {
                    if sec.stoichiometry.contains_key(tot_spec_i) {
                        let sec_id = species_ids[&sec.name];
                        tot_i[sec_id] = sec.stoichiometry[tot_spec_i];
                    }
                }

                for exch in self.exchange_species.iter() {
                    if exch.stoichiometry.contains_key(tot_spec_i) {
                        let exch_id = species_ids[&exch.name];
                        tot_i[exch_id] = exch.stoichiometry[tot_spec_i];
                    }
                }
                total_columns.push(Column::new(tot_spec_i.into(), tot_i.to_vec()));
            }
        }
        // ================================== //

        let total_df: DataFrame = DataFrame::new(self.num_species(), total_columns)
            .expect("Failed to construct mass and charge conservation dataframe")
            .transpose(
                None,
                Some(polars::polars_utils::either::Right(self.species_names())),
            )
            .expect("Failed to transpose mass_charge dataframe:");
        let sec_stoich_df: DataFrame = match sec_stoich_columns.len() == 0 {
            true => DataFrame::empty(),
            false => DataFrame::new(self.num_species(), sec_stoich_columns)
                .expect("Failed to construct mass and charge conservation dataframe")
                .transpose(
                    None,
                    Some(polars::polars_utils::either::Right(self.species_names())),
                )
                .expect("Failed to transpose secondary stoichiometry dataframe"),
        };
        let seq_eq_vec: Series = Series::new("log_keq".into(), log_eq_consts);

        EquilibriumParameters::new(
            PyDataFrame(sec_stoich_df),
            PySeries(seq_eq_vec),
            PyDataFrame(total_df),
        )
    }

    #[getter]
    pub fn tst_params(&self) -> PyResult<TstParameters> {
        let mut mineral_stoich_cols: Vec<Column> = Vec::new();
        let mut dep_cols: Vec<Column> = Vec::new();

        let mut min_eq_vals: Vec<f64> = Vec::new();

        let spec_indices = self.species_indices();
        for m in self.mineral.iter() {
            if self.mineral_kinetics.tst_reactions.contains_key(&m.name) {
                // This reaction is a TST reaction
                min_eq_vals.push((10_f64).powf(m.eq_consts[1]));
                let mut min_stoich: Vec<f64> = Vec::new();
                let mut dep_stoich: Vec<f64> = Vec::new();
                let dep = &self.mineral_kinetics.tst_reactions[&m.name].dependence;
                let db_stoich = &m.stoichiometry;
                for spec in self.species_names() {
                    if dep.contains_key(&spec) {
                        dep_stoich.push(dep[&spec]);
                        min_stoich.push(db_stoich[&spec]);
                    } else {
                        dep_stoich.push(0.0);
                        min_stoich.push(0.0);
                    }
                }

                mineral_stoich_cols.push(Column::new(m.name.clone().into(), dep_stoich));
                dep_cols.push(Column::new(m.name.clone().into(), min_stoich));
            } else {
                // This reaction is NOT a TST reaction
                min_eq_vals.push(1.0);
                let min_stoich: Vec<f64> = vec![0.0; self.num_species()];
                let dep_stoich: Vec<f64> = vec![0.0; self.num_species()];
                mineral_stoich_cols.push(Column::new(m.name.clone().into(), min_stoich));
                dep_cols.push(Column::new(m.name.clone().into(), dep_stoich));
            }
        }

        let min_eq_ser: Series = Series::new("eq_consts".into(), min_eq_vals);

        let dep_df = match dep_cols.len() == 0 {
            true => DataFrame::empty(),
            false => DataFrame::new(self.num_species(), dep_cols)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .transpose(
                    None,
                    Some(polars::polars_utils::either::Right(self.species_names())),
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };

        let mineral_stoich_df = match mineral_stoich_cols.len() == 0 {
            true => DataFrame::empty(),
            false => DataFrame::new(self.num_species(), mineral_stoich_cols)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .transpose(
                    None,
                    Some(polars::polars_utils::either::Right(self.species_names())),
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };

        TstParameters::new(
            PyDataFrame(mineral_stoich_df),
            PyDataFrame(dep_df),
            PySeries(min_eq_ser),
        )
    }

    #[getter]
    pub fn monod_params(&self) -> PyResult<MonodParameters> {
        let mut monod_cols: Vec<Column> = Vec::new();
        let mut inhib_cols: Vec<Column> = Vec::new();

        for m in self.mineral.iter() {
            let mut monod_i: Vec<f64> = Vec::new();
            let mut inhib_i: Vec<f64> = Vec::new();

            if self.mineral_kinetics.monod_reactions.contains_key(&m.name) {
                let rxn = &self.mineral_kinetics.monod_reactions[&m.name];
                for spec in &self.species_names() {
                    if rxn.monod_terms.contains_key(spec) {
                        monod_i.push(rxn.monod_terms[spec]);
                    } else {
                        monod_i.push(f64::NAN)
                    }

                    if rxn.inhib_terms.contains_key(spec) {
                        inhib_i.push(rxn.inhib_terms[spec]);
                    } else {
                        inhib_i.push(f64::NAN)
                    }
                }
            } else {
                monod_i = vec![0.0; self.num_species()];
                inhib_i = vec![0.0; self.num_species()];
            }

            monod_cols.push(Column::new(m.name.clone().into(), monod_i));
            inhib_cols.push(Column::new(m.name.clone().into(), inhib_i));
        }

        let monod_df = match monod_cols.len() == 0 {
            true => DataFrame::empty(),
            false => DataFrame::new(self.num_species(), monod_cols)
                .expect("Failed to create Monod Matrix")
                .transpose(
                    None,
                    Some(polars::polars_utils::either::Right(self.species_names())),
                )
                .expect("Failed to transpose Monod dataframe"),
        };

        let inhib_df = match inhib_cols.len() == 0 {
            true => DataFrame::empty(),
            false => DataFrame::new(self.num_species(), inhib_cols)
                .expect("Failed to create Inhibition Matrix")
                .transpose(
                    None,
                    Some(polars::polars_utils::either::Right(self.species_names())),
                )
                .expect("Failed to transpose Inhibition dataframe"),
        };

        // let monod_df: DataFrame = ;
        let monod_mat: PyDataFrame = PyDataFrame(monod_df);

        let inhib_mat: PyDataFrame = PyDataFrame(inhib_df);

        MonodParameters::new(monod_mat, inhib_mat)
    }

    #[getter]
    pub fn species_names(&self) -> Vec<String> {
        let mut spec_names: Vec<String> = Vec::new();

        for x in &self.primary_aqueous {
            spec_names.push(x.name.clone());
        }

        for x in &self.secondary {
            spec_names.push(x.name.clone());
        }

        for x in &self.mineral {
            spec_names.push(x.name.clone());
        }

        if self.has_exchange() {
            spec_names.push("X-".to_string());

            for exch in self.exchange_species.iter() {
                spec_names.push(exch.name.clone());
            }
        }

        spec_names
    }

    #[getter]
    pub fn mineral_species_names(&self) -> Vec<String> {
        self.mineral.iter().map(|x| x.name.clone()).collect()
    }

    #[getter]
    pub fn mineral_stoichiometry(&self) -> PyResult<PyDataFrame> {
        let mut columns: Vec<Column> = Vec::new();

        for m in &self.mineral {
            let mut stoich_i: Vec<f64> = Vec::new();
            for s_i in self.species_names() {
                if m.stoichiometry.contains_key(&s_i) {
                    stoich_i.push(m.stoichiometry[&s_i]);
                } else {
                    stoich_i.push(0.0)
                }
            }
            columns.push(Column::new(m.name.clone().into(), stoich_i));
        }

        let df: DataFrame = DataFrame::new(self.num_species(), columns)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyDataFrame(df))
    }

    #[getter]
    pub fn transport_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let mut mask: Array1<bool> = Array1::from_elem(self.num_species(), true);

        for i in self.num_aqueous_species()..self.num_species() {
            mask[i] = false;
        }

        mask.to_pyarray(py)
    }

    #[getter]
    pub fn mineral_molar_masses<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let molar_masses: Vec<f64> = self.mineral.iter().map(|x| x.molar_mass).collect();
        PyArray1::from_vec(py, molar_masses)
    }

    #[getter]
    pub fn rate_consts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut log10_rate_consts: Vec<f64> = Vec::new();

        for m in &self.mineral {
            if self.mineral_kinetics.monod_reactions.contains_key(&m.name) {
                log10_rate_consts
                    .push(self.mineral_kinetics.monod_reactions[&m.name].rate_constant);
            } else if self.mineral_kinetics.tst_reactions.contains_key(&m.name) {
                log10_rate_consts.push(self.mineral_kinetics.tst_reactions[&m.name].rate_constant);
            } else {
                let msg: String = format!("Cannot find mineral {}", &m.name);
                return Err(PyKeyError::new_err(msg));
            }
        }

        let log_arr: Array1<f64> = Array1::from_vec(log10_rate_consts);
        let arr: Array1<f64> = log_arr.map(|x| (10f64).powf(*x));

        Ok(arr.to_pyarray(py))
    }

    #[getter]
    pub fn num_minerals(&self) -> usize {
        self.mineral.len()
    }

    #[getter]
    pub fn num_exchange_species(&self) -> usize {
        if self.has_exchange() {
            self.exchange_species.len() + 1
        } else {
            0
        }
    }

    #[getter]
    pub fn primary_ids(&self) -> Vec<usize> {
        let mut id_vals: Vec<usize> = Vec::new();

        for i in 0..self.primary_aqueous.len() {
            id_vals.push(i);
        }

        id_vals
    }

    #[getter]
    pub fn secondary_ids(&self) -> Vec<usize> {
        let mut id_vals: Vec<usize> = Vec::new();

        let num_prim = self.primary_aqueous.len();

        for i in 0..self.num_secondary() {
            id_vals.push(i + num_prim);
        }

        id_vals
    }

    #[getter]
    pub fn mineral_ids(&self) -> Vec<usize> {
        let mut id_vals: Vec<usize> = Vec::new();

        let offset = self.primary_aqueous.len() + self.num_secondary();

        for i in 0..self.num_minerals() {
            id_vals.push(i + offset);
        }

        id_vals
    }

    #[getter]
    pub fn exchange_ids(&self) -> Vec<usize> {
        let mut id_vals: Vec<usize> = Vec::new();

        let offset = self.primary_aqueous.len() + self.num_secondary() + self.num_minerals();

        for i in 0..self.num_exchange_species() {
            id_vals.push(i + offset);
        }

        id_vals
    }

    #[getter]
    pub fn num_species(&self) -> usize {
        self.primary_aqueous.len() + self.secondary.len() + self.mineral.len() + {
            if self.has_exchange() {
                1 + self.exchange_species.len()
            } else {
                0
            }
        }
    }

    #[getter]
    pub fn num_secondary(&self) -> usize {
        self.secondary.len()
    }

    #[getter]
    pub fn num_mineral_parameters(&self) -> usize {
        5 * self.num_minerals()
    }

    #[getter]
    pub fn species_types(&self) -> HashMap<String, String> {
        let mut s: HashMap<String, String> = HashMap::new();

        for x in self.primary_aqueous.iter() {
            s.insert(x.name.clone(), "primary".into());
        }
        for x in self.secondary.iter() {
            s.insert(x.name.clone(), "secondary".into());
        }
        for x in self.mineral.iter() {
            s.insert(x.name.clone(), "mineral".into());
        }

        if self.has_exchange() {
            s.insert("X-".to_string(), "exchange".into());

            for x in self.exchange_species.iter() {
                s.insert(x.name.clone(), "exchange".into());
            }
        }

        s
    }

    #[getter]
    pub fn species_indices(&self) -> HashMap<String, usize> {
        let mut s: HashMap<String, usize> = HashMap::new();
        let mut counter: usize = 0;

        for x in self.primary_aqueous.iter() {
            s.insert(x.name.clone(), counter);
            counter += 1;
        }
        for x in self.secondary.iter() {
            s.insert(x.name.clone(), counter);
            counter += 1;
        }
        for x in self.mineral.iter() {
            s.insert(x.name.clone(), counter);
            counter += 1;
        }

        if self.has_exchange() {
            s.insert("X-".to_string(), counter);
            counter += 1;

            for x in self.exchange_species.iter() {
                s.insert(x.name.clone(), counter);
                counter += 1;
            }
        }

        s
    }

    #[getter]
    pub fn num_aqueous_species(&self) -> usize {
        self.primary_aqueous.len() + self.secondary.len()
    }

    #[getter]
    pub fn num_total_species(&self) -> usize {
        self.primary_aqueous.len() + self.mineral.len() + self.num_exchange_species()
    }

    #[getter]
    pub fn total_species(&self) -> Vec<String> {
        let mut tot: Vec<String> = Vec::new();

        for x in self.primary_aqueous.iter() {
            tot.push(x.name.clone())
        }

        for x in self.mineral.iter() {
            tot.push(x.name.clone())
        }

        if self.has_exchange() {
            tot.push("X-".to_string());
        }

        tot
    }

    #[pyo3(signature = (init_conc=1e-6))]
    pub fn get_default_aqueous_initial_state<'py>(
        &self,
        py: Python<'py>,
        init_conc: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let conc: Array1<f64> = Array1::from_elem(self.num_aqueous_species(), init_conc);
        conc.to_pyarray(py)
    }

    #[getter]
    pub fn mineral_names(&self) -> Vec<String> {
        self.mineral.iter().map(|x| x.name.clone()).collect()
    }

    #[getter]
    pub fn primary_names(&self) -> Vec<String> {
        self.primary_aqueous
            .iter()
            .map(|x| x.name.clone())
            .collect()
    }

    #[getter]
    pub fn secondary_names(&self) -> Vec<String> {
        self.secondary.iter().map(|x| x.name.clone()).collect()
    }

    #[getter]
    pub fn exchange_names(&self) -> Vec<String> {
        let mut spec_names: Vec<String> = vec!["X-".to_string()];

        for x in self.exchange_species.iter() {
            spec_names.push(x.name.clone());
        }

        spec_names
    }

    pub fn __getstate__(
        &self,
    ) -> (
        Vec<PrimaryAqueousSpecies>,
        Vec<MineralSpecies>,
        Vec<SecondarySpecies>,
        MineralKineticData,
        Vec<ExchangeReaction>,
    ) {
        (
            self.primary_aqueous.clone(),
            self.mineral.clone(),
            self.secondary.clone(),
            self.mineral_kinetics.clone(),
            self.exchange_species.clone(),
        )
    }

    pub fn __setstate__(
        &mut self,
        state: (
            Vec<PrimaryAqueousSpecies>,
            Vec<MineralSpecies>,
            Vec<SecondarySpecies>,
            MineralKineticData,
            Vec<ExchangeReaction>,
        ),
    ) {
        let (pa, m, s, mk, exch) = state;

        // Regenerate the derived species DataFrame
        let mut species_types = Vec::new();
        let mut names = Vec::new();

        species_types.extend(vec!["primary".to_string(); pa.len()]);
        species_types.extend(vec!["secondary".to_string(); s.len()]);
        species_types.extend(vec!["mineral".to_string(); m.len()]);

        for x in &pa {
            names.push(x.name.clone());
        }
        for x in &s {
            names.push(x.name.clone());
        }
        for x in &m {
            names.push(x.name.clone());
        }

        if self.has_exchange() {
            names.push("X-".to_string());
            for x in &exch {
                names.push(x.name.clone());
            }

            let num_exchange = 1 + exch.len();
            species_types.extend(vec!["exchange".to_string(); num_exchange]);
        }

        let species_df = df!(
            "name" => names,
            "type" => species_types
        )
        .expect("Failed to reconstruct dataframe");

        self.primary_aqueous = pa;
        self.mineral = m;
        self.secondary = s;
        self.mineral_kinetics = mk;
        self.exchange_species = exch;
        self.species = PyDataFrame(species_df);
    }

    pub fn __getnewargs__(
        &self,
    ) -> (
        Vec<PrimaryAqueousSpecies>,
        Vec<MineralSpecies>,
        Vec<SecondarySpecies>,
        MineralKineticData,
        Vec<ExchangeReaction>,
    ) {
        (
            self.primary_aqueous.clone(),
            self.mineral.clone(),
            self.secondary.clone(),
            self.mineral_kinetics.clone(),
            self.exchange_species.clone(),
        )
    }
}
