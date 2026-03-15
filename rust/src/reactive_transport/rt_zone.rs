use numpy::{
    ndarray::{Array1, s},
    PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{
    common_types::RtForcing,
    math::find_root_multi,
    reactive_transport::{kinetic_structures::{EquilibriumParameters, MineralParameters, MonodParameters, RtParameters, TstParameters}, reaction_network::ReactionNetwork},
};

pub const ZERO_CONC: f64 = 1e-20;

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RtStep {
    pub state: Array1<f64>,
    pub conc_in: Array1<f64>,
    pub mass_in: Array1<f64>,
    pub lat_conc: Array1<f64>,
    pub vert_conc: Array1<f64>,
    pub lat_mass: Array1<f64>,
    pub vert_mass: Array1<f64>,
    pub mineral_rates: Array1<f64>,
}

impl RtStep {
    pub fn from_arrays(
        state: Array1<f64>,
        conc_in: Array1<f64>,
        mass_in: Array1<f64>,
        lat_conc: Array1<f64>,
        vert_conc: Array1<f64>,
        lat_mass: Array1<f64>,
        vert_mass: Array1<f64>,
        mineral_rates: Array1<f64>,
    ) -> Self {
        Self {
            state,
            conc_in,
            mass_in,
            lat_conc,
            vert_conc,
            lat_mass,
            vert_mass,
            mineral_rates,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct MiscData {
    pub mineral_stoichiometry: Array1<f64>,
    pub species_mobility: Array1<bool>,
    pub mineral_molar_mass: Array1<f64>,
    pub rate_const: Array1<f64>,
}

#[pymethods]
impl MiscData {
    #[new]
    pub fn new(
        mineral_stoichiometry: PyReadonlyArray1<f64>,
        species_mobility: PyReadonlyArray1<f64>,
        mineral_molar_mass: PyReadonlyArray1<f64>,
        rate_const: PyReadonlyArray1<f64>,
    ) -> Self {
        unimplemented!()
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RtZone {
    pub network: ReactionNetwork,
    pub parameters: RtParameters,
    pub do_reactions: bool,
    pub do_speciation: bool,
    pub name: String,
    pub monod: MonodParameters,
    pub tst: TstParameters,
    pub eq: EquilibriumParameters,
    pub aux: MineralParameters,
    pub misc: MiscData,
}

impl RtZone {
    fn mass_balance_ode_rust(&self, chms: &Array1<f64>, d: &RtForcing) -> Array1<f64> {
        let transport_rate_vec: Array1<f64> = self.transport_rate_rust(chms, d);

        match self.do_reactions {
            true => transport_rate_vec + self.reaction_rate_rust(chms, d, false),
            false => transport_rate_vec,
        }
    }

    fn reaction_rate_rust<'py>(
        &self,
        chms: &Array1<f64>,
        d: &RtForcing,
        minerals_only: bool,
    ) -> Array1<f64> {
        let monod_rate: Array1<f64> = self.monod.rate_rust(&chms);
        let tst_rate: Array1<f64> = self.tst.rate_rust(&chms);
        let aux_rate: Array1<f64> = self.aux.factor_rust(d);

        let num_min: usize = self.monod.inhib_np.shape()[0];
        let num_spec: usize = chms.len();
        let min_start_ind: usize = num_spec - num_min - 1;
        let min_conc = chms.slice(s![min_start_ind..]).to_owned();

        let mineral_rates: Array1<f64> = 86_400.0 * &self.misc.rate_const * &self.aux.ssa * &self.misc.mineral_molar_mass * min_conc * aux_rate * (monod_rate + tst_rate);

        if minerals_only {
            return mineral_rates
        } else {
            let mut all_species_rates: Array1<f64> = &self.misc.mineral_stoichiometry * mineral_rates;
            for i in min_start_ind..num_spec {
                all_species_rates[i] = 0.0
            }

            return all_species_rates
        }
    }

    fn transport_rate_rust(&self, chms: &Array1<f64>, d: &RtForcing) -> Array1<f64> {
        let q_int: f64 = d.hydro_step.lat_flux + d.hydro_step.vert_flux;
        let q_ext: f64 = d.hydro_step.lat_flux_ext + d.hydro_step.vert_flux_ext - q_int;
        
        let mass_in: Array1<f64> = d.hydro_forc.q_in * &d.conc_in;

        let mass_out_internal: Array1<f64> = q_int * chms;
        let mass_out_external: Array1<f64> = q_ext * &d.conc_in;
        let mass_out: Array1<f64> = mass_out_external + mass_out_internal;

        let mut mass_change: Array1<f64> = mass_in - mass_out;

        for (i, m_i) in self.misc.species_mobility.iter().enumerate() {
            if !(*m_i) {
                mass_change[i] = 0.0
            }
        }
        mass_change
    }
}

#[pymethods]
impl RtZone {
    #[new]
    pub fn new(
        network: ReactionNetwork,
        params: RtParameters,
        do_reactions: bool,
        do_speciation: bool,
        name: String,
    ) -> PyResult<Self> {
        unimplemented!()
    }

    pub fn mass_balance_ode<'py>(
        &self,
        py: Python<'py>,
        chms: PyReadonlyArray1<f64>,
        d: &RtForcing,
    ) -> Bound<'py, PyArray1<f64>> {
        let chms_arr: Array1<f64> = chms.to_owned_array();
        self.mass_balance_ode_rust(&chms_arr, d).to_pyarray(py)
    }

    pub fn reaction_rate<'py>(
        &self,
        py: Python<'py>,
        chms: PyReadonlyArray1<f64>,
        d: &RtForcing,
        minerals_only: bool,
    ) -> Bound<'py, PyArray1<f64>> {
        let chms_arr: Array1<f64> = chms.to_owned_array();
        self.reaction_rate_rust(&chms_arr, d, minerals_only)
            .to_pyarray(py)
    }

    pub fn transport_rate<'py>(
        &self,
        py: Python<'py>,
        chms: PyReadonlyArray1<f64>,
        d: &RtForcing,
    ) -> Bound<'py, PyArray1<f64>> {
        let chms_arr: Array1<f64> = chms.to_owned_array();
        self.transport_rate_rust(&chms_arr, d).to_pyarray(py)
    }

    pub fn step(
        &self,
        c_0: PyReadonlyArray1<f64>,
        d: &RtForcing,
        dt_days: f64,
    ) -> PyResult<RtStep> {
        // Solve kinetic reactions first
        let c_0_arr: Array1<f64> = c_0.to_owned_array();

        let residual = |c: &Array1<f64>| (&c_0_arr - c) + dt_days * self.mass_balance_ode_rust(c, d);

        let c_after_rt: Array1<f64> = match find_root_multi(&residual, c_0_arr.clone()) {
            Ok(v) => v,
            Err(_) => return Err(PyValueError::new_err("Failed to solve RT step during simulation")),
        };

        let c_after_eq = match self.do_speciation {
            false => c_after_rt.clone(),
            true => {
                match self.eq.solve_equilibrium_rust(&c_after_rt) {
                    Ok(v) => v,
                    Err(_) => return Err(PyValueError::new_err("Failed to solve speciation after RT"))
                }
            }
        };

        let q_int: f64 = d.hydro_step.lat_flux + d.hydro_step.vert_flux;
        let q_ext: f64 = d.hydro_step.lat_flux_ext + d.hydro_step.vert_flux_ext - q_int;
        let total_q_out_water: f64 = q_int + q_ext;

        let lat_conc: Array1<f64>;
        let vert_conc: Array1<f64>;
        let (lat_conc, vert_conc) = match q_int + q_ext > 1e-6 {
            true => {
                let total_mass_out_flux: Array1<f64> = q_int * &c_after_eq + q_ext * &d.conc_in;

                let lat: Array1<f64> = &total_mass_out_flux * d.hydro_step.lat_flux_ext / total_q_out_water;
                let vert: Array1<f64> = &total_mass_out_flux * d.hydro_step.vert_flux_ext / total_q_out_water;

                (lat, vert)
            },
            false => {
                let num_species = c_0_arr.len();
                let z: Array1<f64> = Array1::from_elem(num_species, ZERO_CONC);

                (z.clone(), z.clone())
            },
        };

        let mineral_rates: Array1<f64> = self.reaction_rate_rust(&c_after_eq, d, true);

        Ok(RtStep {
            state: c_after_eq,
            conc_in: d.conc_in.clone(),
            mass_in: &d.conc_in * d.hydro_step.q_in,
            lat_conc: lat_conc.clone(),
            vert_conc: vert_conc.clone(),
            lat_mass: lat_conc * d.hydro_step.lat_flux_ext,
            vert_mass: vert_conc * d.hydro_step.vert_flux_ext,
            mineral_rates,
        })
    }

    #[getter]
    pub fn all_species(&self) -> Vec<String> {
        self.network.species_names()
    }

    #[getter]
    pub fn mineral_species(&self) -> Vec<String> {
        self.network.mineral_species_names()
    }

    pub fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.parameters.to_array(py)
    }

    #[staticmethod]
    fn from_array<'py>(
        py: Python<'py>,
        arr: PyReadonlyArray1<f64>,
        network: &ReactionNetwork,
        do_reactions: bool,
        do_speciation: bool,
        name: String,
    ) -> PyResult<Self> {
        match RtParameters::from_array(py, arr) {
            Ok(v) => {
                match Self::new(network.clone(), v, do_reactions, do_speciation, name) {
                    Ok(new_self) => return Ok(new_self),
                    Err(e) => return Err(e)
                }
            },
            Err(e) => return Err(e)
        }
    }
}
