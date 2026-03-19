use numpy::{
    ndarray::{s, Array1, Array2},
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
use polars::prelude::Float64Type;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

use crate::{
    common_types::RtForcing,
    math::{find_root_multi, RootFindingError},
    reactive_transport::{
        kinetic_structures::{
            EquilibriumParameters, MineralParameters, MonodParameters, RtParameters, TstParameters,
            ZoneDimensions,
        },
        reaction_network::ReactionNetwork,
    },
};

pub const ZERO_CONC: f64 = 1e-20;

#[pyclass]
#[derive(Debug)]
pub struct RtStep {
    #[pyo3(get)]
    pub state: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub conc_in: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub mass_in: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub lat_conc: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub vert_conc: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub lat_mass: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub vert_mass: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub mineral_rates: Py<PyArray1<f64>>,
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct MiscData {
    pub mineral_stoichiometry: Array2<f64>,
    pub species_mobility: Array1<bool>,
    pub mineral_molar_mass: Array1<f64>,
    pub rate_const: Array1<f64>,
}

#[pymethods]
impl MiscData {
    #[new]
    pub fn new(
        mineral_stoichiometry: PyReadonlyArray2<f64>,
        species_mobility: PyReadonlyArray1<bool>,
        mineral_molar_mass: PyReadonlyArray1<f64>,
        rate_const: PyReadonlyArray1<f64>,
    ) -> Self {
        Self {
            mineral_stoichiometry: mineral_stoichiometry.to_owned_array(),
            species_mobility: species_mobility.to_owned_array(),
            mineral_molar_mass: mineral_molar_mass.to_owned_array(),
            rate_const: rate_const.to_owned_array(),
        }
    }

    pub fn get_mineral_stoichiometry<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.mineral_stoichiometry.to_pyarray(py)
    }

    pub fn get_species_mobility<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.species_mobility.to_pyarray(py)
    }

    pub fn get_mineral_molar_mass<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.mineral_molar_mass.to_pyarray(py)
    }

    pub fn get_rate_const<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.rate_const.to_pyarray(py)
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RtZone {
    #[pyo3(get)]
    pub network: ReactionNetwork,
    #[pyo3(get)]
    pub parameters: RtParameters,
    #[pyo3(get)]
    pub do_reactions: bool,
    #[pyo3(get)]
    pub do_speciation: bool,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub monod: MonodParameters,
    #[pyo3(get)]
    pub tst: TstParameters,
    #[pyo3(get)]
    pub eq: EquilibriumParameters,
    #[pyo3(get)]
    pub aux: MineralParameters,
    #[pyo3(get)]
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
        let min_start_ind: usize = num_spec - num_min;
        let min_conc: Array1<f64> = chms.slice(s![min_start_ind..]).to_owned();

        let mineral_rates: Array1<f64> = 86_400.0
            * &self.misc.rate_const
            * &self.aux.ssa
            * &self.misc.mineral_molar_mass
            * min_conc
            * aux_rate
            * (monod_rate + tst_rate);

        if minerals_only {
            return mineral_rates;
        } else {
            let mut all_species_rates: Array1<f64> =
                self.misc.mineral_stoichiometry.dot(&mineral_rates);
            for i in min_start_ind..num_spec {
                all_species_rates[i] = 0.0
            }

            return all_species_rates;
        }
    }

    fn transport_rate_rust(&self, chms: &Array1<f64>, d: &RtForcing) -> Array1<f64> {
        let q_int: f64 = d.hydro_step.lat_flux + d.hydro_step.vert_flux;
        let q_ext: f64 = d.hydro_step.lat_flux_ext + d.hydro_step.vert_flux_ext - q_int;

        let mass_in: Array1<f64> = d.hydro_forc.q_in * &d._conc_in;

        let mass_out_internal: Array1<f64> = q_int * chms;
        let mass_out_external: Array1<f64> = q_ext * &d._conc_in;
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
    #[pyo3(signature = (network, params, do_reactions = true, do_speciation = true, name = "unnamed".to_string()))]
    pub fn new<'py>(
        py: Python<'py>,
        network: ReactionNetwork,
        params: RtParameters,
        do_reactions: bool,
        do_speciation: bool,
        name: String,
    ) -> PyResult<Self> {
        let monod: MonodParameters = match network.monod_params() {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("Failed to create monod parameters: {}", e.to_string());
                return Err(PyValueError::new_err(msg));
            }
        };
        let tst: TstParameters = match network.tst_params() {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("Failed to create Tst parameters: {}", e.to_string());
                return Err(PyValueError::new_err(msg));
            }
        };
        let eq: EquilibriumParameters = match network.equilibrium_parameters() {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("Failed to create equilibrium parameters: {}", e.to_string());
                return Err(PyValueError::new_err(msg));
            }
        };
        let minerals: MineralParameters = params.mineral_params.clone();
        let stoich = network
            .mineral_stoichiometry()?
            .0
            .to_ndarray::<Float64Type>(polars::prelude::IndexOrder::C)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mobility = network.transport_mask(py);
        let min_molar_mass = network.mineral_molar_masses(py);

        let misc = MiscData {
            mineral_stoichiometry: stoich,
            species_mobility: mobility.to_owned_array(),
            mineral_molar_mass: min_molar_mass.to_owned_array(),
            rate_const: network
                .rate_consts(py)
                .expect("Failed to get rate constants")
                .to_owned_array(),
        };

        Ok(Self {
            network: network,
            parameters: params,
            do_reactions: do_reactions,
            do_speciation: do_speciation,
            name: name,
            monod: monod,
            tst: tst,
            eq: eq,
            aux: minerals,
            misc: misc,
        })
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

    #[pyo3(signature = (chms, d, minerals_only = false))]
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

    pub fn step<'py>(
        &self,
        py: Python<'py>,
        c_0: PyReadonlyArray1<f64>,
        d: &RtForcing,
        dt_days: f64,
    ) -> PyResult<RtStep> {
        // Solve kinetic reactions first
        let c_0_arr: Array1<f64> = c_0.to_owned_array();

        let residual =
            |c: &Array1<f64>| (&c_0_arr - c) + dt_days * self.mass_balance_ode_rust(c, d);

        let c_after_rt: Array1<f64> = match find_root_multi(&residual, c_0_arr.clone()) {
            Ok(v) => v.map(|x| x.max(ZERO_CONC)),
            Err(e) => match e {
                // Match tuple-style variant with no data
                RootFindingError::IterationError() => {
                    return Err(crate::IterationError::new_err(
                        "Exceeded iterations when solving RT system",
                    ));
                }
                // Match and bind the data from LinearSystemError
                RootFindingError::LinearSystemError(description) => {
                    return Err(crate::LinearSystemError::new_err(
                        "Exceeded iterations when solving RT system",
                    ));
                }
                // Match the 'Other' variant
                RootFindingError::Other() => {
                    return Err(PyRuntimeError::new_err(
                        "Failed to solve RT system with other error",
                    ));
                }
            },
        };

        let c_after_eq = match self.do_speciation {
            false => c_after_rt.clone(),
            true => match self.eq.solve_equilibrium_rust(&c_after_rt) {
                Ok(v) => {
                    // eprintln!("Solved speciation");
                    v.map(|x| x.max(ZERO_CONC))
                }
                Err(e) => match e {
                    // Match tuple-style variant with no data
                    RootFindingError::IterationError() => {
                        return Err(crate::IterationError::new_err(
                            "Exceeded iterations when solving speciation",
                        ));
                    }
                    // Match and bind the data from LinearSystemError
                    RootFindingError::LinearSystemError(description) => {
                        return Err(crate::LinearSystemError::new_err(
                            "Exceeded iterations when solving speciation",
                        ));
                    }
                    // Match the 'Other' variant
                    RootFindingError::Other() => {
                        return Err(PyRuntimeError::new_err(
                            "Failed to solve speciation with other error",
                        ));
                    }
                },
            },
        };

        let q_int: f64 = d.hydro_step.lat_flux + d.hydro_step.vert_flux;
        let q_ext: f64 = d.hydro_step.lat_flux_ext + d.hydro_step.vert_flux_ext - q_int;
        let total_q_out_water: f64 = q_int + q_ext;

        let lat_conc: Array1<f64>;
        let vert_conc: Array1<f64>;
        let (lat_conc, vert_conc) = match q_int + q_ext > 1e-6 {
            true => {
                let total_mass_out_flux: Array1<f64> = q_int * &c_after_eq + q_ext * &d._conc_in;

                let lat: Array1<f64> =
                    &total_mass_out_flux * d.hydro_step.lat_flux_ext / total_q_out_water;
                let vert: Array1<f64> =
                    &total_mass_out_flux * d.hydro_step.vert_flux_ext / total_q_out_water;

                (lat, vert)
            }
            false => {
                let num_species = c_0_arr.len();
                let z: Array1<f64> = Array1::from_elem(num_species, ZERO_CONC);

                (z.clone(), z.clone())
            }
        };

        let mineral_rates: Array1<f64> = self.reaction_rate_rust(&c_after_eq, d, true);

        Ok(RtStep {
            state: c_after_eq.to_pyarray(py).unbind(),
            conc_in: d._conc_in.clone().to_pyarray(py).unbind(),
            mass_in: (&d._conc_in * d.hydro_step.q_in).to_pyarray(py).unbind(),
            lat_conc: lat_conc.clone().to_pyarray(py).unbind(),
            vert_conc: vert_conc.clone().to_pyarray(py).unbind(),
            lat_mass: (lat_conc * d.hydro_step.lat_flux_ext)
                .to_pyarray(py)
                .unbind(),
            vert_mass: (vert_conc * d.hydro_step.vert_flux_ext)
                .to_pyarray(py)
                .unbind(),
            mineral_rates: mineral_rates.to_pyarray(py).unbind(),
        })
    }

    pub fn monod_rate<'py>(
        &self,
        py: Python<'py>,
        chms: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let chms_arr: Array1<f64> = chms.to_owned_array();
        self.monod.rate_rust(&chms_arr).to_pyarray(py)
    }

    pub fn tst_rate<'py>(
        &self,
        py: Python<'py>,
        chms: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let chms_arr: Array1<f64> = chms.to_owned_array();
        self.tst.rate_rust(&chms_arr).to_pyarray(py)
    }

    pub fn aux_factor<'py>(&self, py: Python<'py>, d: RtForcing) -> Bound<'py, PyArray1<f64>> {
        self.aux.factor(py, d)
    }

    #[getter]
    pub fn all_species(&self) -> Vec<String> {
        self.network.species_names()
    }

    #[getter]
    pub fn mineral_species(&self) -> Vec<String> {
        self.network.mineral_species_names()
    }

    #[getter]
    pub fn dimensions(&self) -> ZoneDimensions {
        self.parameters.dimensions.clone()
    }

    #[getter]
    pub fn num_species(&self) -> usize {
        self.network.num_species()
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
            Ok(v) => match Self::new(py, network.clone(), v, do_reactions, do_speciation, name) {
                Ok(new_self) => return Ok(new_self),
                Err(e) => return Err(e),
            },
            Err(e) => return Err(e),
        }
    }
}
