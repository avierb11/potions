use numpy::{
    ndarray::{s, Array1},
    PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray,
};
use polars::prelude::Float64Type;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

use crate::{
    common_types::{MiscData, RtForcing, RtStep, ZERO_CONC},
    math::find_root_multi,
    molar, molar_per_time, moles, moles_per_time,
    reactive_transport::{
        kinetic_structures::{
            EquilibriumParameters, MineralParameters, MonodParameters, RiverDimensions,
            RiverParameters, TstParameters,
        },
        reaction_network::ReactionNetwork,
    },
};

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RiverZone {
    #[pyo3(get)]
    pub network: ReactionNetwork,
    #[pyo3(get)]
    pub parameters: RiverParameters,
    #[pyo3(get, set)]
    pub do_reactions: bool,
    #[pyo3(get, set)]
    pub do_speciation: bool,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get)]
    pub monod: MonodParameters,
    #[pyo3(get)]
    pub tst: TstParameters,
    #[pyo3(get)]
    pub eq: EquilibriumParameters,
    #[pyo3(get)]
    pub aux: Option<MineralParameters>,
    #[pyo3(get)]
    pub misc: MiscData,
}

impl RiverZone {
    fn mass_balance_ode_rust(&self, chms: &Array1<f64>, d: &RtForcing) -> Array1<molar_per_time> {
        let transport_rate_vec: Array1<molar_per_time> = self.transport_rate_rust(chms, d);

        match self.do_reactions {
            true => transport_rate_vec + self.reaction_rate_rust(chms, d, false),
            false => transport_rate_vec,
        }
    }

    fn reaction_rate_rust(
        &self,
        tot_moles: &Array1<molar>,
        d: &RtForcing,
        minerals_only: bool,
    ) -> Array1<molar_per_time> {
        let chms = self.moles_to_conc_rust(tot_moles, d);

        let monod_rate: Array1<molar_per_time> = self.monod.rate_rust(&chms);
        let tst_rate: Array1<molar_per_time> = self.tst.rate_rust(&chms);
        let aux_rate: Array1<f64>;
        let ssa: &Array1<f64>;
        // let aux_rate: Array1<f64> = self.aux.unwrap().factor_rust(d);

        match &self.aux {
            Some(v) => {
                ssa = &v.ssa;
                aux_rate = v.factor_rust(d);
            }
            None => {
                let msg = format!(
                    "Trying to calculate kinetic rate in zone '{}' with kinetic rates disabled",
                    self.name.clone()
                );
                panic!("{}", msg);
            }
        }

        let num_min: usize = self.monod.inhib_np.shape()[0];
        let num_spec: usize = chms.len();
        let min_start_ind: usize = num_spec - num_min;
        let min_conc: Array1<molar> = chms.slice(s![min_start_ind..]).to_owned();

        let mineral_conc_rates: Array1<molar_per_time> = 86_400.0
            * &self.misc.rate_const
            * ssa
            * &self.misc.mineral_molar_mass
            * min_conc
            * aux_rate
            * (monod_rate + tst_rate);

        // Convert the molar rate of production to the total amount per unit area by multiplying by the depth
        // Because 1 mm * m^2 = 1 L, so (moles/L/T) * (depth) = (moles/T) to get the total production rate

        if minerals_only {
            return mineral_conc_rates;
        } else {
            let mut all_species_rates: Array1<f64> =
                self.misc.mineral_stoichiometry.dot(&mineral_conc_rates);
            for i in min_start_ind..num_spec {
                all_species_rates[i] = 0.0
            }

            return all_species_rates;
        }
    }

    fn transport_rate_rust(&self, chms: &Array1<molar>, d: &RtForcing) -> Array1<molar_per_time> {
        if d.hydro_step.state.abs() < 1e-6 {
            return Array1::from_elem((chms.len(),), ZERO_CONC);
        }

        let q_in: f64 = d.hydro_step.q_internal();
        let q_out: f64 = d.hydro_step.vap_flux + d.hydro_step.lat_flux + d.hydro_step.vert_flux;
        let v_0: f64 = d.hydro_step.state;
        // let v_t: f64 = v_0 + (q_in - q_out);
        let c_in: &Array1<molar> = &d._conc_in;

        let mut transport_rate: Array1<moles_per_time> = (q_in / v_0) * (c_in - chms);

        for i in self.network.num_aqueous_species()..self.network.num_species() {
            transport_rate[i] = 0.0
        }

        transport_rate
    }

    fn get_tot_moles_rust(&self, chms: &Array1<molar>, d: &RtForcing) -> Array1<moles> {
        let mut tot_moles: Array1<moles> = Array1::zeros(chms.len());
        let num_aqueous: usize = self.network.num_aqueous_species();

        for (i, c_i) in chms.iter().enumerate() {
            if i < num_aqueous {
                tot_moles[i] = c_i * d.hydro_step.state;
            } else {
                tot_moles[i] = c_i * self.dimensions().bed_depth;
            }
        }

        tot_moles
    }

    fn moles_to_conc_rust(&self, tot_moles: &Array1<moles>, d: &RtForcing) -> Array1<moles> {
        let mut moles_arr: Array1<f64> = Array1::zeros(tot_moles.len());
        let num_aqueous: usize = self.network.num_aqueous_species();
        let water_volume = d.hydro_step.state;

        for (i, m_i) in tot_moles.iter().enumerate() {
            if i < num_aqueous {
                if water_volume < 1e-6 {
                    // If there is no water, set concentration to zero
                    moles_arr[i] = ZERO_CONC;
                } else {
                    moles_arr[i] = m_i / d.hydro_step.state;
                }
            } else {
                moles_arr[i] = m_i / self.dimensions().bed_depth;
            }
        }

        moles_arr
    }
}

#[pymethods]
impl RiverZone {
    #[new]
    #[pyo3(signature = (network, params, do_reactions = true, do_speciation = true, name = "river".to_string()))]
    pub fn new<'py>(
        py: Python<'py>,
        network: ReactionNetwork,
        params: RiverParameters,
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
        let minerals: Option<MineralParameters> = match &params.mineral_params {
            Some(v) => Some(v.clone()),
            None => None,
        };

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
        tot_moles: PyReadonlyArray1<f64>,
        d: &RtForcing,
    ) -> Bound<'py, PyArray1<f64>> {
        let chms_arr: Array1<f64> = tot_moles.to_owned_array();
        self.mass_balance_ode_rust(&chms_arr, d).to_pyarray(py)
    }

    #[pyo3(signature = (chms, d, minerals_only = false))]
    pub fn reaction_rate<'py>(
        &self,
        py: Python<'py>,
        chms: PyReadonlyArray1<f64>,
        d: &RtForcing,
        minerals_only: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match self.do_reactions {
            true => {
                let chms_arr: Array1<f64> = chms.to_owned_array();
                Ok(self
                    .reaction_rate_rust(&chms_arr, d, minerals_only)
                    .to_pyarray(py))
            }
            false => {
                let msg = format!(
                    "Trying to calculate reaction rate on zone '{}' without kinetic properties",
                    self.name.clone()
                );
                Err(PyValueError::new_err(msg))
            }
        }
    }

    pub fn transport_rate<'py>(
        &self,
        py: Python<'py>,
        chms: PyReadonlyArray1<molar>,
        d: &RtForcing,
    ) -> Bound<'py, PyArray1<molar_per_time>> {
        let chms_arr: Array1<molar> = chms.to_owned_array();
        self.transport_rate_rust(&chms_arr, d).to_pyarray(py)
    }

    pub fn get_tot_moles<'py>(
        &self,
        py: Python<'py>,
        chms: PyReadonlyArray1<f64>,
        d: RtForcing,
    ) -> Bound<'py, PyArray1<f64>> {
        self.get_tot_moles_rust(&chms.to_owned_array(), &d)
            .to_pyarray(py)
    }

    pub fn moles_to_conc<'py>(
        &self,
        py: Python<'py>,
        tot_moles: PyReadonlyArray1<f64>,
        d: RtForcing,
    ) -> Bound<'py, PyArray1<f64>> {
        self.moles_to_conc_rust(&tot_moles.to_owned_array(), &d)
            .to_pyarray(py)
    }

    #[pyo3(signature = (c_0, d, dt_days, verbose=false))]
    pub fn step<'py>(
        &self,
        py: Python<'py>,
        c_0: PyReadonlyArray1<f64>,
        d: &RtForcing,
        dt_days: f64,
        verbose: bool,
    ) -> PyResult<RtStep> {
        // Solve kinetic reactions first
        let c_0_arr: Array1<molar> = c_0.to_owned_array(); // Initial concentrations in the zone
        let c_in: &Array1<molar> = &d._conc_in;
        let tot_moles_init: Array1<moles> = self.get_tot_moles_rust(&c_0_arr, d); // Initial moles of each species at the start of the step
        let tot_mass_in: Array1<moles> = &d._conc_in * d.hydro_step.q_in; // Total mass entering the system
        let q_int: f64 = d.hydro_step.q_internal(); // Water flux entering the zone
        let q_ext: f64 = d.hydro_step.q_external(); // Water flux passing by the zone
        let tot_moles_ext: Array1<f64> = q_ext * c_in; // Total moles that just pass through the zone and do not interact with mass balance

        let set_minerals_to_zero = |conc: Array1<f64>| {
            let mut x = conc.clone();
            for i in self.network.num_aqueous_species()..self.network.num_species() {
                x[i] = 0.0
            }
            x
        };

        let residual = |conc: &Array1<molar>| {
            (&c_0_arr - conc) + dt_days * self.mass_balance_ode_rust(conc, d)
        };

        let c_after_rt: Array1<molar> = find_root_multi(&residual, c_0_arr.clone(), verbose)?;

        let c_after_eq = match self.do_speciation {
            false => c_after_rt.clone(),
            true => self.eq.solve_equilibrium_rust(&c_after_rt, verbose)?,
        };

        let tot_moles_after_eq: Array1<moles> = self.get_tot_moles_rust(&c_after_eq, d);

        let tot_moles_out_internal: Array1<moles> = set_minerals_to_zero(&c_after_eq * q_int); // Minerals are immobile

        let tot_moles_out: Array1<moles> =
            set_minerals_to_zero(&tot_moles_ext + tot_moles_out_internal);

        let total_q_out_water: f64 = q_int + q_ext;
        let frac_lat = d.hydro_step.lat_flux_ext / total_q_out_water;
        let frac_vert = d.hydro_step.vert_flux_ext / total_q_out_water;

        let lat_mass: Array1<moles>;
        let vert_mass: Array1<moles>;
        let lat_conc: Array1<molar>;
        let vert_conc: Array1<molar>;
        let (lat_mass, vert_mass, lat_conc, vert_conc) = match q_int + q_ext > 1e-6 {
            true => {
                let lat: Array1<moles> = &tot_moles_out * frac_lat;
                let vert: Array1<moles> = &tot_moles_out * frac_vert;
                let mut lc: Array1<molar> = lat.clone();
                let mut vc: Array1<molar> = vert.clone();

                let q_lat = d.hydro_step.lat_flux_ext;
                let q_vert = d.hydro_step.vert_flux_ext;

                for (i, (l_i, v_i)) in lat.iter().zip(&vert).enumerate() {
                    if i < self.network.num_aqueous_species() {
                        if q_lat.abs() <= 1e-6 {
                            lc[i] = ZERO_CONC;
                        } else {
                            lc[i] = l_i / q_lat;
                        }
                        if q_vert.abs() <= 1e-6 {
                            vc[i] = ZERO_CONC;
                        } else {
                            vc[i] = v_i / q_vert;
                        }
                    }
                }

                (lat, vert, lc, vc)
            }
            false => {
                let num_species = c_0_arr.len();
                let z: Array1<f64> = Array1::from_elem(num_species, ZERO_CONC);

                (z.clone(), z.clone(), z.clone(), z.clone())
            }
        };

        let mineral_rates: Array1<f64> = match self.do_reactions {
            true => self.reaction_rate_rust(&c_after_eq, d, true),
            false => Array1::zeros(self.network.num_minerals()),
        };

        Ok(RtStep {
            state: c_after_eq.to_pyarray(py).unbind(),
            total_moles: tot_moles_after_eq.to_pyarray(py).unbind(),
            conc_in: d._conc_in.clone().to_pyarray(py).unbind(),
            mass_in: (&d._conc_in * d.hydro_step.q_in).to_pyarray(py).unbind(),
            lat_conc: lat_conc.clone().to_pyarray(py).unbind(),
            vert_conc: vert_conc.clone().to_pyarray(py).unbind(),
            lat_mass: lat_mass.to_pyarray(py).unbind(),
            vert_mass: vert_mass.to_pyarray(py).unbind(),
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

    pub fn aux_factor<'py>(
        &self,
        py: Python<'py>,
        d: RtForcing,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match &self.aux {
            Some(v) => Ok(v.factor(py, d)),
            None => {
                let msg = format!("Tried to call `aux_factor` without kinetic parameters");
                Err(PyValueError::new_err(msg))
            }
        }
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
    pub fn dimensions(&self) -> RiverDimensions {
        self.parameters.dimensions.clone()
    }

    #[getter]
    pub fn num_species(&self) -> usize {
        self.network.num_species()
    }

    pub fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.parameters.to_array(py, self.do_reactions)
    }

    #[staticmethod]
    #[pyo3(signature=(arr, network, do_reactions, do_speciation, name="river".to_string(), natural_scales=true))]
    fn from_array<'py>(
        py: Python<'py>,
        arr: PyReadonlyArray1<f64>,
        network: &ReactionNetwork,
        do_reactions: bool,
        do_speciation: bool,
        name: String,
        natural_scales: bool,
    ) -> PyResult<Self> {
        match RiverParameters::from_array(py, arr, natural_scales) {
            Ok(v) => match Self::new(py, network.clone(), v, do_reactions, do_speciation, name) {
                Ok(new_self) => return Ok(new_self),
                Err(e) => return Err(e),
            },
            Err(e) => return Err(e),
        }
    }
}
