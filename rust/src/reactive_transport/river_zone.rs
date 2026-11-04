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
    math::find_root_multi_analytic_fused_then_lm,
    molar, molar_per_time, moles, moles_per_time,
    reactive_transport::{
        kinetic_structures::{
            EquilibriumParameters, MineralParameters, MonodParameters, RiverDimensions,
            RiverParameters, TstParameters,
        },
        reaction_network::ReactionNetwork,
        rt_zone::{kinetic_residual_and_jacobian_impl, SpeciationCache},
    },
    units::{liter, mole, mole_per_liter, AmountOfSubstance, MolarConcentration, Volume},
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
    last_x_free: SpeciationCache,
}

impl RiverZone {
    fn num_free(&self) -> usize {
        self.eq.num_free()
    }

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
        let k_a: &Array1<f64>;
        // let aux_rate: Array1<f64> = self.aux.unwrap().factor_rust(d);

        match &self.aux {
            Some(v) => {
                k_a = &v.k_a;
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

        let num_spec: usize = chms.len();
        // Minerals immediately follow the aqueous species, so the mineral block
        // starts at `num_aq` (the same block `RtZone` uses). `X-` and the
        // exchange species sit *after* the minerals and must not be treated as
        // mineral concentrations.
        let min_start_ind: usize = self.network.num_aqueous_species();

        // Mineral reaction rate: the `86400 * k_a` factor converts the
        // per-second rate constant to a per-day one (with the dimensionless aux
        // factor) — a rate-parameter conversion, kept raw f64.
        let mineral_conc_rates: Array1<molar_per_time> =
            86_400.0 * k_a * aux_rate * (monod_rate + tst_rate);

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

        // q_in/v_0 is an exchange rate (L/d ÷ L = 1/d) and this is an array-level
        // product: uom 0.38 has no volume-flow kind (compounded 1/time), so kept
        // raw f64. Result: [mol/(L·d)].
        let mut transport_rate: Array1<moles_per_time> = (q_in / v_0) * (c_in - chms);

        for i in self.network.num_aqueous_species()..self.network.num_species() {
            transport_rate[i] = 0.0
        }

        transport_rate
    }

    fn get_tot_moles_rust(&self, chms: &Array1<molar>, d: &RtForcing) -> Array1<moles> {
        let mut tot_moles: Array1<moles> = Array1::zeros(chms.len());
        let num_aqueous: usize = self.network.num_aqueous_species();

        // mol/L × L → mol: typed as MolarConcentration × Volume →
        // AmountOfSubstance (M0 spike test 5), extracted in moles; water volumes
        // are liters per the unit-area convention (1 mm · m² = 1 L).
        for (i, c_i) in chms.iter().enumerate() {
            let c = MolarConcentration::new::<mole_per_liter>(*c_i);
            if i < num_aqueous {
                let vol = Volume::new::<liter>(d.hydro_step.state);
                tot_moles[i] = (c * vol).get::<mole>();
            } else {
                let vol = Volume::new::<liter>(self.dimensions().bed_depth);
                tot_moles[i] = (c * vol).get::<mole>();
            }
        }

        tot_moles
    }

    fn moles_to_conc_rust(&self, tot_moles: &Array1<moles>, d: &RtForcing) -> Array1<moles> {
        let mut moles_arr: Array1<f64> = Array1::zeros(tot_moles.len());
        let num_aqueous: usize = self.network.num_aqueous_species();
        let water_volume = d.hydro_step.state;

        // mol ÷ L → mol/L: compounded dimension in uom 0.38 (Amount/Volume) —
        // `.value` is the base-SI value (mol/m³), so ÷1000 → mol/L (M0 spike
        // kind-mismatch note).
        for (i, m_i) in tot_moles.iter().enumerate() {
            let n = AmountOfSubstance::new::<mole>(*m_i);
            if i < num_aqueous {
                if water_volume < 1e-6 {
                    // If there is no water, set concentration to zero
                    moles_arr[i] = ZERO_CONC;
                } else {
                    let vol = Volume::new::<liter>(d.hydro_step.state);
                    moles_arr[i] = (n / vol).value / 1000.0;
                }
            } else {
                let vol = Volume::new::<liter>(self.dimensions().bed_depth);
                moles_arr[i] = (n / vol).value / 1000.0;
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
            last_x_free: SpeciationCache::default(),
        })
    }

    /// Show the zone name, the reaction / speciation flags, and the size of
    /// the reaction network this zone integrates.
    pub fn __repr__(&self) -> String {
        format!(
            "RiverZone(name='{}',do_reactions={},do_speciation={},num_species={})",
            self.name,
            if self.do_reactions { "True" } else { "False" },
            if self.do_speciation { "True" } else { "False" },
            self.network.num_species()
        )
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

        // For RiverZone the water flux driving transport is q_internal(). The
        // mineral block sits right after the aqueous species, i.e.
        // `num_aqueous_species() .. num_aqueous_species() + num_minerals`,
        // exactly as in `RtZone`. `X-` and the exchange species come after the
        // minerals, so the older `num_species - num_minerals` convention was
        // wrong whenever exchange species were present.
        let min_start = self.network.num_aqueous_species();
        let bed_depth = self.dimensions().bed_depth;
        let q_in = d.hydro_step.q_internal();
        let network = &self.network;
        let monod = &self.monod;
        let tst = &self.tst;
        let aux = self.aux.as_ref();
        let misc = &self.misc;
        let do_reactions = self.do_reactions;
        let moles_to_conc = |s: &Array1<f64>| self.moles_to_conc_rust(s, d);

        // Non-aqueous species are frozen for the step (their ODE rows are zero),
        // so solve only the aqueous block and re-use the frozen tail as the
        // mineral/exchange background for the kinetic rates.
        let num_aq = min_start;
        let num_spec = self.network.num_species();
        let bg = c_0_arr.slice(s![num_aq..]).to_owned();

        // Fused narrow residual + analytic Jacobian: rebuild the full-width
        // concentration (aqueous guess + frozen non-aqueous tail), evaluate the
        // shared full-width kernel once per Newton iteration, and keep only the
        // aqueous block of the residual / Jacobian.
        let fused = |c_aq: &Array1<molar>| {
            let mut vf: Vec<f64> = Vec::with_capacity(num_spec);
            vf.extend_from_slice(c_aq.as_slice().unwrap());
            vf.extend_from_slice(bg.as_slice().unwrap());
            let conc_full = Array1::from_vec(vf);
            let (res, jac) = kinetic_residual_and_jacobian_impl(
                network,
                monod,
                tst,
                aux,
                misc,
                bed_depth,
                q_in,
                &moles_to_conc,
                &c_0_arr,
                &conc_full,
                d,
                dt_days,
                min_start,
                do_reactions,
            );
            let res_aq = res.slice(s![0..num_aq]).to_owned();
            let jac_aq = jac.slice(s![0..num_aq, 0..num_aq]).to_owned();
            (res_aq, jac_aq)
        };

        // Newton's method with the fused (single-evaluation) residual + analytic
        // Jacobian over the aqueous block only, falling back to
        // Levenberg-Marquardt (using the fused analytic Jacobian) resumed from
        // Newton's best iterate.
        let sol_aq = find_root_multi_analytic_fused_then_lm(
            &fused,
            c_0_arr.slice(s![0..num_aq]).to_owned(),
            verbose,
        )?;

        let mut vf: Vec<f64> = Vec::with_capacity(num_spec);
        vf.extend_from_slice(sol_aq.as_slice().unwrap());
        vf.extend_from_slice(bg.as_slice().unwrap());
        let c_after_rt: Array1<molar> = Array1::from_vec(vf);
        debug_assert_eq!(c_after_rt.len(), num_spec);

        let c_after_eq = match self.do_speciation {
            false => c_after_rt.clone(),
            true => {
                if self.num_free() == 0 {
                    // No equilibrium reactions in this network (empty null
                    // space), so there is nothing to re-speciate; the kinetic
                    // concentrations are already the final answer.
                    c_after_rt.clone()
                } else {
                    let initial_x = self.last_x_free.get(self.num_free());
                    let x_free = self
                        .eq
                        .x_free_solve_rust(&c_after_rt, &initial_x, verbose)?;
                    self.last_x_free.set(x_free.clone());
                    self.eq.conc_func_rust(&x_free)
                }
            }
        };

        let mineral_rates: Array1<f64> = match self.do_reactions {
            true => self.reaction_rate_rust(&c_after_eq, d, true),
            false => Array1::zeros(self.network.num_minerals()),
        };

        Ok(RtStep {
            state: c_after_eq.to_pyarray(py).unbind(),
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
