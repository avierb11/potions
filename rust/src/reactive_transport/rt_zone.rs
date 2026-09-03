use numpy::{
    ndarray::{s, Array1, Array2},
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, ToPyArray,
};
use polars::prelude::Float64Type;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

use crate::{
    common_types::{MiscData, RtForcing, RtStep, ZERO_CONC},
    math::{find_root_multi_analytic_fused, levenberg_marquardt},
    molar, molar_per_time, moles, moles_per_time,
    reactive_transport::{
        kinetic_structures::{
            EquilibriumParameters, MineralParameters, MonodParameters, RtParameters, TstParameters,
            ZoneDimensions,
        },
        reaction_network::ReactionNetwork,
    },
};

/// Warm-start cache for the speciation solve. Stores the `x_free` solution from
/// the previous time step, which is usually an excellent initial guess for the
/// next one (concentrations move slowly between steps).
///
/// It wraps an `Arc<Mutex<_>>` so the owning zone can update the cache in place
/// during `step` without `&mut self`, while remaining `Send + Sync + Clone +
/// Debug` (a requirement of `#[pyclass]`, which a bare `RefCell` is not).
#[derive(Debug, Clone, Default)]
pub struct SpeciationCache {
    inner: std::sync::Arc<std::sync::Mutex<Option<Array1<f64>>> >,
}

impl SpeciationCache {
    /// Return the cached `x_free` with the expected length, or zeros if it is
    /// not cached / mismatched.
    pub fn get(&self, n: usize) -> Array1<f64> {
        let g = self.inner.lock().unwrap();
        match g.as_ref() {
            Some(p) if p.len() == n => p.clone(),
            _ => Array1::zeros(n),
        }
    }
    pub fn set(&self, v: Array1<f64>) {
        *self.inner.lock().unwrap() = Some(v);
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RtZone {
    #[pyo3(get)]
    pub network: ReactionNetwork,
    #[pyo3(get)]
    pub parameters: RtParameters,
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
    // Warm start for the speciation solve
    last_x_free: SpeciationCache,
}

impl RtZone {
    fn mass_balance_ode_rust(&self, chms: &Array1<f64>, d: &RtForcing) -> Array1<molar_per_time> {
        let transport_rate_vec: Array1<molar_per_time> = self.transport_rate_rust(chms, d);

        let mass_balance_vec: Array1<f64> = match self.do_reactions {
            true => transport_rate_vec + self.reaction_rate_rust(chms, d, false),
            false => transport_rate_vec,
        };

        // Get only the mobile species
        // let num_mobile = self.network.num_aqueous_species();
        // let mut mobile_mass_balance: Array1<f64> = Array1::zeros(num_mobile);

        // for (i, c_i) in mass_balance_vec.into_iter().enumerate() {
        //     mobile_mass_balance[i] = c_i;
        // }

        // mobile_mass_balance
        mass_balance_vec
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
        let num_ads: usize = self.network.num_exchange_species();
        let num_aq: usize = self.network.num_aqueous_species();
        let num_spec: usize = chms.len();
        let min_start_ind: usize = num_aq;
        let min_end_ind: usize = num_aq + num_min;
        let min_conc: Array1<molar> = chms.slice(s![min_start_ind..min_end_ind]).to_owned();

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

        // let q_internal: f64 = d.hydro_step.q_internal();
        let q_in: f64 = d.hydro_step.q_in;
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
                tot_moles[i] = c_i * self.dimensions().depth;
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
                moles_arr[i] = m_i / self.dimensions().depth;
            }
        }

        moles_arr
    }

    pub fn residual_function_rust(
        &self,
        c_0: &Array1<f64>,
        conc: &Array1<f64>,
        d: &RtForcing,
        dt_days: f64,
    ) -> Array1<f64> {
        // if c_0.len() != conc.len() {
        //     panic!("c_0 size: {}, conc size: {}", c_0.len(), conc.len());
        // }

        let res = (c_0 - conc) + dt_days * self.mass_balance_ode_rust(conc, d);

        if cfg!(debug_assertions) {
            if res.is_any_nan() {
                eprintln!("Error: nan value in residual");
            }
        }

        res
    }

    pub fn jacobian_residual_function_rust(
        &self,
        c_0: &Array1<f64>,
        conc: &Array1<f64>,
        d: &RtForcing,
        dt_days: f64,
    ) -> Array2<f64> {
        let n: usize = conc.len();
        let ode_jac: Array2<f64> = self.odes_jacobian_rust(conc, d);
        // d/dc [ c_0 - c + dt * ODE(c) ] = -I + dt * dODE/dc
        let jac_x: Array2<f64> = (-Array2::eye(n)) + (dt_days * ode_jac);
        jac_x
    }

    // The free-variable dimension of the speciation problem (used to size the
    // warm-start vector)
    fn num_free(&self) -> usize {
        self.eq.num_free()
    }

    // The analytical Jacobian of the *unscaled* mass-balance ODE,
    // d(transport + reaction)/d(conc), i.e. the derivative of
    // `mass_balance_ode_rust` with respect to its second argument.
    fn odes_jacobian_rust(&self, conc: &Array1<f64>, d: &RtForcing) -> Array2<f64> {
        let moles_to_conc = |s: &Array1<f64>| self.moles_to_conc_rust(s, d);
        residual_jacobian_impl(
            &self.network,
            &self.monod,
            &self.tst,
            self.aux.as_ref(),
            &self.misc,
            self.dimensions().depth,
            d.hydro_step.q_in,
            &moles_to_conc,
            conc,
            d,
            self.do_reactions,
        )
    }

    pub fn solve_rt_step_rust(
        &self,
        py: Python<'_>,
        c_0: &Array1<f64>,
        d: &RtForcing,
        dt_days: f64,
        verbose: bool,
    ) -> PyResult<Array1<f64>> {
        let depth = self.dimensions().depth;
        let num_aq = self.network.num_aqueous_species();
        // For RtZone the mineral concentrations sit immediately after the
        // aqueous species, and transport is driven by q_in directly.
        let min_start = num_aq;
        let q_in = d.hydro_step.q_in;
        let network = &self.network;
        let monod = &self.monod;
        let tst = &self.tst;
        let aux = self.aux.as_ref();
        let misc = &self.misc;
        let do_reactions = self.do_reactions;

        let moles_to_conc = |s: &Array1<f64>| self.moles_to_conc_rust(s, d);
        let residual = |conc: &Array1<molar>| self.residual_function_rust(c_0, conc, d, dt_days);
        // Fused residual + analytic Jacobian: the expensive Monod/TST kinetics
        // are evaluated once per Newton iteration (shared by the residual and the
        // Jacobian) rather than once by each. Falls back to Levenberg-Marquardt
        // only on non-convergence, as before.
        let fused = |conc: &Array1<molar>| {
            kinetic_residual_and_jacobian_impl(
                network,
                monod,
                tst,
                aux,
                misc,
                depth,
                q_in,
                &moles_to_conc,
                c_0,
                conc,
                d,
                dt_days,
                min_start,
                do_reactions,
            )
        };

        // Use Newton's method with the analytical Jacobian. This is much faster
        // than the finite-difference version because each iteration only needs a
        // single residual and a single Jacobian evaluation. If it fails to
        // converge, fall back to Levenberg-Marquardt.
        let res = match find_root_multi_analytic_fused(&fused, c_0.clone(), verbose) {
            Ok(v) => Ok(v),
            Err(_) => levenberg_marquardt(&residual, c_0.clone(), verbose),
        };

        res
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

    pub fn residual_function<'py>(
        &self,
        py: Python<'py>,
        c_0: PyReadonlyArray1<f64>,
        conc: PyReadonlyArray1<f64>,
        d: &RtForcing,
        dt_days: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let c_0_arr = c_0.to_owned_array();
        let conc_arr = conc.to_owned_array();
        self.residual_function_rust(&c_0_arr, &conc_arr, d, dt_days)
            .to_pyarray(py)
    }

    pub fn jacobian_residual_function<'py>(
        &self,
        py: Python<'py>,
        c_0: PyReadonlyArray1<f64>,
        conc: PyReadonlyArray1<f64>,
        d: &RtForcing,
        dt_days: f64,
    ) -> Bound<'py, PyArray2<f64>> {
        let c_0_arr = c_0.to_owned_array();
        let conc_arr = conc.to_owned_array();
        self.jacobian_residual_function_rust(&c_0_arr, &conc_arr, d, dt_days)
            .to_pyarray(py)
    }

    #[pyo3(signature = (c_0, d, dt_days, verbose=false))]
    pub fn solve_rt_step<'py>(
        &self,
        py: Python<'py>,
        c_0: PyReadonlyArray1<f64>,
        d: &RtForcing,
        dt_days: f64,
        verbose: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let c_0_arr = c_0.to_owned_array();

        let res = self.solve_rt_step_rust(py, &c_0_arr, d, dt_days, verbose);

        match res {
            Ok(x) => Ok(x.to_pyarray(py)),
            Err(e) => Err(e),
        }
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
        let num_spec: usize = self.network.num_species();
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

        // let mobile_mask: Array1<bool> = self.network.mobile_mask_rust();

        // let num_mobile = self.network.num_aqueous_species();
        // let mut c_mobile = Array1::zeros(num_mobile);
        // for i in 0..num_mobile {
        //     c_mobile[i] = c_0_arr[i];
        // }
        let c_after_rt: Array1<molar> =
            self.solve_rt_step_rust(py, &c_0_arr, d, dt_days, verbose)?;
        // let mut c_after_rt = c_0_arr.clone();
        // for (i, c_i) in c_mobile_after_rt.iter().enumerate() {
        //     c_after_rt[i] = *c_i;
        // }

        // dbg!(&c_after_rt);
        if verbose {
            eprintln!("c_after_rt={}", &c_after_rt);
        }

        let c_after_eq = match self.do_speciation {
            false => c_after_rt.clone(),
            true => {
                if self.num_free() == 0 {
                    // The network has no equilibrium reactions (empty null space,
                    // e.g. the simple carbon network), so there are no free
                    // variables to solve for: nothing to re-speciate. The kinetic
                    // concentrations are already the final answer.
                    c_after_rt.clone()
                } else {
                    // Warm-start the speciation Newton iteration with the `x_free`
                    // solution from the previous time step (concentrations move
                    // slowly, so it is an excellent initial guess).
                    let initial_x = self.last_x_free.get(self.num_free());
                    let x_free = self.eq.x_free_solve_rust(&c_after_rt, &initial_x, verbose)?;
                    self.last_x_free.set(x_free.clone());
                    self.eq.conc_func_rust(&x_free)
                }
            }
        };

        if c_after_eq.len() != num_spec {
            let msg = format!(
                "c_after_eq has the wrong shape, should have length {}, but is {}",
                num_spec, &c_after_eq
            );
            return Err(PyValueError::new_err(msg));
        }

        if verbose {
            eprintln!("c_after_eq={}", &c_after_eq);
        }

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
    pub fn dimensions(&self) -> ZoneDimensions {
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
    #[pyo3(signature=(arr, network, do_reactions, do_speciation, name, natural_scales=true))]
    fn from_array<'py>(
        py: Python<'py>,
        arr: PyReadonlyArray1<f64>,
        network: &ReactionNetwork,
        do_reactions: bool,
        do_speciation: bool,
        name: String,
        natural_scales: bool,
    ) -> PyResult<Self> {
        match RtParameters::from_array(py, arr, natural_scales) {
            Ok(v) => match Self::new(py, network.clone(), v, do_reactions, do_speciation, name) {
                Ok(new_self) => return Ok(new_self),
                Err(e) => return Err(e),
            },
            Err(e) => return Err(e),
        }
    }
}

/// Analytical Jacobian of the mass-balance ODE with respect to the concentration
/// vector, i.e. the derivative of `mass_balance_ode_rust` with respect to its
/// second argument.
///
/// # Derivation
///
/// The ODE (for the full species vector `c`, length `num_spec`) is
/// ```text
/// ODE(c) = transport(c) [ if do_reactions: + reaction(c) ]
/// ```
///
/// * **Transport** (aqueous species only, `j < num_aq`):
///   `t_j = (q_in / v0) * (c_in[j] - c[j])` so `dt_j/dc_j = -q_in / v0`
///   (zero elsewhere). This is the only contribution to the diagonal for the
///   `do_reactions == false` case.
///
/// * **Reaction** (both zone types re-normalise the state first):
///   `R(c) = S @ M(chms)` where `chms = moles_to_conc(c)` (a diagonal scaling of
///   `c`), `M` is the per-mineral kinetic rate vector and `S` is the mineral
///   stoichiometry. Its Jacobian is `S @ (dM/dchms) @ P` with
///   `P = d(moles_to_conc)/dc` a diagonal. For mineral `i`
///   `M_i = A_i * chms[m_i] * (Mn_i + T_i)` with `m_i = num_aq + i`, so
///   ```text
///   dM_i/dchms_j = A_i * ( δ_{j,m_i} (Mn_i+T_i) + chms[m_i] * d(Mn_i+T_i)/dchms_j )
///   ```
///   where the log-derivative of the Monod product is
///   `dMn_i/dchms_j = Mn_i * ( Kmonod_ij/(Kmonod_ij+chms_j)² − Kinhib_ij/(Kinhib_ij+chms_j)² )`
///   (a term present only when that entry is finite), and the TST term
///   `T_i = D_i (1 − Q_i/K_i)` with `Q_i = 10^(E_q_i)`, `D_i = 10^(E_d_i)`:
///   ```text
///   dT_i/dchms_j = D_i dep_ij / chms_j (1 − Q_i/K_i)
///                 − D_i Q_i stoich_ij / (chms_j K_i)
///   ```
///
/// Only the aqueous rows `r < num_aq` are non-empty (the mineral/exchange rows of
/// the ODE are fixed at zero, so their Jacobian rows are zero).
///
/// Compared with the previous finite-difference Jacobian, which required `n+1`
/// full ODE evaluations per Newton iteration (each re-evaluating all kinetics),
/// this touches each mineral kinetics value a constant number of times.
///
/// The expensive shared kinetics are factored out into [`reaction_terms_impl`];
/// this entry point simply assembles the Jacobian from the previously computed
/// terms so that it is bit-for-bit identical to the historical monolithic
/// implementation (it is what the finite-difference verification checks).
pub(crate) fn residual_jacobian_impl<M2C>(
    network: &ReactionNetwork,
    monod: &MonodParameters,
    tst: &TstParameters,
    aux: Option<&MineralParameters>,
    misc: &MiscData,
    depth: f64,
    q_in: f64,
    moles_to_conc: &M2C,
    conc: &Array1<f64>,
    d: &RtForcing,
    do_reactions: bool,
) -> Array2<f64>
where
    M2C: Fn(&Array1<f64>) -> Array1<f64>,
{
    let terms = reaction_terms_impl(
        network, monod, tst, aux, misc, moles_to_conc, conc, d, do_reactions,
    );
    assemble_ode_jacobian(network, &terms, misc, depth, q_in, d, do_reactions)
}

/// Shared kinetic terms for the mass-balance ODE at a single state `conc`,
/// computed once so that both the residual (the ODE itself) and the Jacobian
/// (its derivative) reuse the same Monod / TST rate values, solubility products
/// and dependence terms instead of recomputing them separately.
#[derive(Debug, Clone)]
pub(crate) struct KineticReactionTerms {
    /// Renormalised concentrations `chms = moles_to_conc(conc)`.
    pub(crate) chms: Array1<f64>,
    /// `dM/dchms`, shape (num_min × num_spec). Row `i` is the derivative of the
    /// per-mineral kinetic rate with respect to the (renormalised) concentration
    /// vector. Rows of inactive minerals are zero.
    pub(crate) dm: Array2<f64>,
    /// Per-mineral kinetic prefactor `A_i = 86400 · rc_i · ssa_i · mm_i · aux_i`.
    pub(crate) base: Array1<f64>,
    /// Per-mineral full kinetic rate `Mn_i + D_i·(1 − Q_i/K_i)`.
    pub(crate) r_full: Array1<f64>,
}

/// Compute the shared kinetic terms at state `conc`. This mirrors, factor for
/// factor, the reaction block that [`residual_jacobian_impl`] used to contain,
/// so the resulting `dm` is bit-for-bit identical to the historical one. When
/// `do_reactions` is false the terms hold the renormalised concentrations and
/// empty kinetic blocks (the Jacobian / ODE then reduce to pure transport).
pub(crate) fn reaction_terms_impl<M2C>(
    network: &ReactionNetwork,
    monod: &MonodParameters,
    tst: &TstParameters,
    aux: Option<&MineralParameters>,
    misc: &MiscData,
    moles_to_conc: &M2C,
    conc: &Array1<f64>,
    d: &RtForcing,
    do_reactions: bool,
) -> KineticReactionTerms
where
    M2C: Fn(&Array1<f64>) -> Array1<f64>,
{
    let num_aq = network.num_aqueous_species();
    let num_spec = network.num_species();
    let chms = moles_to_conc(conc);

    if !do_reactions {
        return KineticReactionTerms {
            chms,
            dm: Array2::zeros((0, num_spec)),
            base: Array1::zeros(0),
            r_full: Array1::zeros(0),
        };
    }

    // ---- Reaction contribution ----
    let num_min = monod.inhib_np().shape()[0];
    let monod_np = monod.monod_np();
    let inhib_np = monod.inhib_np();
    let stoich_np = tst.stoich_np();
    let dep_np = tst.dep_np();
    let k_min = tst.eq_const_np();
    let rate_const = &misc.rate_const;
    let molar_mass = &misc.mineral_molar_mass;

    // Kinetic values (computed once, reused for M and dM/dchms)
    let mn_vals = monod.rate_rust(&chms); // Mn_i = monod_i * inhib_i
    let (q_vals, _) = tst.calculate_solubility_product(&chms); // Q_i
    let (d_vals, _) = tst.calculate_dependence_term(&chms); // D_i

    // Which minerals may actually have a non-zero reaction term at this state.
    // This subsumes all possible NaN paths (e.g. missing aux parameters) and
    // keeps the Jacobian finite.
    let mineral_active: Vec<bool> = (0..num_min).map(|i| {
        rate_const[i].is_finite()
            && molar_mass[i].is_finite()
            && chms[num_aq + i].is_finite()
            && mn_vals[i].is_finite()
            && d_vals[i].is_finite()
            && q_vals[i].is_finite()
            && k_min[i].is_finite()
            && k_min[i].abs() > 0.0
    }).collect();

    let aux_present = aux.is_some();
    let empty_ssa = Array1::<f64>::zeros(0);
    let ssa: &Array1<f64> = match aux {
        Some(a) => &a.ssa,
        None => &empty_ssa,
    };
    let aux_factor: Array1<f64> = match aux {
        Some(a) => a.factor_rust(d),
        None => Array1::zeros(num_min),
    };

    // Per-mineral kinetic prefactor and full rate (used by the residual ODE).
    let mut base: Array1<f64> = Array1::zeros(num_min);
    let mut r_full: Array1<f64> = Array1::zeros(num_min);
    for i in 0..num_min {
        r_full[i] = mn_vals[i] + d_vals[i] * (1.0 - q_vals[i] / k_min[i]);
        if aux_present && ssa[i].is_finite() {
            base[i] = 86_400.0 * rate_const[i] * ssa[i] * molar_mass[i] * aux_factor[i];
        }
    }

    // dM/dchms : (num_min x num_spec)
    let mut dm: Array2<f64> = Array2::zeros((num_min, num_spec));
    for i in 0..num_min {
        if !mineral_active[i] || !aux_present || !ssa[i].is_finite() {
            continue; // this mineral contributes nothing to the ODE (or its aux is undefined)
        }
        let a = 86_400.0 * rate_const[i] * ssa[i] * molar_mass[i] * aux_factor[i];
        let r_full_i = mn_vals[i] + d_vals[i] * (1.0 - q_vals[i] / k_min[i]);
        let c_min = chms[num_aq + i];
        let mn_i = mn_vals[i];
        let d_i = d_vals[i];
        let q_i = q_vals[i];
        let k_i = k_min[i];
        for jidx in 0..num_spec {
            let c_j = chms[jidx];
            let (inv_cj, safe_cj) = if c_j.abs() > 0.0 {
                (1.0 / c_j, c_j)
            } else {
                (0.0, 0.0)
            };

            // dMn_i/dchms_j, where Mn_i = Π_f c/(K_f+c) · Π_h K_h/(K_h+c)
            //   d ln(Mn)/dc_j = [ K_f/(c_j(c_j+K_f)) ]_finite-monod
            //                  − [ K_h/(c_j(c_j+K_h)) ]_finite-inhib
            let mut g = 0.0;
            let mk = monod_np[(i, jidx)];
            if mk.is_finite() && safe_cj.abs() > 0.0 {
                g += mk / (safe_cj * (mk + safe_cj));
            }
            let ikv = inhib_np[(i, jidx)];
            if ikv.is_finite() && safe_cj.abs() > 0.0 {
                g -= ikv / (safe_cj * (ikv + safe_cj));
            }
            let dmn = mn_i * g;

            // dT_i/dchms_j, where T_i = D_i (1 − Q_i/K_i), Q_i=D_i·10^(Σ ν log10 c)
            //   dT = D·dep_j (1−Q/K)/c_j  − D·Q·stoich_j/(K·c_j)
            let dts = if safe_cj.abs() > 0.0 {
                d_i * dep_np[(i, jidx)] * inv_cj * (1.0 - q_i / k_i)
                    - d_i * q_i * stoich_np[(i, jidx)] * inv_cj / k_i
            } else {
                0.0
            };

            let delta = if jidx == num_aq + i { 1.0 } else { 0.0 };
            dm[(i, jidx)] = a * (delta * r_full_i + c_min * (dmn + dts));
        }
    }

    KineticReactionTerms {
        chms,
        dm,
        base,
        r_full,
    }
}

/// Assemble the analytic ODE Jacobian `d(transport + reaction)/d(conc)` from the
/// pre-computed [`KineticReactionTerms`]: the transport diagonal over the aqueous
/// species plus `S @ dm @ P`, where `P` is `d(moles_to_conc)/dc`.
fn assemble_ode_jacobian(
    network: &ReactionNetwork,
    terms: &KineticReactionTerms,
    misc: &MiscData,
    depth: f64,
    q_in: f64,
    d: &RtForcing,
    do_reactions: bool,
) -> Array2<f64> {
    let num_aq = network.num_aqueous_species();
    let num_spec = network.num_species();
    let v0 = d.hydro_step.state;
    let v0_ok = v0.abs() >= 1e-6;

    let mut j: Array2<f64> = Array2::zeros((num_spec, num_spec));

    // ---- Transport contribution (diagonal over aqueous species) ----
    if v0_ok {
        let f = q_in / v0;
        for r in 0..num_aq {
            j[(r, r)] = -f;
        }
    }

    if !do_reactions {
        return j;
    }

    let num_min = terms.dm.shape()[0];

    // Diagonal P = d(moles_to_conc)/dc. Note: in the true `moles_to_conc_rust` a
    // very small water volume makes the aqueous entries a constant (zero
    // derivative). We keep the smooth 1/v0 here; the difference is bounded by the
    // reaction term and is negligible compared with the finite-difference noise.
    let mut p: Array1<f64> = Array1::zeros(num_spec);
    for i in 0..num_spec {
        p[i] = if i < num_aq {
            if v0_ok {
                1.0 / v0
            } else {
                0.0
            }
        } else {
            if depth.abs() > 0.0 {
                1.0 / depth
            } else {
                0.0
            }
        };
    }

    // J += S @ dM @ P  (only aqueous rows r < num_aq); zero out r >= num_aq.
    let s = &misc.mineral_stoichiometry; // (num_spec x num_min)
    for r in 0..num_aq {
        for jidx in 0..num_spec {
            let pj = p[jidx];
            if pj == 0.0 {
                continue;
            }
            let mut acc = 0.0;
            for m in 0..num_min {
                acc += s[(r, m)] * terms.dm[(m, jidx)];
            }
            j[(r, jidx)] += acc * pj;
        }
    }

    j
}

/// Evaluate the mass-balance ODE (transport + reaction) at state `conc` from the
/// pre-computed [`KineticReactionTerms`]. `min_start` is the index of the first
/// mineral concentration in the (renormalised) state; it is `num_aqueous` for
/// `RtZone` and `num_species - num_minerals` for `RiverZone` (the two are equal
/// for networks without exchange species). `q_in` is the water flux that drives
/// transport into the zone (per-zone; `d.hydro_step.q_in` for `RtZone`,
/// `q_internal()` for `RiverZone`).
fn kinetic_ode_from_terms(
    network: &ReactionNetwork,
    terms: &KineticReactionTerms,
    misc: &MiscData,
    d: &RtForcing,
    conc: &Array1<f64>,
    q_in: f64,
    min_start: usize,
    do_reactions: bool,
) -> Array1<f64> {
    let num_aq = network.num_aqueous_species();
    let num_spec = network.num_species();
    let v0 = d.hydro_step.state;

    let mut ode: Array1<f64> = Array1::zeros(num_spec);

    // ---- Transport (aqueous species only) ----
    if v0.abs() >= 1e-6 {
        let f = q_in / v0;
        let c_in: &Array1<molar> = &d._conc_in;
        for r in 0..num_aq {
            ode[r] = f * (c_in[r] - conc[r]);
        }
    }

    // ---- Reaction (S @ M(chms), mineral/exchange rows fixed at zero) ----
    if do_reactions {
        let num_min = terms.base.shape()[0];
        let mut mcr: Array1<f64> = Array1::zeros(num_min);
        for i in 0..num_min {
            mcr[i] = terms.base[i] * terms.chms[min_start + i] * terms.r_full[i];
        }
        let s = &misc.mineral_stoichiometry; // (num_spec x num_min)
        let mut react = s.dot(&mcr);
        for i in min_start..num_spec {
            react[i] = 0.0;
        }
        for r in 0..num_spec {
            ode[r] += react[r];
        }
    }

    ode
}

/// Fused evaluation: computes the residual
/// `r(c) = c_0 - c + dt · ODE(c)` and its analytic Jacobian
/// `J(c) = -I + dt · dODE/dc` in a single pass, sharing the kinetic terms.
///
/// This is what the Newton driver consumes: one invocation yields both, so each
/// iteration performs a single Monod/TST rate evaluation instead of the separate
/// ones the residual and Jacobian used to each trigger.
pub(crate) fn kinetic_residual_and_jacobian_impl<M2C>(
    network: &ReactionNetwork,
    monod: &MonodParameters,
    tst: &TstParameters,
    aux: Option<&MineralParameters>,
    misc: &MiscData,
    depth: f64,
    q_in: f64,
    moles_to_conc: &M2C,
    c_0: &Array1<f64>,
    conc: &Array1<f64>,
    d: &RtForcing,
    dt_days: f64,
    min_start: usize,
    do_reactions: bool,
) -> (Array1<f64>, Array2<f64>)
where
    M2C: Fn(&Array1<f64>) -> Array1<f64>,
{
    let terms = reaction_terms_impl(
        network, monod, tst, aux, misc, moles_to_conc, conc, d, do_reactions,
    );
    let num_spec = network.num_species();
    let j_ode = assemble_ode_jacobian(network, &terms, misc, depth, q_in, d, do_reactions);
    let jac_x: Array2<f64> = (-Array2::eye(num_spec)) + (dt_days * j_ode);
    let ode = kinetic_ode_from_terms(network, &terms, misc, d, conc, q_in, min_start, do_reactions);
    let res = (c_0 - conc) + (dt_days * ode);
    (res, jac_x)
}
