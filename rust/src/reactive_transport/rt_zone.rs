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
    common_types::{MiscData, RtForcing, RtStep, ZERO_CONC},
    math::{
        find_root_multi_analytic_fused_then_lm, invalid_state_error, max_abs, tag_subsystem,
        OptimizerState,
    },
    molar, molar_per_time, moles, moles_per_time,
    reactive_transport::{
        kinetic_structures::{
            smoothed_rel_deriv, EquilibriumParameters, MineralParameters, MonodParameters,
            RtParameters, TstParameters, ZoneDimensions,
        },
        reaction_network::ReactionNetwork,
    },
    units::{liter, mole, mole_per_liter, AmountOfSubstance, MolarConcentration, Volume},
};

/// Absolute tolerance (mol/m^3) by which an accepted kinetic state may dip
/// below zero in any *aqueous* entry before it is rejected as physically
/// invalid.
///
/// A converged kinetic solve routinely leaves trace species (notably CO3--)
/// slightly negative: the step residual is insensitive to the sign of a
/// near-zero concentration (the rate-law log terms clamp), and the
/// speciation re-solve repairs the excursion. Across the East River corpora
/// the observed excursions reach at most ~2e-7 mol/m^3 (against a CO3--
/// equilibrium of ~2.5e-11), while genuinely non-physical "monster" states
/// (e.g. DOC = -1.198) exceed this tolerance by more than five orders of
/// magnitude and remain fatal.
const KINETIC_AQ_TOLERANCE: f64 = 2.0e-6;

/// Warm-start cache for the speciation solve. Stores the `x_free` solution from
/// the previous time step, which is usually an excellent initial guess for the
/// next one (concentrations move slowly between steps).
///
/// It wraps an `Arc<Mutex<_>>` so the owning zone can update the cache in place
/// during `step` without `&mut self`, while remaining `Send + Sync + Clone +
/// Debug` (a requirement of `#[pyclass]`, which a bare `RefCell` is not).
#[derive(Debug, Clone, Default)]
pub struct SpeciationCache {
    inner: std::sync::Arc<std::sync::Mutex<Option<Array1<f64>>>>,
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

/// Plain (Python-free) output of one `RtZone` step. Kept separate from
/// [`crate::common_types::RtStep`] so that the whole multi-zone / multi-step
/// driver can run without ever touching the GIL or allocating per-element
/// NumPy arrays: `state` and `mineral_rates` become the returned [`RtStep`],
/// while `lat_mass` / `vert_mass` are accumulated into the driver's strided
/// coupling buffers for the downstream zones of the same step.
#[derive(Debug, Clone, Default)]
pub struct StepResultAll {
    pub state: Array1<f64>,
    pub lat_mass: Array1<f64>,
    pub vert_mass: Array1<f64>,
    pub mineral_rates: Array1<f64>,
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

        let num_aq: usize = self.network.num_aqueous_species();
        let num_spec: usize = chms.len();
        let min_start_ind: usize = num_aq;

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

        // let q_internal: f64 = d.hydro_step.q_internal();
        let q_in: f64 = d.hydro_step.q_in;
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

        let max_water = self.dimensions().max_water;

        // mol/L × L → mol: typed as MolarConcentration × Volume →
        // AmountOfSubstance (M0 spike test 5), extracted in moles; water volumes
        // are liters per the unit-area convention (1 mm · m² = 1 L).
        for (i, c_i) in chms.iter().enumerate() {
            let c = MolarConcentration::new::<mole_per_liter>(*c_i);
            if i < num_aqueous {
                let vol = Volume::new::<liter>(d.hydro_step.state);
                tot_moles[i] = (c * vol).get::<mole>();
            } else {
                let vol = Volume::new::<liter>(max_water);
                tot_moles[i] = (c * vol).get::<mole>();
            }
        }

        tot_moles
    }

    fn moles_to_conc_rust(&self, tot_moles: &Array1<moles>, d: &RtForcing) -> Array1<moles> {
        let mut moles_arr: Array1<f64> = Array1::zeros(tot_moles.len());
        let num_aqueous: usize = self.network.num_aqueous_species();
        let water_volume = d.hydro_step.state;

        let max_water = self.dimensions().max_water;

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
                let vol = Volume::new::<liter>(max_water);
                moles_arr[i] = (n / vol).value / 1000.0;
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
            self.dimensions().max_water,
            d.hydro_step.q_in,
            &moles_to_conc,
            conc,
            d,
            self.do_reactions,
        )
    }

    pub fn solve_rt_step_rust(
        &self,
        c_0: &Array1<f64>,
        d: &RtForcing,
        dt_days: f64,
        verbose: bool,
    ) -> PyResult<Array1<f64>> {
        let max_water = self.dimensions().max_water;
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

        // Non-aqueous species are frozen for the step: their ODE rows are zero
        // (transport only moves the aqueous block and reaction zeroes the
        // mineral/exchange rows), so the full-width Newton already holds them
        // constant at `c_0`. We solve only the aqueous block here and re-use the
        // frozen tail as the mineral/exchange background for the kinetic rates.
        let num_spec = self.network.num_species();
        let bg = c_0.slice(s![num_aq..]).to_owned();
        let c_0_full = c_0.clone();

        // Fused narrow residual + analytic Jacobian: rebuild the full-width
        // concentration (aqueous guess + frozen non-aqueous tail), evaluate the
        // shared full-width kernel once per Newton iteration, and keep only the
        // aqueous block of the residual / Jacobian. Falls back to
        // Levenberg-Marquardt only on non-convergence, as before.
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
                max_water,
                q_in,
                &moles_to_conc,
                &c_0_full,
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

        // Use Newton's method with the analytical Jacobian over the aqueous
        // block only. This is much faster than the finite-difference version
        // (each iteration needs a single fused residual + Jacobian) and smaller
        // than the full-width system since the non-aqueous rows are identically
        // zero. If it fails to converge, fall back to Levenberg-Marquardt
        // (itself using the fused analytic Jacobian) resumed from Newton's best
        // iterate.
        let sol_aq = match find_root_multi_analytic_fused_then_lm(
            &fused,
            c_0_full.slice(s![0..num_aq]).to_owned(),
            verbose,
        ) {
            Ok(x) => x,
            // The shared Newton / LM chain does not know which subsystem it
            // is serving (its message carries only the hard-coded
            // "reactive-transport" label). Tag the error at this boundary so
            // the Python driver can route the fsolve backup to the kinetic
            // sub-problem via `err.subsystem`.
            Err(e) => return Err(tag_subsystem(e, "kinetic")),
        };

        let mut vf: Vec<f64> = Vec::with_capacity(num_spec);
        vf.extend_from_slice(sol_aq.as_slice().unwrap());
        vf.extend_from_slice(bg.as_slice().unwrap());
        let c_after = Array1::from_vec(vf);
        debug_assert_eq!(c_after.len(), num_spec);
        Ok(c_after)
    }

    /// Pure-Rust per-zone step returning all output fields as plain
    /// `ndarray` vectors (no Python objects). This is the fast path the
    /// `run_zone_steps` driver consumes; the public `step` pymethod simply
    /// wraps this result into a `RtStep` for backwards compatibility.
    ///
    /// The Newton / LM kinetic solve runs here; all post-solve bookkeeping
    /// happens in [`Self::finish_step_internal`].
    ///
    /// `x_free_override` is `None` on the normal path. When the speciation
    /// re-equilibration has failed and the Python-side `scipy.optimize.fsolve`
    /// backup has converged its own free-variable vector, `step` passes it
    /// here and the speciation branch adopts it directly (gated on the same
    /// physical-validity check as the internal solve).
    pub fn step_internal(
        &self,
        c_0: &Array1<f64>,
        d: &RtForcing,
        dt_days: f64,
        verbose: bool,
        x_free_override: Option<&Array1<f64>>,
    ) -> PyResult<StepResultAll> {
        let c_0_arr: Array1<molar> = c_0.clone(); // Initial concentrations in the zone

        // let mobile_mask: Array1<bool> = self.network.mobile_mask_rust();

        // let num_mobile = self.network.num_aqueous_species();
        // let mut c_mobile = Array1::zeros(num_mobile);
        // for i in 0..num_mobile {
        //     c_mobile[i] = c_0_arr[i];
        // }
        let c_after_rt: Array1<molar> = self.solve_rt_step_rust(&c_0_arr, d, dt_days, verbose)?;
        // let mut c_after_rt = c_0_arr.clone();
        // for (i, c_i) in c_mobile_after_rt.iter().enumerate() {
        //     c_after_rt[i] = *c_i;
        // }

        self.finish_step_internal(&c_0_arr, &c_after_rt, d, dt_days, verbose, x_free_override)
    }

    /// Finalize a zone step from an already-solved post-reaction state.
    ///
    /// The post-solve half of [`Self::step_internal`]: speciation
    /// re-equilibration, then the mass / flux bookkeeping that produces
    /// [`StepResultAll`]. `step_internal` feeds it the state from the
    /// internal Newton / LM kinetic solve, while the `step_with_solution`
    /// pymethod feeds it a state solved elsewhere (the
    /// `scipy.optimize.fsolve` backup in
    /// `ReactiveTransportModel.step_rt_model`), so every finalized step
    /// passes through exactly this one post-solve path.
    ///
    /// `dt_days` is not consumed by the bookkeeping (it belongs to the
    /// kinetic solve) but is kept for signature symmetry with
    /// [`Self::step_internal`].
    ///
    /// `x_free_override` is `None` on the normal path and on
    /// `step_with_solution`. When the speciation re-equilibration itself
    /// failed and the Python-side fsolve backup converged its own
    /// free-variable vector, `step` passes it here: the speciation branch
    /// then adopts the vector directly (gated on the same physical-validity
    /// check as the internal solve) instead of re-solving.
    fn finish_step_internal(
        &self,
        c_0: &Array1<molar>,
        c_after_rt: &Array1<molar>,
        d: &RtForcing,
        dt_days: f64,
        verbose: bool,
        x_free_override: Option<&Array1<f64>>,
    ) -> PyResult<StepResultAll> {
        // Physical-validity gate on the accepted post-reaction state: every
        // *aqueous* entry must be finite and at least -KINETIC_AQ_TOLERANCE.
        // The kinetic solve minimizes the step residual, which is insensitive
        // to the sign of a near-zero concentration (the log terms clamp), so
        // a stalled or non-unique solve can "converge" onto a non-physical
        // branch (a largely negative or zeroed aqueous species, huge exchange
        // values) while still meeting the residual bar. Without this gate such
        // a state is accepted here and only rejected one step later by the
        // speciation solve, with the failure mis-attributed to that solve;
        // catching it here fails the run at the step that *created* the
        // invalid state, with correct subsystem attribution.
        //
        // The tolerance (rather than strict positivity) is essential:
        // healthy converged solves routinely leave trace species slightly
        // negative (see KINETIC_AQ_TOLERANCE), and rejecting those turned
        // every affected row into a failure.
        //
        // The non-aqueous tail (minerals, exchange) is deliberately not
        // gated: those entries are legitimately zero in many steps. The
        // final speciation state is still screened by B1a after
        // re-equilibration, so no previously rejected state passes through.
        let num_aq: usize = self.network.num_aqueous_species();
        if !c_after_rt
            .slice(s![0..num_aq])
            .iter()
            .all(|c| c.is_finite() && *c >= -KINETIC_AQ_TOLERANCE)
        {
            let last_f_x: Array1<f64> = self.residual_function_rust(c_0, c_after_rt, d, dt_days);
            let jacobian: Array2<f64> =
                self.jacobian_residual_function_rust(c_0, c_after_rt, d, dt_days);
            let error: f64 = max_abs(&last_f_x);
            let state = OptimizerState {
                iteration: 0,
                final_x: c_after_rt.clone(),
                last_f_x,
                initial_x: c_0.clone(),
                jacobian,
                error,
                errors: Vec::new(),
                xs: Vec::new(),
                fxs: Vec::new(),
                jacobians: Vec::new(),
                lambdas: Vec::new(),
                best_x: c_after_rt.clone(),
                best_err: error,
            };
            return Err(invalid_state_error(
                &state,
                "kinetic",
                "accepted kinetic state is physically invalid (non-finite or unreasonably negative aqueous concentration)",
            ));
        }

        let num_spec: usize = self.network.num_species();

        // dbg!(&c_after_rt);
        if verbose {
            eprintln!("c_after_rt={}", c_after_rt);
        }

        let c_after_eq = match self.do_speciation {
            false => (*c_after_rt).clone(),
            true => {
                if self.num_free() == 0 {
                    // The network has no equilibrium reactions (empty null space,
                    // e.g. the simple carbon network), so there are no free
                    // variables to solve for: nothing to re-speciate. The kinetic
                    // concentrations are already the final answer.
                    (*c_after_rt).clone()
                } else if let Some(x_override) = x_free_override {
                    // The internal speciation solve failed and the Python-side
                    // fsolve backup converged its own free-variable vector:
                    // adopt it directly. Gate it on the same physical-validity
                    // check the internal solve applies, so a non-finite /
                    // non-positive "solution" is rejected with a typed
                    // `OptimizationError` (tagged "speciation") instead of
                    // propagating into the zone state.
                    let concs: Array1<f64> = self.eq.conc_func_rust(x_override);
                    if !concs.iter().all(|c| c.is_finite() && *c > 0.0) {
                        let f_x: Array1<f64> = self
                            .eq
                            .residual_rust(x_override, &self.eq.totals_rust(c_after_rt));
                        let err: f64 = max_abs(&f_x);
                        let state = OptimizerState {
                            iteration: 0,
                            final_x: x_override.clone(),
                            last_f_x: f_x.clone(),
                            initial_x: x_override.clone(),
                            jacobian: self.eq.jacobian_residual_rust(x_override),
                            error: err,
                            errors: vec![err],
                            xs: vec![x_override.clone()],
                            fxs: vec![f_x.clone()],
                            jacobians: Vec::new(),
                            lambdas: Vec::new(),
                            best_x: x_override.clone(),
                            best_err: err,
                        };
                        return Err(invalid_state_error(
                            &state,
                            "speciation",
                            "supplied free-variable vector is physically invalid (non-finite or non-positive concentration)",
                        ));
                    }
                    // The gate proved the state is physically valid, so it is a
                    // safe warm start for the next step's speciation solve
                    // (equivalent to the conditional cache write in the normal
                    // branch below, which caches after the same gate).
                    self.last_x_free.set(x_override.clone());
                    concs
                } else {
                    // Warm-start the speciation Newton iteration with the `x_free`
                    // solution from the previous time step (concentrations move
                    // slowly, so it is an excellent initial guess).
                    let initial_x = self.last_x_free.get(self.num_free());
                    let x_free = self.eq.x_free_solve_rust(c_after_rt, &initial_x, verbose)?;
                    let c_after_eq = self.eq.conc_func_rust(&x_free);
                    // Cache the warm start only for a physically valid state
                    // (all concentrations finite and positive): `last_x_free`
                    // is the initial guess for the *next* step's speciation
                    // solve, so a poisoned guess must never be stored, even
                    // though `x_free_solve_rust` already rejects invalid
                    // plateaus.
                    if c_after_eq.iter().all(|c| c.is_finite() && *c > 0.0) {
                        self.last_x_free.set(x_free);
                    }
                    c_after_eq
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

        let (lat_mass, vert_mass) = self.outgoing_mass_rust(&c_after_eq, d);

        let mineral_rates: Array1<f64> = match self.do_reactions {
            true => self.reaction_rate_rust(&c_after_eq, d, true),
            false => Array1::zeros(self.network.num_minerals()),
        };

        Ok(StepResultAll {
            state: c_after_eq,
            lat_mass,
            vert_mass,
            mineral_rates,
        })
    }

    /// Compute the lateral / vertical outgoing mass of a zone's post-reaction
    /// (post-speciation) state: the moles of each species the zone exports,
    /// split between the two outgoing water routes.
    ///
    /// This is the pure bookkeeping the Gauss-Seidel coupling consumes:
    /// downstream zones of the same step derive their incoming concentration
    /// from these masses. The arithmetic is exactly the one a normal step
    /// used to inline before returning the masses as part of its result:
    ///
    /// ```text
    /// tot_moles_out = min_zero(c_after_eq * q_internal + q_external * c_in)
    /// lat_mass      = tot_moles_out * lat_flux_ext / (q_internal + q_external)
    /// vert_mass     = tot_moles_out * vert_flux_ext / (q_internal + q_external)
    /// ```
    ///
    /// with `min_zero` zeroing the non-aqueous (mineral / exchange) entries,
    /// since only the aqueous species are mobile. When the zone has no
    /// outgoing water (`q_internal + q_external <= 1e-6`), both masses are
    /// the `ZERO_CONC` sentinel instead.
    fn outgoing_mass_rust(
        &self,
        c_after_eq: &Array1<molar>,
        d: &RtForcing,
    ) -> (Array1<moles>, Array1<moles>) {
        let c_in: &Array1<molar> = &d._conc_in;
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

        let tot_moles_out_internal: Array1<moles> = set_minerals_to_zero(c_after_eq * q_int); // Minerals are immobile

        let tot_moles_out: Array1<moles> =
            set_minerals_to_zero(&tot_moles_ext + tot_moles_out_internal);

        let total_q_out_water: f64 = q_int + q_ext;
        let frac_lat = d.hydro_step.lat_flux_ext / total_q_out_water;
        let frac_vert = d.hydro_step.vert_flux_ext / total_q_out_water;

        match q_int + q_ext > 1e-6 {
            true => {
                let lat: Array1<moles> = &tot_moles_out * frac_lat;
                let vert: Array1<moles> = &tot_moles_out * frac_vert;
                (lat, vert)
            }
            false => {
                let z: Array1<f64> = Array1::from_elem(c_after_eq.len(), ZERO_CONC);
                (z.clone(), z)
            }
        }
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

    /// Show the zone name, the reaction / speciation flags, and the size of
    /// the reaction network this zone integrates.
    pub fn __repr__(&self) -> String {
        format!(
            "RtZone(name='{}',do_reactions={},do_speciation={},num_species={})",
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

        let res = self.solve_rt_step_rust(&c_0_arr, d, dt_days, verbose);

        match res {
            Ok(x) => Ok(x.to_pyarray(py)),
            Err(e) => Err(e),
        }
    }

    #[pyo3(signature = (c_0, d, dt_days, verbose=false, x_free_override=None))]
    pub fn step<'py>(
        &self,
        py: Python<'py>,
        c_0: PyReadonlyArray1<f64>,
        d: &RtForcing,
        dt_days: f64,
        verbose: bool,
        x_free_override: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<RtStep> {
        let c_0_arr: Array1<f64> = c_0.to_owned_array();
        let x_override: Option<Array1<f64>> = x_free_override.map(|a| a.to_owned_array());
        let r = self.step_internal(&c_0_arr, d, dt_days, verbose, x_override.as_ref())?;
        Ok(make_rt_step(py, &r))
    }

    /// Finalize a zone step from an externally computed post-reaction state.
    ///
    /// Runs the exact same post-solve path as [`Self::step`] (speciation
    /// re-equilibration, mass / flux bookkeeping, outgoing lateral / vertical
    /// mass) but skips the internal Newton / LM kinetic solve: the caller
    /// supplies `c_after_rt`, the full post-reaction concentration vector it
    /// has already solved. `ReactiveTransportModel.step_rt_model` uses this
    /// to rescue a step whose Newton / LM chain stalled, by re-solving the
    /// aqueous block with `scipy.optimize.fsolve`.
    ///
    /// `c_0` must be the initial state the failed solve started from, and
    /// `d` the same forcing (in particular the incoming concentration the
    /// driver computed), so the bookkeeping is identical to a normal step.
    #[pyo3(signature = (c_0, c_after_rt, d, dt_days, verbose=false))]
    pub fn step_with_solution<'py>(
        &self,
        py: Python<'py>,
        c_0: PyReadonlyArray1<f64>,
        c_after_rt: PyReadonlyArray1<f64>,
        d: &RtForcing,
        dt_days: f64,
        verbose: bool,
    ) -> PyResult<RtStep> {
        let c_0_arr: Array1<f64> = c_0.to_owned_array();
        let c_after_rt_arr: Array1<f64> = c_after_rt.to_owned_array();
        let r = self.finish_step_internal(&c_0_arr, &c_after_rt_arr, d, dt_days, verbose, None)?;
        Ok(make_rt_step(py, &r))
    }

    /// Compute this zone's lateral / vertical outgoing mass for a
    /// post-reaction state, exactly as the internal Gauss-Seidel coupling
    /// does (see [`Self::outgoing_mass_rust`]).
    ///
    /// Exposed to Python because the `RtStep` a rescue finalizes no longer
    /// carries the outgoing masses, and `run_zone_steps` only reports them
    /// for the zones it solved before a failure: the Python driver calls
    /// this for a rescued zone to re-seed its coupling buffers so a resumed
    /// sweep matches an uninterrupted one.
    ///
    /// `c_after_eq` must be the full post-speciation state (as in an
    /// `RtStep.state`), and `d` the same forcing the step ran with.
    #[pyo3(signature = (c_after_eq, d))]
    pub fn outgoing_mass<'py>(
        &self,
        py: Python<'py>,
        c_after_eq: PyReadonlyArray1<f64>,
        d: &RtForcing,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let (lat, vert) = self.outgoing_mass_rust(&c_after_eq.to_owned_array(), d);
        (lat.to_pyarray(py), vert.to_pyarray(py))
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
///   where the log-derivative of the Monod product
///   `Mn_i = Π_f c/(K_f+c) · Π_h K_h/(K_h+c)` is
///   `dMn_i/dchms_j = Mn_i * ( Kmonod_ij/(chms_j·(Kmonod_ij+chms_j)) − 1/(Kinhib_ij+chms_j) )`
///   (a term present only when that entry is finite), and the TST term
///   `T_i = D_i (1 − Q_i/K_i)` with `Q_i = 10^(E_q_i)`, `D_i = 10^(E_d_i)`:
///   Q and D are power products (10^(Σ ν·log10 c) = Π c^ν exactly), so their
///   derivatives are V·ν·s'/s — no ln(10), which cancels — where s'/s is the
///   relative derivative of the smoothed concentration `s`
///   ([`kinetic_structures::smoothed_conc`]):
///   ```text
///   dT_i/dchms_j = D_i dep_ij (1 − Q_i/K_i) · s'(chms_j)/s(chms_j)
///                 − D_i Q_i stoich_ij · s'(chms_j)/s(chms_j) / K_i
///   ```
///   with `s'/s = sigmoid(chms_j/δ)/s(chms_j)`: bit-identical to the
///   historical `1/chms_j` for every chms_j ≥ 1e-25, and smoothly decaying
///   through zero on the non-physical side of the zero plane (the historical
///   form jumped 0 → 1/chms_j there, a kink that stalled Newton / LM).
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
    min_moles_volume: f64,
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
        network,
        monod,
        tst,
        aux,
        misc,
        moles_to_conc,
        conc,
        d,
        do_reactions,
    );
    assemble_ode_jacobian(
        network,
        &terms,
        misc,
        min_moles_volume,
        q_in,
        d,
        do_reactions,
    )
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
    /// Per-mineral kinetic prefactor `A_i = 86400 · rc_i · k_a_i · mm_i · aux_i`.
    pub(crate) base: Array1<f64>,
    /// Per-mineral full kinetic rate `Mn_i + D_i·(1 − Q_i/K_i)`.
    pub(crate) r_full: Array1<f64>,
}

/// Compute the shared kinetic terms at state `conc`. This mirrors, factor for
/// factor, the reaction block that [`residual_jacobian_impl`] used to contain.
/// For every physical state (concentrations ≥ 1e-25 mol/L) all values and
/// derivatives here are bit-for-bit identical to the historical ones:
/// [`kinetic_structures::smoothed_conc`] and its relative derivative coincide
/// exactly with the historical hard clamp and `1/c` there. The change is
/// confined to the non-physical side of the zero-concentration plane, where
/// the TST value and derivative now vary smoothly instead of kinked (the
/// historical hard clamp kinked dQ/dc from 0 to a finite value at every
/// zero-concentration plane — the stall that fix 3 removes). When
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

    // Kinetic values (computed once, reused for M and dM/dchms)
    let mn_vals = monod.rate_rust(&chms); // Mn_i = monod_i * inhib_i
    let (q_vals, _) = tst.calculate_solubility_product(&chms); // Q_i
    let (d_vals, _) = tst.calculate_dependence_term(&chms); // D_i

    // Which minerals may actually have a non-zero reaction term at this state.
    // This subsumes all possible NaN paths (e.g. missing aux parameters) and
    // keeps the Jacobian finite.
    let mineral_active: Vec<bool> = (0..num_min)
        .map(|i| {
            chms[num_aq + i].is_finite()
                && mn_vals[i].is_finite()
                && d_vals[i].is_finite()
                && q_vals[i].is_finite()
                && k_min[i].is_finite()
                && k_min[i].abs() > 0.0
        })
        .collect();

    let aux_present = aux.is_some();
    let empty_k_a = Array1::<f64>::zeros(0);
    let k_a: &Array1<f64> = match aux {
        Some(a) => &a.k_a,
        None => &empty_k_a,
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
        if aux_present && k_a[i].is_finite() {
            base[i] = 86_400.0 * k_a[i] * aux_factor[i];
        }
    }

    // dM/dchms : (num_min x num_spec). The relative derivative s'/s of the
    // smoothed concentration for every species is computed once here and
    // reused by the TST derivative below (see `smoothed_rel_deriv`).
    let rel_dsc: Array1<f64> = chms.map(|x| smoothed_rel_deriv(*x));
    let mut dm: Array2<f64> = Array2::zeros((num_min, num_spec));
    for i in 0..num_min {
        if !mineral_active[i] || !aux_present || !k_a[i].is_finite() {
            continue; // this mineral contributes nothing to the ODE (or its aux is undefined)
        }
        let a = 86_400.0 * k_a[i] * aux_factor[i];
        let r_full_i = r_full[i];
        let mn_i = mn_vals[i];
        let d_i = d_vals[i];
        let q_i = q_vals[i];
        let k_i = k_min[i];
        for jidx in 0..num_spec {
            let c_j = chms[jidx];
            let safe_cj: f64 = if c_j.abs() > 0.0 { c_j } else { 0.0 };

            // dMn_i/dchms_j, where Mn_i = Π_f c/(K_f+c) · Π_h K_h/(K_h+c)
            //   d ln(Mn)/dc_j = [ K_f/(c_j(c_j+K_f)) ]_finite-monod
            //                  − [ 1/(K_h+c_j) ]_finite-inhib
            // The Monod term keeps its c_j ≠ 0 guard (its derivative has a
            // true pole at c_j = 0); the inhibitor 1/(K_h+c_j) has no pole at
            // c_j = 0, so only its finiteness is guarded.
            let mut g = 0.0;
            let mk = monod_np[(i, jidx)];
            if mk.is_finite() && safe_cj.abs() > 0.0 {
                g += mk / (safe_cj * (mk + safe_cj));
            }
            let ikv = inhib_np[(i, jidx)];
            if ikv.is_finite() {
                g -= 1.0 / (ikv + c_j);
            }
            let dmn = mn_i * g;

            // dT_i/dchms_j, where T_i = D_i (1 − Q_i/K_i) and
            // Q_i = D_i·Π c^ν (a power product: 10^(Σ ν log10 c) = Π c^ν
            // exactly). Its derivative is V·ν·s'/s — no ln(10), which
            // cancels — with s'/s = `smoothed_rel_deriv`. For every physical
            // state (c_j ≥ 1e-25) this is bit-identical to the old 1/c_j
            // form; on the non-physical side it decays smoothly through the
            // zero plane instead of jumping 0 → 1/c_j, which removes the
            // kink that stalled Newton / LM at zero-concentration planes.
            let rel_j: f64 = rel_dsc[jidx];
            let dts = d_i * dep_np[(i, jidx)] * rel_j * (1.0 - q_i / k_i)
                - d_i * q_i * stoich_np[(i, jidx)] * rel_j / k_i;

            dm[(i, jidx)] = a * (dmn + dts);
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
    min_moles_volume: f64,
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
        // Exchange rate q_in/v0 (L/d ÷ L = 1/d): compounded dimension in uom 0.38
        // (no volume-flow kind), kept raw f64.
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
            if min_moles_volume.abs() > 0.0 {
                1.0 / min_moles_volume
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
/// mineral concentration in the (renormalised) state; it is
/// `num_aqueous_species()` for both `RtZone` and `RiverZone` (minerals directly
/// follow the aqueous species, with `X-` and the exchange species after them).
/// `q_in` is the water flux that drives transport into the zone (per-zone;
/// `d.hydro_step.q_in` for `RtZone`, `q_internal()` for `RiverZone`).
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
        // Exchange rate q_in/v0 (L/d ÷ L = 1/d): compounded dimension in uom 0.38
        // (no volume-flow kind), kept raw f64.
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
            mcr[i] = terms.base[i] * terms.r_full[i];
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
    min_moles_volume: f64,
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
        network,
        monod,
        tst,
        aux,
        misc,
        moles_to_conc,
        conc,
        d,
        do_reactions,
    );
    let num_spec = network.num_species();
    let j_ode = assemble_ode_jacobian(
        network,
        &terms,
        misc,
        min_moles_volume,
        q_in,
        d,
        do_reactions,
    );
    let jac_x: Array2<f64> = (-Array2::eye(num_spec)) + (dt_days * j_ode);
    let ode = kinetic_ode_from_terms(
        network,
        &terms,
        misc,
        d,
        conc,
        q_in,
        min_start,
        do_reactions,
    );
    let res = (c_0 - conc) + (dt_days * ode);
    (res, jac_x)
}

/// Convert a pure-Rust [`StepResultAll`] into the Python-visible [`RtStep`],
/// mirroring exactly what the single-zone `step` pymethod builds.
fn make_rt_step<'py>(py: Python<'py>, r: &StepResultAll) -> RtStep {
    RtStep {
        state: r.state.to_pyarray(py).unbind(),
        mineral_rates: r.mineral_rates.to_pyarray(py).unbind(),
    }
}

/// Pure-Rust driver for one time step across every reactive-transport zone.
///
/// This replaces the per-zone Python loop in `ReactiveTransportModel.step_rt_model`
/// with a single GIL boundary crossing. Zones are advanced in order `i = 0..n`,
/// and each zone's incoming concentration is computed from the lateral/vertical
/// mass fluxes of the zones already solved earlier in the same step
/// (Gauss-Seidel coupling) exactly as the Python loop did:
///
/// ```text
/// mass_in      = lat_mass.T @ lat_mat[i]  +  vert_mass.T @ vert_mat[i]
/// conc_in      = mass_in / (q_in + forc_flux)   if > 1e-6 else 0
/// ```
///
/// Surface zones skip the coupling (their precipitation `_conc_in` is already set).
/// On success, returns one [`RtStep`] per zone in the same order as the input
/// `zones`; on failure, returns the steps computed so far together with
/// `(failing zone index, the original exception, the incoming concentration
/// that zone was about to step from, and this driver's lat / vert coupling
/// buffers as they stand — rows `start..i` filled, the failing row and later
/// rows zero)`.
///
/// `start` supports resuming a sweep the Python driver interrupted to rescue a
/// failed zone: the sweep then runs from zone `start` instead of zone 0 and
/// returns only the steps for zones `start..n_zones`. A resumed sweep requires
/// `lat_mass_init` / `vert_mass_init` — the outgoing mass produced by the zones
/// solved before the interruption — so that the Gauss-Seidel coupling yields
/// exactly the same incoming concentrations an uninterrupted sweep would.
#[pyfunction]
#[pyo3(signature = (zones, states, forcings, lat_mat, vert_mat, surface_zone_ids, dt, verbose=false, start=0, lat_mass_init=None, vert_mass_init=None))]
pub fn run_zone_steps<'py>(
    py: Python<'py>,
    zones: Vec<Bound<'py, RtZone>>,
    states: PyReadonlyArray2<f64>,
    forcings: Vec<RtForcing>,
    lat_mat: PyReadonlyArray2<f64>,
    vert_mat: PyReadonlyArray2<f64>,
    surface_zone_ids: Vec<usize>,
    dt: f64,
    verbose: bool,
    start: usize,
    lat_mass_init: Option<PyReadonlyArray2<f64>>,
    vert_mass_init: Option<PyReadonlyArray2<f64>>,
) -> (
    Vec<RtStep>,
    Option<(
        usize,
        PyErr,
        Bound<'py, PyArray1<molar>>,
        Bound<'py, PyArray2<molar>>,
        Bound<'py, PyArray2<molar>>,
    )>,
) {
    let n_zones: usize = zones.len();
    // Get plain ndarray views upfront (cheaper than calling `.shape()` /
    // indexing on `PyReadonlyArray2` inside the loop).
    let states_arr: Array2<f64> = states.as_array().to_owned();
    let lat_mat_arr: Array2<f64> = lat_mat.as_array().to_owned();
    let vert_mat_arr: Array2<f64> = vert_mat.as_array().to_owned();
    let n_spec: usize = states_arr.shape()[1];

    if states_arr.shape()[0] != n_zones {
        return (
            Vec::new(),
            Some((
                0,
                PyValueError::new_err("`states` first dimension must equal the number of zones"),
                Array1::<f64>::zeros(0).to_pyarray(py),
                Array2::<f64>::zeros((0, n_spec)).to_pyarray(py),
                Array2::<f64>::zeros((0, n_spec)).to_pyarray(py),
            )),
        );
    }
    if forcings.len() != n_zones {
        return (
            Vec::new(),
            Some((
                0,
                PyValueError::new_err("number of forcings must equal the number of zones"),
                Array1::<f64>::zeros(0).to_pyarray(py),
                Array2::<f64>::zeros((0, n_spec)).to_pyarray(py),
                Array2::<f64>::zeros((0, n_spec)).to_pyarray(py),
            )),
        );
    }
    if n_zones == 0 {
        return (Vec::new(), None);
    }

    // Set of surface zones (their incoming concentration is precipitation, not
    // the upstream mass flux, so the coupling step is skipped for them).
    // An out-of-range surface id is reported through the same error channel as a
    // step failure (using the last zone index as a best-effort marker), rather
    // than aborting the whole GIL boundary.
    let mut surface = vec![false; n_zones];
    let mut bad_id = None;
    for &id in &surface_zone_ids {
        if id >= n_zones {
            bad_id = Some(id);
            break;
        }
        surface[id] = true;
    }
    if let Some(id) = bad_id {
        return (
            Vec::new(),
            Some((
                0,
                PyValueError::new_err(format!(
                    "surface zone id {id} is out of range (num zones {n_zones})"
                )),
                Array1::<f64>::zeros(0).to_pyarray(py),
                Array2::<f64>::zeros((0, n_spec)).to_pyarray(py),
                Array2::<f64>::zeros((0, n_spec)).to_pyarray(py),
            )),
        );
    }

    // `forcings` is already an owned clone of the Python list (extraction copies
    // each pyclass), so we can write each zone's `_conc_in` in place without ever
    // touching the caller's objects.
    let mut owns: Vec<RtForcing> = forcings;

    // A resumed sweep (`start > 0`) must be seeded with the outgoing mass of the
    // zones solved before the interruption, so the coupling for the remaining
    // zones is identical to an uninterrupted sweep.
    if start > n_zones {
        return (
            Vec::new(),
            Some((
                0,
                PyValueError::new_err(format!(
                    "`start` index {start} exceeds the number of zones {n_zones}"
                )),
                Array1::<f64>::zeros(0).to_pyarray(py),
                Array2::<f64>::zeros((0, n_spec)).to_pyarray(py),
                Array2::<f64>::zeros((0, n_spec)).to_pyarray(py),
            )),
        );
    }
    let seed_mass_buffer =
        |init: &Option<PyReadonlyArray2<f64>>, name: &str| -> PyResult<Array2<molar>> {
            if start == 0 {
                return Ok(Array2::zeros((n_zones, n_spec)));
            }
            let Some(init) = init else {
                return Err(PyValueError::new_err(format!(
                    "{name} is required when resuming a sweep (start > 0)"
                )));
            };
            let init_arr: Array2<f64> = init.as_array().to_owned();
            if init_arr.shape()[0] != n_zones || init_arr.shape()[1] != n_spec {
                return Err(PyValueError::new_err(format!(
                    "{name} must have shape [{n_zones} {n_spec}] but has [{}, {}]",
                    init_arr.shape()[0],
                    init_arr.shape()[1]
                )));
            }
            Ok(init_arr)
        };
    let mut lat_mass: Array2<molar> = match seed_mass_buffer(&lat_mass_init, "lat_mass_init") {
        Ok(buf) => buf,
        Err(e) => {
            return (
                Vec::new(),
                Some((
                    0,
                    e,
                    Array1::<f64>::zeros(0).to_pyarray(py),
                    Array2::<f64>::zeros((0, n_spec)).to_pyarray(py),
                    Array2::<f64>::zeros((0, n_spec)).to_pyarray(py),
                )),
            );
        }
    };
    let mut vert_mass: Array2<molar> = match seed_mass_buffer(&vert_mass_init, "vert_mass_init") {
        Ok(buf) => buf,
        Err(e) => {
            return (
                Vec::new(),
                Some((
                    0,
                    e,
                    Array1::<f64>::zeros(0).to_pyarray(py),
                    Array2::<f64>::zeros((0, n_spec)).to_pyarray(py),
                    Array2::<f64>::zeros((0, n_spec)).to_pyarray(py),
                )),
            );
        }
    };

    let mut out: Vec<RtStep> = Vec::with_capacity(n_zones - start);

    for i in start..n_zones {
        let d: &mut RtForcing = &mut owns[i];
        let tot_water_in: f64 = d.hydro_step.q_in + d.hydro_step.forc_flux;

        if !surface[i] {
            // Mass arriving from the zones already solved earlier this step.
            // Equivalent to the Python `mass_in = lat_mass.T @ lat_row + vert_mass.T @ vert_row`
            // (both accumulate over the *rows* k of the mass matrices that have
            // been filled so far, i.e. the previously-solved zones).
            let mut mass_in: Array1<molar> = Array1::zeros(n_spec);
            let lat_row = &lat_mat_arr.row(i);
            let vert_row = &vert_mat_arr.row(i);
            for sp in 0..n_spec {
                let mut li = 0.0;
                let mut vi = 0.0;
                for k in 0..n_zones {
                    li += lat_mass[[k, sp]] * lat_row[k];
                    vi += vert_mass[[k, sp]] * vert_row[k];
                }
                mass_in[sp] = li + vi;
            }
            if tot_water_in > 1e-6 {
                for sp in 0..n_spec {
                    mass_in[sp] /= tot_water_in;
                }
            } else {
                mass_in.fill(0.0);
            }
            d._conc_in = mass_in;
        }

        let s_i: Array1<f64> = states_arr.row(i).to_owned();
        // Borrow the zone (no deep clone): `Bound::borrow()` yields a `PyRef<T>`
        // that derefs to `&T`. The zone is only read (`step_internal` is `&self`),
        // so a shared borrow is all we need; the speciation cache (an `Arc`)
        // stays shared with the caller across steps, so warm-starts still work.
        let zone = zones[i].borrow();
        match zone.step_internal(&s_i, d, dt, verbose, None) {
            Ok(r) => {
                // Record outgoing mass so downstream zones (k > i) can consume it.
                lat_mass.row_mut(i).assign(&r.lat_mass);
                vert_mass.row_mut(i).assign(&r.vert_mass);
                out.push(make_rt_step(py, &r));
            }
            Err(e) => {
                // Hand back everything needed to reproduce the failure in
                // Python: the failing zone index, the *original* exception
                // (type preserved, e.g. OptimizationError), the incoming
                // concentration that zone was about to step from, and the
                // driver's lat / vert coupling buffers as they stand (rows
                // `start..i` filled, the rest zero). The Python driver needs
                // them to re-seed its own buffers after rescuing zone `i`,
                // since the `RtStep`s no longer carry outgoing masses.
                let conc_in = d._conc_in.to_pyarray(py);
                return (
                    out,
                    Some((
                        i,
                        e,
                        conc_in,
                        lat_mass.to_pyarray(py),
                        vert_mass.to_pyarray(py),
                    )),
                );
            }
        }
    }

    (out, None)
}
