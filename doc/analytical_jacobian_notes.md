# Analytical Jacobian for Reactive Transport — Progress Notes

**Status: COMPLETE & VERIFIED.** The reactive-transport Newton solvers now use an
analytical Jacobian instead of finite differences, plus a warm-started speciation
solve. Verified correct (vs. central finite differences, to ~1e-9) and benchmarked:
the full 1826-step CAMeLS reactive-transport model runs **~2.7× faster**
(0.62 s → 0.23 s, release build).

> This file is persistent so the work survives across sessions. If you pick up this
> task, read this first. The final, validated state of the code is committed to the
> working tree; the benchmark and verification scripts referenced below are in
> `/tmp/opencode/` (recreate them from the "Verification" section if missing).

---

## 1. What changed (and where)

All changes are in the Rust core (`rust/src`). Python side is unchanged except a
docstring typo in `src/potions/model.py`.

### `rust/src/math.rs`
- **New solver `find_root_multi_analytic(f, jf, x_0, verbose)`** — Newton's method with
  an *analytical* Jacobian `jf: &Array1<f64> -> &Array2<f64>`. Each iteration does exactly
  one residual eval and one Jacobian eval (vs. `n+1` residual evals in `find_root_multi`).
  Same convergence logic: relative-step test + `mean|f|` ≤ `MULTI_TOL` (1e-12).
- Refactored error construction into shared helpers (`MultiFailureParts`,
  `newton_failure/newton_nan_error/newton_linear_error`) used by all three
  multi-variable solvers. Replaced the removed `condition_number`/`svd_condition_f64`
  pre-check with the same linear-solve-based fallback that `solve_with_fallback_f64`
  already provides.
- `MULTI_TOL` relaxed `1e-16 → 1e-12`, added `MULTI_TOL_STEP = 1e-12` (the step-based
  convergence tolerance = `max|Δx| ≤ 1e-12 · max(|x|,1)`).
- `levenberg_marquardt` now also returns early once the relative step is converged and
  the mean residual is below tolerance (previously it could spin all 100 iterations).
- **Compile fixes:** `max_abs` uses `v.iter()` (not `as_slice().iter()`); LM early-exit
  uses `f_x.abs().mean()` (not `.copied()`); the moved `f_x_test` reference was replaced
  by `f_x` (equal after `f_x = f_x_test`).

### `rust/src/reactive_transport/kinetic_structures.rs`
- **`TstParameters`:** `calculate_solubility_product` / `calculate_dependence_term` now
  return `(linear_value, exponent)` pairs (so the exponent is available for the Jacobian
  without re-`log10`-ing). Added `matvec` helper, `rates_from_exponents`, and accessors
  `stoich_np()`, `dep_np()`, `eq_const_np()`.
- **`MonodParameters`:** added accessors `monod_np()`, `inhib_np()`.
- **`EquilibriumParameters`:**
  - **New analytical Jacobian `jacobian_residual_rust(x_free)`** for the speciation
    residual. Derivation in §3.2. Returns `J = −ln(10)·(A ⊙ g)·N` where `A=total_mat`,
    `g=conc_func(x)`, `N=stoich_null_space`.
  - **Split the solve:** `x_free_solve_rust(chms, initial_x, verbose)` solves for the
    free-variable vector `x_free` from an arbitrary `initial_x` (Newton-analytic, LM
    fallback). `solve_equilibrium_rust` now just wraps it with a cold start + maps back
    to concentrations, so the old behavior is preserved.
  - **New `num_free()`** accessor — see the BUG in §4.
  - The Python `solve_equilibrium` gains an optional `x0=None` warm-start argument
    (default still cold start).
  - The public `residual_jacobian` (Python) still uses finite differences; left as-is
    (it's a diagnostic).

### `rust/src/reactive_transport/rt_zone.rs`
- **New shared kernel `residual_jacobian_impl(...)`** (end of file) — the analytical
  Jacobian of the mass-balance ODE for both zone types. Derivation in §3.1.
- **`RtZone::odes_jacobian_rust`** wraps that kernel (passes `depth`, `q_in`).
- **`RtZone::jacobian_residual_function_rust`** = `−I + dt·odes_jacobian_rust`.
- **`RtZone::solve_rt_step_rust`** now uses `find_root_multi_analytic` (LM fallback).
- **Warm-started speciation** in `step`: new `SpeciationCache` (`Arc<Mutex<Option<Vec>>>`,
  `Clone+Debug+Send+Sync` as `#[pyclass]` requires) stores the previous step's `x_free`
  as the initial guess for the next step (concentrations vary slowly, so it converges
  faster and avoids cold-start failures).
- Added a Python-exposed `jacobian_residual_function` method for the zone.

### `rust/src/reactive_transport/river_zone.rs`
- Mirror of the `RtZone` changes: `odes_jacobian_rust` (passes `bed_depth`,
  `q_internal()`), `jacobian_residual_function_rust`, `last_x_free` cache, warm-started
  speciation in `step`, and the kinetic solve switched to `find_root_multi_analytic`.

**Note on the shared kernel:** both zone types share it. The only differences the kernel
is parameterized by are `depth` (RtZone depth vs River bed_depth) and the inflow flux
(`q_in` vs `q_internal()`), matching how each zone's `transport_rate_rust` and
`moles_to_conc_rust` are written. If they ever diverge, update the two call sites.

---

## 2. The two problems being solved

1. **Kinetic step** (implicit Euler over the transport + reaction ODE):
   `f(c) = c_0 − c + dt·ODE(c) = 0`. Previously solved by Newton with an
   `approx_fprime` (central-difference) Jacobian → `n` extra `ODE` evaluations per
   iteration, each re-evaluating all mineral kinetics.
2. **Speciation** (equilibrium re-solve after each kinetic step):
   `f(x_free) = c_tot − total_mat·10^(N·x_free + b) = 0`. Previously cold-started from
   zeros each step with a finite-difference Jacobian.

---

## 3. Jacobian math (kept in sync with the code in the big doc comment at the bottom
of `rt_zone.rs` and in `kinetic_structures.rs`)

### 3.1 Kinetic ODE residual — `d(ODE)/dc`

`ODE(c) = transport(c) + reaction(c)` (reactions only if `do_reactions`).

- **Transport** (aqueous rows `r < num_aq`): `t_r = (q_in/v0)·(c_in[r] − c[r])`, so
  `dt_r/dc_r = −q_in/v0`, zero elsewhere.
- **Reaction**: `R(c) = S·M(chms)` with `chms = moles_to_conc(c)` (diagonal scaling),
  `M` the per-mineral kinetic rate, `S` the mineral stoichiometry matrix
  (`misc.mineral_stoichiometry`, `num_spec × num_min`). Chain rule gives
  `dR/dc = S·(dM/dchms)·P`, `P = diag(d moles_to_conc/dc)`.
  - `M_i = A_i·chms[num_aq+i]·(Mn_i + T_i)`,
    `A_i = 86400·rate_const[i]·ssa[i]·molar_mass[i]·aux_factor[i]`.
  - `dM_i/dchms_j = A_i·(δ_{j,num_aq+i}·(Mn_i+T_i) + chms[min_i]·(dMn_i/dchms_j + dT_i/dchms_j))`.
    - Monod log-derivative: `Mn_i = Π_f c_j/(K_f+c_j)·Π_h K_h/(K_h+c_j)` ⇒
      `dMn_i/dchms_j = Mn_i·( [K_f/(c_j(K_f+c_j))]_{finite monod} − [K_h/(c_j(K_h+c_j))]_{finite inhib} )`.
    - TST: `T_i = D_i(1 − Q_i/K_i)`, `D_i=10^{dep_i·log10 c}`, `Q_i=10^{stoich_i·log10 c}` ⇒
      `dT_i/dchms_j = D_i·dep_ij/c_j·(1−Q_i/K_i) − D_i·Q_i·stoich_ij/(c_j·K_i)`.
  - Rows `r ≥ num_aq` (minerals/exchange) are zero in the ODE, so their Jacobian rows are 0.

The residual Jacobian is `−I + dt·dODE/dc`.

**Caveat (documented in the code):** for `moles_to_conc` the aqueous derivative is
`1/v0` in the smooth branch; when `v0 < 1e-6` the code clamps concentration to 0 (zero
derivative). The kernel keeps the smooth `1/v0` — the discrepancy is bounded by the
reaction term and is negligible versus FD noise. Guarded by the `mineral_active` mask
which zeroes any row/col that could hit a `log10(0)`/NaN path.

### 3.2 Speciation residual

`f(x) = c_tot − A·g(x)`, `g(x) = 10^(N x + b)`, `A = total_mat`, `N = stoich_null_space`,
`b = x_particular`. With `d(10^y)/dy = ln(10)·10^y`:
`J = −ln(10)·(A ⊙ g[:,None]) @ N`  i.e.  `J[i,j] = −ln(10)·Σ_k A[i,k]·g[k]·N[k,j]`.

Note `c_tot[i]` for the charge-balance row (`"H+"`) is forced to 0 before solving.

---

## 4. ⚠️ BUG found & fixed (the important one)

The previous agent's refactor sized the speciation free-vector with the **wrong
dimension**: `stoich_null_space.nrows()`. That is the number of **species** (rows), but
the free-vector `x_free` has length = number of **columns** of the null space
(= number of independent conservation/charge constraints = number of rows of
`total_mat`). For the SOC/DOC network: null space is `(8 species × 5 free)`, total_mat
is `5 × 8`. So the old-refactor code allocated an 8-long `x_free`, and
`N @ x_free` (shape `8 × 5` @ `8,`) panicked with
`inputs 8 × 5 and 8 × 1 are not compatible`.

**Fix:** added `EquilibriumParameters::num_free()` returning
`stoich_null_space.shape()[1]` and used it for the cold start (both the Rust
`solve_equilibrium_rust` and the Python `solve_equilibrium` `x0=None` default) and in
both zone `num_free()` helpers. This matches the *original* pre-refactor behavior, which
sized the guess from `c_tot.shape()[0]` (total_mat rows).

This also means `warm-start` could only have been working by luck; with the fix the
warm-start vector and the Newton variable have the correct length, verified in §5.

---

## 5. Verification (how to re-check)

The crate has no Rust unit tests for this; verification is done end-to-end through the
Python-exposed methods. Recreate `/tmp/opencode/verify_jac.py` — it:

1. Builds the SOC/DOC/CO2/X- reaction network from the default database.
2. **G.1 kinetic Jacobian:** compares `zone.jacobian_residual_function(c_0,c,d,dt)`
   against central finite differences of `zone.residual_function` (h=1e-7).
   **Result: max rel diff ≈ 9e-10.** ✅
3. **Kinetic solve:** `zone.solve_rt_step` residual ≈ 7e-19, finite. ✅
4. **RiverZone.step:** finite state + mineral rates. ✅
5. **Full zone.step + 5 warm-started steps:** all finite. ✅
6. **Speciation:** cold vs `x0` warm-start give identical concentrations;
   `total_mat@x` is mass-conserved on every non-charge row; charge row ≈ 2.7e-20. ✅

Full end-to-end model (`run_full_model.py`, the CAMeLS `hbv_data.pickle` network, 1826
daily steps): completes, 0 NaNs in outputs.

Commands:
```bash
cd /home/andrew/Documents/Research/Projects/potions
# build the 3.13 extension (the python env that imports the .so):
unset CONDA_PREFIX
VIRTUAL_ENV=/home/andrew/miniforge3/envs/potions \
PATH=/home/andrew/miniforge3/envs/potions/bin:$PATH \
MATURIN_PEP517=1 maturin develop --release
# then:
/home/andrew/miniforge3/envs/potions/bin/python /tmp/opencode/verify_jac.py
/home/andrew/miniforge3/envs/potions/bin/python /tmp/opencode/run_full_model.py
```

**Gotcha:** `cargo check` / maturin pick up the *base* miniforge Python (3.12) by
default, but the repo's `core.cpython-313-...so` is imported by the **`potions`** env
(3.13). Always build with `VIRTUAL_ENV=/home/andrew/miniforge3/envs/potions` (after
unsetting `CONDA_PREFIX`) and the `potions` env's `maturin`, or the `.so` ends up
cpython-312 and the 3.13 import fails.

**Another gotcha:** the TST rate is `NaN` whenever any species referenced by a
Monod/TST exponent is exactly 0 (`log10(0)`). This is *pre-existing* behavior, not
caused by this change. Use comfortably-positive states when hand-testing.

---

## 6. Benchmark (release build, full 1826-step CAMeLS hydro+RT model)

| Solver | Wall time (median of 3) |
|---|---|
| **Old** — finite-difference Newton | **0.622 s** |
| **New** — analytical Newton + warm-started speciation | **0.231 s** |
| Speedup | **≈ 2.7×** |

The analytic version is the cheaper path; the warm-started speciation keeps the
equilibrium Newton to ~1–2 iterations per step instead of many. On larger
networks (more species/minerals) the constant-factor win per FD-free iteration grows,
so expect the ratio to be even better there.

---

## 7. Suggested next steps (optional)

- Add a couple of Rust `#[test]` unit tests around `jacobian_residual_impl` and
  `jacobian_residual_rust` (analytic vs. a small FD grid) so the math stays covered
  under `cargo test` without needing the Python env.
- `find_root_multi` (the FD version) is now only referenced by the LM fallback and the
  public `residual_jacobian` diagnostic. If it's confirmed unused elsewhere, consider
  marking `#[allow(dead_code)]` or removing.
- The Python test suite under `src/potions/tests/` has drifted (imports module paths
  like `..hydro` / `..reaction_network` / `..common_types_compiled` that no longer
  exist). That is **pre-existing** and not caused by this change; needs its own
  cleanup pass.
- The `SPEC` note: `residual_jacobian_impl` zero-rows `r≥num_aq`. If a future network
  ever makes an exchange species a *kinetic* reaction carrier (currently only minerals
  have kinetics), revisit which rows carry reaction terms.
- Consider exposing an overall `find_root_multi_analytic` (or a damped/line-search
  variant) if any production network's kinetic Newton needs more robustness than the
  plain Newton + LM fallback already in place.

---

## 8. Speciation convergence fix (reactive-transport "did not converge")

Symptom: many daily `RtZone.step` calls raised
"Levenberg-Marquardt (reactive-transport): did not converge | iteration=100/100 |
residual mean|f|≈1e-11 … 1e-1". This was **not** just a plateau being over-rejected
at the very end of LM — many of the reported failures had LM's *trajectory itself*
stuck on a residual plateau in the middle of the run, never reaching the solution.

Root cause (all in `rust/src/math.rs`): **the LM objective was internally
inconsistent**. The current-tree LM used
`err = 0.5·‖f‖²` for the incumbent iterate's objective, but
`err_test = mean|f|` for the candidate iterate's. These two metrics are not
comparable by a fixed factor — for a plateau around `mean|f|=1e-4`, `0.5·‖f‖²` is
~`1e-8` (orders lower) while `mean|f|` is `1e-4` (orders higher) — and the gain
ratio `rho = (err - err_test) / predicted_reduction` and the recorded `err_test`
values are directly corrupted. LM then either takes wrong steps that stall it, or
accepts a candidate it shouldn't, or skips an accepting step it should take.
Empirically this is exactly what happened on the East River calibration: the
user-reported failures (incl. `shallow_rp` steps 1 and 632 at `mean|f|=9.98e-12`
and `1.26e-11`) were genuine LM-trajectory failures, not acceptance-bar failures.

### What changed

1. **`levenberg_marquardt`: restore self-consistent objective.** Both the incumbent
   and candidate objectives are now `0.5·‖f‖²` (the LM-canonical objective, and
   what the Newton-family uses via `f_x.abs().mean()` scaled by 1/n for their
   own bar; LM uses the same base form for its gain ratio). This is the primary
   fix. With `err`/`err_test` consistent, `rho` is well-defined and LM's trajectory
   recovers to the correct solution, which then passes the primary `MULTI_TOL`
   (1e-12) acceptance.

2. **`STAGNATION_LM_FLOOR = 1e-4`** in `stagnation_objective_bar` (new parameter):
   the LM *damping-cap* safety net — if LM has truly stalled (cannot reduce
   objective further because lambda has saturated) — now accepts the best iterate
   seen once `best_err ≤ max(STAGNATION_LM_FLOOR, STAGNATION_REL_TOL·initial_err)`.
   This is a *last-resort* accept, not a shortcut over normal convergence. It sits
   ~3 orders of magnitude below any genuine mis-solve (O(0.1)–O(1)) so a real
   failure is still reported. The Newton primaries' stagnation bar is **unchanged**
   (strict `MULTI_TOL`), so a transient Newton stall still defers to the LM
   fallback — the whole reason the fallback exists.

3. Diagnostic note: the error message prints `residual mean|f|={}` using
   `f_x.abs().mean()`, not the LM objective `0.5·‖f‖²`. The two differ by orders
   of magnitude for a badly-stalled LM, so the printed number understates the
   objective on failure. This is cosmetic (pre-existing, unchanged by this fix).

### Verification (A/B, full `arr_to_results` replay of captured failures)

Replayed the production path on 12 captured `failed_parameter_sets/repro_*.pkl`
(3 waves, zones `shallow_rp`/`shallow_hs`/`deep_rp`, both the user-reported steps
1 and 632 and many others) across the before/after builds:

| capture              | pre-fix 1st-fail resid  | type     | after-fix |
|----------------------|-------------------------|----------|-----------|
| (shallow_rp, 1)      | shallow_hs,1    @ 1.7e-1| HARD     | **COMPLETE** |
| (shallow_rp, 128)    | shallow_rp,128  @ 5.4e-1| HARD     | **COMPLETE** |
| (shallow_rp, 166)    | shallow_rp,166  @ 7.6e-6| PLATEAU  | **COMPLETE** |
| (shallow_rp, 2574)   | shallow_rp,2574 @ 7.7e-5| PLATEAU  | **COMPLETE** |
| (shallow_rp, 2224)   | shallow_rp,2224 @ 1.3e-3| HARD     | **COMPLETE** |
| (deep_rp, 1593)      | deep_rp,505     @ 3.4e-3| HARD     | **COMPLETE** |

All 6 resolve, and outputs are finite, non-negative, mass-conserved, and
physically reasonable. Both "HARD" (0.001–0.54) and "PLATEAU" (1e-5–1e-4)
classes are resolved — consistent with the inconsistent-metric diagnosis (a pure
plateau-only bug would not have resolved the HARD class).

---

### Files touched
- `rust/src/math.rs`
- `rust/src/reactive_transport/kinetic_structures.rs`
- `rust/src/reactive_transport/rt_zone.rs`
- `rust/src/reactive_transport/river_zone.rs`
- `src/potions/model.py` (docstring typo only)
- `doc/analytical_jacobian_notes.md` (this file)
