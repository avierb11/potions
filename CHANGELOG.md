# Changelog

All notable changes to the `potions` package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **East River calibration GUI** in `potions.gui`: a PySide6 desktop
  workbench (replacing the earlier Streamlit app) with the model's
  three figures in one panel (flipped by arrow buttons or the Left /
  Right keys), an always-visible per-run objective-function table under
  the figure, and every calibrated parameter in a panel below, grouped
  and themed by zone (hydrology in blue, reactive transport in red).
  Pinned (read-only) parameters can be unlocked by double-click to
  choose an adjustable range; parameter edits re-run the hydrology and
  reactive-transport models on a background thread.  Start it with the
  `potions-gui` console script; the East River series CSV ships with
  the package, and *Save run* writes a fresh timestamped directory
  under `results/` in the current working directory, including the
  session state (`gui_state.json`) that the app restores on its next
  start.  PySide6 (>= 6.4) is now a core dependency (the `potions`
  conda environment's PySide6 install carries a local `py.typed`
  marker so `mypy --strict` can type-check the GUI).
- The calibration GUI's window now restores the maximized state of
  the run it reopens: a run saved while maximized opens maximized
  (Home+Up toggles maximize), any restored or default window size is
  clamped to the screen, and a save while maximized records the
  window's normal geometry instead of the maximized client size.
  `gui_state.json` format version 8 (version 7 files still restore).
- **Pre-run model validation** with descriptive, typed errors:
  - New error types in `potions.utils`: `ModelStructureError` and
    `ModelInputError` (both subclass `ValueError` and `HydrologyError`).
    Structure problems are caught at model construction; input problems are
    caught before the first model step.
  - **Bottom-zone placement rules**: the last layer of a model must consist of
    bottom zones, and nothing may be placed below a bottom zone. A model whose
    bottom is any other zone type now raises `ModelStructureError` at
    construction (see *Changed* — this is a **breaking change**).
  - **Python bottom-zone registry** in `potions.model_components`:
    `register_bottom_zone` (decorator), `BOTTOM_ZONE_CLASSES` (the registry
    set), and `is_bottom_zone()` (membership test). `SubsurfaceZoneB` is
    registered by default.
  - **Structural checks** (all `ModelStructureError`): a non-empty structure;
    finite zone parameters (including `zones=` overrides); and partition scales
    that are finite, non-negative, and sum to one.
  - **Input-data checks** (all `ModelInputError`): no NaN/Inf in forcing
    series; no negative precipitation or PET; consistent time-series lengths;
    non-empty, numeric forcing; measured-streamflow length, non-negativity,
    and rejection of infinite values (NaN entries mark missing measurements
    and are excluded from the objective functions); and `init_state` length,
    finiteness, and non-negativity.
  - **Reactive-transport forcing checks**: `precip_conc` must have the correct
    `(num_surface_zones, n_steps, num_species)` shape and be finite.
- **Opt-in water-balance assertion**: `run_hydro_model(..., check_water_balance=True)`
  now raises `HydrologyNumericalError` (with full step context) when a step's
  mass balance does not close, instead of the previous silent/printed result.
  Disabled by default, so normal runs are unaffected.
- **Reactive-transport calibration suite** in `potions.calibration` (exported
  from `potions`):
  - `calibrate(data, spec, bounds, options)` — global parameter search
    (differential evolution, `scipy.optimize`) of hydro + RT parameters against
    streamflow and target-solute concentration.
  - `polish(...)` — local refinement of a calibration result.
  - `run_ensemble(...)` — ensemble of the top-ranked parameter sets.
  - `build_rt_bounds(...)`, `make_objective(...)`, `arr_to_results(...)`.
  - Composable objective specification: `ObjectiveSpec`, `ObjectiveComponent`,
    `evaluate_components(...)`, `METRIC_REGISTRY`, and a default
    `DEFAULT_RT_OBJECTIVE_SPEC`.
  - New dataclasses: `RtCalibrationData`, `CalibrationOptions`,
    `CalibrationResults`, `EnsembleResults`.
- **New metrics** in `potions.objective_functions`: flow-weighted variants
  `flow_weighted_nse`, `flow_weighted_log_nse`, and `flow_weighted_kge`,
  registered in `METRIC_REGISTRY`.
- **Parameter documentation**: `describe_parameters()` on both the
  hydrological and reactive-transport models returns a DataFrame of
  parameter name, value, and description. Backed by a new
  `potions.reactive_transport.parameter_docs` helper module.
- **Tests**: `tests/test_validation.py` (31), `tests/test_rt_calibration.py`
  (17), `tests/test_parameter_docs.py` (15), plus structure-diagram layout
  tests in `tests/test_hydro_model.py`.
- **Result saving, on by default**: `run_hydro_model`, `run_rt_model`, and
  `run_both_models` now save each run's results and model parameters to a
  fresh `results/run_<timestamp>/` directory (`run_both_models` writes both
  models into one folder under `hydro/` and `rt/`). Each run folder contains
  the simulation time series (`simulation.csv`), the full model parameters
  (`parameters.csv`, from `describe_parameters()`), the evaluated objective
  functions when measured data was provided (`objective_functions.csv`), and a
  `run_info.json` with metadata (version, run length, objective values, ...);
  RT runs additionally store their results object in
  `results.pkl` (with a text note if it cannot be pickled).
  `run_batch` writes each member's time series and parameters into a
  timestamped batch folder, plus a batch-level `run_info.json`.
  - Saving is configurable: pass `save_results=False` (or keep
    `write_time_series_results=False` for batches) to disable, and
    `results_dir=` / `output_dir=` to redirect.
  - Results objects (`HydroModelResults`, `RtModelResults`) gain a
    `saved_dir: Optional[str]` field pointing at the written folder.
  - New `potions.result_saving` module with the building blocks
    (`is_save_enabled`, `resolve_results_dir`, `make_run_dir`,
    `save_hydro_run`, `save_rt_run`, `save_model_run`, `write_batch_info`)
    for saving results from custom entry points.
- **Area-weighting API on `HydrologicalModel`**: `zone_scales` (per-zone area
  fractions propagated from the surface `scales`), `lat_mat_scaled` (the
  area-weighted lateral connectivity matrix used by the hydro step and the
  outlet), and `vert_mat_scaled` (the area-weighted vertical matrix, which
  matters when a layer aggregates N zones into one). The plain `lat_mat` and
  `vert_mat` (0/1 adjacency) are retained for the reactive-transport
  concentration coupling.
- **Model-level derived measurements** (e.g. DIC as the sum of the aqueous
  carbonate species): a new `DerivedSpecies` type (name + component species
  names), exportable from `potions`, accepted by `Model`,
  `ReactiveTransportModel`, and `Model.from_array` via a
  `derived_species=` keyword. A model that declares a derived measurement
  adds a `{name}_riv` column to the RT results (the sum of its
  components' river concentrations), and a `meas_river_conc` column named
  after the measurement is scored against that sum, exactly like a measured
  network species. Components must be aqueous (primary or secondary) species
  of the network; names must not collide with a network species or another
  derived measurement. Unknown `meas_river_conc` columns (neither a network
  species nor a declared derived measurement) raise `ModelInputError`
  before the simulation starts.
- **GUI: selectable line-search objectives**: the objective table fits its
  content with no scrollbars, sits top-aligned, and each cell carries a
  checkbox for selecting line-search objectives; the line search optimizes
  the equal-weighted sum of every selected objective (e.g. NSE + LogNSE +
  KGE across Q, DOC, DIC, Ca++).
- **GUI: static parameters in the range dialog**: the range dialog can make
  a value static and disable the range (the parameter becomes pinned;
  double-click the pinned cell to give it a range again).
- **GUI: per-job progress in the header**: a header spinner and a
  determinate progress bar show each job's completed points as it runs.
- **GUI: parallel point evaluation**: gradient and line-search points are
  evaluated in parallel in a `multiprocessing.Pool` using all CPUs by
  default; a failed point is shown as a missing value instead of failing
  the whole job.
- **GUI: mode-tinted window background**: the window background tints light
  blue in hydrology-only mode and warm in reactive-transport mode.

### Changed

> **Breaking changes** — the following alter public behavior or APIs.

- Model runs **write to disk by default** (see *Added*). Opt out per call with
  `save_results=False` / `write_time_series_results=False`.
- `run_batch`'s `write_time_series_results` was `bool = False` (off) and is
  now `Optional[bool] = None`, meaning "use the built-in default (saving on)".
  Passing `True`/`False` explicitly behaves as before.
- `run_batch`'s `output_dir` default changed from `"batch_results"` to
  `"results"`, and saved output now lands in a timestamped
  `<output_dir>/run_<timestamp>/` subfolder (member files are named
  `run_<i>.csv` / `run_<i>_parameters.csv`).
- The bottom layer of a model **must** consist of bottom zones (e.g.
  `SubsurfaceZoneB`). Models whose bottom is now any other zone type (e.g. a
  plain `SubsurfaceZone`) raise `ModelStructureError` at construction.
- **Lateral transfer and outlet discharge are now area-weighted.** Previously
  only the forcing was scaled by `scales`; inter-zone lateral fluxes and the
  reported streamflow were unweighted sums of intensive (mm/d) zone fluxes,
  which let the calibration implicitly absorb zone size. Now:
  - a lateral edge from donor `d` to receiver `r` carries the area ratio
    `s_d / s_r` (via the new `lat_mat_scaled`), so runoff from a large
    hillslope is concentrated onto a smaller riparian footprint;
  - the outlet is the area-weighted sum of river-zone lateral fluxes
    (`streamflow_sim`), i.e. a catchment-mean depth (mm/d);
  - zone fluxes themselves stay intensive (mm/d) — area factors live only on
    the graph *boundaries*.
  Multi-zone runs with unequal `scales` will therefore produce different
  (correct) discharge than before. Single-zone models (`scales=[1.0]`) are
  numerically unchanged.
- Partition scales must now be **strictly positive** (previously
  non-negative): a zero-area zone would make the donor/receiver ratio
  undefined. `ModelStructureError` is raised for `s <= 0`.
- The subsurface hydro zones are renamed `GroundZone` -> `SubsurfaceZone`
  and `GroundZoneB` -> `SubsurfaceZoneB` (Rust classes and Python exports
  alike); the old names are not retained.
- `ZoneDimensions` is specified with `max_water` directly (plus
  `passive_water_storage`) instead of `porosity` and `depth`; the derived
  `max_water_volume` property is removed.
- **Water-table depth is removed entirely**: the `z_w` field (hydro step /
  RT forcing), `calculate_water_table_depth()`, and the `n_alpha`
  (water-table factor) RT parameter are all gone.
- Mineral RT parameters went from 5 to **4 per mineral**: `ssa` (specific
  surface area) and the water-table `n_alpha` factor are replaced by a single
  `reactivity` prefactor. A zone's base RT parameters go from 3 to 2
  (`max_water`, `passive_water_storage`), so the RT calibration parameter
  vector is correspondingly shorter.
- `HydroStep` / `RtForcing` pickled state no longer includes `z_w`; states
  pickled by a previous release cannot be unpickled by this one (and vice
  versa).
- The example notebook
  `notebooks/Run Reactive Transport Model.ipynb` is updated to the new
  `ZoneDimensions` / `reactivity` API.
- The reactive-transport step result `RtStep` is shrunk from nine fields
  to two: `state` (the final concentrations) and `mineral_rates`. The
  derivable per-step bookkeeping fields `conc_in`, `total_moles`,
  `mass_in`, `lat_conc`, `vert_conc`, `lat_mass`, and `vert_mass` are
  removed from the step result and from the `simulation.csv` output,
  which now carries only the `{species}_{zone}` state columns, the
  `{mineral}_rate_{zone}` columns, and the `{species}_riv` river
  columns. Each step's incoming concentration remains available through
  the forcing (`RtForcing.conc_in`, i.e. `results.rt_forcings`), and a
  new `RtZone.outgoing_mass(c_after_eq, d)` method computes a zone's
  lateral / vertical outgoing mass for a post-reaction state on demand.
  `RtStep` objects pickled in `results.pkl` by a previous release (nine
  fields) can no longer be unpickled.
- **`RtModelResults` is reduced to its tabular output.** The `rt_forcings`,
  `steps`, and `other` fields are removed: the per-step `RtStep` objects and
  the stacked state arrays were redundant with the `simulation` DataFrame
  (which is assembled from them), and the forcing matrix is a deterministic
  function of the hydro results and `precip_conc`. `run_rt_model` still
  computes the forcing and the per-step states during the run, but only the
  assembled `simulation` DataFrame (plus the objective functions) is kept in
  the returned and pickled results object. `RtModelResults` pickled by a
  previous release can no longer be unpickled.
- **`{mineral}_rate_{zone}` columns are emitted only for zones in which
  reactions run** (the zone's `do_reactions` flag). Zones with reactions
  disabled no longer carry their (always zero) rate columns, so the RT
  `simulation.csv` and `results.pkl` shrink by one rate column per mineral
  per non-reacting zone.
- **The calibration GUI's chrome is reorganized.** The window's header
  row is replaced by standard window chrome: a **File menu** (*Load
  data…*, *Load network file…*, *Save network as…*, *Edit network…*);
  an **Options menu** (*Auto-run*, *Save run history*, *Reset bounds*,
  and *Quit* with the standard Ctrl+Q shortcut); a fixed (non-movable,
  text-only) **toolbar** carrying the model and network selectors, the
  running-job spinner, the *Run model*, *Gradients*, *Line search*,
  *Calibrate*, and *Save run* actions, and the *Hydrology only* toggle;
  and a bottom **status bar** for the status line. The bold header title
  label is dropped (the same text was already the window title), and the
  header's minimum-width squeeze floors are replaced by the toolbar's
  overflow (») button, so the chrome can no longer pin the window wider
  than the screen. Session state is unchanged: the same
  `gui_state.json` (format version 8) restores the parameters, mode,
  toggles, geometry, and splitter sizes.

### Fixed

- **River-zone mineral-block indexing**: minerals now read starting
  immediately after the aqueous species, so exchange species that sit after the
  mineral block are no longer miscounted as mineral concentrations.
- `reaction_network` species masks now correctly cover the aqueous + mineral
  species and are consistent with the 4-parameters-per-mineral scheme.
- **GUI: a line search could race a job in flight**: starting a line search
  while a job (a model run or a gradient job) was still running was not
  refused; when that job finished, its older outcome could clobber the
  line search's newer, already-adopted results, so the simulation appeared
  to revert to stale results. A line search started while a job is in
  flight is now refused with a status message — a job cannot be
  interrupted, so the two kinds of job no longer race.
- **GUI: the window's minimum size exceeded a common screen**: every
  header item reported its full text width as its minimum, summing to a
  window `minimumSizeHint` of roughly 2050 px that window managers honor,
  so the calibration window could not be opened or shrunk onto a 1920-px
  display. The header's items now squeeze down to fixed floors (title,
  combos, buttons and checkboxes), so a narrow window squeezes the row —
  clipped text stays available in tooltips and combo popups — and the
  window's minimum (roughly 1150 px) fits inside its default size.

### Performance

- **Narrow Newton solve** in `RtZone` / `RiverZone`: each iteration solves only
  the aqueous species block (fused residual + analytic Jacobian) with the
  non-aqueous species held fixed as background, replacing the wider Newton
  step and reducing per-iteration cost.
- `get_hydro_steps()` now reads each column with a single `to_numpy()` call
  instead of per-step `.iloc` indexing, removing large numbers of slow pandas
  lookups on long simulations.
