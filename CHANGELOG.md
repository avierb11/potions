# Changelog

All notable changes to the `potions` package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
    set), and `is_bottom_zone()` (membership test). `GroundZoneB` is
    registered by default.
  - **Structural checks** (all `ModelStructureError`): a non-empty structure;
    finite zone parameters (including `zones=` overrides); and partition scales
    that are finite, non-negative, and sum to one.
  - **Input-data checks** (all `ModelInputError`): no NaN/Inf in forcing
    series; no negative precipitation or PET; consistent time-series lengths;
    non-empty, numeric forcing; and measured-streamflow/`init_state` length,
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

### Changed

> **Breaking changes** — the following alter public behavior or APIs.

- The bottom layer of a model **must** consist of bottom zones (e.g.
  `GroundZoneB`). Models whose bottom is now any other zone type (e.g. a
  plain `GroundZone`) raise `ModelStructureError` at construction.
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

### Fixed

- **River-zone mineral-block indexing**: minerals now read starting
  immediately after the aqueous species, so exchange species that sit after the
  mineral block are no longer miscounted as mineral concentrations.
- `reaction_network` species masks now correctly cover the aqueous + mineral
  species and are consistent with the 4-parameters-per-mineral scheme.

### Performance

- **Narrow Newton solve** in `RtZone` / `RiverZone`: each iteration solves only
  the aqueous species block (fused residual + analytic Jacobian) with the
  non-aqueous species held fixed as background, replacing the wider Newton
  step and reducing per-iteration cost.
- `get_hydro_steps()` now reads each column with a single `to_numpy()` call
  instead of per-step `.iloc` indexing, removing large numbers of slow pandas
  lookups on long simulations.
