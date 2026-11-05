"""
Persistence and restoration of the East River lateral-model
calibration GUI state.

A run saved from the GUI (the *Save run* button) carries, besides the
standard run outputs of `potions.result_saving`, a ``gui_state.json``
file in the run directory with the full state of the GUI at save time:
the series CSV path, the value and bounds of every parameter (including
bounds the user has edited), the plot selections, and the window
geometry.  At startup the GUI looks for the most recent saved run
directory under the ``results`` directory of the current working
directory and restores from it:

* a run saved by this GUI (a ``gui_state.json`` is present) restores
  everything exactly, and the figures are built straight from the
  run's saved simulation CSVs, so the model is not re-run;
* an older run without the file restores the parameter values
  best-effort from the saved ``parameters.csv`` files (the values are
  4-decimal in the CSV, so the restore is slightly lossy, and the
  initial-state parameters are absent from the CSV, so they fall back
  to their defaults); the figures still come from the saved CSVs.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import potions as pt

from potions.gui.data import GuiInputData, PlotSelection, default_plot_selection

# ==== Constants ==== #
# The name of the GUI state file in a saved run directory.
GUI_STATE_FILENAME: Final[str] = "gui_state.json"
# The version of the gui_state.json format written and accepted here.
GUI_STATE_VERSION: Final[int] = 1
# The name pattern of the run directories `potions.result_saving`
# creates: a zero-padded timestamp and, when two runs landed in the
# same second, a counter suffix.
_RUN_DIR_NAME: Final[re.Pattern[str]] = re.compile(
    r"^run_(\d{8})_(\d{6})(?:_(\d+))?$"
)
# =================== #


# ==== Types ==== #
@dataclass(frozen=True, slots=True)
class GuiState:
    """
    The persisted state of the GUI at the time of a *Save run*.

    Attributes:
        csv_path: The path of the series CSV the session used, or
            None for the record shipped with the package.
        param_values: The value of every parameter, keyed by bounds
            key, in the bounds' key order.
        param_bounds: The bounds of every parameter (the user-edited
            bounds, or the defaults when none were edited).
        selection: The plot selections of the session.
        window_geometry: The window position and size, as a mapping
            with ``x`` / ``y`` / ``width`` / ``height`` keys; None when
            the state predates geometry persistence.
        splitter_sizes: The sizes of the main window's three
            splitter sections, in order; None when the state predates
            splitter persistence.
    """

    csv_path: str | None
    param_values: dict[str, float]
    param_bounds: dict[str, tuple[float, float]]
    selection: PlotSelection
    window_geometry: dict[str, int] | None
    splitter_sizes: list[int] | None

    def to_dict(self) -> dict[str, object]:
        """
        The JSON-serializable form of the state.

        Returns:
            A plain dict (JSON-ready) with the state's fields.
        """
        return {
            "version": GUI_STATE_VERSION,
            "csv_path": self.csv_path,
            "param_values": {
                name: float(value)
                for name, value in self.param_values.items()
            },
            "param_bounds": {
                name: [float(lo), float(hi)]
                for name, (lo, hi) in self.param_bounds.items()
            },
            "selection": {
                "species": list(self.selection.species),
                "zones": list(self.selection.zones),
                "minerals": list(self.selection.minerals),
            },
            "window_geometry": (
                dict(self.window_geometry)
                if self.window_geometry is not None
                else None
            ),
            "splitter_sizes": (
                list(self.splitter_sizes)
                if self.splitter_sizes is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GuiState:
        """
        Build the state from a parsed ``gui_state.json`` mapping.

        Args:
            data: The parsed JSON mapping.

        Returns:
            The restored `GuiState`.

        Raises:
            ValueError: Raised when the file's version is not
                `GUI_STATE_VERSION` or a required field is missing or
                has an unexpected shape.
        """
        version: object = data.get("version", GUI_STATE_VERSION)
        if version != GUI_STATE_VERSION:
            raise ValueError(
                f"Unsupported gui_state.json version {version!r} "
                f"(expected {GUI_STATE_VERSION})."
            )

        csv_path: object = data.get("csv_path")
        if csv_path is not None and not isinstance(csv_path, str):
            raise ValueError(
                f"gui_state.json 'csv_path' must be a string or null, "
                f"got {type(csv_path).__name__}."
            )

        return cls(
            csv_path=csv_path,
            param_values=_require_values(data),
            param_bounds=_require_bounds(data),
            selection=_require_selection(data),
            window_geometry=_optional_geometry(data),
            splitter_sizes=_optional_splitter_sizes(data),
        )
# =============== #


# ==== Functions ==== #
def _require_values(data: dict[str, object]) -> dict[str, float]:
    """
    The ``param_values`` field of a gui_state.json mapping.

    Args:
        data: The parsed JSON mapping.

    Returns:
        The mapping of parameter key to value.

    Raises:
        ValueError: Raised when the field is missing, empty, or
            carries a non-numeric entry.
    """
    raw: object = data.get("param_values")
    if not isinstance(raw, dict) or len(raw) == 0:
        raise ValueError("gui_state.json is missing the 'param_values' mapping.")
    values: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or _not_number(value):
            raise ValueError(
                f"gui_state.json param_values entry {name!r} is not a number."
            )
        values[name] = float(value)
    return values


def _require_bounds(data: dict[str, object]) -> dict[str, tuple[float, float]]:
    """
    The ``param_bounds`` field of a gui_state.json mapping.

    Args:
        data: The parsed JSON mapping.

    Returns:
        The mapping of parameter key to (min, max) bound.

    Raises:
        ValueError: Raised when the field is missing, empty, or
            carries an entry that is not a two-number sequence.
    """
    raw: object = data.get("param_bounds")
    if not isinstance(raw, dict) or len(raw) == 0:
        raise ValueError("gui_state.json is missing the 'param_bounds' mapping.")
    bounds: dict[str, tuple[float, float]] = {}
    for name, value in raw.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, (list, tuple))
            or len(value) != 2
            or _not_number(value[0])
            or _not_number(value[1])
        ):
            raise ValueError(
                f"gui_state.json param_bounds entry {name!r} "
                f"is not a two-number sequence."
            )
        bounds[name] = (float(value[0]), float(value[1]))
    return bounds


def _require_selection(data: dict[str, object]) -> PlotSelection:
    """
    The ``selection`` field of a gui_state.json mapping.

    Args:
        data: The parsed JSON mapping.

    Returns:
        The restored `PlotSelection`.

    Raises:
        ValueError: Raised when the field is missing or one of the
            three selection lists is not a list of strings.
    """
    raw: object = data.get("selection")
    if not isinstance(raw, dict):
        raise ValueError("gui_state.json is missing the 'selection' mapping.")

    def _strings(key: str) -> list[str]:
        entry: object = raw.get(key)
        if not isinstance(entry, list) or not all(
            isinstance(item, str) for item in entry
        ):
            raise ValueError(
                f"gui_state.json selection field {key!r} is not a "
                f"list of strings."
            )
        return list(entry)

    return PlotSelection(
        species=_strings("species"),
        zones=_strings("zones"),
        minerals=_strings("minerals"),
    )


def _optional_geometry(data: dict[str, object]) -> dict[str, int] | None:
    """
    The optional ``window_geometry`` field of a gui_state.json mapping.

    Args:
        data: The parsed JSON mapping.

    Returns:
        The geometry mapping (``x`` / ``y`` / ``width`` / ``height``)
        or None.

    Raises:
        ValueError: Raised when the field is present but is not a
            mapping of integers.
    """
    raw: object = data.get("window_geometry")
    if raw is None:
        return None
    if not isinstance(raw, dict) or not all(
        isinstance(key, str) and _is_int(value) for key, value in raw.items()
    ):
        raise ValueError(
            "gui_state.json 'window_geometry' must be a mapping of integers."
        )
    return {key: int(value) for key, value in raw.items()}


def _optional_splitter_sizes(data: dict[str, object]) -> list[int] | None:
    """
    The optional ``splitter_sizes`` field of a gui_state.json mapping.

    Args:
        data: The parsed JSON mapping.

    Returns:
        The list of splitter section sizes or None.

    Raises:
        ValueError: Raised when the field is present but is not a
            list of integers.
    """
    raw: object = data.get("splitter_sizes")
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(
        _is_int(item) for item in raw
    ):
        raise ValueError(
            "gui_state.json 'splitter_sizes' must be a list of integers."
        )
    return [int(item) for item in raw]


def _not_number(value: object) -> bool:
    """Whether a JSON value is not a (non-boolean) number."""
    return not isinstance(value, (int, float)) or isinstance(value, bool)


def _is_int(value: object) -> bool:
    """Whether a JSON value is an integer (a boolean is not)."""
    return isinstance(value, int) and not isinstance(value, bool)


def write_gui_state(run_dir: Path, state: GuiState) -> Path:
    """
    Write the GUI state into a saved run directory.

    The state is written as ``gui_state.json`` directly in the run
    directory, alongside the ``hydro`` and ``rt`` output
    subdirectories of `potions.result_saving.save_model_run`.

    Args:
        run_dir: The run directory to write the state file into.
        state: The state to persist.

    Returns:
        The path of the written state file.
    """
    path: Path = run_dir / GUI_STATE_FILENAME
    path.write_text(json.dumps(state.to_dict(), indent=2) + "\n")
    return path


def read_gui_state(run_dir: Path) -> GuiState | None:
    """
    Read the GUI state from a saved run directory.

    Args:
        run_dir: The run directory to read the state file from.

    Returns:
        The restored state, or None when the file is absent (a run
        saved by an older version of the GUI or by the command-line
        tools) or malformed (a note is printed to the standard error
        in that case; a bad file must never prevent the GUI from
        starting).
    """
    path: Path = run_dir / GUI_STATE_FILENAME
    if not path.is_file():
        return None
    try:
        data: object = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("the top-level JSON value is not an object")
        return GuiState.from_dict(data)
    except (ValueError, TypeError, OSError) as error:
        print(
            f"Could not restore the GUI state from {path}: {error}",
            file=sys.stderr,
        )
        return None


def find_latest_run_dir(results_dir: str | Path = "results") -> Path | None:
    """
    The most recent saved run directory under the results directory.

    Runs are the ``run_<YYYYMMDD_HHMMSS>[_<counter>]`` directories
    created by `potions.result_saving`; the most recent one is the
    one with the latest timestamp (ties broken by the counter).

    Args:
        results_dir: The results directory to scan; defaults to
            ``results`` relative to the current working directory, the
            run saver's built-in default.

    Returns:
        The path of the most recent run directory, or None when the
        directory does not exist or contains no run directories.
    """
    base: Path = Path(results_dir)
    if not base.is_dir():
        return None

    candidates: list[tuple[str, int, Path]] = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        match: re.Match[str] | None = _RUN_DIR_NAME.match(entry.name)
        if match is None:
            continue
        stamp: str = f"{match.group(1)}{match.group(2)}"
        counter_part: str | None = match.group(3)
        counter: int = int(counter_part) if counter_part is not None else 0
        candidates.append((stamp, counter, entry))

    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][2]


@dataclass(frozen=True, slots=True)
class SavedRun:
    """
    The results and GUI state restored from a saved run directory.

    Attributes:
        results: The model results reconstructed from the run's saved
            simulation CSVs (see `load_saved_run`).
        param_values: The parameter values to start from.
        param_bounds: The parameter bounds to start from (the
            user-edited bounds when the state file carried them, the
            defaults otherwise).
        selection: The plot selections to start from.
        window_geometry: The window position and size to restore, as
            a mapping with ``x`` / ``y`` / ``width`` / ``height`` keys;
            None when the state did not carry them.
        splitter_sizes: The sizes of the main window's three
            splitter sections to restore, in order; None when the
            state did not carry them.
        restore_note: A note for the status bar explaining how the
            state was restored; None for a full gui_state.json restore.
    """

    results: pt.ModelResults
    param_values: dict[str, float]
    param_bounds: dict[str, tuple[float, float]]
    selection: PlotSelection
    window_geometry: dict[str, int] | None
    splitter_sizes: list[int] | None
    restore_note: str | None
# =============== #


# ==== Saved-run restoration ==== #
def load_saved_run(run_dir: Path, input_data: GuiInputData) -> SavedRun:
    """
    Restore the GUI state and the model results from a saved run.

    When the run carries a ``gui_state.json`` (a run saved by this
    GUI), the parameter values, the user-edited bounds, and the plot
    selections are restored exactly.  Otherwise (a run saved by an
    older version of the GUI or by the command-line tools) the
    parameter values are recovered best-effort from the saved
    ``parameters.csv`` files: the values in that CSV are 4-decimal, so
    the restore is slightly lossy, and the initial-state parameters
    are absent from it, so those fall back to their default (midpoint)
    values.  In both cases the results are reconstructed from the
    run's saved simulation CSVs, so the model is not re-run and the
    figures are available immediately.

    Args:
        run_dir: The run directory to restore from.
        input_data: The assembled GUI inputs (the parameter bounds and
            the reaction network, which the parameter mapping needs).

    Returns:
        The restored results and GUI state.

    Raises:
        ValueError: Raised when the run directory is incomplete (a
            missing simulation CSV or parameters CSV) or carries a
            parameter value it cannot parse.
    """
    state: GuiState | None = read_gui_state(run_dir)
    results: pt.ModelResults = _reconstruct_results(run_dir, input_data)

    if state is not None:
        return SavedRun(
            results=results,
            param_values=dict(state.param_values),
            param_bounds=dict(state.param_bounds),
            selection=state.selection,
            window_geometry=dict(state.window_geometry)
            if state.window_geometry is not None
            else None,
            splitter_sizes=list(state.splitter_sizes)
            if state.splitter_sizes is not None
            else None,
            restore_note=None,
        )

    parsed: tuple[dict[str, float], str] = _parse_saved_parameters(
        run_dir, input_data
    )
    values: dict[str, float] = parsed[0]
    note: str = parsed[1]
    return SavedRun(
        results=results,
        param_values=values,
        param_bounds=dict(input_data.bounds),
        selection=default_plot_selection(),
        window_geometry=None,
        splitter_sizes=None,
        restore_note=note,
    )


def _read_simulation_csv(path: Path) -> pd.DataFrame:
    """
    Read a saved simulation CSV.

    Args:
        path: The path of the simulation CSV.

    Returns:
        The simulation frame, indexed by date.

    Raises:
        ValueError: Raised when the file does not exist.
    """
    if not path.is_file():
        raise ValueError(f"Saved run is incomplete: {path} not found.")
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def _read_hydro_objectives(path: Path) -> pd.Series:
    """
    Read a saved hydrology objective-functions CSV.

    The file is the hydrology objective Series written to a one-column
    frame (a ``value`` column); a missing file yields an empty Series
    (the run was not scored against measured streamflow).

    Args:
        path: The path of the objective-functions CSV.

    Returns:
        The objective values as a Series, keyed by metric name.

    Raises:
        ValueError: Raised when the file exists but carries no
            ``value`` column.
    """
    if not path.is_file():
        return pd.Series(dtype=float)
    df: pd.DataFrame = pd.read_csv(path, index_col=0)
    if "value" not in df.columns:
        raise ValueError(
            f"Unrecognized hydrology objective-functions CSV {path}: "
            f"expected a 'value' column."
        )
    return df["value"].astype(float)


def _read_rt_objectives(path: Path) -> pd.DataFrame | None:
    """
    Read a saved reactive-transport objective-functions CSV.

    Args:
        path: The path of the objective-functions CSV.

    Returns:
        The objective frame (one row per measured species) or None
        when the file is absent (the run was not scored against
        measured river concentrations).
    """
    if not path.is_file():
        return None
    return pd.read_csv(path, index_col=0)


def _reconstruct_results(run_dir: Path, input_data: GuiInputData) -> pt.ModelResults:
    """
    Rebuild the model results from a run's saved simulation CSVs.

    The hydrology results carry the session's forcing (the one the
    saved run was made with; the record is part of the package) and
    the saved simulation frame; the reactive-transport results reuse
    the hydrology frame's date index, as the model does when it runs
    the two models in sequence.

    Args:
        run_dir: The run directory to read the simulation CSVs from.
        input_data: The assembled GUI inputs (for the forcing).

    Returns:
        The reconstructed results.

    Raises:
        ValueError: Raised when either simulation CSV is missing or an
            objective-functions CSV is unparseable.
    """
    hydro_sim: pd.DataFrame = _read_simulation_csv(
        run_dir / "hydro" / "simulation.csv"
    )
    rt_sim: pd.DataFrame = _read_simulation_csv(run_dir / "rt" / "simulation.csv")
    hydro_objectives: pd.Series = _read_hydro_objectives(
        run_dir / "hydro" / "objective_functions.csv"
    )
    rt_objectives: pd.DataFrame | None = _read_rt_objectives(
        run_dir / "rt" / "objective_functions.csv"
    )

    hydro_results: pt.HydroModelResults = pt.HydroModelResults(
        forcing=input_data.forc,
        simulation=hydro_sim,
        objective_functions=hydro_objectives,
        saved_dir=str(run_dir),
    )
    rt_results: pt.RtModelResults = pt.RtModelResults(
        simulation=rt_sim,
        objective_functions=rt_objectives,
        saved_dir=str(run_dir),
    )
    return pt.ModelResults(hydro=hydro_results, reactive_transport=rt_results)


def _parse_saved_parameters(
    run_dir: Path, input_data: GuiInputData
) -> tuple[dict[str, float], str]:
    """
    Recover the parameter values from the parameters CSV of a saved
    run that carries no ``gui_state.json``.

    The CSV is the model's full ``describe_parameters`` table: the
    hydrology rows (``zone`` is the zone name and ``name`` the
    parameter), the reactive-transport storage rows, the per-mineral
    rows (``name`` is the mineral's label and field, e.g.
    ``SOC(s).sw_threshold``), the river-zone rows, and the catchment
    proportion rows (``zone`` is ``model``).  Each row is mapped onto
    a bounds key and only keys present in `input_data.bounds` are
    kept, which drops the river-zone rows uniformly.  The values are
    4-decimal in the CSV, so the restore is slightly lossy; the
    initial-state parameters are absent from the CSV, so their keys
    fall back to the default (midpoint) values.

    Args:
        run_dir: The run directory to parse (the ``hydro``
            ``parameters.csv`` is used, with the ``rt`` copy as a
            fallback).
        input_data: The assembled GUI inputs (the bounds and the
            network, for the mineral-label mapping).

    Returns:
        The recovered values, in the bounds' key order, and a note
        for the status bar describing the lossy restore.

    Raises:
        ValueError: Raised when neither ``hydro/parameters.csv`` nor
            ``rt/parameters.csv`` exists, or a kept row carries a
            value that cannot be parsed as a number.
    """
    params_path: Path | None = None
    for sub in ("hydro", "rt"):
        candidate: Path = run_dir / sub / "parameters.csv"
        if candidate.is_file():
            params_path = candidate
            break
    if params_path is None:
        raise ValueError(
            f"Saved run is incomplete: no parameters.csv in {run_dir} "
            f"(looked in hydro/ and rt/)."
        )

    df: pd.DataFrame = pd.read_csv(params_path)
    mineral_names: list[str] = list(input_data.network.mineral_names)
    mineral_index: dict[str, int] = {
        name: i for i, name in enumerate(mineral_names)
    }
    bounds: dict[str, tuple[float, float]] = input_data.bounds

    recovered: dict[str, float] = {}
    found: int = 0
    for row in df.itertuples(index=False):
        key: str | None = _row_key(row.zone, row.name, mineral_index)
        if key is None or key not in bounds:
            continue
        try:
            values_row: float = float(str(row.value))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Could not parse the value {row.value!r} of the "
                f"parameter row (zone={row.zone!r}, name={row.name!r}) "
                f"in {params_path}."
            ) from error
        recovered[key] = values_row
        found += 1

    # Default (midpoint) values for the keys the CSV does not carry
    # (the initial-state parameters), in the bounds' key order.
    values: dict[str, float] = {}
    for name, (lo, hi) in bounds.items():
        if name in recovered:
            values[name] = recovered[name]
        else:
            values[name] = float(lo + hi) / 2.0 if lo != hi else float(lo)

    note: str = (
        f"Restored parameters from the saved run's parameters.csv "
        f"({found} values, 4-decimal; initial-state parameters at "
        f"their defaults)."
    )
    return values, note


def _row_key(
    zone: object, name: object, mineral_index: dict[str, int]
) -> str | None:
    """
    Map a row of a saved parameters CSV onto a bounds key.

    The hydrology and reactive-transport storage rows name the
    parameter directly (``name`` is the parameter); the per-mineral
    rows name it as ``{label}.{field}`` (the mineral's label and the
    field) and map to ``{zone}.{field}[{mineral_id}]`` by the label's
    position in the network's mineral names; the catchment proportion
    rows carry ``zone`` equal to ``model`` and are keyed by their
    ``name`` alone; the river-zone rows are not part of the
    calibrated array and map to None.

    Args:
        zone: The row's ``zone`` field.
        name: The row's ``name`` field.
        mineral_index: The mapping of mineral label to network index.

    Returns:
        The bounds key of the row, or None when the row is not part
        of the calibrated array (the river zone) or names an
        unrecognized mineral (the row is skipped, so an unfamiliar
        model layout never blocks a restore).
    """
    zone_name: str = str(zone)
    name_str: str = str(name)

    if zone_name == "model":
        return name_str
    if zone_name == "river":
        return None
    if "." not in name_str:
        return f"{zone_name}.{name_str}"

    label, field = name_str.split(".", 1)
    if label not in mineral_index:
        return None
    return f"{zone_name}.{field}[{mineral_index[label]}]"
# =================== #
