"""
Persistence and restoration of the East River lateral-model
calibration GUI state.

A run saved from the GUI (the *Save run* button) carries, besides the
standard run outputs of `potions.result_saving`, a ``gui_state.json``
file in the run directory with the full state of the GUI at save time:
the series CSV path, the value and bounds of every parameter (including
bounds the user has edited), the plot selections, the run mode, and the
window geometry.  At startup the GUI looks for the most recent saved run
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

The parameter panel's *History* group lists every run directory
`list_run_dirs` finds (newest first) beside the session's own runs,
so any saved run can be restored without a re-run, not only the
most recent one.

Besides the per-run state, the GUI keeps an abbreviated session
history: with the *Save run history* checkbox on, every finished run
appends one JSON object to a ``gui_run_history.jsonl`` file in the
results directory, holding the timestamp, the run mode, the
parameter values, and the objective values (see
`append_run_history`).
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import potions as pt

from potions.gui.common import TIME_SERIES_OPTIONS
from potions.gui.data import (
    GuiInputData,
    GuiRunResults,
    PlotSelection,
    default_plot_selection,
)

# ==== Constants ==== #
# The name of the GUI state file in a saved run directory.
GUI_STATE_FILENAME: Final[str] = "gui_state.json"
# The version of the gui_state.json format written and accepted here.
GUI_STATE_VERSION: Final[int] = 1
# The name of the session's abbreviated run-history file, kept in
# the results directory: one JSON object per finished run (see
# `append_run_history`).
RUN_HISTORY_FILENAME: Final[str] = "gui_run_history.jsonl"
# The two run modes: "full" runs the hydrology and
# reactive-transport models together; "hydro" runs only the
# hydrology model (the GUI's hydrology-only mode).
MODE_FULL: Final[str] = "full"
MODE_HYDRO: Final[str] = "hydro"
# The set of the valid run modes.
_VALID_MODES: Final[frozenset[str]] = frozenset({MODE_FULL, MODE_HYDRO})
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
        splitter_sizes: The sizes of the window's two splitters, in
            order: the figure and the lower (table-and-parameters)
            row of the main vertical splitter, then the objective-
            table and parameter-panel widths of the inner horizontal
            splitter; None when the state predates splitter
            persistence.
        save_history: Whether the *Save run history* checkbox was
            on in the session; an absent field in a state file
            means a state saved before the toggle existed, which
            restores it off.
        mode: The run mode of the session (`MODE_FULL` or
            `MODE_HYDRO`); an absent field in a state file means a
            state saved before the mode existed, which restores
            full mode.
    """

    csv_path: str | None
    param_values: dict[str, float]
    param_bounds: dict[str, tuple[float, float]]
    selection: PlotSelection
    window_geometry: dict[str, int] | None
    splitter_sizes: list[int] | None
    save_history: bool = False
    mode: str = MODE_FULL

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
                "time_series": list(self.selection.time_series),
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
            "save_history": self.save_history,
            "mode": self.mode,
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
            save_history=_optional_save_history(data),
            mode=_optional_mode(data),
        )


@dataclass(frozen=True, slots=True)
class RunHistoryEntry:
    """
    One finished run, in the abbreviated form the session's
    run history stores.

    The history is the ``gui_run_history.jsonl`` file in the results
    directory, one JSON object per line (see `append_run_history`),
    kept so the GUI can list previously used parameter sets and the
    objective values they produced.

    Attributes:
        timestamp: When the run finished, as an ISO 8601 UTC
            timestamp with second precision.
        mode: The run's mode, which models it ran: `MODE_FULL`
            runs the hydrology and reactive-transport models
            together, `MODE_HYDRO` the hydrology model alone.
        param_values: The parameter values the run used, keyed by
            bounds key in the bounds' key order.
        objectives: The objective-function values the run reported:
            one mapping per measured target (the streamflow and the
            river species), each mapping metric name to value.
    """

    timestamp: str
    mode: str
    param_values: dict[str, float]
    objectives: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        """
        The JSON-serializable form of the entry.

        Returns:
            A plain dict (JSON-ready) with the entry's fields.
        """
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "param_values": {
                name: float(value)
                for name, value in self.param_values.items()
            },
            "objectives": {
                target: {
                    metric: float(value)
                    for metric, value in metrics.items()
                }
                for target, metrics in self.objectives.items()
            },
        }
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
            selection lists (the time-series list is optional; an
            absent one keeps every series on) is not a list of
            strings.
    """
    raw: object = data.get("selection")
    if not isinstance(raw, dict):
        raise ValueError("gui_state.json is missing the 'selection' mapping.")

    def _strings(
        key: str, optional_default: Sequence[str] | None = None
    ) -> list[str]:
        entry: object = raw.get(key)
        # An absent optional field falls back to its default: the
        # time-series list of a state saved before the time-series
        # checkboxes existed means every series stays on.
        if entry is None and optional_default is not None:
            return list(optional_default)
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
        time_series=_strings(
            "time_series", optional_default=TIME_SERIES_OPTIONS
        ),
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

    The current layout writes four sizes (the two sections of the
    main vertical splitter, then the two of the inner horizontal
    one).  A state saved by the single-splitter layout writes three
    (figure, objective table, parameters stacked vertically) and is
    mapped onto the current layout: the figure keeps its height, and
    the table and the parameters move side by side into the lower
    row.

    Args:
        data: The parsed JSON mapping.

    Returns:
        The list of splitter sizes in the current four-size layout,
        or None.

    Raises:
        ValueError: Raised when the field is present but is not a
            list of three or four integers.
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
    values: list[int] = [int(item) for item in raw]
    if len(values) == 3:
        figure, objective, parameters = values
        return [figure, objective + parameters, objective, parameters]
    if len(values) == 4:
        return values
    raise ValueError(
        f"gui_state.json 'splitter_sizes' must hold three or four "
        f"sizes, got {len(values)}."
    )


def _optional_save_history(data: dict[str, object]) -> bool:
    """
    The optional ``save_history`` field of a gui_state.json mapping.

    Args:
        data: The parsed JSON mapping.

    Returns:
        The field's boolean value, or False when it is absent (a
        state saved before the toggle existed restores the checkbox
        off).

    Raises:
        ValueError: Raised when the field is present but is not a
            boolean.
    """
    raw: object = data.get("save_history")
    if raw is None:
        return False
    if not isinstance(raw, bool):
        raise ValueError(
            f"gui_state.json 'save_history' must be a boolean, "
            f"got {type(raw).__name__}."
        )
    return raw


def _optional_mode(data: dict[str, object]) -> str:
    """
    The optional ``mode`` field of a gui_state.json mapping.

    Args:
        data: The parsed JSON mapping.

    Returns:
        The field's run mode, or `MODE_FULL` when it is absent (a
        state saved before the mode existed restores full mode).

    Raises:
        ValueError: Raised when the field is present but is not a
            valid mode string.
    """
    raw: object = data.get("mode")
    if raw is None:
        return MODE_FULL
    if not isinstance(raw, str) or raw not in _VALID_MODES:
        raise ValueError(
            f"gui_state.json 'mode' must be 'full' or 'hydro', "
            f"got {raw!r}."
        )
    return raw


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


def list_run_dirs(results_dir: str | Path = "results") -> list[Path]:
    """
    The saved run directories under the results directory, newest
    first.

    Runs are the ``run_<YYYYMMDD_HHMMSS>[_<counter>]`` directories
    created by `potions.result_saving`; the list is sorted by the
    timestamp (ties broken by the counter), newest first.

    Args:
        results_dir: The results directory to scan; defaults to
            ``results`` relative to the current working directory, the
            run saver's built-in default.

    Returns:
        The run directory paths in newest-first order; empty when
        the directory does not exist or contains no run directories.
    """
    base: Path = Path(results_dir)
    if not base.is_dir():
        return []

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

    candidates.sort()
    return [path for _, _, path in reversed(candidates)]


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
    runs: list[Path] = list_run_dirs(results_dir)
    return runs[0] if runs else None


# ==== Run history ==== #
def append_run_history(
    results_dir: str | Path, entry: RunHistoryEntry
) -> Path:
    """
    Append one finished run to the session's run-history file.

    The history is the ``gui_run_history.jsonl`` file in the results
    directory (created on first use): each finished run appends one
    JSON object with the entry's fields, so the file grows by one
    line per run.  A failure to create or write the file is
    reported on the standard error and does not raise, so a bad
    history can never disturb a finished run.

    Args:
        results_dir: The results directory the history file lives
            in; defaults to ``results`` relative to the current
            working directory, the run saver's built-in default.
        entry: The finished run's abbreviated record.

    Returns:
        The path of the history file.
    """
    path: Path = Path(results_dir) / RUN_HISTORY_FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict()) + "\n")
    except OSError as error:
        print(
            f"Could not append to the run history: {error}",
            file=sys.stderr,
        )
    return path


def objectives_from_results(
    results: pt.ModelResults | GuiRunResults,
) -> dict[str, dict[str, float]]:
    """
    The objective values of a finished run, keyed by measured target.

    This is the source of a run-history entry's ``objectives`` field
    (see `append_run_history`): the streamflow objectives when the
    run was scored against measured streamflow, and one mapping per
    measured river species when the reactive transport model
    reported any.  A target with no reported objectives is omitted,
    as the on-screen table omits its column.  The hydrology-only
    mode's `GuiRunResults` carries no reactive-transport results, so
    its entry holds the streamflow objectives alone.

    Args:
        results: The finished run's results (a `pt.ModelResults` for
            a full-mode run, a `GuiRunResults` for a hydrology-only
            run).

    Returns:
        The mapping of target name to (metric name to value).
    """
    if isinstance(results, GuiRunResults):
        hydro_results: pt.HydroModelResults = results.hydro
        rt_results: pt.RtModelResults | None = results.rt
    else:
        hydro_results = results.hydro
        rt_results = results.reactive_transport

    objectives: dict[str, dict[str, float]] = {}
    hydro_objectives: pd.Series = hydro_results.objective_functions
    if len(hydro_objectives) > 0:
        objectives["Streamflow"] = {
            str(name): float(value)
            for name, value in hydro_objectives.items()
        }
    rt_objectives: pd.DataFrame | None = (
        rt_results.objective_functions if rt_results is not None else None
    )
    if rt_objectives is not None and len(rt_objectives) > 0:
        for species in rt_objectives.index:
            objectives[str(species)] = {
                str(metric): float(rt_objectives.loc[species, metric])
                for metric in rt_objectives.columns
            }
    return objectives


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
        splitter_sizes: The sizes of the window's two splitters to
            restore (the main vertical pair, then the inner
            horizontal pair); None when the state did not carry them.
        save_history: Whether the session's *Save run history*
            checkbox was on when the run was saved; False for a run
            saved before the toggle existed.
        mode: The run mode the saved run used (`MODE_FULL` or
            `MODE_HYDRO`); for a run saved before the mode existed,
            inferred from the saved outputs (hydrology-only when the
            run carried no reactive-transport results).
        restore_note: A note for the status bar explaining how the
            state was restored; None for a full gui_state.json restore.
    """

    results: GuiRunResults
    param_values: dict[str, float]
    param_bounds: dict[str, tuple[float, float]]
    selection: PlotSelection
    window_geometry: dict[str, int] | None
    splitter_sizes: list[int] | None
    save_history: bool
    mode: str
    restore_note: str | None
# =============== #


# ==== Saved-run restoration ==== #
def load_saved_run(run_dir: Path, input_data: GuiInputData) -> SavedRun:
    """
    Restore the GUI state and the model results from a saved run.

    When the run carries a ``gui_state.json`` (a run saved by this
    GUI), the parameter values, the user-edited bounds, the plot
    selections, the run mode, and the save-history toggle are
    restored exactly.  Otherwise (a run saved by an
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
    results: GuiRunResults = _reconstruct_results(run_dir, input_data)

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
            save_history=state.save_history,
            mode=state.mode,
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
        save_history=False,
        # A stateless run's mode is inferred from its saved outputs:
        # a run with no reactive-transport results is a hydrology-
        # only run.
        mode=MODE_HYDRO if results.rt is None else MODE_FULL,
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


def _reconstruct_results(run_dir: Path, input_data: GuiInputData) -> GuiRunResults:
    """
    Rebuild the model results from a run's saved simulation CSVs.

    Two layouts are recognized: a full-mode run carries its
    simulation CSVs in the run's ``hydro`` and ``rt`` subdirectories
    (the layout of `potions.result_saving.save_model_run`), and a
    hydrology-only run carries a single ``simulation.csv`` (and
    ``objective_functions.csv``) directly in the run directory (the
    layout of `potions.result_saving.save_hydro_run`), with no
    reactive-transport results.

    The hydrology results carry the session's forcing (the one the
    saved run was made with; the record is part of the package) and
    the saved simulation frame; the reactive-transport results reuse
    the hydrology frame's date index, as the model does when it runs
    the two models in sequence.

    Args:
        run_dir: The run directory to read the simulation CSVs from.
        input_data: The assembled GUI inputs (for the forcing).

    Returns:
        The reconstructed results, with no reactive-transport
        results for a hydrology-only run.

    Raises:
        ValueError: Raised when a simulation CSV is missing or an
            objective-functions CSV is unparseable.
    """
    hydro_sim: pd.DataFrame
    rt_sim: pd.DataFrame | None
    hydro_objectives: pd.Series
    rt_objectives: pd.DataFrame | None
    if (run_dir / "hydro" / "simulation.csv").is_file():
        hydro_sim = _read_simulation_csv(run_dir / "hydro" / "simulation.csv")
        rt_sim = _read_simulation_csv(run_dir / "rt" / "simulation.csv")
        hydro_objectives = _read_hydro_objectives(
            run_dir / "hydro" / "objective_functions.csv"
        )
        rt_objectives = _read_rt_objectives(
            run_dir / "rt" / "objective_functions.csv"
        )
    else:
        hydro_sim = _read_simulation_csv(run_dir / "simulation.csv")
        rt_sim = None
        hydro_objectives = _read_hydro_objectives(
            run_dir / "objective_functions.csv"
        )
        rt_objectives = None

    hydro_results: pt.HydroModelResults = pt.HydroModelResults(
        forcing=input_data.forc,
        simulation=hydro_sim,
        objective_functions=hydro_objectives,
        saved_dir=str(run_dir),
    )
    if rt_sim is not None:
        rt_results: pt.RtModelResults | None = pt.RtModelResults(
            simulation=rt_sim,
            objective_functions=rt_objectives,
            saved_dir=str(run_dir),
        )
    else:
        rt_results = None
    return GuiRunResults(hydro=hydro_results, rt=rt_results)


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
        run_dir: The run directory to parse (a ``parameters.csv`` at
            the run's top level, as a hydrology-only run saves it,
            is used first, then the ``hydro`` and ``rt`` copies).
        input_data: The assembled GUI inputs (the bounds and the
            network, for the mineral-label mapping).

    Returns:
        The recovered values, in the bounds' key order, and a note
        for the status bar describing the lossy restore.

    Raises:
        ValueError: Raised when no ``parameters.csv`` exists (at the
            run's top level, in ``hydro/``, or in ``rt/``), or a kept
            row carries a value that cannot be parsed as a number.
    """
    candidates: list[Path] = [
        run_dir / "parameters.csv",
        run_dir / "hydro" / "parameters.csv",
        run_dir / "rt" / "parameters.csv",
    ]
    params_path: Path | None = next(
        (candidate for candidate in candidates if candidate.is_file()),
        None,
    )
    if params_path is None:
        raise ValueError(
            f"Saved run is incomplete: no parameters.csv in {run_dir} "
            f"(looked in the run's top level, hydro/, and rt/)."
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

    # A hydrology-only run's top-level parameters.csv carries no
    # hydro-zone rows that map onto bounds keys (they are layer-
    # named), so those parameters fall back to their defaults along
    # with the initial-state ones.
    if params_path.parent == run_dir:
        note: str = (
            f"Restored parameters from the saved run's parameters.csv "
            f"({found} values, 4-decimal; hydro-zone and initial-"
            f"state parameters at their defaults)."
        )
    else:
        note = (
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
