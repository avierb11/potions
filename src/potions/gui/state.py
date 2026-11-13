"""
Persistence and restoration of the East River calibration GUI state.

A run saved from the GUI (the *Save run* button) carries, besides the
standard run outputs of `potions.result_saving`, a ``gui_state.json``
file in the run directory with the full state of the GUI at save time:
the series CSV path, the selected model, the value and bounds of every
parameter (including bounds the user has edited), the plot selections,
the run mode, and the window geometry.  At startup the GUI looks for
the most recent saved run directory under the ``results`` directory of
the current working directory and restores from it:

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

from potions.gui.common import RIVER_SPECIES, WATER_YEARS, water_year_window
from potions.gui.data import (
    DEFAULT_WINDOW_END,
    DEFAULT_WINDOW_START,
    GuiInputData,
    GuiRunResults,
    PlotSelection,
    default_plot_mineral,
    default_plot_selection,
)
from potions.gui.model_specs import (
    DEFAULT_MODEL_KEY,
    get_model_spec,
    model_key_for_class_name,
)

# ==== Constants ==== #
# The name of the GUI state file in a saved run directory.
GUI_STATE_FILENAME: Final[str] = "gui_state.json"
# The name of the run's network file in a saved run directory: a
# human-editable YAML copy of the reaction network the run used (see
# `potions.reactive_transport.network_file`).
NETWORK_FILENAME: Final[str] = "network.yaml"
# The version of the gui_state.json format written here.
GUI_STATE_VERSION: Final[int] = 9
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
VALID_MODES: Final[frozenset[str]] = frozenset({MODE_FULL, MODE_HYDRO})
# The versions of the gui_state.json format accepted on restore:
# the current one (9, the dock-layout blob), version 8 (the
# maximized-window flag), version 7 (the run's network fingerprint),
# version 6 (the date-based plotting window), version 5 (the selected
# model's key), version 4 (the objective-table selection), version 3
# (the time-series figure's selection list), version 2 (the last
# version this GUI has written), and version 1 (the two-splitter
# layout, whose splitter sizes are mapped onto the current one).
ACCEPTED_VERSIONS: Final[frozenset[int]] = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 8, GUI_STATE_VERSION}
)
# The name pattern of the run directories `potions.result_saving`
# creates: a zero-padded timestamp and, when two runs landed in the
# same second, a counter suffix.
RUN_DIR_NAME: Final[re.Pattern[str]] = re.compile(
    r"^run_(\d{8})_(\d{6})(?:_(\d+))?$"
)
# =================== #


# ==== Types ==== #
@dataclass(frozen=True, slots=True)
class GuiState:
    """
    The persisted state of the GUI at the time of a *Save run*.

    Attributes:
        csv_path: The path of the record file (a CSV or XLSX
            spreadsheet) the session used, or None for the record
            shipped with the package.
        param_values: The value of every parameter, keyed by bounds
            key, in the bounds' key order.
        param_bounds: The bounds of every parameter (the user-edited
            bounds, or the defaults when none were edited).
        selection: The plot selections of the session.
        window_geometry: The window position and size, as a mapping
            with ``x`` / ``y`` / ``width`` / ``height`` keys; None when
            the state predates geometry persistence.
        splitter_sizes: The sizes of the main vertical splitter, in
            order: the figure and the lower (table-and-parameters)
            row; None when the state predates splitter
            persistence. Accepted from state files of version 8 or
            older; version 9 writes null because the dockable figure
            panels replaced that splitter.
        save_history: Whether the *Save run history* checkbox was
            on in the session; an absent field in a state file
            means a state saved before the toggle existed, which
            restores it off.
        mode: The run mode of the session (`MODE_FULL` or
            `MODE_HYDRO`); an absent field in a state file means a
            state saved before the mode existed, which restores
            full mode.
        objective_terms: The cells of the objective table the session
            had checked, as ``"<target>: <metric>"`` labels; None
            when the session made no explicit selection (an unset
            selection searches every selectable term) or the state
            predates the selection.
        model_key: The key of the model spec the session ran; an
            absent field in a state file means a state saved before
            the model selector existed, which restores the lateral
            three-layer model.
        network_fingerprint: The fingerprint of the reaction network
            the session ran; an absent field in a state file means a
            state saved before network persistence existed, which
            restores the default network.
        column_map: The role-to-column mapping the session used to
            read the record, or None when the record was read with
            auto-detected roles (the shipped record always is); an
            absent field in a state file means a state saved before
            mapping persistence existed, which restores None.
        window_maximized: Whether the window was maximized when the
            run was saved; restoring such a run opens it maximized.
            An absent field in a state file means a state saved
            before the maximized flag existed, which restores a
            normal window.
        dock_state: The base64 encoding of the main window's
            `saveState()` dock layout (the per-figure dock panels'
            placement, sizing, and tabbing); None means the session
            recorded no dock layout, which restores the window's
            default layout.
    """

    csv_path: str | None
    param_values: dict[str, float]
    param_bounds: dict[str, tuple[float, float]]
    selection: PlotSelection
    window_geometry: dict[str, int] | None
    splitter_sizes: list[int] | None
    save_history: bool = False
    mode: str = MODE_FULL
    objective_terms: list[str] | None = None
    model_key: str = DEFAULT_MODEL_KEY
    network_fingerprint: str | None = None
    column_map: dict[str, str | None] | None = None
    window_maximized: bool = False
    dock_state: str | None = None

    def to_dict(self) -> dict[str, object]:
        """
        The JSON-serializable form of the state.

        The ``column_map`` key is present only when the session read
        the record with an explicit mapping: its absence marks an
        old-format state (a record read with auto-detected roles).

        Returns:
            A plain dict (JSON-ready) with the state's fields.
        """
        data: dict[str, object] = {
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
                "mineral": self.selection.mineral,
                "streamflow": self.selection.streamflow,
                "window_start": self.selection.window_start,
                "window_end": self.selection.window_end,
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
            "objective_terms": (
                list(self.objective_terms)
                if self.objective_terms is not None
                else None
            ),
            "model_key": self.model_key,
            "network_fingerprint": self.network_fingerprint,
            "window_maximized": self.window_maximized,
            "dock_state": self.dock_state,
        }
        if self.column_map is not None:
            data["column_map"] = dict(self.column_map)
        return data

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
        years: list[int] = WATER_YEARS,
    ) -> GuiState:
        """
        Build the state from a parsed ``gui_state.json`` mapping.

        Args:
            data: The parsed JSON mapping.
            years: The calendar years of the record the state is
                restored against; a version-5 state's water years
                are validated against it (the site record's water
                years, when no record is known).

        Returns:
            The restored `GuiState`.

        Raises:
            ValueError: Raised when the file's version is not one of
                the accepted versions (`ACCEPTED_VERSIONS`) or a
                required field is missing or has an unexpected
                shape.
        """
        version: object = data.get("version", GUI_STATE_VERSION)
        if version not in ACCEPTED_VERSIONS:
            raise ValueError(
                f"Unsupported gui_state.json version {version!r} "
                f"(expected {sorted(ACCEPTED_VERSIONS)})."
            )

        csv_path: object = data.get("csv_path")
        if csv_path is not None and not isinstance(csv_path, str):
            raise ValueError(
                f"gui_state.json 'csv_path' must be a string or null, "
                f"got {type(csv_path).__name__}."
            )

        return cls(
            csv_path=csv_path,
            param_values=require_values(data),
            param_bounds=require_bounds(data),
            selection=require_selection(data, years),
            window_geometry=optional_geometry(data),
            splitter_sizes=optional_splitter_sizes(data),
            save_history=optional_save_history(data),
            mode=optional_mode(data),
            objective_terms=optional_objective_terms(data),
            model_key=optional_model_key(data),
            network_fingerprint=optional_network_fingerprint(data),
            column_map=optional_column_map(data),
            window_maximized=optional_maximized(data),
            dock_state=optional_dock_state(data),
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
        model_key: The key of the model spec the run used; an absent
            field in a history line means a run recorded before the
            model selector existed, which belongs to the lateral
            three-layer model.
        network_fingerprint: The fingerprint of the run's reaction
            network; an absent field in a history line means a run
            recorded before network persistence existed, which
            belongs to the default network.
    """

    timestamp: str
    mode: str
    param_values: dict[str, float]
    objectives: dict[str, dict[str, float]]
    model_key: str = DEFAULT_MODEL_KEY
    network_fingerprint: str | None = None

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
            "model_key": self.model_key,
            "network_fingerprint": self.network_fingerprint,
        }
# =============== #


# ==== Functions ==== #
def require_values(data: dict[str, object]) -> dict[str, float]:
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
        if not isinstance(name, str) or not_number(value):
            raise ValueError(
                f"gui_state.json param_values entry {name!r} is not a number."
            )
        values[name] = float(value)
    return values


def require_bounds(data: dict[str, object]) -> dict[str, tuple[float, float]]:
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
            or not_number(value[0])
            or not_number(value[1])
        ):
            raise ValueError(
                f"gui_state.json param_bounds entry {name!r} "
                f"is not a two-number sequence."
            )
        bounds[name] = (float(value[0]), float(value[1]))
    return bounds


def require_selection(
    data: dict[str, object],
    years: list[int] = WATER_YEARS,
) -> PlotSelection:
    """
    The ``selection`` field of a gui_state.json mapping.

    A mapping saved by the current version carries the current
    shape: the shared river species, the zones, the single
    rate-figure mineral, the streamflow toggle, and the date-based
    plotting window.  A version-5 mapping carries the old water-year
    window instead and is migrated by converting the two years to
    the water-year span's first and last days.  A mapping saved by
    version 3 or earlier carries the previous shape — the zone
    figure's species, the time-series figure's streamflow and
    river-species list, and a list of minerals — and is migrated onto
    the current shape: the species are the river species that either
    of the old figures had on (in the panel's order), the mineral is
    the first of the old list (the dominant SOC mineral when the
    list is absent), the streamflow axis is on when the old
    time-series list carried it (every series stays on when the list
    is absent, as before), and the plotting window is the whole
    record.

    Args:
        data: The parsed JSON mapping.
        years: The calendar years of the record the state is
            restored against; a version-5 state's water years must
            be members of it (the site record's water years, when
            no record is known).

    Returns:
        The restored `PlotSelection`.

    Raises:
        ValueError: Raised when the field is missing or a selection
            entry is not of its expected type (a string list, a
            string, a boolean, a date, or a water year).
    """
    raw: object = data.get("selection")
    if not isinstance(raw, dict):
        raise ValueError("gui_state.json is missing the 'selection' mapping.")

    def strings(
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

    def read_bool(key: str, default: bool) -> bool:
        entry: object = raw.get(key)
        if entry is None:
            return default
        if not isinstance(entry, bool):
            raise ValueError(
                f"gui_state.json selection field {key!r} is not a "
                f"boolean."
            )
        return entry

    def year(key: str, default: int) -> int:
        entry: object = raw.get(key)
        if entry is None:
            return default
        if not is_int(entry) or entry not in years:
            raise ValueError(
                f"gui_state.json selection field {key!r} is not a "
                f"water year."
            )
        return int(entry)

    def date(key: str, default: str) -> str:
        entry: object = raw.get(key)
        if entry is None:
            return default
        if not isinstance(entry, str):
            raise ValueError(
                f"gui_state.json selection field {key!r} is not a "
                f"date."
            )
        try:
            pd.Timestamp(entry)
        except ValueError as err:
            raise ValueError(
                f"gui_state.json selection field {key!r} is not a "
                f"date."
            ) from err
        return entry

    # The current shape carries a single `mineral` string; version 3
    # and earlier carry a `minerals` list and no `streamflow` or
    # plotting-window fields.
    if isinstance(raw.get("mineral"), str):
        # Version 5 stores the window as water years; the current
        # shape stores its first and last days.
        if "window_start" in raw or "window_end" in raw:
            start: str = date("window_start", DEFAULT_WINDOW_START)
            end: str = date("window_end", DEFAULT_WINDOW_END)
        elif "water_years_from" in raw or "water_years_to" in raw:
            legacy_start, legacy_end = water_year_window(
                year("water_years_from", years[0]),
                year("water_years_to", years[-1]),
            )
            start = legacy_start.strftime("%Y-%m-%d")
            end = legacy_end.strftime("%Y-%m-%d")
        else:
            start = DEFAULT_WINDOW_START
            end = DEFAULT_WINDOW_END
        if pd.Timestamp(start) > pd.Timestamp(end):
            raise ValueError(
                f"gui_state.json selection window starts on {start!r}, "
                f"after its end {end!r}."
            )
        return PlotSelection(
            species=strings("species"),
            zones=strings("zones"),
            mineral=str(raw["mineral"]),
            streamflow=read_bool("streamflow", True),
            window_start=start,
            window_end=end,
        )

    # The version 3 or earlier shape: migrate it onto the current
    # one (see the docstring for the mapping).
    legacy_species: list[str] = strings("species")
    legacy_time_series: list[str] = strings(
        "time_series", optional_default=["Streamflow", *RIVER_SPECIES]
    )
    legacy_minerals: list[str] = strings("minerals")
    return PlotSelection(
        species=[
            name
            for name in RIVER_SPECIES
            if name in legacy_species or name in legacy_time_series
        ],
        zones=strings("zones"),
        mineral=legacy_minerals[0]
        if legacy_minerals
        else default_plot_mineral(),
        streamflow="Streamflow" in legacy_time_series,
        window_start=DEFAULT_WINDOW_START,
        window_end=DEFAULT_WINDOW_END,
    )


def optional_geometry(data: dict[str, object]) -> dict[str, int] | None:
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
        isinstance(key, str) and is_int(value) for key, value in raw.items()
    ):
        raise ValueError(
            "gui_state.json 'window_geometry' must be a mapping of integers."
        )
    return {key: int(value) for key, value in raw.items()}


def optional_splitter_sizes(data: dict[str, object]) -> list[int] | None:
    """
    The optional ``splitter_sizes`` field of a gui_state.json mapping.

    The current layout writes two sizes (the main vertical
    splitter's sections: the figure and the lower
    table-and-parameters row).  The two-splitter layout of version 1
    wrote four (the two main sections, then the two of the inner
    horizontal one) and is mapped onto the current layout by
    keeping the first two; the single-splitter layout before that
    wrote three (figure, objective table, parameters stacked
    vertically) and is mapped by summing the lower two into the
    lower row.

    Args:
        data: The parsed JSON mapping.

    Returns:
        The list of splitter sizes in the current two-size layout,
        or None.

    Raises:
        ValueError: Raised when the field is present but is not a
            list of two, three, or four integers.
    """
    raw: object = data.get("splitter_sizes")
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(
        is_int(item) for item in raw
    ):
        raise ValueError(
            "gui_state.json 'splitter_sizes' must be a list of integers."
        )
    values: list[int] = [int(item) for item in raw]
    if len(values) == 2:
        return values
    if len(values) == 3:
        figure, objective, parameters = values
        return [figure, objective + parameters]
    if len(values) == 4:
        return values[:2]
    raise ValueError(
        f"gui_state.json 'splitter_sizes' must hold two, three, or "
        f"four sizes, got {len(values)}."
    )


def optional_save_history(data: dict[str, object]) -> bool:
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


def optional_maximized(data: dict[str, object]) -> bool:
    """
    The optional ``window_maximized`` field of a gui_state.json mapping.

    Args:
        data: The parsed JSON mapping.

    Returns:
        The field's boolean value, or False when it is absent (a state
        saved before the maximized flag existed restores a normal
        window).

    Raises:
        ValueError: Raised when the field is present but is not a
            boolean.
    """
    raw: object = data.get("window_maximized")
    if raw is None:
        return False
    if not isinstance(raw, bool):
        raise ValueError(
            f"gui_state.json 'window_maximized' must be a boolean, "
            f"got {type(raw).__name__}."
        )
    return raw


def optional_dock_state(data: dict[str, object]) -> str | None:
    """
    The optional ``dock_state`` field of a gui_state.json mapping.

    Args:
        data: The parsed JSON mapping.

    Returns:
        The field's base64 dock-layout string, or None when it is
        absent or null (a state saved before the dockable figure
        panels existed, or one that recorded no layout, restores the
        window's default layout).

    Raises:
        ValueError: Raised when the field is present but is neither a
            string nor null.
    """
    raw: object = data.get("dock_state")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(
            f"gui_state.json 'dock_state' must be a string or null, "
            f"got {type(raw).__name__}."
        )
    return raw


def optional_mode(data: dict[str, object]) -> str:
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
    if not isinstance(raw, str) or raw not in VALID_MODES:
        raise ValueError(
            f"gui_state.json 'mode' must be 'full' or 'hydro', "
            f"got {raw!r}."
        )
    return raw


def optional_objective_terms(data: dict[str, object]) -> list[str] | None:
    """
    The optional ``objective_terms`` field of a gui_state.json mapping.

    Args:
        data: The parsed JSON mapping.

    Returns:
        The field's list of objective terms, or None when it is
        absent or null (a state saved before the selection existed
        restores an unset selection, which searches every
        selectable term).

    Raises:
        ValueError: Raised when the field is present but is not a
            list of strings.
    """
    raw: object = data.get("objective_terms")
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(t, str) for t in raw):
        raise ValueError(
            f"gui_state.json 'objective_terms' must be a list of "
            f"objective terms, got {raw!r}."
        )
    return list(raw)


def optional_model_key(data: dict[str, object]) -> str:
    """
    The optional ``model_key`` field of a gui_state.json mapping.

    Args:
        data: The parsed JSON mapping.

    Returns:
        The field's model key, or `DEFAULT_MODEL_KEY` when it is
        absent (a state saved before the model selector existed
        restores the lateral three-layer model).

    Raises:
        ValueError: Raised when the field is present but is not a
            string or does not name a registered model spec.
    """
    raw: object = data.get("model_key", DEFAULT_MODEL_KEY)
    if not isinstance(raw, str):
        raise ValueError(
            f"gui_state.json 'model_key' must be a string, "
            f"got {type(raw).__name__}."
        )
    try:
        return get_model_spec(raw).key
    except ValueError as error:
        raise ValueError(f"gui_state.json {error}") from error


def optional_network_fingerprint(data: dict[str, object]) -> str | None:
    """
    The optional ``network_fingerprint`` field of a gui_state.json
    mapping.

    The value is the fingerprint of the run's network file, a
    content hash, so unlike ``model_key`` no registry validation
    applies here.

    Args:
        data: The parsed JSON mapping.

    Returns:
        The field's fingerprint, or None when it is absent (a state
        saved before network persistence existed).

    Raises:
        ValueError: Raised when the field is present but is not a
            string.
    """
    raw: object = data.get("network_fingerprint")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(
            "gui_state.json 'network_fingerprint' must be a string or "
            f"null, got {type(raw).__name__}."
        )
    return raw


def optional_column_map(
    data: dict[str, object],
) -> dict[str, str | None] | None:
    """
    The optional ``column_map`` field of a gui_state.json mapping.

    The value is the role-to-column mapping the session used to read
    its record: internal role name to the column it read, or null
    when the dialog marked the role absent.  Unlike
    ``network_fingerprint`` no registry validation applies here: the
    mapping was already checked against the record when the run was
    saved, and a record switch makes a stale mapping inert (the
    window applies it only to its own record).

    Args:
        data: The parsed JSON mapping.

    Returns:
        The field's mapping, or None when it is absent (a state
        saved before mapping persistence existed, whose record was
        read with auto-detected roles).

    Raises:
        ValueError: Raised when the field is present but is not a
            mapping of role names to column names or null.
    """
    raw: object = data.get("column_map")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            "gui_state.json 'column_map' must be a mapping of role "
            "name to column name or null, got "
            f"{type(raw).__name__}."
        )
    mapping: dict[str, str | None] = {}
    for role, column in raw.items():
        if (
            not isinstance(role, str)
            or (column is not None and not isinstance(column, str))
        ):
            raise ValueError(
                "gui_state.json 'column_map' must map every role to a "
                "column name or null, got "
                f"{role!r}: {column!r}."
            )
        mapping[role] = column
    return mapping


def not_number(value: object) -> bool:
    """Whether a JSON value is not a (non-boolean) number."""
    return not isinstance(value, (int, float)) or isinstance(value, bool)


def is_int(value: object) -> bool:
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


def read_gui_state(
    run_dir: Path,
    years: list[int] = WATER_YEARS,
) -> GuiState | None:
    """
    Read the GUI state from a saved run directory.

    Args:
        run_dir: The run directory to read the state file from.
        years: The calendar years of the record the state is
            restored against; a version-5 state's water years are
            validated against it (the site record's water years,
            when no record is known).

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
        return GuiState.from_dict(data, years)
    except (ValueError, TypeError, OSError) as error:
        print(
            f"Could not restore the GUI state from {path}: {error}",
            file=sys.stderr,
        )
        return None


def saved_run_model_key(run_dir: Path) -> str:
    """
    The model key a saved run was produced with.

    The run's ``gui_state.json`` is authoritative (a run saved by
    this GUI records the selected model there).  A run without one (a
    run saved by an older version of the GUI or by the command-line
    tools) is identified by the ``model_class`` field of its
    ``run_info.json``: the top-level file of a hydrology-only run,
    then the ``hydro`` and ``rt`` copies of a full run.  A run that
    names no recognized model class, or carries no ``run_info.json``
    at all, is assumed to be the lateral three-layer model, the
    GUI's startup default.

    Args:
        run_dir: The saved run directory to inspect.

    Returns:
        The key of the run's model spec.
    """
    state: GuiState | None = read_gui_state(run_dir)
    if state is not None:
        return state.model_key
    for info_path in (
        run_dir / "run_info.json",
        run_dir / "hydro" / "run_info.json",
        run_dir / "rt" / "run_info.json",
    ):
        if not info_path.is_file():
            continue
        try:
            data: object = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        model_class: object = data.get("model_class")
        if isinstance(model_class, str):
            key: str | None = model_key_for_class_name(model_class)
            if key is not None:
                return key
    return DEFAULT_MODEL_KEY


def saved_run_network_fingerprint(run_dir: Path) -> str | None:
    """
    The network fingerprint a saved run was produced with.

    The run's ``gui_state.json`` is the only source: a run without
    one (a run saved by an older version of the GUI or by the
    command-line tools), or a state that predates the field, has no
    fingerprint.

    Args:
        run_dir: The saved run directory to inspect.

    Returns:
        The fingerprint recorded in the run's state, or None when the
        run predates network persistence.
    """
    state: GuiState | None = read_gui_state(run_dir)
    return state.network_fingerprint if state is not None else None


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
        match: re.Match[str] | None = RUN_DIR_NAME.match(entry.name)
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
        splitter_sizes: The sizes of the main vertical splitter to
            restore (the figure and the lower
            table-and-parameters row); None when the state did not
            carry them.
        save_history: Whether the session's *Save run history*
            checkbox was on when the run was saved; False for a run
            saved before the toggle existed.
        mode: The run mode the saved run used (`MODE_FULL` or
            `MODE_HYDRO`); for a run saved before the mode existed,
            inferred from the saved outputs (hydrology-only when the
            run carried no reactive-transport results).
        restore_note: A note for the status bar explaining how the
            state was restored; None for a full gui_state.json restore.
        objective_terms: The cells of the objective table the run's
            gui_state selected, as ``"<target>: <metric>"`` labels;
            None for a run saved before the selection existed or
            without a state file (its search falls back to every
            selectable term).
        model_key: The key of the model spec the run was produced
            with, resolved by `saved_run_model_key`.
        network_fingerprint: The fingerprint of the run's reaction
            network; None for a run that predates network
            persistence (no state file, or a pre-v7 state).
        csv_path: The path of the record file the run's session
            used, or None for a run saved from the record shipped
            with the package (or one that predates the field).
        column_map: The role-to-column mapping the run's session
            used to read its record, or None for a record read with
            auto-detected roles (or a run predating the field).
        window_maximized: Whether the window was maximized when the
            run was saved; restoring the run reopens it maximized.
            False for a run saved before the flag existed or without
            a state file.
        dock_state: The saved run's encoded dock-layout blob (the
            base64 of the main window's ``saveState()``, restored via
            ``restoreState()``); None when the state predates the
            field or the run was saved without a state file.
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
    objective_terms: list[str] | None
    model_key: str
    network_fingerprint: str | None
    csv_path: str | None
    column_map: dict[str, str | None] | None
    window_maximized: bool
    dock_state: str | None
# =============== #


# ==== Saved-run restoration ==== #
def load_saved_run(run_dir: Path, input_data: GuiInputData) -> SavedRun:
    """
    Restore the GUI state and the model results from a saved run.

    When the run carries a ``gui_state.json`` (a run saved by this
    GUI), the parameter values, the user-edited bounds, the plot
    selections, the run mode, the save-history toggle, and the
    objective-table selection are restored exactly.  Otherwise (a run
    saved by an
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
            the reaction network, which the parameter mapping needs);
            its record's calendar years, which a version-5 state's
            water years are validated against.

    Returns:
        The restored results and GUI state.

    Raises:
        ValueError: Raised when the run directory is incomplete (a
            missing simulation CSV or parameters CSV) or carries a
            parameter value it cannot parse.
    """
    # A record's calendar years, from its forcing span (a daily
    # record spans every calendar year between its first and last
    # day).
    years: list[int] = list(
        range(input_data.window[0].year, input_data.window[1].year + 1)
    )
    state: GuiState | None = read_gui_state(run_dir, years)
    results: GuiRunResults = reconstruct_results(run_dir, input_data)

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
            objective_terms=state.objective_terms,
            model_key=state.model_key,
            network_fingerprint=state.network_fingerprint,
            csv_path=state.csv_path,
            column_map=state.column_map,
            window_maximized=state.window_maximized,
            dock_state=state.dock_state,
        )

    parsed: tuple[dict[str, float], str] = parse_saved_parameters(
        run_dir, input_data
    )
    values: dict[str, float] = parsed[0]
    note: str = parsed[1]
    return SavedRun(
        results=results,
        param_values=values,
        param_bounds=dict(input_data.bounds),
        selection=default_plot_selection(
            input_data.spec,
            network=input_data.network,
            window=input_data.window,
        ),
        window_geometry=None,
        splitter_sizes=None,
        save_history=False,
        # A stateless run's mode is inferred from its saved outputs:
        # a run with no reactive-transport results is a hydrology-
        # only run.
        mode=MODE_HYDRO if results.rt is None else MODE_FULL,
        restore_note=note,
        # A stateless run never recorded an objective selection.
        objective_terms=None,
        model_key=saved_run_model_key(run_dir),
        network_fingerprint=saved_run_network_fingerprint(run_dir),
        # A stateless run never recorded a record file or a mapping.
        csv_path=None,
        column_map=None,
        # A stateless run never recorded a maximized state.
        window_maximized=False,
        # A stateless run never recorded a dock layout.
        dock_state=None,
    )


def read_simulation_csv(path: Path) -> pd.DataFrame:
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


def read_hydro_objectives(path: Path) -> pd.Series:
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


def read_rt_objectives(path: Path) -> pd.DataFrame | None:
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


def zone_names_from_hydro_sim(hydro_sim: pd.DataFrame) -> list[str]:
    """
    Recover the ordered zone names from a saved hydrology simulation frame.

    The flat frame carries one ``s_{zone}`` storage column per zone in
    model (layer-major) order, and no other column uses that prefix, so
    the prefix alone identifies every zone.  A reactive-transport run
    uses the same zone names (an RT model must declare an RT zone for
    every model zone), so the same list serves both reconstructed
    result objects.

    Args:
        hydro_sim: The loaded hydrology simulation frame.

    Returns:
        The zone names in the order the frame stores them.

    Raises:
        ValueError: Raised when the frame carries no ``s_`` storage
            columns, so no zone names can be recovered.
    """
    names: list[str] = [
        c[len("s_"):] for c in hydro_sim.columns if c.startswith("s_")
    ]
    if not names:
        raise ValueError(
            "The loaded hydrology simulation frame carries no `s_` "
            "storage columns, so its zone names cannot be recovered."
        )
    return names


def reconstruct_results(run_dir: Path, input_data: GuiInputData) -> GuiRunResults:
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
    the two models in sequence. Both results record the zone names
    recovered from the hydrology frame, so their per-zone accessors
    keep working on a loaded run.

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
        hydro_sim = read_simulation_csv(run_dir / "hydro" / "simulation.csv")
        rt_sim = read_simulation_csv(run_dir / "rt" / "simulation.csv")
        hydro_objectives = read_hydro_objectives(
            run_dir / "hydro" / "objective_functions.csv"
        )
        rt_objectives = read_rt_objectives(
            run_dir / "rt" / "objective_functions.csv"
        )
    else:
        hydro_sim = read_simulation_csv(run_dir / "simulation.csv")
        rt_sim = None
        hydro_objectives = read_hydro_objectives(
            run_dir / "objective_functions.csv"
        )
        rt_objectives = None

    zone_names = zone_names_from_hydro_sim(hydro_sim)

    hydro_results: pt.HydroModelResults = pt.HydroModelResults(
        forcing=input_data.forc,
        simulation=hydro_sim,
        objective_functions=hydro_objectives,
        saved_dir=str(run_dir),
        zone_names=zone_names,
    )
    if rt_sim is not None:
        rt_results: pt.RtModelResults | None = pt.RtModelResults(
            simulation=rt_sim,
            objective_functions=rt_objectives,
            saved_dir=str(run_dir),
            zone_names=zone_names,
        )
    else:
        rt_results = None
    return GuiRunResults(hydro=hydro_results, rt=rt_results)


def parse_saved_parameters(
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
        key: str | None = row_key(row.zone, row.name, mineral_index)
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


def row_key(
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
