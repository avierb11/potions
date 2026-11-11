"""
Non-Qt tests for the general record loader of `potions.gui.data`.

Covers the tasks of the general-data-loading plan:

* column-role resolution — `BASE_COLUMN_ROLES`, `measured_roles`, and
  `resolve_columns` (canonical-name auto-detection with the explicit
  column-map override);
* `read_spreadsheet` — the CSV / XLSX reader with its datetime index;
* `trim_record` — the forcing-span trim and the daily-completeness
  validation;
* `read_record` and the generalized `load_input_data` (the bundled
  East River series as the default record, any mapped CSV / XLSX as
  the general case);
* the None-safe run builders (`arr_to_results` /
  `arr_to_hydro_results` with absent measured series) and the
  worker-pool plumbing (`model_pool.init_worker` / `map_model_runs`
  with a column map).

The Qt-layer tests (the column-mapping dialog, the Load Data action,
the state round trip, the CLI) live in `test_gui_qt.py`.
"""

import pandas as pd
import pytest

from ..gui import data as gui_data
from ..reactive_transport.common_networks import (
    get_chloride_network,
    get_soc_calcite_network,
)


# ------------------------------------------------------------------ #
# Column-role resolution
# ------------------------------------------------------------------ #


class TestColumnRoles:
    """`resolve_columns` maps a spreadsheet's columns to the record's
    roles: explicit map first, then the canonical name, then the
    aliases (case-insensitive and trimmed); the ``date`` role is
    excluded (the reader owns the index)."""

    def test_east_river_columns_auto_resolve(self) -> None:
        """The bundled record's raw column names auto-resolve to the
        internal roles; the species keys come after the forcing keys
        in species order with DIC last."""
        df: pd.DataFrame = pd.DataFrame(
            columns=[
                "date",
                "ppt",
                "temp",
                "pet",
                "q_meas",
                "doc_molL",
                "dic_molL",
                "ca_molL",
            ]
        )
        resolved: dict[str, str] = gui_data.resolve_columns(
            df, gui_data.default_network()
        )
        assert resolved == {
            "precip": "ppt",
            "temp": "temp",
            "pet": "pet",
            "q_meas": "q_meas",
            "DOC": "doc_molL",
            "DIC": "dic_molL",
            "Ca++": "ca_molL",
        }
        assert list(resolved) == [
            "precip",
            "temp",
            "pet",
            "q_meas",
            "DOC",
            "Ca++",
            "DIC",
        ]

    def test_canonical_names_resolve(self) -> None:
        """Columns that already carry the internal names map to
        themselves (``streamflow`` is a streamflow alias)."""
        df: pd.DataFrame = pd.DataFrame(
            columns=["date", "precip", "temp", "pet", "streamflow", "DOC"]
        )
        resolved: dict[str, str] = gui_data.resolve_columns(
            df, gui_data.default_network()
        )
        assert resolved == {
            "precip": "precip",
            "temp": "temp",
            "pet": "pet",
            "q_meas": "streamflow",
            "DOC": "DOC",
        }

    def test_case_insensitive_and_trimmed(self) -> None:
        """Matching ignores case and surrounding whitespace, but the
        original column spelling is what the result carries."""
        df: pd.DataFrame = pd.DataFrame(
            columns=["date", " PPT ", "TEMP", "pet", "Doc_MolL"]
        )
        resolved: dict[str, str] = gui_data.resolve_columns(
            df, gui_data.default_network()
        )
        assert resolved["precip"] == " PPT "
        assert resolved["temp"] == "TEMP"
        assert resolved["DOC"] == "Doc_MolL"

    def test_column_map_overrides(self) -> None:
        """An explicit map value wins over detection, even when a
        detectable column also exists."""
        df: pd.DataFrame = pd.DataFrame(
            columns=["date", "precip", "wet", "temp", "pet", "q_meas"]
        )
        resolved: dict[str, str] = gui_data.resolve_columns(
            df, gui_data.default_network(), column_map={"precip": "wet"}
        )
        assert resolved["precip"] == "wet"

    def test_column_map_none_clears_optional(self) -> None:
        """A ``None`` map value deliberately marks an optional role
        absent, even when a detectable column exists."""
        df: pd.DataFrame = pd.DataFrame(
            columns=["date", "ppt", "temp", "pet", "q_meas"]
        )
        resolved: dict[str, str] = gui_data.resolve_columns(
            df, gui_data.default_network(), column_map={"q_meas": None}
        )
        assert "q_meas" not in resolved

    def test_missing_required_rejected(self) -> None:
        """A required role with no match is a ValueError that names
        the missing role and the columns that do exist."""
        df: pd.DataFrame = pd.DataFrame(columns=["date", "temp", "pet"])
        with pytest.raises(ValueError) as excinfo:
            gui_data.resolve_columns(df, gui_data.default_network())
        message: str = str(excinfo.value)
        assert "precip" in message
        assert "temp" in message

    def test_collision_rejected(self) -> None:
        """Two roles mapped to the same column are a ValueError that
        names the column and both roles."""
        df: pd.DataFrame = pd.DataFrame(columns=["date", "x", "y", "z"])
        with pytest.raises(ValueError) as excinfo:
            gui_data.resolve_columns(
                df,
                gui_data.default_network(),
                column_map={"precip": "x", "temp": "x"},
            )
        message: str = str(excinfo.value)
        assert "x" in message
        assert "precip" in message
        assert "temp" in message

    def test_ambiguous_alias_priority(self) -> None:
        """When both the canonical name and an alias are present, the
        canonical name wins (Review Focus 1: no silent swap)."""
        df: pd.DataFrame = pd.DataFrame(
            columns=["date", "ppt", "precip", "temp", "pet"]
        )
        resolved: dict[str, str] = gui_data.resolve_columns(
            df, gui_data.default_network()
        )
        assert resolved["precip"] == "precip"

    def test_measured_roles_trio_and_non_trio(self) -> None:
        """The measured species roles are the network's river species
        in species order with DIC last; a network without the
        carbonate trio carries no DIC role (and the chloride network
        carries no river species at all)."""
        assert gui_data.measured_roles(gui_data.default_network()) == [
            "DOC",
            "Ca++",
            "DIC",
        ]
        assert gui_data.measured_roles(get_chloride_network()) == []
        assert "DIC" in gui_data.measured_roles(get_soc_calcite_network())
