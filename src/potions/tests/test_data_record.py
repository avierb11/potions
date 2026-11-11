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

from pathlib import Path

import numpy as np
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


# ------------------------------------------------------------------ #
# Spreadsheet reading
# ------------------------------------------------------------------ #


class TestReadSpreadsheet:
    """`read_spreadsheet` parses a CSV or XLSX record into an indexed
    frame: a sorted daily `DatetimeIndex` (detected or mapped),
    float64 data columns, and precise errors for a bad extension, a
    missing or duplicated date column, and non-numeric data."""

    def test_read_csv_sorted(self, tmp_path: Path) -> None:
        """Rows out of order in the file come back in date order."""
        path: Path = tmp_path / "record.csv"
        pd.DataFrame(
            {
                "date": ["2019-03-01", "2019-01-01", "2019-02-01"],
                "precip": [1.0, 2.0, 3.0],
            }
        ).to_csv(path, index=False)
        df: pd.DataFrame = gui_data.read_spreadsheet(path)
        assert list(df.index) == [
            pd.Timestamp("2019-01-01"),
            pd.Timestamp("2019-02-01"),
            pd.Timestamp("2019-03-01"),
        ]
        assert list(df["precip"]) == [2.0, 3.0, 1.0]

    def test_read_xlsx(self, tmp_path: Path) -> None:
        """An XLSX spreadsheet reads back with a datetime index and
        float columns."""
        path: Path = tmp_path / "record.xlsx"
        pd.DataFrame(
            {
                "date": ["2019-01-01", "2019-01-02"],
                "temp": [5.5, 6.5],
            }
        ).to_excel(path, index=False)
        df: pd.DataFrame = gui_data.read_spreadsheet(path)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.index) == [
            pd.Timestamp("2019-01-01"),
            pd.Timestamp("2019-01-02"),
        ]
        assert list(df["temp"]) == [5.5, 6.5]

    def test_xlsx_text_dates(self, tmp_path: Path) -> None:
        """Date cells stored as text (Review Focus 3) parse to the
        right timestamps instead of failing or misparsing."""
        path: Path = tmp_path / "record.xlsx"
        pd.DataFrame(
            {"date": ["1/15/2019", "1/16/2019"], "ppt": [1.0, 2.0]}
        ).to_excel(path, index=False)
        df: pd.DataFrame = gui_data.read_spreadsheet(path)
        assert df.index[0] == pd.Timestamp("2019-01-15")
        assert df.index[1] == pd.Timestamp("2019-01-16")

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        """A non-spreadsheet file is rejected with the accepted
        extensions named."""
        path: Path = tmp_path / "record.txt"
        path.write_text("date,ppt\n2019-01-01,1\n")
        with pytest.raises(ValueError) as excinfo:
            gui_data.read_spreadsheet(path)
        message: str = str(excinfo.value)
        assert "csv" in message
        assert "xlsx" in message

    def test_duplicate_dates_rejected(self, tmp_path: Path) -> None:
        """Two rows on the same date are a ValueError that names the
        duplicated date."""
        path: Path = tmp_path / "record.csv"
        pd.DataFrame(
            {"date": ["2019-01-01", "2019-01-01"], "ppt": [1.0, 2.0]}
        ).to_csv(path, index=False)
        with pytest.raises(ValueError) as excinfo:
            gui_data.read_spreadsheet(path)
        assert "2019-01-01" in str(excinfo.value)

    def test_non_numeric_column_rejected(self, tmp_path: Path) -> None:
        """A data column holding a non-numeric value is a ValueError
        that names the column and the offending value."""
        path: Path = tmp_path / "record.csv"
        pd.DataFrame(
            {"date": ["2019-01-01", "2019-01-02"], "temp": ["x", "5.0"]}
        ).to_csv(path, index=False)
        with pytest.raises(ValueError) as excinfo:
            gui_data.read_spreadsheet(path)
        message: str = str(excinfo.value)
        assert "temp" in message
        assert "x" in message

    def test_datetime_from_map(self, tmp_path: Path) -> None:
        """An explicit map value owns the index, even when the column
        name is not a date-role name."""
        path: Path = tmp_path / "record.csv"
        pd.DataFrame(
            {"day": ["2019-01-02", "2019-01-01"], "ppt": [2.0, 1.0]}
        ).to_csv(path, index=False)
        df: pd.DataFrame = gui_data.read_spreadsheet(
            path, column_map={"date": "day"}
        )
        assert list(df.index) == [
            pd.Timestamp("2019-01-01"),
            pd.Timestamp("2019-01-02"),
        ]

    def test_missing_datetime_rejected(self, tmp_path: Path) -> None:
        """No date-role column is a ValueError that lists the
        columns that do exist."""
        path: Path = tmp_path / "record.csv"
        pd.DataFrame({"a": [1.0], "b": [2.0]}).to_csv(path, index=False)
        with pytest.raises(ValueError) as excinfo:
            gui_data.read_spreadsheet(path)
        message: str = str(excinfo.value)
        assert "a" in message
        assert "b" in message


# ------------------------------------------------------------------ #
# Forcing-span trim and validation
# ------------------------------------------------------------------ #


class TestTrimRecord:
    """`trim_record` trims an indexed, renamed record to its forcing
    span — rows outside the span (observations beyond the forcing
    period) are the only rows ever dropped — and rejects interior
    forcing NaNs, gaps, and non-daily cadence."""

    def test_observations_beyond_span_trimmed(self) -> None:
        """Rows before and after the forcing span are dropped, with
        their measured values on the surviving rows intact."""
        index: pd.DatetimeIndex = pd.date_range(
            "2019-01-01", periods=20, freq="D"
        )
        df: pd.DataFrame = pd.DataFrame(
            {
                "precip": np.full(20, 1.0),
                "temp": np.full(20, 5.0),
                "pet": np.full(20, 2.0),
                "q_meas": np.arange(20, dtype=float),
            },
            index=index,
        )
        df.iloc[:5, :3] = np.nan
        df.iloc[15:, :3] = np.nan
        out: pd.DataFrame = gui_data.trim_record(df)
        assert len(out) == 10
        assert out.index[0] == pd.Timestamp("2019-01-06")
        assert out.index[-1] == pd.Timestamp("2019-01-15")
        assert list(out["q_meas"]) == list(range(5, 15))

    def test_forcing_nan_inside_rejected(self) -> None:
        """An interior forcing NaN (inside the span) is a ValueError
        that names the column and the affected date."""
        index: pd.DatetimeIndex = pd.date_range(
            "2019-01-01", periods=10, freq="D"
        )
        df: pd.DataFrame = pd.DataFrame(
            {
                "precip": [1.0] * 10,
                "temp": np.full(10, 5.0),
                "pet": np.full(10, 2.0),
            },
            index=index,
        )
        df.iloc[4, 0] = np.nan
        with pytest.raises(ValueError) as excinfo:
            gui_data.trim_record(df)
        message: str = str(excinfo.value)
        assert "precip" in message
        assert "2019-01-05" in message

    def test_missing_date_rejected(self) -> None:
        """A gap inside the forcing span is a ValueError that names
        the missing date."""
        index: pd.DatetimeIndex = pd.date_range(
            "2019-01-01", periods=10, freq="D"
        )
        df: pd.DataFrame = pd.DataFrame(
            {
                "precip": np.full(10, 1.0),
                "temp": np.full(10, 5.0),
                "pet": np.full(10, 2.0),
            },
            index=index,
        ).drop(index[4])
        with pytest.raises(ValueError) as excinfo:
            gui_data.trim_record(df)
        assert "2019-01-05" in str(excinfo.value)

    def test_non_daily_rejected(self) -> None:
        """A bi-daily cadence inside the span is rejected (the
        record must be exactly daily)."""
        index: pd.DatetimeIndex = pd.date_range(
            "2019-01-01", periods=10, freq="2D"
        )
        df: pd.DataFrame = pd.DataFrame(
            {
                "precip": np.full(10, 1.0),
                "temp": np.full(10, 5.0),
                "pet": np.full(10, 2.0),
            },
            index=index,
        )
        with pytest.raises(ValueError):
            gui_data.trim_record(df)

    def test_empty_rejected(self) -> None:
        """No row with complete forcing is a ValueError."""
        index: pd.DatetimeIndex = pd.date_range(
            "2019-01-01", periods=3, freq="D"
        )
        df: pd.DataFrame = pd.DataFrame(
            {
                "precip": np.full(3, np.nan),
                "temp": np.full(3, 5.0),
                "pet": np.full(3, 2.0),
            },
            index=index,
        )
        with pytest.raises(ValueError):
            gui_data.trim_record(df)

    def test_bundled_span(self) -> None:
        """The bundled record trims to its forcing span: 2,557 rows
        over 2014-10-01 → 2021-09-30 with zero forcing NaN (the
        standing invariant the CSV trim in Task 4 relies on)."""
        raw: pd.DataFrame = pd.read_csv(
            gui_data.DEFAULT_CSV_PATH, parse_dates=["date"], index_col="date"
        )
        df: pd.DataFrame = raw.rename(columns={"ppt": "precip"})
        out: pd.DataFrame = gui_data.trim_record(df)
        assert len(out) == 2557
        assert out.index[0] == pd.Timestamp("2014-10-01")
        assert out.index[-1] == pd.Timestamp("2021-09-30")
        assert not out[["precip", "temp", "pet"]].isna().any().any()
