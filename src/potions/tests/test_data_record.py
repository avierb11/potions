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
  worker-pool plumbing (`proc_pool.init_worker` /
  `map_model_runs` with a column map).

The Qt-layer tests (the column-mapping dialog, the Load Data action,
the state round trip, the CLI) live in `test_gui_qt.py`.
"""

import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ..gui import data as gui_data
from ..gui import model_pool
from ..gui import model_specs
from ..gui import proc_pool
from ..gui import state as gui_state
from ..hydro_model import HydrologicalModel
from ..model import Model
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

    def test_corrupt_xlsx_rejected(self, tmp_path: Path) -> None:
        """A file that is not a valid zip (a text file renamed
        ``.xlsx``) is a ``ValueError``, not an uncaught
        ``zipfile.BadZipFile``: the Load data slot reports any
        ``ValueError`` and keeps the session running."""
        path: Path = tmp_path / "renamed_text.xlsx"
        path.write_text("this is not a zip file\n")
        with pytest.raises(ValueError) as excinfo:
            gui_data.read_spreadsheet(path)
        message: str = str(excinfo.value)
        assert "Could not read the record spreadsheet" in message

    def test_corrupt_xml_xlsx_rejected(self, tmp_path: Path) -> None:
        """A valid zip whose XML parts are malformed is a
        ``ValueError``, not an uncaught xml ``ParseError``."""
        path: Path = tmp_path / "bad_xml.xlsx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types><Type invalid=")
            archive.writestr("xl/workbook.xml", "<<<not xml>>>")
        with pytest.raises(ValueError) as excinfo:
            gui_data.read_spreadsheet(path)
        message: str = str(excinfo.value)
        assert "Could not read the record spreadsheet" in message


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


# ------------------------------------------------------------------ #
# read_record and the generalized load_input_data
# ------------------------------------------------------------------ #


def forcing_frame(
    index: pd.DatetimeIndex, with_streamflow: bool = True
) -> pd.DataFrame:
    """A complete daily forcing frame for the record tests (ISO date
    column, constant forcing, optional integer streamflow)."""
    n: int = len(index)
    df: pd.DataFrame = pd.DataFrame(
        {
            "date": index.strftime("%Y-%m-%d"),
            "precip": np.ones(n),
            "temp": np.full(n, 5.0),
            "pet": np.full(n, 2.0),
        }
    )
    if with_streamflow:
        df["q_meas"] = np.arange(n, dtype=float)
    return df


class TestReadRecord:
    """`read_record` composes the reader, the role resolver, and the
    span trim into a runnable record frame; the generalized
    `load_input_data` assembles `GuiInputData` from any record, with
    the spin-up capped to half the record."""

    def test_bundled_invariants(self) -> None:
        """The bundled record loads as 2,557 rows over
        2014-10-01 → 2021-09-30 with zero forcing or streamflow NaN
        (the standing invariant the CSV trim relies on)."""
        df: pd.DataFrame = gui_data.read_record(
            gui_data.DEFAULT_CSV_PATH, gui_data.default_network()
        )
        assert len(df) == 2557
        assert df.index[0] == pd.Timestamp("2014-10-01")
        assert df.index[-1] == pd.Timestamp("2021-09-30")
        assert not df[["precip", "temp", "pet", "q_meas"]].isna().any().any()

    def test_spinup_cap_short_record(self, tmp_path: Path) -> None:
        """A 300-day record gets a 150-day spin-up (half the record),
        so 450 total rows: the measured values are NaN over the
        spin-up and actual over the record."""
        path: Path = tmp_path / "short.csv"
        index: pd.DatetimeIndex = pd.date_range(
            "2019-01-01", periods=300, freq="D"
        )
        forcing_frame(index).to_csv(path, index=False)
        data: gui_data.GuiInputData = gui_data.load_input_data(path)
        assert len(data.forc.precip) == 450
        assert data.q_meas is not None
        assert data.q_meas.iloc[:150].isna().all()
        assert data.q_meas.iloc[150:].notna().all()
        assert list(data.q_meas.iloc[150:]) == list(range(300))

    def test_spinup_default_east_river(self) -> None:
        """The bundled record (longer than 2×730) keeps the full
        default two-year spin-up."""
        data: gui_data.GuiInputData = gui_data.load_input_data()
        assert len(data.forc.precip) == 2557 + gui_data.NUM_SPINUP_DAYS

    def test_spinup_param(self, tmp_path: Path) -> None:
        """An explicit spin-up length is honored when it fits under
        the half-record cap (100-day record, 10-day spin-up)."""
        path: Path = tmp_path / "hundred.csv"
        index: pd.DatetimeIndex = pd.date_range(
            "2019-01-01", periods=100, freq="D"
        )
        forcing_frame(index).to_csv(path, index=False)
        data: gui_data.GuiInputData = gui_data.load_input_data(
            path, spinup_days=10
        )
        assert len(data.forc.precip) == 110

    def test_no_streamflow_column(self, tmp_path: Path) -> None:
        """A forcing-only record loads with ``q_meas = None`` and
        complete forcing."""
        path: Path = tmp_path / "forcing_only.csv"
        index: pd.DatetimeIndex = pd.date_range(
            "2019-01-01", periods=30, freq="D"
        )
        forcing_frame(index, with_streamflow=False).to_csv(path, index=False)
        data: gui_data.GuiInputData = gui_data.load_input_data(path)
        assert data.q_meas is None
        assert not data.forc.precip.isna().any()
        assert not data.forc.temp.isna().any()
        assert not data.forc.pet.isna().any()

    def test_no_concentrations(self, tmp_path: Path) -> None:
        """A forcing-only record yields a concentration frame with no
        columns, indexed like the rest of the record."""
        path: Path = tmp_path / "forcing_only.csv"
        index: pd.DatetimeIndex = pd.date_range(
            "2019-01-01", periods=30, freq="D"
        )
        forcing_frame(index, with_streamflow=False).to_csv(path, index=False)
        data: gui_data.GuiInputData = gui_data.load_input_data(path)
        assert list(data.conc_df.columns) == []
        assert data.conc_df.index.equals(data.forc.precip.index)

    def test_window_field(self) -> None:
        """The window is the record's forcing span, not the East
        River default."""
        data: gui_data.GuiInputData = gui_data.load_input_data()
        assert data.window == (
            pd.Timestamp("2014-10-01"),
            pd.Timestamp("2021-09-30"),
        )

    def test_one_day_record(self, tmp_path: Path) -> None:
        """A one-day record (Review Focus 4) loads: the window is
        that day, the spin-up cap yields zero spin-up rows, and no
        index errors occur."""
        path: Path = tmp_path / "one_day.csv"
        index: pd.DatetimeIndex = pd.date_range("2019-06-15", periods=1)
        forcing_frame(index, with_streamflow=False).to_csv(path, index=False)
        data: gui_data.GuiInputData = gui_data.load_input_data(path)
        assert len(data.forc.precip) == 1
        assert data.window == (
            pd.Timestamp("2019-06-15"),
            pd.Timestamp("2019-06-15"),
        )
        assert data.q_meas is None

    def test_xlsx_with_map(self, tmp_path: Path) -> None:
        """An XLSX whose columns only resolve through an explicit map
        loads (and its short span caps the spin-up to half)."""
        path: Path = tmp_path / "mapped.xlsx"
        index: pd.DatetimeIndex = pd.date_range("2019-01-01", periods=10)
        frame: pd.DataFrame = forcing_frame(index, with_streamflow=False)
        frame.rename(
            columns={
                "date": "day",
                "precip": "wet",
                "temp": "degC",
                "pet": "evap",
            }
        ).to_excel(path, index=False)
        data: gui_data.GuiInputData = gui_data.load_input_data(
            path,
            column_map={
                "date": "day",
                "precip": "wet",
                "temp": "degC",
                "pet": "evap",
            },
        )
        assert len(data.forc.precip) == 15
        assert data.q_meas is None


# ------------------------------------------------------------------ #
# The None-safe run builders
# ------------------------------------------------------------------ #


def midpoint_vector(input_data: gui_data.GuiInputData) -> np.ndarray:
    """The parameter vector at every bound's midpoint."""
    return np.array(
        [0.5 * (lo + hi) for lo, hi in input_data.bounds.values()]
    )


class TestNoneSafeRunBuilders:
    """The run builders hand the core the measured series the record
    carries: the identical objects when present, ``None`` (the
    core's sentinel) when the record has no column for them."""

    def test_forcing_only_run_passes_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A record without a streamflow or concentration column
        runs with every measured series passed to the models as
        ``None`` (Review Focus 2)."""
        index: pd.DatetimeIndex = pd.date_range("2019-01-01", periods=10)
        path: Path = tmp_path / "forcing_only.csv"
        forcing_frame(index, with_streamflow=False).to_csv(
            path, index=False
        )
        input_data: gui_data.GuiInputData = gui_data.load_input_data(path)
        assert input_data.q_meas is None

        seen_both: list[dict[str, object]] = []
        seen_hydro: list[dict[str, object]] = []

        def fake_run_both_models(
            self: object, **kwargs: object
        ) -> object:
            seen_both.append(dict(kwargs))
            return SimpleNamespace(hydro=None, reactive_transport=None)

        def fake_run_hydro_model(
            self: object, _forc: object, **kwargs: object
        ) -> object:
            seen_hydro.append(dict(kwargs))
            return SimpleNamespace()

        monkeypatch.setattr(Model, "run_both_models", fake_run_both_models)
        monkeypatch.setattr(
            HydrologicalModel, "run_hydro_model", fake_run_hydro_model
        )

        gui_data.arr_to_results(midpoint_vector(input_data), input_data)
        gui_data.arr_to_hydro_results(midpoint_vector(input_data), input_data)

        assert len(seen_both) == 1
        assert seen_both[0]["meas_streamflow"] is None
        assert seen_both[0]["meas_river_conc"] is None
        assert len(seen_hydro) == 1
        assert seen_hydro[0]["meas_streamflow"] is None

    def test_measured_run_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bundled record's measured series reach the model
        unchanged: the recorder receives the identical objects, so
        the ``None`` mapping applies to absent columns only."""
        input_data: gui_data.GuiInputData = gui_data.load_input_data()
        assert input_data.q_meas is not None

        seen: list[dict[str, object]] = []

        def fake_run_both_models(
            self: object, **kwargs: object
        ) -> object:
            seen.append(dict(kwargs))
            return SimpleNamespace(hydro=None, reactive_transport=None)

        monkeypatch.setattr(Model, "run_both_models", fake_run_both_models)

        gui_data.arr_to_results(midpoint_vector(input_data), input_data)

        assert len(seen) == 1
        assert seen[0]["meas_streamflow"] is input_data.q_meas
        assert seen[0]["meas_river_conc"] is input_data.conc_df


# ------------------------------------------------------------------ #
# Record-derived plot windows
# ------------------------------------------------------------------ #


class TestDefaultPlotSelectionWindow:
    """The default selection's plotting window follows a given record
    span; without one, the default window is kept."""

    def test_default_selection_with_window(self) -> None:
        """A record span becomes the selection's ISO-date window, and
        only the window changes."""
        window: tuple[pd.Timestamp, pd.Timestamp] = (
            pd.Timestamp("2019-01-01"),
            pd.Timestamp("2020-12-31"),
        )
        selection = gui_data.default_plot_selection(window=window)
        assert selection.window_start == "2019-01-01"
        assert selection.window_end == "2020-12-31"
        default = gui_data.default_plot_selection()
        assert selection.species == default.species
        assert selection.zones == default.zones
        assert selection.mineral == default.mineral
        assert selection.streamflow == default.streamflow

    def test_default_selection_default_unchanged(self) -> None:
        """Without a span, the selection keeps the default window."""
        selection = gui_data.default_plot_selection()
        assert selection.window_start == gui_data.DEFAULT_WINDOW_START
        assert selection.window_end == gui_data.DEFAULT_WINDOW_END


# ------------------------------------------------------------------ #
# Worker-pool plumbing
# ------------------------------------------------------------------ #


class TestWorkerPoolPlumbing:
    """The pool's initializer and phase runner carry the column map
    into the record loader: a mapped record loads in a worker exactly
    as it does in the parent, and a failed load reports through the
    pool's error-string contract."""

    def test_init_worker_with_map(self, tmp_path: Path) -> None:
        """An xlsx whose columns only resolve through a map loads in
        the worker's initializer: the mapped streamflow column
        becomes ``q_meas`` (NaN over the capped spin-up)."""
        index: pd.DatetimeIndex = pd.date_range("2020-01-01", periods=10)
        frame: pd.DataFrame = forcing_frame(index, with_streamflow=True)
        frame.rename(
            columns={
                "date": "day",
                "precip": "wet",
                "temp": "degC",
                "pet": "evap",
                "q_meas": "flow",
            }
        ).to_excel(tmp_path / "mapped.xlsx", index=False)

        mapping: dict[str, str | None] = {
            "date": "day",
            "precip": "wet",
            "temp": "degC",
            "pet": "evap",
            "q_meas": "flow",
        }
        try:
            proc_pool.init_worker(
                SimpleNamespace(value=0.0),
                str(tmp_path / "mapped.xlsx"),
                model_specs.DEFAULT_MODEL_KEY,
                column_map=mapping,
            )
            assert proc_pool.INIT_ERROR is None
            loaded: gui_data.GuiInputData | None = proc_pool.INPUT_DATA
            assert loaded is not None
            q_meas: pd.Series | None = loaded.q_meas
            assert q_meas is not None
            # A 10-day record caps the spin-up to 5 days (half the
            # record): 15 rows, measurements NaN over the spin-up.
            assert len(q_meas) == 15
            assert q_meas.iloc[:5].isna().all()
            assert list(q_meas.iloc[5:]) == list(range(10))
        finally:
            # In-process init leaves the module globals set.
            proc_pool.INPUT_DATA = None
            proc_pool.INIT_ERROR = None

    def test_init_worker_bad_map_reports(self, tmp_path: Path) -> None:
        """A map that points a role at a missing column does not
        crash the worker: the initializer's failure-string contract
        holds, and no input data is loaded."""
        index: pd.DatetimeIndex = pd.date_range("2020-01-01", periods=10)
        path: Path = tmp_path / "bad_map.csv"
        forcing_frame(index, with_streamflow=True).to_csv(
            path, index=False
        )

        captured_error: str | None = None
        try:
            proc_pool.init_worker(
                SimpleNamespace(value=0.0),
                str(path),
                model_specs.DEFAULT_MODEL_KEY,
                column_map={"q_meas": "nope"},
            )
            assert proc_pool.INPUT_DATA is None
            captured_error = proc_pool.INIT_ERROR
        finally:
            # In-process init leaves the module globals set.
            proc_pool.INPUT_DATA = None
            proc_pool.INIT_ERROR = None
        assert captured_error is not None
        assert captured_error.startswith(
            "ValueError: The column map points the role 'q_meas' at "
            "the column 'nope'"
        )

    def test_map_model_runs_default_unchanged(self) -> None:
        """A phase on the bundled record (no map, hydro mode, one
        point) still yields one results object: the default path is
        unchanged by the mapping parameter (the ~0.3 s cost is the
        pool's one record re-parse plus one hydro run)."""
        input_data: gui_data.GuiInputData = gui_data.load_input_data()
        point: np.ndarray = midpoint_vector(input_data)
        runs: list[gui_data.GuiRunResults | str] = model_pool.map_model_runs(
            [point],
            gui_state.MODE_HYDRO,
            str(gui_data.DEFAULT_CSV_PATH),
            workers=1,
        )
        assert len(runs) == 1
        assert isinstance(runs[0], gui_data.GuiRunResults)
