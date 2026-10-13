"""Tests for the default run-result saving behavior.

These tests opt in to saving explicitly (the test suite conftest sets
``POTIONS_SAVE_RESULTS=0`` globally, so the default-on behavior would
otherwise clutter the working directory) and always write under pytest's
``tmp_path``.
"""

import datetime
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from numpy import float64 as f64
from pandas import DataFrame, Series
from pathlib import Path

from ..common_models import HbvModel
from ..common_types import ForcingData, RtModelResults
from ..hydro_model import HydrologicalModel
from ..result_saving import (
    DEFAULT_RESULTS_DIR,
    RESULTS_DIR_ENV,
    SAVE_RESULTS_ENV,
    is_save_enabled,
    make_run_dir,
    resolve_results_dir,
    save_model_run,
    save_rt_run,
)

NUM_STEPS = 30


def _make_forcing(num_steps: int = NUM_STEPS) -> ForcingData:
    start = datetime.date(2000, 1, 1)
    dates: list[datetime.date] = [
        start + datetime.timedelta(days=i) for i in range(num_steps)
    ]
    index = Series(dates)

    def const(v: float) -> Series:
        return Series([v] * num_steps, index=index, dtype=f64)

    return ForcingData(precip=const(2.0), temp=const(15.0), pet=const(1.0))


def _make_rt_results(sim_df: DataFrame) -> RtModelResults:
    """A minimal, hand-built RT result object (fast, no Newton solve needed)."""
    return RtModelResults(
        simulation=sim_df,
        objective_functions=None,
        rt_forcings=np.zeros((len(sim_df), 2)),
        steps=[],
        other={"old_states": np.zeros((2, 6)), "new_states": np.zeros((2, 6))},
    )


# ==== Configuration helpers ==== #


def test_is_save_enabled_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SAVE_RESULTS_ENV, raising=False)
    assert is_save_enabled(None) is True  # built-in default: on
    assert is_save_enabled(True) is True
    assert is_save_enabled(False) is False

    monkeypatch.setenv(SAVE_RESULTS_ENV, "0")
    assert is_save_enabled(None) is False
    assert is_save_enabled(False) is False
    assert is_save_enabled(True) is True  # an explicit argument always wins

    monkeypatch.setenv(SAVE_RESULTS_ENV, "1")
    assert is_save_enabled(None) is True
    assert is_save_enabled(False) is False


def test_resolve_results_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(RESULTS_DIR_ENV, raising=False)
    assert resolve_results_dir() == Path(DEFAULT_RESULTS_DIR)
    assert resolve_results_dir("mydir") == Path("mydir")

    monkeypatch.setenv(RESULTS_DIR_ENV, str(tmp_path))
    assert resolve_results_dir() == tmp_path
    assert resolve_results_dir("explicit") == Path("explicit")


def test_make_run_dir_collision(tmp_path: Path) -> None:
    d1 = make_run_dir(tmp_path)
    d2 = make_run_dir(tmp_path)
    d3 = make_run_dir(tmp_path)

    assert d1.parent == tmp_path
    assert d1.name.startswith("run_")
    assert d1.exists() and d2.exists() and d3.exists()
    assert len({d1, d2, d3}) == 3  # never clobber an existing directory


# ==== Hydro model runs ==== #


def test_run_hydro_model_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SAVE_RESULTS_ENV, "1")
    model = HbvModel()
    res = model.run_hydro_model(
        forc=_make_forcing(),
        save_results=True,
        results_dir=str(tmp_path / "out"),
    )

    assert res.saved_dir is not None
    run_dir = Path(res.saved_dir)
    assert run_dir.parent == tmp_path / "out"
    assert run_dir.name.startswith("run_")
    for name in ("run_info.json", "simulation.csv", "parameters.csv"):
        assert (run_dir / name).is_file()

    sim = pd.read_csv(run_dir / "simulation.csv", index_col=0)
    assert len(sim) == NUM_STEPS

    params = pd.read_csv(run_dir / "parameters.csv")
    assert len(params) > 0
    assert "name" in params.columns
    assert "value" in params.columns

    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["run_id"] == run_dir.name
    assert "HbvModel" in info["model_class"]
    assert info["num_time_steps"] == NUM_STEPS
    assert "potions_version" in info
    assert "python_version" in info


def test_run_hydro_model_with_measured_streamflow_saves_objectives(
    tmp_path: Path
) -> None:
    model = HbvModel()
    num_steps = 90
    forc = _make_forcing(num_steps=num_steps)
    index = forc.precip.index
    meas = Series([5.0] * num_steps, index=index, dtype=f64)

    res = model.run_hydro_model(
        forc=forc,
        meas_streamflow=meas,
        save_results=True,
        results_dir=str(tmp_path),
    )

    assert res.saved_dir is not None
    run_dir = Path(res.saved_dir)
    assert (run_dir / "objective_functions.csv").is_file()

    info = json.loads((run_dir / "run_info.json").read_text())
    # json.loads succeeded, so all values were strict JSON (no NaN/inf)
    assert "objective_functions" in info
    assert len(info["objective_functions"]) > 0


def test_run_hydro_model_save_explicit_off(tmp_path: Path) -> None:
    model = HbvModel()
    res = model.run_hydro_model(
        forc=_make_forcing(),
        save_results=False,
        results_dir=str(tmp_path),
    )
    assert res.saved_dir is None
    assert not any(tmp_path.iterdir())


def test_env_can_disable_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SAVE_RESULTS_ENV, "0")
    model = HbvModel()
    res = model.run_hydro_model(
        forc=_make_forcing(), results_dir=str(tmp_path)
    )
    assert res.saved_dir is None
    assert not any(tmp_path.iterdir())


def test_explicit_save_wins_over_off_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SAVE_RESULTS_ENV, "0")
    model = HbvModel()
    res = model.run_hydro_model(
        forc=_make_forcing(), save_results=True, results_dir=str(tmp_path)
    )
    assert res.saved_dir is not None


# ==== RT results ==== #


def test_save_rt_run_writes_pickle(tmp_path: Path) -> None:
    model = HbvModel()
    num = 10
    sim_df = DataFrame(
        {"DOC_b": np.full(num, 1e-3)},
        index=Series(
            [
                datetime.date(2000, 1, 1) + datetime.timedelta(days=i)
                for i in range(num)
            ]
        ),
    )
    rt_res = _make_rt_results(sim_df)

    out = save_rt_run(model, rt_res, str(tmp_path / "out"))
    pkl_path = out / "results.pkl"

    assert pkl_path.is_file()
    with pkl_path.open("rb") as fh:
        loaded = pickle.load(fh)
    assert loaded.simulation.equals(rt_res.simulation)

    info = json.loads((out / "run_info.json").read_text())
    assert info.get("results_pickle_saved") is True


def test_save_model_run_combined_layout(tmp_path: Path) -> None:
    model = HbvModel()
    hydro_res = model.run_hydro_model(forc=_make_forcing(), save_results=False)
    rt_res = _make_rt_results(hydro_res.simulation)

    out = save_model_run(model, hydro_res, rt_res, str(tmp_path / "out"))

    assert (out / "hydro" / "simulation.csv").is_file()
    assert (out / "hydro" / "parameters.csv").is_file()
    assert (out / "hydro" / "run_info.json").is_file()
    assert (out / "rt" / "simulation.csv").is_file()
    assert (out / "rt" / "results.pkl").is_file()

    hydro_info = json.loads((out / "hydro" / "run_info.json").read_text())
    rt_info = json.loads((out / "rt" / "run_info.json").read_text())
    assert hydro_info["run_id"] == rt_info["run_id"] == out.name


# ==== Batch runs ==== #
#
# `run_batch` builds each worker's model via `hydro_from_series`, which in
# turn calls each zone's `from_dict`. That Rust method is currently
# unimplemented for some zone types (pyo3 `todo!()`), so the tests patch it
# to construct a default `HbvModel` instead. This keeps them exercising the
# saving logic under test (Pool, worker write, parent aggregation).


def _fake_hydro_from_series(cls, series: Series) -> HydrologicalModel:
    del series
    return HbvModel()


def test_run_batch_saves_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        HydrologicalModel, "hydro_from_series", classmethod(_fake_hydro_from_series)
    )
    model = HbvModel()
    params = DataFrame([model.to_series(), model.to_series()])
    forc = _make_forcing(num_steps=20)
    index = forc.precip.index
    meas = Series([5.0] * 20, index=index, dtype=f64)

    res = HbvModel.run_batch(
        params=params,
        forc=forc,
        meas_streamflow=meas,
        num_threads=1,
        write_time_series_results=True,
        output_dir=str(tmp_path / "batch"),
    )

    base = tmp_path / "batch"
    run_dirs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    member_files = sorted(p.name for p in run_dir.iterdir() if p.is_file())
    assert any(n.endswith(".csv") and "parameters" not in n for n in member_files)
    assert any(n.endswith("parameters.csv") for n in member_files)

    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["num_runs"] == 2

    assert len(res["objective_functions"]) == 2
    assert len(res["simulations"]) == 2


def test_run_batch_save_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        HydrologicalModel, "hydro_from_series", classmethod(_fake_hydro_from_series)
    )
    model = HbvModel()
    params = DataFrame([model.to_series(), model.to_series()])
    forc = _make_forcing(num_steps=20)

    res = HbvModel.run_batch(
        params=params,
        forc=forc,
        num_threads=1,
        write_time_series_results=False,
        output_dir=str(tmp_path / "batch"),
    )

    # No output directory should be created at all
    assert not (tmp_path / "batch").exists()
    assert len(res["objective_functions"]) == 2
    assert len(res["simulations"]) == 2
