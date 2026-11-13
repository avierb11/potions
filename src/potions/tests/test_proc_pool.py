"""
Tests for the heartbeat-guarded forkserver pool and the model-run session.

The pool is exercised through a *real* forkserver worker pool against
module-level fakes: a forkserver child re-imports this module fresh, so
parent-side monkeypatches and closures are invisible to it and only
module-level names pickle across the boundary.  The fakes model the
failure mode the watchdog exists for — workers that never finish (or
never start) their initializer and so never return a task — and the
watchdog tests assert the stall is reported quickly and by point index,
not that it hangs.  No model is ever run: the per-point-error test
fails through a genuinely missing input record in the child's own
initializer.
"""

from __future__ import annotations

import multiprocessing
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import potions as pt
from potions.gui import data as gui_data
from potions.gui import proc_pool
from potions.gui.model_specs import DEFAULT_MODEL_KEY
from potions.gui.proc_pool import (
    HeartbeatPool,
    ModelRunSession,
    ModelRunStalledError,
)
from potions.gui.state import MODE_FULL

FORKSERVER_AVAILABLE: bool = "forkserver" in multiprocessing.get_all_start_methods()
pytestmark = pytest.mark.skipif(
    not FORKSERVER_AVAILABLE, reason="forkserver start method unavailable"
)

# A heartbeat value no monotonic clock reading can reach naturally, so
# a child's write of it is unmistakable when read back in the parent.
SENTINEL_HEARTBEAT: float = 777_777.0


# ------------------------------------------------------------------ #
# Module-level fakes (they must be picklable by name and re-importable
# by a fresh forkserver child)
# ------------------------------------------------------------------ #


def log_initializer(heartbeat: Any, log_path: str) -> None:
    """Append one line per worker process and heartbeat the start."""
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"init {multiprocessing.current_process().pid}\n")
    heartbeat.value = time.monotonic()


def noop_task(index_and_arg: tuple[int, Any]) -> tuple[int, float]:
    """Return the point's input index and its argument doubled."""
    index, arg = index_and_arg
    return index, float(arg) * 2.0


def sleeping_task(index_and_arg: tuple[int, Any]) -> tuple[int, float]:
    """Never return: models a worker whose model run stalls forever."""
    time.sleep(60.0)
    return index_and_arg[0], 0.0


def stuck_initializer(heartbeat: Any, _log_path: str) -> None:
    """
    Never finish: models a worker parked before its initializer
    returns — the deadlock shape the watchdog bounds.  It heartbeats
    nothing, so the parent must conclude "stalled" and stop waiting.
    """
    time.sleep(60.0)


def echo_initializer(heartbeat: Any, _payload: Any) -> None:
    """Accept any picklable initarg and heartbeat the completed start."""
    heartbeat.value = time.monotonic()


def sentinel_task(index_and_arg: tuple[int, Any]) -> tuple[int, float]:
    """Write the sentinel into the shared heartbeat from the child."""
    index, _arg = index_and_arg
    heartbeat = proc_pool.HEARTBEAT
    heartbeat.value = SENTINEL_HEARTBEAT
    return index, float(heartbeat.value)


# ------------------------------------------------------------------ #
# Local fixtures and helpers (kept out of test_gui so the forkserver
# children re-import only this module's lean chain)
# ------------------------------------------------------------------ #


@pytest.fixture(scope="session")
def input_data() -> gui_data.GuiInputData:
    """The package's own East River input data (the app's default)."""
    return gui_data.load_input_data(gui_data.DEFAULT_CSV_PATH)


def picklable_payload(
    input_data: gui_data.GuiInputData,
) -> gui_data.GuiRunResults:
    """A small, fully synthetic ``GuiRunResults`` for a pickle round trip."""
    index: pd.DatetimeIndex = input_data.forc.precip.index
    zones: list[str] = list(input_data.spec.zones)
    hydro: pt.HydroModelResults = pt.HydroModelResults(
        forcing=input_data.forc,
        simulation=pd.DataFrame(
            {"streamflow_sim": np.arange(len(index), dtype=float)}, index=index
        ),
        objective_functions=pd.Series({"KGE": 0.9}, dtype=float),
        zone_names=zones,
    )
    rt: pt.RtModelResults = pt.RtModelResults(
        simulation=pd.DataFrame(
            {"DOC_riv": np.arange(len(index), dtype=float)}, index=index
        ),
        objective_functions=pd.DataFrame({"DOC": [0.5]}, index=["KGE"]),
        saved_dir=None,
        zone_names=zones,
    )
    return gui_data.GuiRunResults(hydro, rt)


class TestHeartbeatPool:
    """The generic heartbeat-guarded forkserver pool itself."""

    def test_map_returns_results_in_input_order(
        self, tmp_path: Path
    ) -> None:
        """Every submitted point comes back, aligned to its input slot."""
        log_file: Path = tmp_path / "workers.log"
        pool: HeartbeatPool = HeartbeatPool(
            noop_task, log_initializer, (str(log_file),), workers=2
        )
        try:
            seen: list[tuple[int, Any]] = []
            results: list[Any] = pool.map(
                [1.0, 2.0, 3.0],
                on_result=lambda index, entry: seen.append((index, entry)),
            )
            assert [entry[1] for entry in results] == [2.0, 4.0, 6.0]
            assert {index for index, _entry in seen} == {0, 1, 2}
            assert pool.map([]) == []
        finally:
            pool.close()

    def test_initializer_runs_once_per_worker(self, tmp_path: Path) -> None:
        """Five points across two workers still initialize exactly two."""
        log_file: Path = tmp_path / "workers.log"
        pool: HeartbeatPool = HeartbeatPool(
            noop_task, log_initializer, (str(log_file),), workers=2
        )
        try:
            results: list[Any] = pool.map([1.0, 2.0, 3.0, 4.0, 5.0])
        finally:
            pool.close()
        assert len(results) == 5
        lines: list[str] = log_file.read_text(encoding="utf-8").splitlines()
        init_lines: list[str] = [line for line in lines if line.startswith("init ")]
        assert len(init_lines) == 2
        assert len({line.split()[1] for line in init_lines}) == 2

    def test_initargs_are_pickled_across_the_boundary(
        self, tmp_path: Path
    ) -> None:
        """A non-trivial initarg must survive the forkserver pickling."""
        pool: HeartbeatPool = HeartbeatPool(
            noop_task, echo_initializer, ({"a": [1, 2]},), workers=1
        )
        try:
            results: list[Any] = pool.map([1.0])
        finally:
            pool.close()
        assert results[0][1] == pytest.approx(2.0)

    def test_heartbeat_is_shared_with_the_children(self) -> None:
        """
        A child's heartbeat write must be visible in the parent.

        The watchdog reads the heartbeat in the parent; if the shared
        value were copied per child instead of shared, the parent's
        copy would never advance and every long batch would look
        stalled.  The sentinel proves one child's write lands in the
        parent's copy.
        """
        pool: HeartbeatPool = HeartbeatPool(
            sentinel_task, echo_initializer, (None,), workers=1
        )
        try:
            results: list[Any] = pool.map([1.0])
            assert results[0][1] == pytest.approx(SENTINEL_HEARTBEAT)
            assert pool.heartbeat.value == pytest.approx(SENTINEL_HEARTBEAT)
        finally:
            pool.close()

    def test_watchdog_raises_when_nothing_returns(self, tmp_path: Path) -> None:
        """
        A worker stuck in its initializer is reported, not waited on.

        The batch must fail with ``ModelRunStalledError`` naming the
        pending point, within roughly the stall timeout — never
        anywhere near the fake's 60 s sleep.
        """
        pool: HeartbeatPool = HeartbeatPool(
            noop_task,
            stuck_initializer,
            (str(tmp_path / "unused.log"),),
            workers=1,
            stall_timeout=1.0,
        )
        started: float = time.monotonic()
        try:
            with pytest.raises(ModelRunStalledError) as excinfo:
                pool.map([1.0])
            elapsed: float = time.monotonic() - started
            assert "point(s) [0]" in str(excinfo.value)
            assert elapsed < 30.0
        finally:
            pool.close()

    def test_watchdog_names_all_pending_points(self) -> None:
        """Every point whose result never arrived is named in the error."""
        pool: HeartbeatPool = HeartbeatPool(
            sleeping_task, echo_initializer, (None,), workers=2, stall_timeout=1.0
        )
        started: float = time.monotonic()
        try:
            with pytest.raises(ModelRunStalledError) as excinfo:
                pool.map([1.0, 2.0, 3.0, 4.0])
            elapsed: float = time.monotonic() - started
            assert "point(s) [0, 1, 2, 3]" in str(excinfo.value)
            assert elapsed < 40.0
        finally:
            pool.close()

    def test_map_after_close_is_rejected(self, tmp_path: Path) -> None:
        """A closed pool refuses new work with an informative error."""
        pool: HeartbeatPool = HeartbeatPool(
            noop_task, echo_initializer, (None,), workers=1
        )
        pool.close()
        with pytest.raises(RuntimeError, match="closed"):
            pool.map([1.0])


class TestModelRunSession:
    """The GUI-facing warm pool wrapper."""

    def test_per_point_errors_without_monkeypatching(
        self, tmp_path: Path, input_data: gui_data.GuiInputData
    ) -> None:
        """
        A missing record fails every point in the children, naturally.

        Nothing is patched anywhere: each fresh child runs the real
        loader against a genuinely missing file, and every point
        reports that child-side failure as an error string — exactly
        the old pool's contract.  This also pins the reaction network
        pickling through the forkserver boundary.
        """
        session: ModelRunSession = ModelRunSession(
            str(tmp_path / "missing.csv"),
            DEFAULT_MODEL_KEY,
            network=input_data.network,
            workers=2,
        )
        try:
            payloads: list[gui_data.GuiRunResults | str] = session.map(
                [np.array([1.0]), np.array([2.0])], MODE_FULL
            )
        finally:
            session.close()
        assert all(isinstance(payload, str) for payload in payloads)
        assert all("FileNotFoundError" in payload for payload in payloads)

    def test_map_of_no_points_returns_empty(
        self, tmp_path: Path, input_data: gui_data.GuiInputData
    ) -> None:
        """An empty batch yields no entries and never touches the pool."""
        session: ModelRunSession = ModelRunSession(
            str(tmp_path / "missing.csv"),
            DEFAULT_MODEL_KEY,
            network=input_data.network,
            workers=1,
        )
        try:
            assert session.map([], MODE_FULL) == []
        finally:
            session.close()


class TestPicklability:
    """The payload and network types the forkserver boundary requires."""

    def test_gui_run_results_are_picklable(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A full-mode payload survives a pickle round trip."""
        payload: gui_data.GuiRunResults = picklable_payload(input_data)
        restored: gui_data.GuiRunResults = pickle.loads(pickle.dumps(payload))
        assert isinstance(restored, gui_data.GuiRunResults)

    def test_reaction_network_is_picklable(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The network handed to a session survives a pickle round trip."""
        restored: pt.ReactionNetwork = pickle.loads(pickle.dumps(input_data.network))
        assert isinstance(restored, pt.ReactionNetwork)
