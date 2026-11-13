"""Surrogate tests for the calibration module's objective machinery.

The polish tests patch the module's objective, batch-map, and run
helpers with in-process stand-ins, so they exercise the descent and
wave-search plumbing without a model simulation.  The pickling and
forkserver tests deliberately use the real machinery: forkserver pool
initargs are pickled, so the objective must survive a pickle round
trip.
"""

from __future__ import annotations

import datetime as dt
import pickle
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
from pandas import DataFrame, Series

from .. import calibration
from ..calibration import (
    RtObjective,
    make_objective,
    parallel_numerical_gradient,
)
from ..common_models import HbvModel
from ..common_types import (
    CalibrationResults,
    ForcingData,
    ModelResults,
    RtCalibrationData,
    RtZoneConfiguration,
)
from ..objective_functions import DEFAULT_RT_OBJECTIVE_SPEC
from ..reactive_transport.common_networks import get_simple_carbon_network


def _quadratic(x: np.ndarray) -> float:
    # f = x0^2 + 2*x1^2 + 3*x2^2  =>  grad = [2*x0, 4*x1, 6*x2]
    return float(x[0] ** 2 + 2.0 * x[1] ** 2 + 3.0 * x[2] ** 2)


def _quad2(x: np.ndarray) -> float:
    # f = x0^2 + x1^2  =>  grad = [2*x0, 2*x1]
    return float(x[0] ** 2 + x[1] ** 2)


def _linear1(x: np.ndarray) -> float:
    # f = 2*x0  =>  grad = [2]
    return float(2.0 * x[0])


def test_parallel_numerical_gradient_matches_analytic() -> None:
    x = np.array([1.0, 2.0, 3.0])
    grad = parallel_numerical_gradient(_quadratic, x, num_threads=2)
    assert np.allclose(grad, np.array([2.0, 8.0, 18.0]), atol=1e-3)


def test_parallel_numerical_gradient_zero_axis() -> None:
    # At x0 = 0, the step size falls back to the absolute `dx` rather than the
    # relative one; the gradient should still be correct.
    x = np.array([0.0, 5.0])
    grad = parallel_numerical_gradient(_quad2, x, num_threads=2)
    assert np.allclose(grad, np.array([0.0, 10.0]), atol=1e-3)


def test_parallel_numerical_gradient_single_axis() -> None:
    x = np.array([3.0])
    grad = parallel_numerical_gradient(_linear1, x, num_threads=1)
    assert np.allclose(grad, np.array([2.0]), atol=1e-3)


# ---------------------------------------------------------------------------
# polish — surrogate objective, fake map, no model runs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _QuadraticObjective:
    """Picklable stand-in mirroring `RtObjective`'s call interface."""

    center: np.ndarray
    scale: float

    def __call__(self, arr: np.ndarray) -> float:
        return float(self.scale * np.sum((np.asarray(arr) - self.center) ** 2))


def _start(params: list[float], objective: float, sentinel: Any) -> CalibrationResults:
    """A `CalibrationResults` whose `best_results` is the sentinel."""
    return CalibrationResults(
        param_names=[f"p{index}" for index in range(len(params))],
        best_params=np.array(params, dtype=np.float64),
        best_results=cast(ModelResults, sentinel),
        objective=objective,
        optimize_result=cast(Any, None),
    )


def _patch_polish_surrogate(
    monkeypatch: pytest.MonkeyPatch,
    objective: _QuadraticObjective,
) -> Any:
    """Patch `calibration` so `polish` scores points in-process.

    Returns the sentinel the patched `arr_to_results` hands back, so a
    test can check which results the polish adopted.
    """
    sentinel = object()
    monkeypatch.setattr(calibration, "make_objective", lambda data, spec=None: objective)
    monkeypatch.setattr(
        calibration,
        "_map_objectives",
        lambda obj, points, workers=None: [float(obj(np.asarray(p))) for p in points],
    )
    monkeypatch.setattr(calibration, "arr_to_results", lambda arr, data: sentinel)
    return sentinel


_NO_DATA = cast(RtCalibrationData, None)


def test_polish_descends_to_the_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    # The corner start sits on the descent line straight into the
    # minimum, so the wave search must land near the center.
    center = np.array([0.0, 0.0, 0.0])
    objective = _QuadraticObjective(center=center, scale=1.0)
    _patch_polish_surrogate(monkeypatch, objective)
    start = _start([-1.0, -1.0, -1.0], objective(np.array([-1.0, -1.0, -1.0])), object())

    polished = calibration.polish(_NO_DATA, start, bounds=[(-1.0, 1.0)] * 3)

    np.testing.assert_allclose(polished.best_params, center, atol=0.2)
    assert polished.objective < start.objective
    assert polished.param_names == start.param_names
    assert polished.optimize_result.status == 0
    assert polished.optimize_result.message in {
        "schedule",
        "parabola",
        "improvement",
        "bracket",
        "budget",
    }


def test_polish_bounds_are_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    # The minimum lies outside the box, so the best reachable point is
    # on the box's boundary and no coordinate may leave its (lo, hi).
    center = np.array([2.0, 2.0, 2.0])
    objective = _QuadraticObjective(center=center, scale=1.0)
    _patch_polish_surrogate(monkeypatch, objective)
    start = _start([0.5, 0.5, 0.5], objective(np.array([0.5, 0.5, 0.5])), object())

    bounds = [(0.0, 1.0)] * 3
    polished = calibration.polish(_NO_DATA, start, bounds=bounds)

    for value, (lo, hi) in zip(polished.best_params, bounds):
        assert lo <= value <= hi
    assert polished.objective < start.objective


def test_polish_flat_objective_returns_start(monkeypatch: pytest.MonkeyPatch) -> None:
    objective = _QuadraticObjective(center=np.array([0.5, 0.5, 0.5]), scale=0.0)
    sentinel = _patch_polish_surrogate(monkeypatch, objective)
    # The start's objective must be the objective's own value at the
    # start point: the engine seeds the base value from it instead of
    # re-running the point.
    start_params = [0.2, 0.4, 0.6]
    start = _start(start_params, objective(np.array(start_params)), sentinel)

    polished = calibration.polish(_NO_DATA, start, bounds=[(-1.0, 1.0)] * 3)

    np.testing.assert_allclose(polished.best_params, start.best_params)
    assert polished.objective == start.objective
    assert polished.best_results is sentinel
    assert polished.optimize_result.nit == 0
    assert polished.optimize_result.message == "no descent direction: flat objective"


# ---------------------------------------------------------------------------
# Picklability — the forkserver pool pickles the objective as an initarg
# ---------------------------------------------------------------------------


def _minimal_rt_data() -> RtCalibrationData:
    """The minimal four-timestep fixture used by `test_rt_calibration`."""
    t_idx = [dt.datetime(2020, 1, 1) + dt.timedelta(days=i) for i in range(4)]
    return RtCalibrationData(
        model_type=HbvModel,
        network=get_simple_carbon_network(),
        zone_configs={
            "surface": RtZoneConfiguration(do_reactions=True, do_speciation=True),
            "shallow": RtZoneConfiguration(do_reactions=True, do_speciation=True),
            "deep": RtZoneConfiguration(do_reactions=False, do_speciation=True),
        },
        forc=ForcingData(
            precip=Series([1.0, 1.0, 1.0, 1.0], index=t_idx),
            temp=Series([5.0, 5.0, 5.0, 5.0], index=t_idx),
            pet=Series([0.5, 0.5, 0.5, 0.5], index=t_idx),
        ),
        precip_conc=np.zeros((1, 4, 1)),
        q_meas=Series([1.0, 2.0, 3.0, 4.0], index=t_idx),
        conc_df=DataFrame({"DOC": [1.0, 2.0, 3.0, 4.0]}, index=t_idx),
        target_species="DOC",
    )


def test_rt_objective_is_picklable() -> None:
    data = _minimal_rt_data()
    objective = make_objective(data, DEFAULT_RT_OBJECTIVE_SPEC)

    restored = pickle.loads(pickle.dumps(objective))

    assert isinstance(restored, RtObjective)
    # Dataclass field equality would choke on the arrays' truth values,
    # so compare the fields a pickle round trip must preserve.
    assert restored.data.model_type is data.model_type
    assert restored.data.target_species == data.target_species
    assert restored.spec == DEFAULT_RT_OBJECTIVE_SPEC


def test_map_objectives_round_trips_through_a_real_forkserver_pool() -> None:
    # Not monkeypatched: the objective really is pickled into the
    # worker processes as the pool's initarg.
    objective = _QuadraticObjective(center=np.array([1.0, 2.0]), scale=2.0)
    points = [np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([1.0, 4.0])]

    values = calibration._map_objectives(objective, points, workers=2)

    assert values == pytest.approx([0.0, 2.0, 8.0])
