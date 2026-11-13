"""
Tests for the shared wave-search line-search engine (`potions.line_search`).

The engine is tested as a pure-numerical component: a surrogate
`evaluate` replaces the model runs, so every test is fast, deterministic,
and independent of Qt, multiprocessing, and the model.  Each surrogate is
a module-level function (the shape the real pool pickles), and the
`Surrogate` wrapper records every point the engine handed to
`evaluate`, which lets the tests pin the engine's contracts: batched
evaluation, no point ever evaluated twice, and each `stopped_by`
condition reachable.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pytest

from ..line_search import (
    BatchEvaluator,
    LineSearchConfig,
    LineSearchResult,
    feasible_t_interval,
    forward_gradient,
    point_key,
    project_direction,
    search_t_points,
    wave_search,
)

# ------------------------------------------------------------------ #
# Surrogates
# ------------------------------------------------------------------ #


class Surrogate:
    """
    A surrogate objective that records every point `evaluate` received.

    `points` is the flat record of every point handed to `evaluate`
    across all batches; `batches` is one entry per `evaluate` call, so
    a test can assert both the batching and the deduplication.
    """

    def __init__(self, value_of: Callable[[np.ndarray], float]) -> None:
        self._value_of = value_of
        self.points: list[np.ndarray] = []
        self.batches: list[list[np.ndarray]] = []

    def evaluate(self, points: Sequence[np.ndarray]) -> list[float]:
        arrays: list[np.ndarray] = [
            np.asarray(point, dtype=np.float64) for point in points
        ]
        self.batches.append(arrays)
        self.points.extend(arrays)
        return [self._value_of(point) for point in arrays]


def peak_at_04(point: np.ndarray) -> float:
    """The 1-D bump `-(t - 0.4)^2 + 1` read straight off the point."""
    t: float = float(point[0])
    return -(t - 0.4) ** 2 + 1.0


def bump_at_045(point: np.ndarray) -> float:
    """The 1-D quadratic bump `1 - (t - 0.45)^2` (peak off-center)."""
    t: float = float(point[0])
    return 1.0 - (t - 0.45) ** 2


def valley_at_04(point: np.ndarray) -> float:
    """The 1-D valley `(t - 0.4)^2` — the inverted bump, for minimize."""
    t: float = float(point[0])
    return (t - 0.4) ** 2


def quadratic_2d(point: np.ndarray) -> float:
    """`-(x0 - 1)^2 - (x1 - 2)^2`, the test-2 gradient surrogate."""
    return -(float(point[0]) - 1.0) ** 2 - (float(point[1]) - 2.0) ** 2


def flat(point: np.ndarray) -> float:
    """A constant: nothing a wave evaluates can improve on the base."""
    return 1.0


def sharp_quartic_at_045(point: np.ndarray) -> float:
    """A sharp quartic peak: `1 - 40 (t - 0.45)^4`."""
    t: float = float(point[0])
    return 1.0 - 40.0 * (t - 0.45) ** 4


def two_finite(point: np.ndarray) -> float:
    """Finite only at t = 0 and t = -1; NaN everywhere else."""
    t: float = float(point[0])
    if abs(t) <= 1e-12:
        return 1.0
    if abs(t + 1.0) <= 1e-12:
        return 0.5
    return float("nan")


def keys_of(points: Sequence[np.ndarray]) -> list[tuple[float, ...]]:
    """The point keys of a batch record, for the no-duplicates asserts."""
    return [point_key(point) for point in points]


# ------------------------------------------------------------------ #
# The engine
# ------------------------------------------------------------------ #


class TestSearchTPoints:
    """The canonical grid builder the engine and the GUI share."""

    def test_search_t_points_snaps_zero(self) -> None:
        """
        The grid is equally spaced across the feasible interval with
        the nearest point snapped to ``t = 0``: a grid that would miss
        the base point exactly still evaluates it, and the snapped grid
        stays strictly increasing.
        """
        symmetric: np.ndarray = search_t_points(-1.0, 1.0, 5)
        np.testing.assert_allclose(
            symmetric, np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        )
        snapped: np.ndarray = search_t_points(-1.0, 2.0, 5)
        np.testing.assert_allclose(
            snapped, np.array([-1.0, 0.0, 0.5, 1.25, 2.0])
        )
        assert (np.diff(snapped) > 0).all()


class TestForwardGradient:
    """The forward-difference gradient the line search uses for its direction."""

    def test_forward_gradient_matches_analytic_and_dedupes(self) -> None:
        """
        Each component is the one-sided difference at its position, the
        base and zero-step offsets never enter a batch, and every new
        point is evaluated exactly once in a single batch.
        """
        surrogate = Surrogate(quadratic_2d)
        base: np.ndarray = np.array([0.5, 3.0])
        offsets: list[np.ndarray] = [
            np.array([0.01, 0.0]),
            np.array([0.0, 0.01]),
            np.array([0.0, 0.0]),
        ]
        positions: list[int] = [0, 1, 0]
        initial: dict[tuple[float, ...], float] = {
            point_key(base): quadratic_2d(base)
        }

        gradient: np.ndarray = forward_gradient(
            surrogate.evaluate, base, offsets, positions, initial
        )

        # Analytic gradient of the surrogate at [0.5, 3.0] is
        # [1.0, -2.0]; the forward differences at h = 0.01 sit within
        # h of it.  The zero-step offset contributes exactly 0.0.
        assert gradient[0] == pytest.approx(1.0, abs=2e-2)
        assert gradient[1] == pytest.approx(-2.0, abs=2e-2)
        assert gradient[2] == 0.0
        # Only the two x + h*e_i points entered a batch — the base came
        # from the memo and the zero offset was never evaluated.
        assert len(surrogate.batches) == 1
        assert len(surrogate.points) == 2
        assert len(set(keys_of(surrogate.points))) == 2
        assert point_key(base) not in keys_of(surrogate.points)


class TestWaveSearch:
    """The wave schedule: bracket-recentered batches, improvement bars, budgets."""

    def test_wave_search_finds_interior_maximum(self) -> None:
        """The schedule homes in on a bump's peak strictly inside the interval."""
        surrogate = Surrogate(peak_at_04)
        result: LineSearchResult = wave_search(
            surrogate.evaluate,
            np.array([0.0]),
            np.array([1.0]),
            -1.0,
            1.0,
            LineSearchConfig(),
            {},
        )
        assert result.t_best == pytest.approx(0.4, abs=0.15)
        assert result.t_best == pytest.approx(0.4, abs=0.15)
        assert (np.diff(result.t_points) > 0).all()
        assert 0.0 in result.t_points
        assert len(result.values) == len(result.t_points)
        # Values stay aligned with the steps: every recorded value is the
        # surrogate evaluated at its own step.
        for t, value in zip(result.t_points, result.values):
            assert value == pytest.approx(peak_at_04(np.array([t])))
        assert result.evaluations <= 16

    def test_wave_search_stops_on_improvement_bar(self) -> None:
        """A flat plateau stops the schedule before the second wave runs."""
        surrogate = Surrogate(flat)
        result: LineSearchResult = wave_search(
            surrogate.evaluate,
            np.array([0.0]),
            np.array([1.0]),
            -1.0,
            1.0,
            LineSearchConfig(),
            {},
        )
        assert result.stopped_by == "improvement"
        assert result.evaluations <= 7 + 5

    def test_wave_search_bracket_shrink_stops(self) -> None:
        """
        A sharp peak collapses the best-three bracket below the width
        bar: the schedule stops by "bracket" before the third wave, and
        the best step lies inside the bracket it collapsed from.
        """
        surrogate = Surrogate(sharp_quartic_at_045)
        base: np.ndarray = np.array([0.0])
        config = LineSearchConfig(bracket_width_fraction=0.25)
        result: LineSearchResult = wave_search(
            surrogate.evaluate,
            base,
            np.array([1.0]),
            -1.0,
            1.0,
            config,
            {point_key(base): sharp_quartic_at_045(base)},
        )
        assert result.stopped_by == "bracket"
        # The wave-1 grid peaks near t = 0.333, so its best three span
        # [0, 0.667]; the collapse leaves the best step inside it.
        assert 0.0 <= result.t_best <= 0.667
        assert result.t_points is not None

    def test_wave_search_no_parabola_falls_back_to_grid(self) -> None:
        """
        With only two finite values the parabola has nothing to fit:
        the search stops by "parabola" at the grid and keeps the NaNs.
        """
        surrogate = Surrogate(two_finite)
        result: LineSearchResult = wave_search(
            surrogate.evaluate,
            np.array([0.0]),
            np.array([1.0]),
            -1.0,
            1.0,
            LineSearchConfig(),
            {},
        )
        assert result.stopped_by == "parabola"
        assert result.t_best == pytest.approx(0.0)
        assert result.best_value == pytest.approx(1.0)
        assert np.isnan(result.values).any()
        assert result.evaluations == 7

    def test_wave_search_budget(self) -> None:
        """A surrogate that never trips a bar runs the schedule inside the budget."""
        surrogate = Surrogate(bump_at_045)
        config = LineSearchConfig()
        result: LineSearchResult = wave_search(
            surrogate.evaluate,
            np.array([0.0]),
            np.array([1.0]),
            -1.0,
            1.0,
            config,
            {},
        )
        assert result.evaluations <= config.max_points
        # The waves sum to 15 < 40, so the schedule exhausts first.
        assert result.stopped_by in ("budget", "schedule")

    def test_wave_search_minimize_flag(self) -> None:
        """With minimize=True the engine reports the minimum raw value."""
        surrogate = Surrogate(valley_at_04)
        result: LineSearchResult = wave_search(
            surrogate.evaluate,
            np.array([0.0]),
            np.array([1.0]),
            -1.0,
            1.0,
            LineSearchConfig(minimize=True),
            {},
        )
        finite: list[float] = [
            float(value) for value in result.values if np.isfinite(value)
        ]
        assert result.best_value == min(finite)
        assert result.t_best == pytest.approx(0.4, abs=0.15)

    def test_dedup_never_reevaluates(self) -> None:
        """Waves that re-hit earlier grid points never hand a point to evaluate twice."""
        surrogate = Surrogate(bump_at_045)
        wave_search(
            surrogate.evaluate,
            np.array([0.0]),
            np.array([1.0]),
            -1.0,
            1.0,
            LineSearchConfig(),
            {},
        )
        keys: list[tuple[float, ...]] = keys_of(surrogate.points)
        assert len(keys) == len(set(keys))


class TestIntervalGeometry:
    """The generalized interval and projection geometry the GUI wrappers delegate to."""

    def test_feasible_t_interval_and_project_direction(self) -> None:
        """
        The interval is the intersection of the per-parameter bounds on
        ``t``, and a component pointing out of the bound its base value
        sits on is projected out — with the failure naming the blocked
        positions.
        """
        positions: list[int] = [0, 1]
        bound_pairs: list[tuple[float, float]] = [(0.0, 1.0), (0.0, 2.0)]
        interval: tuple[float, float] | None = feasible_t_interval(
            np.array([0.5, 1.0]), np.array([1.0, -1.0]), positions, bound_pairs
        )
        assert interval is not None
        assert interval[0] == pytest.approx(-0.5)
        assert interval[1] == pytest.approx(0.5)

        # A zero component whose base sits outside its bounds admits no
        # feasible step at all.
        blocked: tuple[float, float] | None = feasible_t_interval(
            np.array([2.0]), np.array([0.0]), [0], [(0.0, 1.0)]
        )
        assert blocked is None

        # Project direction (the GUI helper's numbers in generalized
        # form): `a` sits on its lower bound pointing down (dropped),
        # `c` on its upper bound pointing up (dropped), the rest survive.
        base_x: np.ndarray = np.array([0.0, 1.0, 1.5, 2.0, 0.0])
        raw: np.ndarray = np.array([-1.0, 2.0, 3.0, 4.0, 1.0])
        expected: np.ndarray = np.array([0.0, 2.0, 0.0, 4.0, 1.0])
        expected = expected / float(np.linalg.norm(expected))
        projected: np.ndarray = project_direction(
            base_x,
            raw,
            [0, 1, 2, 3, 4],
            [(0.0, 1.0), (0.0, 2.0), (0.5, 1.5), (0.0, 3.0), (0.0, 1.0)],
        )
        np.testing.assert_allclose(projected, expected)
        assert float(np.linalg.norm(projected)) == pytest.approx(1.0)

        # No component survives: the failure names the blocked positions.
        with pytest.raises(ValueError) as excinfo:
            project_direction(
                np.array([0.0, 1.5]),
                np.array([-1.0, 1.0]),
                [0, 1],
                [(0.0, 1.0), (0.5, 1.5)],
            )
        message: str = str(excinfo.value)
        assert "no feasible direction" in message
        assert "positions 0, 1" in message

        # No component along any free parameter at all.
        with pytest.raises(ValueError) as excinfo:
            project_direction(
                np.array([0.0]),
                np.array([0.0]),
                [0],
                [(0.0, 1.0)],
            )
        assert "no feasible direction" in str(excinfo.value)
