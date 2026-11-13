"""
The shared wave-search engine behind the line-search jobs.

A line search follows the straight line ``x + t * direction`` through
parameter space and looks for the step ``t`` whose model output is
best.  This module owns the schedule that finds it: an initial grid of
steps across the line's feasible interval, then successively smaller
"waves" of steps recentered on the parabola through the best steps
found so far, with improvement, bracket-width, and evaluation-budget
bars that stop the schedule once further waves stop paying for
themselves.  Each wave is a single batch, so every step of a wave can
be evaluated in parallel.

The engine never evaluates a model itself.  It asks a caller-supplied
`BatchEvaluator` to score each batch of points it selects, so the same
schedule drives any batch evaluator: the GUI's line-search job scores
its batches through a worker-pool session, and calibration polishing
scores its batches through its own parallel map.  A point whose value
is already known — from a gradient run, the base run, or an earlier
wave — is passed in ``initial_values`` and is never scored again.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

BatchEvaluator = Callable[[Sequence[np.ndarray]], list[float]]
PointKey = tuple[float, ...]

BOUND_TOLERANCE: float = 1e-9
VERTEX_CURVATURE_TOLERANCE: float = 1e-12
DEGENERATE_BRACKET_FRACTION: float = 0.25


@dataclass(slots=True)
class LineSearchConfig:
    """
    The wave schedule and the bars that bound one line search.

    Attributes:
        gradient_mode:
            The difference mode the caller's direction was computed
            with.  The engine itself only ships the "forward" mode.

        wave_schedule:
            The number of evaluation points in each wave; the first
            entry sizes the initial grid, the rest size the recentered
            waves.  The schedule is exhaustive: the search runs its
            waves until a bar stops it or the schedule is used up.

        min_relative_improvement:
            A wave must improve the best objective by at least this
            fraction of the previous best, or the search stops
            ("improvement").

        bracket_width_fraction:
            The search stops ("bracket") when the best-three bracket
            is narrower than this fraction of the full feasible
            interval.

        max_points:
            The maximum number of distinct points a search may hand to
            the evaluator; a wave batch is trimmed to the remaining
            budget.

        minimize:
            When True the search minimizes the raw value (calibration
            descent); the default False maximizes it (the GUI's
            ascent).
    """

    gradient_mode: str = "forward"
    wave_schedule: tuple[int, ...] = (7, 5, 3)
    min_relative_improvement: float = 1e-3
    bracket_width_fraction: float = 0.01
    max_points: int = 40
    minimize: bool = False


@dataclass(slots=True)
class LineSearchResult:
    """
    The outcome of one wave search along one line.

    Attributes:
        t_points:
            Every step evaluated on the line, ascending, including
            ``t = 0`` (the base point).

        values:
            The raw objective values aligned with `t_points`; a step
            whose evaluation was not finite keeps its NaN.

        t_best:
            The step with the best finite value.

        best_value:
            The raw value at `t_best` (not an internally sign-flipped
            objective).

        evaluations:
            The number of distinct points handed to the evaluator
            during the search.

        stopped_by:
            Why the schedule ended: "improvement", "bracket",
            "budget", "parabola", or "schedule" (all waves ran).
    """

    t_points: np.ndarray
    values: np.ndarray
    t_best: float
    best_value: float
    evaluations: int
    stopped_by: str


def point_key(x: np.ndarray) -> PointKey:
    """
    The hashable identity of a parameter point.

    Deduplication across waves, gradient offsets, and memoized values
    compares points by their exact float elements, so a step that
    lands on an already-scored point is recognized without a
    re-evaluation.

    Args:
        x: The parameter point.

    Returns:
        The point's elements as a tuple of Python floats.
    """
    return tuple(float(value) for value in x)


def search_t_points(
    t_lo: float,
    t_hi: float,
    n: int,
) -> np.ndarray:
    """
    The equally spaced steps of the line search's grid.

    The grid spans the feasible interval ``[t_lo, t_hi]`` with ``n``
    equally spaced points.  A plain linspace over an asymmetric
    interval generally does not land exactly on ``t = 0`` (the base
    point), which the search must always evaluate; the point nearest
    to zero is therefore snapped to exactly zero, so the grid always
    includes the base point while keeping ``n`` points across the
    whole interval.

    Args:
        t_lo: The interval's lower end, the smallest feasible step.
        t_hi: The interval's upper end, the largest feasible step.
        n: The number of grid points.

    Returns:
        The grid of ``n`` steps, strictly increasing, with the point
        nearest to zero set to exactly 0.0.
    """
    points: np.ndarray = np.linspace(t_lo, t_hi, n)
    points[int(np.argmin(np.abs(points)))] = 0.0
    return points


def feasible_t_interval(
    x: np.ndarray,
    direction: np.ndarray,
    positions: Sequence[int],
    bound_pairs: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
    """
    The interval of ascent-line steps that stays inside the bounds.

    The ascent line is ``x + t * direction``; a step ``t`` is feasible
    when every parameter that moves stays inside its current range.
    Each such parameter bounds ``t`` on one or both ends (a parameter
    whose direction component is zero does not move at all and merely
    has to sit in bounds at the base point), and the feasible interval
    is the intersection of those bounds.  When the base point is in
    bounds, the interval contains ``t = 0``.

    Args:
        x: The base parameter vector.
        direction: The unit-length ascent direction, in the same
            order as `x`.
        positions: The indices in `x` of the parameters that must
            stay in bounds.
        bound_pairs: The (low, high) bound of each of those
            parameters, in `positions`' order.

    Returns:
        The (lower, upper) feasible interval of ``t``, or None when no
        step keeps every listed parameter inside its bounds.
    """
    t_lo: float = float("-inf")
    t_hi: float = float("inf")
    for position, (low, high) in zip(positions, bound_pairs):
        base: float = float(x[position])
        delta: float = float(direction[position])
        if delta == 0.0:
            if not low <= base <= high:
                return None
            continue
        if delta > 0.0:
            t_lo = max(t_lo, (low - base) / delta)
            t_hi = min(t_hi, (high - base) / delta)
        else:
            t_lo = max(t_lo, (high - base) / delta)
            t_hi = min(t_hi, (low - base) / delta)
    if t_lo >= t_hi or not (np.isfinite(t_lo) and np.isfinite(t_hi)):
        return None
    return (t_lo, t_hi)


def project_direction(
    x: np.ndarray,
    direction: np.ndarray,
    positions: Sequence[int],
    bound_pairs: Sequence[tuple[float, float]],
    names: Sequence[str] | None = None,
) -> np.ndarray:
    """
    The ascent direction with every out-of-bounds component dropped.

    An ascent direction computed at a base point can point out of the
    bounds for a parameter that sits exactly on a bound: for such a
    parameter, stepping outward leaves the bounds immediately, so the
    interval of feasible steps collapses to the base point itself.
    This zeroes out each component that points out of the bound its
    base value sits on (a negative component at the lower bound, a
    positive one at the upper bound) and renormalizes the surviving
    vector to unit length, so the search only ever moves along
    directions that stay feasible near the base point.

    A component is treated as pointing out of a bound when the base
    value sits within a small relative tolerance of that bound
    (`1e-9` times the parameter's range), to absorb float error around
    a value that lands exactly on a bound.

    Args:
        x: The base parameter vector.
        direction: The ascent direction, in the same order as `x`.
        positions: The indices in `x` of the parameters whose
            components may need dropping.
        bound_pairs: The (low, high) bound of each of those
            parameters, in `positions`' order.
        names: Optional human-readable names for the parameters, in
            `positions`' order.  When given, a failure from blocked
            parameters names them instead of their positions.

    Returns:
        The projected direction as a unit-length vector.

    Raises:
        ValueError:
            Raised when no component survives the projection: every
            moving parameter sits on the bound its component points
            out of (the message names the blocked parameters, by
            `names` when they were supplied and by position
            otherwise), or the direction carries no component along
            any of them.
    """
    blocked: list[int] = []
    blocked_names: list[str] = []
    projected: np.ndarray = np.array(direction, dtype=np.float64, copy=True)
    for entry, (position, (low, high)) in enumerate(
        zip(positions, bound_pairs)
    ):
        component: float = float(projected[position])
        if component == 0.0:
            continue
        base: float = float(x[position])
        tolerance: float = BOUND_TOLERANCE * (high - low)
        if component < 0.0 and base - low <= tolerance:
            projected[position] = 0.0
            blocked.append(position)
            if names is not None:
                blocked_names.append(names[entry])
        elif component > 0.0 and high - base <= tolerance:
            projected[position] = 0.0
            blocked.append(position)
            if names is not None:
                blocked_names.append(names[entry])
    norm: float = float(np.linalg.norm(projected))
    if norm == 0.0:
        if blocked:
            if names is not None:
                blocked_labels: str = ", ".join(blocked_names)
            else:
                blocked_labels = "positions " + ", ".join(
                    str(position) for position in blocked
                )
            raise ValueError(
                "The line search has no feasible direction: the ascent "
                "direction points out of the current bounds at every "
                f"free parameter that can move ({blocked_labels})."
            )
        raise ValueError(
            "The line search has no feasible direction: the ascent "
            "direction carries no component along any free parameter."
        )
    return projected / norm


def expand_direction(
    row: np.ndarray,
    positions: Sequence[int],
    length: int,
) -> np.ndarray:
    """
    The full-length vector of a free-space direction row.

    A line search's direction is computed (or taken from a finished
    gradient job) as a row in the free parameters' space — one
    component per free parameter.  The search's arithmetic works in
    the full parameter vector's space, so this spreads the row's
    components into that space at the given positions, leaving the
    pinned parameters' components at zero (they never move).

    Args:
        row: The direction's components, in `positions`' order.
        positions: The indices in the full vector each component
            belongs to.
        length: The full parameter vector's length.

    Returns:
        The direction as a full-length vector.
    """
    full: np.ndarray = np.zeros(length, dtype=np.float64)
    for position, component in zip(positions, row):
        full[position] = component
    return full


def forward_gradient(
    evaluate: BatchEvaluator,
    x: np.ndarray,
    offsets: Sequence[np.ndarray],
    positions: Sequence[int],
    initial_values: Mapping[PointKey, float],
) -> np.ndarray:
    """
    The forward-difference gradient of a batch evaluator at a point.

    Each component is the one-sided difference along its offset:
    ``(f(x + offset) - f(x)) / offset[position]``, where ``position``
    is the offset's perturbed parameter.  Every new perturbation point
    is evaluated in a single batch (so the differences can run in
    parallel); a point already in `initial_values` — the base point or
    a duplicate offset — comes from the memo and is never re-evaluated,
    and an offset whose effective step is zero contributes exactly
    ``0.0`` without being evaluated.

    Args:
        evaluate: The batch evaluator scoring the perturbation points.
        x: The point the gradient is taken at.
        offsets: One full-length perturbation vector per component.
        positions: One per offset: the index of its perturbed
            parameter.
        initial_values: Already-known values keyed by `point_key`,
            including at least the base point's when it is worth
            preserving an existing evaluation.

    Returns:
        The gradient components, in `offsets`' order.
    """
    base: np.ndarray = np.asarray(x, dtype=np.float64)
    base_key: PointKey = point_key(base)
    memo: dict[PointKey, float] = dict(initial_values)

    steps: list[tuple[PointKey, float]] = []
    batch: list[np.ndarray] = []
    for offset, position in zip(offsets, positions):
        step: float = float(offset[position])
        if step == 0.0:
            continue
        key: PointKey = point_key(base + np.asarray(offset, dtype=np.float64))
        if key in memo or any(seen == key for seen, _ in steps):
            continue
        steps.append((key, step))
        batch.append(base + np.asarray(offset, dtype=np.float64))

    points_to_score: list[np.ndarray] = (
        [base] if base_key not in memo else []
    ) + batch
    _evaluate_new_points(evaluate, memo, points_to_score)

    base_value: float = memo[base_key]
    components: list[float] = []
    for offset, position in zip(offsets, positions):
        delta: float = float(offset[position])
        if delta == 0.0:
            components.append(0.0)
        else:
            perturbed: PointKey = point_key(
                base + np.asarray(offset, dtype=np.float64)
            )
            components.append((memo[perturbed] - base_value) / delta)
    return np.asarray(components, dtype=np.float64)


def _evaluate_new_points(
    evaluate: BatchEvaluator,
    memo: dict[PointKey, float],
    points: Sequence[np.ndarray],
) -> int:
    """
    Score every not-yet-scored point, exactly once, in one batch.

    Args:
        evaluate: The batch evaluator.
        memo: The value memo keyed by `point_key`, updated in place
            with each newly scored point.
        points: The candidate points; already-memoized ones are
            skipped.

    Returns:
        The number of points handed to ``evaluate`` (zero, without a
        call, when every point is already known).
    """
    new_points: list[np.ndarray] = [
        point for point in points if point_key(point) not in memo
    ]
    if not new_points:
        return 0
    values: list[float] = evaluate(new_points)
    for point, value in zip(new_points, values):
        memo[point_key(point)] = float(value)
    return len(new_points)


def _finite_objectives(
    steps_by_key: Mapping[PointKey, float],
    memo: Mapping[PointKey, float],
    sign: float,
) -> list[tuple[float, float]]:
    """
    The on-line evaluated steps as ascending (step, objective) pairs.

    Args:
        steps_by_key: Each evaluated on-line point's step, keyed by
            `point_key`.
        memo: The points' raw values keyed by `point_key`.
        sign: The engine's objective sign (see `wave_search`).

    Returns:
        The (step, objective) pairs whose raw value is finite, sorted
        by step.
    """
    records: list[tuple[float, float]] = [
        (step, sign * memo[key])
        for key, step in steps_by_key.items()
        if math.isfinite(memo[key])
    ]
    return sorted(records)


def _best_objective(
    steps_by_key: Mapping[PointKey, float],
    memo: Mapping[PointKey, float],
    sign: float,
) -> float:
    """
    The best objective among the on-line finite values, or -inf.

    Args:
        steps_by_key: Each evaluated on-line point's step, keyed by
            `point_key`.
        memo: The points' raw values keyed by `point_key`.
        sign: The engine's objective sign (see `wave_search`).

    Returns:
        The best objective, or ``float("-inf")`` when no on-line value
        is finite.
    """
    objectives: list[float] = [
        sign * memo[key] for key in steps_by_key if math.isfinite(memo[key])
    ]
    if not objectives:
        return float("-inf")
    return max(objectives)


def _vertex_step(
    steps: Sequence[float],
    objectives: Sequence[float],
    best_step: float,
) -> float:
    """
    The parabola vertex implied by the best three steps.

    A quadratic through the best three (step, objective) pairs
    estimates where along the line the objective peaks.  A fit whose
    leading coefficient is (numerically) zero has no finite vertex —
    the points are collinear or flat — and the vertex falls back to
    the best step itself.

    Args:
        steps: The best three steps' ``t`` values.
        objectives: The objectives at those steps, in the same order.
        best_step: The best of the three steps, the vertex fallback.

    Returns:
        The vertex step, or `best_step` when no finite vertex exists.
    """
    coefficients: np.ndarray = np.polyfit(list(steps), list(objectives), 2)
    quadratic: float = float(coefficients[0])
    linear: float = float(coefficients[1])
    if abs(quadratic) <= VERTEX_CURVATURE_TOLERANCE:
        return best_step
    vertex: float = -linear / (2.0 * quadratic)
    if not math.isfinite(vertex):
        return best_step
    return vertex


def _bracket_interval(
    steps: Sequence[float],
    vertex: float,
    best_step: float,
    t_lo: float,
    t_hi: float,
) -> tuple[float, float]:
    """
    The next wave's bracket: the best three's span, widened and clamped.

    The bracket spans the best three steps, widened to include the
    fitted vertex (it may sit just outside the sampled points), and
    clamped to the feasible interval.  A bracket that clamps to (or
    starts as) a degenerate interval is centered on the best step with
    a quarter of the full interval on each side.

    Args:
        steps: The best three steps' ``t`` values.
        vertex: The fitted vertex step.
        best_step: The best of the three steps.
        t_lo: The feasible interval's lower end.
        t_hi: The feasible interval's upper end.

    Returns:
        The (lower, upper) bracket interval.
    """
    low: float = max(min(min(steps), vertex), t_lo)
    high: float = min(max(max(steps), vertex), t_hi)
    if low >= high:
        half: float = DEGENERATE_BRACKET_FRACTION * (t_hi - t_lo)
        low = max(best_step - half, t_lo)
        high = min(best_step + half, t_hi)
    return (low, high)


def wave_search(
    evaluate: BatchEvaluator,
    x: np.ndarray,
    direction: np.ndarray,
    t_lo: float,
    t_hi: float,
    config: LineSearchConfig,
    initial_values: Mapping[PointKey, float],
) -> LineSearchResult:
    """
    Search a line for its best step with batched, recentered waves.

    The search maximizes the raw value unless ``config.minimize`` is
    set; internally every value is multiplied by a sign so "best" and
    "improvement" always mean the largest objective, and the result
    reports the raw values back.  The schedule runs as follows:

    1. The base point is evaluated only when `initial_values` lacks
       it (a finished gradient job already scored it, typically).
    2. The first wave is the `search_t_points` grid across
       ``[t_lo, t_hi]``, every un-scored grid point evaluated in one
       batch.
    3. Each following wave refits a parabola through the best three
       evaluated steps, brackets its vertex with that triple, and
       evaluates an equally spaced batch inside the bracket — but
       first the bars are consulted: a previous wave that improved the
       best objective by less than ``min_relative_improvement`` stops
       the search ("improvement"), a bracket narrower than
       ``bracket_width_fraction`` of the interval stops it
       ("bracket"), and fewer than three finite values leaves the
       parabola nothing to fit ("parabola", the grid result stands).
       A batch is trimmed to the remaining ``max_points`` budget, and
       a batch with no point left to evaluate stops the search
       ("budget" after trimming, "schedule" when every candidate was
       already scored).
    4. When the schedule runs out, the search stopped by "schedule".

    Args:
        evaluate: The batch evaluator scoring every batch of points.
            It receives only un-scored points and returns one value
            per point, aligned; a value may be NaN.
        x: The line's base parameter vector (the ``t = 0`` point).
        direction: The unit-length line direction, same length as `x`.
        t_lo: The feasible interval's lower end.
        t_hi: The feasible interval's upper end.
        config: The wave schedule, bars, and budget (see
            `LineSearchConfig`).
        initial_values: Already-known values keyed by `point_key`; the
            base point and gradient-run points included here are never
            re-evaluated.

    Returns:
        The `LineSearchResult` with every on-line evaluated step, its
        raw value, the best finite step, the number of distinct points
        evaluated, and the stop reason.

    Raises:
        ValueError:
            Raised when `config.wave_schedule` is empty, or when no
            evaluated step produced a finite value.
    """
    if not config.wave_schedule:
        raise ValueError(
            "The wave schedule must contain at least one wave; "
            f"received {config.wave_schedule}."
        )
    base: np.ndarray = np.asarray(x, dtype=np.float64)
    line: np.ndarray = np.asarray(direction, dtype=np.float64)
    sign: float = -1.0 if config.minimize else 1.0

    memo: dict[PointKey, float] = dict(initial_values)
    base_key: PointKey = point_key(base)
    steps_by_key: dict[PointKey, float] = {base_key: 0.0}
    evaluations: int = _evaluate_new_points(evaluate, memo, [base])
    bests: list[float] = [sign * memo[base_key]]

    grid: list[tuple[float, np.ndarray]] = [
        (float(t), base + float(t) * line)
        for t in search_t_points(t_lo, t_hi, config.wave_schedule[0])
    ]
    steps_by_key.update({point_key(point): step for step, point in grid})
    evaluations += _evaluate_new_points(
        evaluate, memo, [point for _, point in grid]
    )
    bests.append(_best_objective(steps_by_key, memo, sign))

    stopped_by: str = "schedule"
    for wave_index in range(1, len(config.wave_schedule)):
        records: list[tuple[float, float]] = _finite_objectives(
            steps_by_key, memo, sign
        )
        if len(records) < 3:
            stopped_by = "parabola"
            break
        ranked: list[tuple[float, float]] = sorted(
            records, key=lambda record: record[1], reverse=True
        )
        best_three: list[tuple[float, float]] = ranked[:3]
        best_step: float = best_three[0][0]
        vertex: float = _vertex_step(
            [step for step, _ in best_three],
            [objective for _, objective in best_three],
            best_step,
        )
        bracket_lo, bracket_hi = _bracket_interval(
            [step for step, _ in best_three], vertex, best_step, t_lo, t_hi
        )
        if bests[-1] - bests[-2] < config.min_relative_improvement * abs(
            bests[-2]
        ):
            stopped_by = "improvement"
            break
        if bracket_hi - bracket_lo < config.bracket_width_fraction * (
            t_hi - t_lo
        ):
            stopped_by = "bracket"
            break

        candidates: list[tuple[PointKey, float]] = []
        batch: list[np.ndarray] = []
        for raw_step in np.linspace(bracket_lo, bracket_hi, config.wave_schedule[wave_index]):
            step: float = float(raw_step)
            point: np.ndarray = base + step * line
            key: PointKey = point_key(point)
            if key in memo:
                steps_by_key[key] = step
            else:
                candidates.append((key, step))
                batch.append(point)
        if not batch:
            stopped_by = "schedule"
            break
        remaining: int = max(0, config.max_points - evaluations)
        if len(batch) > remaining:
            candidates = candidates[:remaining]
            batch = batch[:remaining]
            if not batch:
                stopped_by = "budget"
                break
        evaluations += _evaluate_new_points(evaluate, memo, batch)
        steps_by_key.update({key: step for key, step in candidates})
        bests.append(_best_objective(steps_by_key, memo, sign))

    ordered: list[tuple[float, PointKey]] = sorted(
        (step, key) for key, step in steps_by_key.items()
    )
    t_points: np.ndarray = np.asarray(
        [step for step, _ in ordered], dtype=np.float64
    )
    values: np.ndarray = np.asarray(
        [memo[key] for _, key in ordered], dtype=np.float64
    )
    finite: list[tuple[float, float, PointKey]] = [
        (sign * memo[key], step, key)
        for step, key in ordered
        if math.isfinite(memo[key])
    ]
    if not finite:
        raise ValueError(
            "The line search evaluated "
            f"{len(ordered)} steps on the interval [{t_lo}, {t_hi}] and "
            "none produced a finite value."
        )
    _, best_t, best_key = max(finite)
    return LineSearchResult(
        t_points=t_points,
        values=values,
        t_best=best_t,
        best_value=float(memo[best_key]),
        evaluations=evaluations,
        stopped_by=stopped_by,
    )
