"""
Batch model-run mapping for the gradient and line-search jobs.

Both jobs evaluate a batch of parameter points by running the model
once per point.  ``map_model_runs`` hands each batch to a
`proc_pool.ModelRunSession` — a heartbeat-guarded forkserver worker
pool — so the points are evaluated across the available cores instead
of one at a time, and a pool that stops making progress raises
``ModelRunStalledError`` after its stall timeout instead of hanging
the job forever.

Cost per ``map_model_runs`` call (a gradient run and each line-search
stage is one call):

* a fresh session (one pool) is built and torn down for every call —
  starting the forkserver workers takes a few hundred milliseconds;
* every worker process loads the input record once through the
  initializer, so a call costs one record parse per worker process
  rather than per point; the loader is not cached, so each call
  re-parses.

The per-point error contract is unchanged: a worker whose initializer
fails (for example, a missing record) stays alive and reports the
initializer's error string for every point it is given.  A batch
therefore always yields exactly one entry per point and the parent
never hangs waiting on a dead worker.
"""

from typing import Callable, Sequence

import numpy as np

import potions as pt

from potions.gui import data as gui_data
from potions.gui.model_specs import DEFAULT_MODEL_KEY
from potions.gui.proc_pool import ModelRunSession


def map_model_runs(
    points: Sequence[np.ndarray],
    mode: str,
    csv_path: str,
    workers: int | None = None,
    on_result: (
        Callable[[int, gui_data.GuiRunResults | str], None] | None
    ) = None,
    model_key: str = DEFAULT_MODEL_KEY,
    network: pt.ReactionNetwork | None = None,
    column_map: dict[str, str | None] | None = None,
) -> list[gui_data.GuiRunResults | str]:
    """
    Run a batch of parameter points in a heartbeat-guarded worker pool.

    Each point is modelled once in a forkserver worker process; the
    workers are initialized with the input data loaded from
    ``csv_path``.  The call blocks until every point has completed.

    Args:
        points:
            The parameter arrays to run, one per pool task.

        mode:
            The run mode, one of the ``state`` module's mode strings
            (``MODE_FULL`` or ``MODE_HYDRO``); it selects which
            model conversion each worker applies.

        csv_path:
            Path of the input record (a CSV or XLSX spreadsheet)
            every worker loads in its initializer.

        workers:
            Number of worker processes.  Defaults to the number of
            available CPUs.

        on_result:
            Optional callback invoked in the parent as each point
            completes, with the point's input index and its results
            or error string.  Completion order is arbitrary.

        model_key:
            The key of the selected model's spec; every worker loads
            that model's inputs, bounds, and zone configurations.
            Defaults to the lateral three-layer model.

        network:
            The run's reaction network; every worker's input data is
            bound to it (the default network when None).

        column_map:
            The role-to-column mapping every worker's record is read
            with; the roles are auto-detected when None.

    Returns:
        One entry per input point, in input order: the point's
        ``GuiRunResults``, or an ``"ExceptionType: message"`` string
        if that point failed (a failed initializer is reported for
        every point).  An empty ``points`` yields ``[]`` without
        building a session.

    Raises:
        ModelRunStalledError:
            Raised when the pool produced no result and no heartbeat
            activity for the whole stall timeout (see
            `proc_pool.ModelRunSession.map`).

    Note:
        The session (and its pool) is created and destroyed for each
        call, and every worker re-parses the record once in its
        initializer (see the module docstring for the cost of both).
    """
    if not points:
        return []
    session: ModelRunSession = ModelRunSession(
        csv_path, model_key, network, column_map, workers
    )
    try:
        return session.map(points, mode, on_result)
    finally:
        session.close()
