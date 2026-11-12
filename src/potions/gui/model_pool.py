"""
Shared worker pool for the gradient and line-search jobs.

Both jobs evaluate a batch of parameter points by running the model
once per point.  ``map_model_runs`` runs each batch in a
``multiprocessing.Pool`` so the points are evaluated across the
available cores instead of one at a time.

Cost per ``map_model_runs`` call (a gradient run and each
line-search stage is one call):

* a fresh pool is created and torn down (``close``/``join``) for
  every phase — forking the workers takes a few milliseconds;
* every worker process loads the input record once through the
  initializer, so a phase costs one record parse per worker process
  rather than per point; the loader is not cached, so each phase
  re-parses.

A worker whose initializer fails (for example, a missing record)
stays alive and reports the initializer's error string for every
point it is given.  A phase therefore always yields exactly one
entry per point and the parent never hangs waiting on a dead
worker: a worker that raised in its initializer would be respawned
forever by the pool, and its tasks would never complete.
"""

import os
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from multiprocessing import Pool

import potions as pt

from potions.gui import data as gui_data
from potions.gui.model_specs import DEFAULT_MODEL_KEY
from potions.gui.state import MODE_HYDRO

# The input data loaded by this process's initializer, and the
# initializer's failure message when the load did not succeed.  Both
# are set once per worker process, before any of its tasks runs.
_INPUT_DATA: gui_data.GuiInputData | None = None
_INIT_ERROR: str | None = None


def init_worker(
    csv_path: str,
    model_key: str,
    network: pt.ReactionNetwork | None = None,
    column_map: dict[str, str | None] | None = None,
) -> None:
    """
    Pool initializer: load the input data once per worker process.

    The record is re-parsed in every worker (the loader is not
    cached); this happens once per process, not once per point.
    Stdout is reset to the raw file descriptor first, so no write
    made in this process can pass through the parent's Python-level
    capturing stream and emit a Qt signal from a foreign process.

    Failures are caught and stored in ``_INIT_ERROR`` rather than
    re-raised: a worker that raised here would be respawned forever
    by the pool and its tasks would never complete, hanging the
    parent's result iterator.

    Args:
        csv_path:
            Path of the input record (a CSV or XLSX spreadsheet)
            this worker should load.
        model_key:
            The key of the selected model's spec; the worker builds
            the inputs, bounds, and zone configurations of that model.
        network:
            The run's reaction network; the worker's input data is
            bound to it (the default network when None).
        column_map:
            The role-to-column mapping the worker's record is read
            with; the roles are auto-detected when None (the
            shipped record always is).
    """
    global _INPUT_DATA, _INIT_ERROR
    try:
        sys.stdout = os.fdopen(1, "w", closefd=False)
        _INPUT_DATA = gui_data.load_input_data(
            Path(csv_path),
            model_key=model_key,
            network=network,
            column_map=column_map,
        )
        _INIT_ERROR = None
    except Exception as err:
        _INPUT_DATA = None
        _INIT_ERROR = f"{type(err).__name__}: {err}"


def _run_one_task(
    task: tuple[int, np.ndarray, str]
) -> tuple[int, gui_data.GuiRunResults | str]:
    """
    Run one point: the pool's task function, executed in a worker.

    Dispatches to the full or hydro-only model conversion based on
    the mode string.  Any exception the conversion raises is
    reported back as an ``"ExceptionType: message"`` string instead
    of crashing the worker, so every point in a batch always yields
    exactly one entry (a ``GuiRunResults`` or an error string) in
    the parent.

    Args:
        task:
            The ``(point index, parameter array, mode)`` triple; the
            mode is one of the ``state`` module's mode strings.

    Returns:
        The point's input index and its ``GuiRunResults``, or the
        error string that explains the failure (an initializer
        failure is reported for every point the worker receives).
    """
    index, point, mode = task
    if _INPUT_DATA is None:
        return index, _INIT_ERROR or "worker was not initialized"
    try:
        if mode == MODE_HYDRO:
            results = gui_data.arr_to_hydro_results(point, _INPUT_DATA)
        else:
            results = gui_data.arr_to_results(point, _INPUT_DATA)
    except Exception as err:
        return index, f"{type(err).__name__}: {err}"
    return index, results


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
    Run a batch of parameter points in a worker pool.

    Each point is modelled once in a forked worker process; the
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
        every point).

    Note:
        The pool is created and destroyed for each call, and every
        worker re-parses the record once in its initializer (see the
        module docstring for the cost of both).
    """
    global _INPUT_DATA, _INIT_ERROR
    _INPUT_DATA = None
    _INIT_ERROR = None

    if not points:
        return []

    tasks: list[tuple[int, np.ndarray, str]] = [
        (index, point, mode) for index, point in enumerate(points)
    ]
    by_index: dict[int, gui_data.GuiRunResults | str] = {}
    pool = Pool(
        processes=workers if workers is not None else (os.cpu_count() or 1),
        initializer=init_worker,
        initargs=(csv_path, model_key, network, column_map),
    )
    try:
        for index, payload in pool.imap_unordered(_run_one_task, tasks):
            by_index[index] = payload
            if on_result is not None:
                on_result(index, payload)
    finally:
        pool.close()
        pool.join()
    return [by_index[index] for index in range(len(tasks))]
