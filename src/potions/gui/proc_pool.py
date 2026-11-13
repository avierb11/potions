"""
Heartbeat-guarded forkserver pool and the GUI's model-run session.

Both the gradient job and the line-search job evaluate a batch of
parameter points by running the model once per point.  ``ModelRunSession``
runs a batch in one warm worker pool that stays open for the whole job,
so the input record is parsed once per worker process for the entire job
rather than once per phase.

Two properties make the pool safe inside a live Qt process:

* **forkserver, not fork.**  Forking a Qt process clones it while
  threads the child will never run still hold their locks: the children
  can park forever inside inherited mutexes.  A forkserver child is
  built from a clean process image instead, so it inherits no Qt
  thread state at all.  (Task and result values are pickled across the
  boundary either way.)
* **a shared heartbeat with a watchdog.**  Every worker stamps a shared
  monotonic timestamp when its initializer completes and when each of
  its tasks starts.  The parent polls task completions without blocking
  and, if neither a result nor a heartbeat advances for ``stall_timeout``
  seconds, terminates the pool and raises ``ModelRunStalledError``
  naming the points that never completed.  A stalled pool is therefore
  reported in about ``stall_timeout`` seconds instead of hanging the
  job forever.

The per-point error contract is unchanged from the old fork pool: a
worker whose initializer fails (for example, a missing record) stays
alive and reports the initializer's error string for every point it is
given, and a point whose run raises is reported as an
``"ExceptionType: message"`` string.  A batch therefore always yields
exactly one entry per point.
"""

import os
import sys
import time
import multiprocessing
from pathlib import Path
from typing import Any, Callable, Final, Sequence

import numpy as np

import potions as pt

from potions.gui import data as gui_data
from potions.gui.model_specs import DEFAULT_MODEL_KEY
from potions.gui.state import MODE_HYDRO

# Seconds a batch may run without a single result or heartbeat before
# the watchdog calls the pool stalled, and how often the parent polls.
DEFAULT_STALL_TIMEOUT: Final[float] = 600.0
POLL_INTERVAL: Final[float] = 0.05

# This process's shared heartbeat proxy, set in every worker by the
# guarded initializer before any task runs.  (``multiprocessing``'s
# shared-value proxy is untyped, hence ``Any``.)
HEARTBEAT: Any = None

# The input data loaded by this process's initializer, and the
# initializer's failure message when the load did not succeed.  Both
# are set once per worker process, before any of its tasks runs.
INPUT_DATA: gui_data.GuiInputData | None = None
INIT_ERROR: str | None = None


class ModelRunStalledError(RuntimeError):
    """
    Raised when a pool produced no result and no heartbeat activity
    for the whole stall timeout.  The pool is terminated before the
    raise; the message names the input indices whose results never
    arrived.
    """


class HeartbeatPool:
    """
    A forkserver worker pool watched through a shared heartbeat.

    The pool is built at construction and stays open across ``map``
    calls; ``close`` shuts it down.  Each submitted argument is handed
    to the task function paired with its input index, and ``map``
    returns the task's return values aligned to the input order.
    Results are collected by polling, never by blocking, so the
    watchdog can judge a batch stalled while its tasks are still
    outstanding.
    """

    def __init__(
        self,
        task: Callable[[tuple[Any, ...]], Any],
        initializer: Callable[..., None],
        initargs: tuple[Any, ...],
        workers: int | None = None,
        stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    ) -> None:
        """
        Start a forkserver pool of ``workers`` processes running ``task``.

        Args:
            task:
                Module-level (picklable) function called in a worker
                with the ``(input index, submitted argument)`` pair.

            initializer:
                Module-level (picklable) worker initializer; it is
                called with the shared heartbeat as its first argument
                followed by ``initargs``, and returning from it is
                what heartbeats the worker's start.

            initargs:
                The initializer's own arguments, pickled to every
                worker at pool start.

            workers:
                Number of worker processes.  Defaults to the number of
                available CPUs.

            stall_timeout:
                Seconds of silence — no completed result and no
                heartbeat advance — after which ``map`` raises
                ``ModelRunStalledError``.

        Raises:
            ValueError:
                Raised by ``multiprocessing`` when an initarg cannot
                be pickled to the workers.
        """
        self._task = task
        self._stall_timeout = stall_timeout
        self._closed = False
        context = multiprocessing.get_context("forkserver")
        # Exposed (not private) because the watchdog's liveness signal
        # is worth inspecting from the outside — tests and callers
        # diagnosing a stall both read it.
        self.heartbeat: Any = context.Value("d", time.monotonic())
        self._pool = context.Pool(
            processes=workers if workers is not None else (os.cpu_count() or 1),
            initializer=guarded_initializer,
            initargs=(self.heartbeat, initializer, *initargs),
        )

    def map(
        self,
        args: Sequence[Any],
        on_result: Callable[[int, Any], None] | None = None,
    ) -> list[Any]:
        """
        Run one argument per pool task and return the task results in
        input order.

        Submission order is the input order; completion order is
        arbitrary and each completion is handed to ``on_result`` in the
        parent as it arrives.  The pool stays open afterwards — call
        ``close`` when the job is done.

        Args:
            args:
                One submitted argument per task; each is paired with
                its input index when the task is called.

            on_result:
                Optional callback invoked in the parent as each point
                completes, with the input index and the task's return
                value.

        Returns:
            The task's return values, aligned to the input order.  An
            empty ``args`` yields ``[]`` without touching the pool.

        Raises:
            ModelRunStalledError:
                Raised when no result completes and the heartbeat does
                not advance for ``stall_timeout`` seconds; the pool is
                terminated and the message names the pending input
                indices.

            RuntimeError:
                Raised when the pool was already closed.
        """
        if self._closed:
            raise RuntimeError(
                "HeartbeatPool.map called on a closed pool: its workers "
                "were terminated and it cannot run new points."
            )
        if not args:
            return []

        pending: dict[int, Any] = {
            index: self._pool.apply_async(guarded_task, (self._task, (index, arg)))
            for index, arg in enumerate(args)
        }
        results: dict[int, Any] = {}
        last_activity: float = time.monotonic()
        while pending:
            for index in sorted(pending):
                ready = pending[index]
                if ready.ready():
                    payload = ready.get()
                    results[index] = payload
                    del pending[index]
                    last_activity = time.monotonic()
                    if on_result is not None:
                        on_result(index, payload)
                    break  # rescan the rest from a fresh state
            if pending:
                silent: float = time.monotonic() - max(
                    self.heartbeat.value, last_activity
                )
                if silent > self._stall_timeout:
                    stalled: list[int] = sorted(pending)
                    self._pool.terminate()
                    self._closed = True
                    raise ModelRunStalledError(
                        f"No worker returned a result for {self._stall_timeout:.0f} "
                        f"s: the point(s) {stalled} never completed, and the pool "
                        "was terminated. The most likely cause is a worker "
                        "stalling before its initializer finished."
                    )
                time.sleep(POLL_INTERVAL)
        return [results[index] for index in range(len(args))]

    def close(self) -> None:
        """
        Terminate and reap the pool's workers.  Calling it more than
        once is harmless, and it is safe (and expected) after the
        watchdog has already terminated the pool.
        """
        if self._closed:
            return
        self._closed = True
        self._pool.terminate()
        self._pool.join()


def guarded_initializer(
    heartbeat: Any, initializer: Callable[..., None], *initargs: Any
) -> None:
    """
    The pool's own initializer: bind this worker's heartbeat global,
    then run the job's initializer with the heartbeat as its first
    argument.  Every worker therefore reaches its task loop with
    ``HEARTBEAT`` bound.
    """
    global HEARTBEAT
    HEARTBEAT = heartbeat
    initializer(heartbeat, *initargs)


def guarded_task(task: Callable[[tuple[Any, ...]], Any], index_and_args: tuple[Any, ...]) -> Any:
    """
    The pool's own task wrapper: heartbeat at the start of every task
    (a worker that starts a new task proves the pool is alive even
    while long tasks run), then run the job's task function.
    """
    if HEARTBEAT is None:
        raise RuntimeError(
            "A pool worker received a task before its guarded initializer "
            "bound the heartbeat."
        )
    HEARTBEAT.value = time.monotonic()
    return task(index_and_args)


def init_worker(
    heartbeat: Any,
    csv_path: str,
    model_key: str,
    network: pt.ReactionNetwork | None = None,
    column_map: dict[str, str | None] | None = None,
) -> None:
    """
    Pool initializer: load the input data once per worker process.

    The record is re-parsed in every worker (the loader is not
    cached); this happens once per process, not once per point.  The
    heartbeat is stamped as soon as the initializer starts, so a slow —
    but progressing — record load never looks stalled.  Stdout is reset
    to the raw file descriptor first, so no write made in this process
    can pass through the parent's Python-level capturing stream and
    emit a Qt signal from a foreign process.

    Failures are caught and stored in ``INIT_ERROR`` rather than
    re-raised: a worker that raised here would be respawned forever by
    the pool and its tasks would never complete, hanging the parent's
    result iterator.

    Args:
        heartbeat:
            This pool's shared heartbeat, handed in by the guarded
            initializer.
        csv_path:
            Path of the input record (a CSV or XLSX spreadsheet) this
            worker should load.
        model_key:
            The key of the selected model's spec; the worker builds the
            inputs, bounds, and zone configurations of that model.
        network:
            The run's reaction network; the worker's input data is
            bound to it (the default network when None).
        column_map:
            The role-to-column mapping the worker's record is read
            with; the roles are auto-detected when None (the shipped
            record always is).
    """
    global INPUT_DATA, INIT_ERROR
    heartbeat.value = time.monotonic()
    try:
        sys.stdout = os.fdopen(1, "w", closefd=False)
        INPUT_DATA = gui_data.load_input_data(
            Path(csv_path),
            model_key=model_key,
            network=network,
            column_map=column_map,
        )
        INIT_ERROR = None
    except Exception as err:
        INPUT_DATA = None
        INIT_ERROR = f"{type(err).__name__}: {err}"


def run_one_task(
    index_and_arg: tuple[int, tuple[np.ndarray, str]]
) -> tuple[int, gui_data.GuiRunResults | str]:
    """
    Run one point: the pool's task function, executed in a worker.

    Dispatches to the full or hydro-only model conversion based on the
    mode string.  Any exception the conversion raises is reported back
    as an ``"ExceptionType: message"`` string instead of crashing the
    worker, so every point in a batch always yields exactly one entry
    (a ``GuiRunResults`` or an error string) in the parent.

    Args:
        index_and_arg:
            The point's input index paired with its ``(parameter
            array, mode)`` argument; the mode is one of the ``state``
            module's mode strings.

    Returns:
        The point's input index and its ``GuiRunResults``, or the error
        string that explains the failure (an initializer failure is
        reported for every point the worker receives).
    """
    index, (point, mode) = index_and_arg
    if INPUT_DATA is None:
        return index, INIT_ERROR or "worker was not initialized"
    try:
        if mode == MODE_HYDRO:
            results = gui_data.arr_to_hydro_results(point, INPUT_DATA)
        else:
            results = gui_data.arr_to_results(point, INPUT_DATA)
    except Exception as err:
        return index, f"{type(err).__name__}: {err}"
    return index, results


class ModelRunSession:
    """
    One warm heartbeat-guarded pool for one job's model-run batches.

    A job (a gradient run, a line search) evaluates its points in
    several batches; a session keeps a single pool open across those
    batches so the record is parsed once per worker for the whole job
    rather than re-parsed per batch.  Each batch is one ``map`` call,
    and the pool is torn down exactly once by ``close``.
    """

    def __init__(
        self,
        csv_path: str,
        model_key: str = DEFAULT_MODEL_KEY,
        network: pt.ReactionNetwork | None = None,
        column_map: dict[str, str | None] | None = None,
        workers: int | None = None,
        stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    ) -> None:
        """
        Start the session's pool: ``workers`` processes, each loading
        the record at ``csv_path`` for ``model_key`` (bound to
        ``network`` and read with ``column_map``) in its initializer.

        Args:
            csv_path:
                Path of the input record every worker loads.
            model_key:
                The selected model's spec key; defaults to the lateral
                three-layer model.
            network:
                The run's reaction network (the default network when
                None).
            column_map:
                The role-to-column mapping for the record; the roles
                are auto-detected when None.
            workers:
                Number of worker processes; defaults to the CPU count.
            stall_timeout:
                The watchdog's stall timeout for every batch.
        """
        self._pool: HeartbeatPool = HeartbeatPool(
            run_one_task,
            init_worker,
            (csv_path, model_key, network, column_map),
            workers,
            stall_timeout,
        )

    def map(
        self,
        points: Sequence[np.ndarray],
        mode: str,
        on_result: (
            Callable[[int, gui_data.GuiRunResults | str], None] | None
        ) = None,
    ) -> list[gui_data.GuiRunResults | str]:
        """
        Run one batch of parameter points and return one payload per
        point in input order.

        Args:
            points:
                The parameter arrays to run, one per pool task.
            mode:
                The run mode, one of the ``state`` module's mode
                strings; it selects which model conversion each point
                gets.
            on_result:
                Optional callback invoked in the parent as each point
                completes, with the point's input index and its
                ``GuiRunResults`` or error string.

        Returns:
            One entry per input point, in input order: the point's
            ``GuiRunResults``, or an ``"ExceptionType: message"``
            string if that point failed.

        Raises:
            ModelRunStalledError:
                Raised when the batch stalls (see ``HeartbeatPool.map``).
        """

        def deliver(index: int, entry: tuple[int, gui_data.GuiRunResults | str]) -> None:
            if on_result is not None:
                on_result(index, entry[1])

        args: list[tuple[np.ndarray, str]] = [
            (np.asarray(point), mode) for point in points
        ]
        entries = self._pool.map(args, deliver if on_result is not None else None)
        return [entry[1] for entry in entries]

    def close(self) -> None:
        """Terminate the session's pool; safe to call more than once."""
        self._pool.close()
