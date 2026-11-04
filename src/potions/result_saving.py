"""Default persistence of model run results.

Every call to `run_hydro_model`, `run_rt_model` and `run_both_models`
saves, by default, the results of the run together with the parameters of
the model that produced them, into a fresh timestamped directory::

    results/run_20261013_213045/
        run_info.json             # metadata: time, model, metrics, versions
        parameters.csv            # every parameter of the model
        simulation.csv            # the simulated time series
        objective_functions.csv   # (only when metrics are available)

`run_both_models` writes both parts into one run directory, under `hydro/`
and `rt/` subdirectories.

The behavior is controlled in two ways, in order of precedence:

1. The per-call `save_results` / `results_dir` arguments (pass `None` to
   fall through to the next level).
2. The built-in defaults: saving is ON, and the base directory is `results`
   (relative to the current working directory).

The module can also be used directly to persist results objects that were
created by hand (`save_hydro_run`, `save_rt_run`, `save_model_run`).
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from pandas import DataFrame, Series

from .common_types import HydroModelResults, RtModelResults

#: Base directory for run outputs (relative to the working directory).
DEFAULT_RESULTS_DIR: Final[str] = "results"


# ==== Configuration ==== #


def is_save_enabled(save_results: bool | None = None) -> bool:
    """Decide whether a run should be saved to disk.

    An explicit `save_results` argument always wins; when it is `None`,
    saving is enabled (the built-in default).
    """
    return True if save_results is None else bool(save_results)


def resolve_results_dir(
    results_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve the base results directory to use for a run.

    An explicit `results_dir` argument wins over the built-in default
    (`"results"`).
    """
    return Path(results_dir) if results_dir is not None else Path(DEFAULT_RESULTS_DIR)


def make_run_dir(
    base_dir: str | os.PathLike[str], prefix: str = "run"
) -> Path:
    """Create a new `<base>/run_<YYYYMMDD_HHMMSS>` directory.

    The directory name is claimed atomically (`mkdir` without `exist_ok`), so
    concurrent callers cannot end up in the same directory. If the name is
    already taken (e.g. two runs within the same second), a counter suffix is
    appended (`..._1`, `..._2`, ...) until a free name is found. The base
    directory is created if it does not yet exist.
    """
    base: Path = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    stamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    counter: int = 0
    while True:
        name: str = f"{prefix}_{stamp}" if counter == 0 else f"{prefix}_{stamp}_{counter}"
        candidate: Path = base / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            counter += 1


# ==== Helpers ==== #


def _json_safe(value: Any) -> Any:
    """Make a scalar value safe for strict JSON serialization."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        fval: float = float(value)
        return fval if math.isfinite(fval) else None
    return value


def _package_version() -> str:
    try:
        return importlib.metadata.version("potions-model")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _num_zones(model: Any) -> int:
    try:
        return int(model.get_num_zones())
    except Exception:
        zones: Any = getattr(model, "zone_names", None)
        return len(zones) if zones is not None else 0


def _parameter_table(model: Any) -> DataFrame | None:
    """Fetch the model's parameter table (``describe_parameters``), if any."""
    fn: Any = getattr(model, "describe_parameters", None)
    if not callable(fn):
        return None
    return fn()


def _objective_dict(obj: Series | DataFrame | None) -> dict[str, Any] | None:
    """Convert objective metrics (Series or DataFrame) to a JSON-safe dict."""
    if obj is None:
        return None
    if isinstance(obj, Series):
        return {str(key): _json_safe(val) for key, val in obj.items()}
    out: dict[str, dict[str, Any]] = {}
    data = obj.to_dict(orient="index")
    for row_name, row_vals in data.items():
        out[str(row_name)] = {str(k): _json_safe(v) for k, v in row_vals.items()}
    return out


def _try_pickle(path: Path, obj: Any) -> bool:
    """Pickle `obj` to `path`; record a text note instead of raising on failure."""
    try:
        with path.open("wb") as fh:
            pickle.dump(obj, fh)
        return True
    except (TypeError, pickle.PicklingError) as err:
        note: Path = path.parent / (path.stem + ".pkl_error.txt")
        note.write_text(
            f"results.pkl could not be saved: {type(err).__name__}: {err}\n"
        )
        return False


def _write_bundle(
    out_dir: Path,
    *,
    run_id: str,
    model: Any,
    simulation: DataFrame,
    objective_functions: Series | DataFrame | None = None,
    pickle_obj: Any | None = None,
) -> None:
    """Write the standard file set for one model's results into `out_dir`."""
    simulation.to_csv(out_dir / "simulation.csv")

    params_df: DataFrame | None = _parameter_table(model)
    if params_df is not None and len(params_df) > 0:
        params_df.to_csv(out_dir / "parameters.csv", index=False)

    obj_dict: dict[str, Any] | None = _objective_dict(objective_functions)
    if objective_functions is not None and len(objective_functions) > 0:
        if isinstance(objective_functions, Series):
            out_df = objective_functions.to_frame(name="value")
        else:
            out_df = objective_functions
        out_df.to_csv(out_dir / "objective_functions.csv")

    pickle_saved: bool | None = None
    if pickle_obj is not None:
        pickle_saved = _try_pickle(out_dir / "results.pkl", pickle_obj)

    index = simulation.index
    info: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "potions_version": _package_version(),
        "python_version": _python_version(),
        "num_zones": _num_zones(model),
        "num_time_steps": len(index),
        "start_time": str(index.min()) if len(index) else None,
        "end_time": str(index.max()) if len(index) else None,
    }
    if obj_dict:
        info["objective_functions"] = obj_dict
    if pickle_saved is not None:
        info["results_pickle_saved"] = pickle_saved
    (out_dir / "run_info.json").write_text(
        json.dumps(info, indent=2, allow_nan=False) + "\n"
    )


# ==== Public save functions ==== #


def save_hydro_run(
    model: Any,
    results: HydroModelResults,
    results_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Save a `HydroModelResults` (plus the model parameters) to a fresh run dir."""
    out_dir: Path = make_run_dir(resolve_results_dir(results_dir))
    _write_bundle(
        out_dir,
        run_id=out_dir.name,
        model=model,
        simulation=results.simulation,
        objective_functions=results.objective_functions,
    )
    return out_dir


def save_rt_run(
    model: Any,
    results: RtModelResults,
    results_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Save an `RtModelResults` (plus parameters and a full pickle) to a run dir."""
    out_dir = make_run_dir(resolve_results_dir(results_dir))
    _write_bundle(
        out_dir,
        run_id=out_dir.name,
        model=model,
        simulation=results.simulation,
        objective_functions=results.objective_functions,
        pickle_obj=results,
    )
    return out_dir


def save_model_run(
    model: Any,
    hydro_results: HydroModelResults,
    rt_results: RtModelResults,
    results_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Save a combined hydro + reactive-transport run to a single run dir.

    Layout::

        results/run_<timestamp>/
            hydro/   parameters.csv, simulation.csv, objective_functions.csv, run_info.json
            rt/      parameters.csv, simulation.csv, objective_functions.csv, run_info.json, results.pkl
    """
    out_dir = make_run_dir(resolve_results_dir(results_dir))
    hydro_dir: Path = out_dir / "hydro"
    rt_dir: Path = out_dir / "rt"
    hydro_dir.mkdir()
    rt_dir.mkdir()
    _write_bundle(
        hydro_dir,
        run_id=out_dir.name,
        model=model,
        simulation=hydro_results.simulation,
        objective_functions=hydro_results.objective_functions,
    )
    _write_bundle(
        rt_dir,
        run_id=out_dir.name,
        model=model,
        simulation=rt_results.simulation,
        objective_functions=rt_results.objective_functions,
        pickle_obj=rt_results,
    )
    return out_dir


def write_batch_info(
    out_dir: str | os.PathLike[str],
    *,
    num_runs: int,
    num_threads: int,
    has_threshold: bool,
) -> None:
    """Write a `run_info.json` describing a batch run into `out_dir`."""
    out_dir = Path(out_dir)
    info: dict[str, Any] = {
        "run_id": out_dir.name,
        "kind": "batch",
        "timestamp": datetime.now().astimezone().isoformat(),
        "model_class": None,
        "potions_version": _package_version(),
        "python_version": _python_version(),
        "num_runs": num_runs,
        "num_threads": num_threads,
        "threshold_function_applied": has_threshold,
    }
    (out_dir / "run_info.json").write_text(
        json.dumps(info, indent=2, allow_nan=False) + "\n"
    )
