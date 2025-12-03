from typing import Callable, Literal, TypeVar
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray
from pandas import DataFrame, Series

from .common_types import ForcingData, HydroForcing

# ==== Types ==== #

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
Vector = np.ndarray[tuple[N], np.dtype[f64]]
Matrix = np.ndarray[tuple[M, N], np.dtype[f64]]
NumTot = TypeVar("NumTot", bound=int)  # Number of total species
NumPrim = TypeVar("NumPrim", bound=int)  # Number of primary aqueous species
NumMin = TypeVar("NumMin", bound=int)  # Number of mineral species
NumSec = TypeVar("NumSec", bound=int)  # Number of secondary aqueous species
NumSpec = TypeVar(
    "NumSpec", bound=int
)  # Number of species in the model - mineral and aqueous


def objective_function(
    x: NDArray,
    cls,
    forc: ForcingData,
    meas_streamflow: Series,
    metric: Literal["kge", "nse", "combined"],
    print_value: bool,
) -> float:
    model = cls.from_array(x, latent=True)
    results: dict[str, float | DataFrame] = model.run(
        init_state=cls.default_init_state(),
        forc=forc,
        meas_streamflow=meas_streamflow,
        verbose=False,
    )

    obj_val: float

    if metric == "kge":
        obj_val = -results["kge"]  # type: ignore
    elif metric == "nse":
        obj_val = -results["nse"]  # type: ignore
    elif metric == "combined":
        obj_val = -results["kge"] - results["nse"]  # type: ignore
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if print_value:
        print(f"{metric.upper()}: {-round(obj_val, 2)}")

    return obj_val


def find_root(f: Callable[[float], float], x_0: float, tol: float = 1e-5) -> float:
    x_1: float = x_0 + 0.1

    err = abs(f(x_1))

    counter = 0
    max_iter = 50

    while err > tol:
        fx_0 = f(x_0)
        fx_1 = f(x_1)
        if abs(fx_1 - fx_0) < 1e-12:
            break  # Avoid division by zero if points are too close
        x_n = (x_0 * fx_1 - x_1 * fx_0) / (fx_1 - fx_0)
        x_0, x_1 = x_1, x_n
        err = abs(f(x_n))
        counter += 1

        if counter >= max_iter:
            raise RuntimeError(
                f"Root finding failed to converge after {max_iter} iterations"
            )

    return x_1


# ==== MCMC functions ==== #
def log_prior(params: np.ndarray, bounds: dict[str, tuple[float, float]]) -> float:
    """
    Computes the log prior probability for a given set of parameters.
    """
    for param, (min_val, max_val) in zip(params, bounds.values()):
        if param < min_val or param > max_val:
            return -np.inf
    return 0.0


def log_probability(
    theta: np.ndarray,
    model_type: type,
    forc: ForcingData | list[ForcingData],
    meas_streamflow: Series,
    bounds: dict[str, tuple[float, float]],
    metric: Callable[[dict], float] | Literal["kge", "nse"],
    elevation: float | list[float] | None = None,
) -> tuple[float, list[float]]:
    """
    Computes the log probability for a given set of parameters.
    """
    lp = log_prior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf, [np.nan, np.nan, np.nan]

    model_res = model_type.from_array(theta, latent=True).run(
        forc=forc, meas_streamflow=meas_streamflow, elevations=elevation
    )

    aux_values: list[float] = [
        model_res["kge"],
        model_res["nse"],
        model_res["bias"],
    ]

    if isinstance(metric, str):
        if metric == "kge":
            return lp + model_res["kge"], aux_values
        elif metric == "nse":
            return lp + model_res["nse"], aux_values
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        return lp + metric(model_res), aux_values


# ======================== #

