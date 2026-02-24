from typing import Callable, Literal, TypeVar, TypedDict
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray
from pandas import DataFrame, Series

from .common_types import ForcingData

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


class HydroModelResults(TypedDict):
    """A dictionary containing the results of a hydrologic model run.

    Attributes:
        simulation (DataFrame): A DataFrame with time series of states and
            fluxes for all zones, plus simulated and measured streamflow.
        objective_functions (Series): A Series with the values of each of the objective functions as keys
    """

    simulation: DataFrame
    objective_functions: Series


def objective_function(
    x: NDArray,
    cls,
    forc: ForcingData,
    meas_streamflow: Series,
    metric: Literal["kge", "nse", "combined"] | Callable[[dict], float],
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
        obj_val = -results["objective_functions"]["kge"]  # type: ignore
    elif metric == "nse":
        obj_val = -results["objective_functions"]["nse"]  # type: ignore
    elif metric == "combined":
        obj_val = -results["objective_functions"]["kge"] - results["objective_functions"]["nse"]  # type: ignore
    else:
        obj_val = metric(results)

    if print_value and isinstance(obj_val, str):
        print(f"{metric.upper()}: {-round(obj_val, 2)}")  # type: ignore

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
    for param, (min_val, max_val) in zip(params, bounds.values(), strict=True):
        if param < min_val or param > max_val:
            return -np.inf
    return 0.0


def log_probability(
    theta: np.ndarray,
    model_type: type,
    forc: ForcingData | list[ForcingData],
    meas_streamflow: Series,
    bounds: dict[str, tuple[float, float]],
    metric: Callable[[HydroModelResults], float] | Literal["kge", "nse"],
    elevation: float | list[float] | None = None,
) -> tuple[float, list[float]]:
    """
    Computes the log probability for a given set of parameters.
    """
    try:
        lp = log_prior(theta, bounds)
        if not np.isfinite(lp):
            return -np.inf, [np.nan, np.nan, np.nan, np.nan, np.nan]

        model_res: HydroModelResults = model_type.from_array(theta, latent=True).run(  # type: ignore
            forc=forc, meas_streamflow=meas_streamflow, elevations=elevation
        )  # type: ignore

        obj = model_res["objective_functions"].to_dict()

        aux_values = [
            obj["kge"] if "kge" in obj else np.nan,
            obj["nse"] if "nse" in obj else np.nan,
            obj["log_kge"] if "log_kge" in obj else np.nan,
            obj["log_nse"] if "log_nse" in obj else np.nan,
            obj["bias"] if "bias" in obj else np.nan,
        ]

        if isinstance(metric, str):
            if metric == "kge":
                return lp + model_res["objective_functions"]["kge"], aux_values
            elif metric == "nse":
                return lp + model_res["objective_functions"]["nse"], aux_values
            else:
                raise ValueError(f"Unknown metric: {metric}")
        else:
            return lp + metric(model_res), aux_values
    except Exception:
        return -np.inf, [np.nan, np.nan, np.nan, np.nan, np.nan]


def jac(
    f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, dx: float = 1e-3
) -> np.ndarray:
    """Numerically estimate the jacobian matrix using a finite difference approximation"""
    jac_mat = np.zeros((x.size, x.size), dtype=np.float64)

    for i, _x_i in enumerate(x):
        x_up = x.copy()
        x_dn = x.copy()
        x_up[i] += dx
        x_dn[i] -= dx

        jac_mat[:, i] = (f(x_up) - f(x_dn)) / (2 * dx)

    return jac_mat


def find_root_multi(
    f: Callable[[NDArray], NDArray],
    x_0: NDArray,
    dx: float = 1e-3,
    max_iter: int = 25,
    tol: float = 1e-6,
    debug: bool = False,
) -> NDArray:
    x: NDArray = x_0.copy()
    f_x: NDArray = f(x)
    err: float = (f_x**2).mean()

    if debug:
        print(f"Initial f(x): {f_x}")
        print(f"Initial error: {err}")

    for i in range(max_iter):
        if err <= tol:
            return x

        jac_x: NDArray = jac(f, x, dx=dx)
        step: NDArray = np.linalg.solve(jac_x, f_x)
        x -= step
        f_x = f(x)
        err = (f_x**2).mean()

        if debug:
            print("-" * 10)
            print(f"Step {i}")
            print(f"f(x): {f_x}")
            print(f"Jacobian matrix: \n{jac_x}")
            print(f"Error: {err}")
            print()

    raise ValueError(f"Failed to find root starting at {x_0=}")


# ======================== #
