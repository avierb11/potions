from typing import Callable, Literal, TypeVar
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray
from pandas import Series

from .common_types import ForcingData, HydroModelResults

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
    metric: Literal["kge", "nse"],
    print_value: bool,
) -> float:
    model = cls.from_array(x, latent=True)
    results: HydroModelResults = model.run(
        init_state=cls.default_init_state(),
        forc=forc,
        meas_streamflow=meas_streamflow,
        verbose=False,
    )

    obj_val: float

    if metric == "kge":
        obj_val = -results.kge  # type: ignore
    elif metric == "nse":
        obj_val = -results.nse  # type: ignore
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
