from dataclasses import dataclass
from typing import Literal, TypeVar
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
        streamflow=meas_streamflow,
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
