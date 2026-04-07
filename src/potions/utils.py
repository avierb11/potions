import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import os
from typing import Callable, Final, Iterable, Literal, Optional, TypeVar
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray
from pandas import DataFrame, Series

from potions.core import (
    HydroForcing,
    HydrologicZone,
    OptimizationError,
    RtForcing,
    RtZone,
)


from .common_types import ForcingData, HydroModelResults

# ==== Logger ==== #
DO_LOGGING = os.environ.get("POTIONSLOGGING") is not None
NOW: Final[datetime] = datetime.now()
NOW_TS: Final[str] = NOW.strftime("%Y%m%d_%H%M%S")
LOGGING_DIR: Final[str] = "./.potions_logs"
LOG_FILE_PATH: Final[str] = os.path.join(LOGGING_DIR, f"{NOW_TS}.log")


def setup_logging(file_path) -> None:
    if DO_LOGGING:
        if not os.path.exists(LOGGING_DIR):
            os.makedirs(LOGGING_DIR)
            with open(LOG_FILE_PATH, "w+"):
                pass
        handler = RotatingFileHandler(
            filename=LOG_FILE_PATH,
            maxBytes=10 * 1024 * 1024,
        )

        logging.basicConfig(
            filename=LOG_FILE_PATH,
            level=logging.INFO,
            format=f"{os.path.basename(file_path)}: %(message)s",
        )

        logging.getLogger().addHandler(handler)
    else:
        return


# ================ #

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
ZERO_CONC: Final[float] = 1e-20


# ==== Error Types ==== #
class PotionsError(Exception):
    pass


class HydrologyError(PotionsError):
    pass


class HydrologyNumericalError(HydrologyError):

    def __init__(
        self,
        model_type: type,
        zone: HydrologicZone,
        parameters: np.ndarray,
        state: float,
        hydro_forcing: HydroForcing,
        *args: object,
    ) -> None:
        super().__init__(*args)
        self.model_type: type = model_type
        self.zone: HydrologicZone = zone
        self.parameters: np.ndarray = parameters
        self.state: float = state
        self.hydro_forcing: HydroForcing = hydro_forcing


class ReactiveTransportError(PotionsError):
    pass


class RtNumericalError(ReactiveTransportError):

    def __init__(
        self,
        error_type: Literal["iteration", "linear_system", "other"],
        model_type: type,
        zone: RtZone,
        parameters: np.ndarray,
        state: np.ndarray,
        rt_forcing: RtForcing,
        math_err: OptimizationError,
        *args: object,
    ) -> None:
        super().__init__(*args)
        self.error_type = error_type
        self.model_type: type = model_type
        self.zone: RtZone = zone
        self.parameters: np.ndarray = parameters
        self.state: np.ndarray = state
        self.rt_forcing: RtForcing = rt_forcing
        self.math_err = math_err


# ===================== #


def objective_function(
    x: NDArray,
    cls,
    forc: ForcingData,
    meas_streamflow: Series,
    metric: Literal["kge", "nse", "combined"] | Callable[[dict], float],
    print_value: bool,
) -> float:
    model = cls.hydro_from_array(x, latent=True)

    try:
        results: dict[str, float | DataFrame] = model.run_hydro_model(
            init_state=cls.default_hydro_init_state(),
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
            obj_val = (
                -results["objective_functions"]["kge"]  # type: ignore
                - results["objective_functions"]["nse"]  # type: ignore
                - results["objective_functions"]["log_nse"]  # type: ignore
                - results["objective_functions"]["log_kge"]  # type: ignore
            )  # type: ignore
        else:
            obj_val = metric(results)

        if print_value and isinstance(obj_val, str):
            print(f"{metric.upper()}: {-round(obj_val, 2)}")  # type: ignore

        return obj_val
    except Exception:
        print(f"Failed with parameters {x}")
        return np.inf


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

        obj = model_res.objective_functions.to_dict()

        aux_values = [
            obj["kge"] if "kge" in obj else np.nan,
            obj["nse"] if "nse" in obj else np.nan,
            obj["log_kge"] if "log_kge" in obj else np.nan,
            obj["log_nse"] if "log_nse" in obj else np.nan,
            obj["bias"] if "bias" in obj else np.nan,
        ]

        if isinstance(metric, str):
            if metric == "kge":
                return lp + model_res.objective_functions["kge"], aux_values
            elif metric == "nse":
                return lp + model_res.objective_functions["nse"], aux_values
            else:
                raise ValueError(f"Unknown metric: {metric}")
        else:
            return lp + metric(model_res), aux_values
    except Exception:
        return -np.inf, [np.nan, np.nan, np.nan, np.nan, np.nan]


def rt_minerals_to_array(
    mineral_conc: Iterable | dict[str, Iterable | dict[str, float]],
    mineral_order: list[str],
    zone_order: list[str],
) -> Optional[NDArray]:

    rows: list[np.ndarray] = []

    if isinstance(mineral_conc, (np.ndarray, list, tuple)):
        if len(mineral_conc) == 0:
            return None
        rows = [x_i for x_i in mineral_conc]
        for row in rows:
            if len(row) != len(mineral_order):
                raise ValueError(
                    f"When passing array-like object as `mineral_conc`, the array must have shape ({
                        len(zone_order)
                    }, {len(mineral_order)})"
                )
    elif isinstance(mineral_conc, dict):
        if set(mineral_conc.keys()) != set(zone_order):
            raise ValueError(f"Must pass all zones in model: need each of {zone_order}")
        else:
            for zone in zone_order:
                zm = mineral_conc[zone]
                if isinstance(zm, dict):
                    vals: list[float] = []
                    for min_name in mineral_order:
                        if min_name not in zm:
                            raise ValueError(
                                f"Zone '{zone}' is missing mineral species '{
                                    min_name}'"
                            )
                        else:
                            vals.append(zm[min_name])
                        rows.append(np.array(vals))
                elif isinstance(zm, (np.ndarray, list, tuple, Series)):
                    rows.append(np.array([x_i for x_i in zm]))

    return np.vstack(rows)


# ======================== #
