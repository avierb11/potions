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


class ModelStructureError(HydrologyError, ValueError):
    """A model's structure violates a structural rule.

    Raised at model construction when zone placement or zone parameter
    values make the model physically incoherent, e.g.:

    - a bottom zone not in the last layer (a zone exists below a no-throughflow zone),
    - a zone other than a bottom zone in the last layer (water would be lost
      below the model),
    - non-finite or negative zone parameters,
    - negative or non-finite size scales.
    """


class ModelInputError(HydrologyError, ValueError):
    """The inputs to a model run are invalid or corrupted.

    Raised at run time before any model steps when the provided data cannot
    be physically simulated, e.g.:

    - non-finite values (``NaN``/``Inf``) in the forcing series,
    - negative precipitation, PET, or measured streamflow,
    - forcing series of mismatched lengths,
    - an initial state with the wrong length, non-finite values, or
      negative zone states,
    - a timestep ``dt`` that is non-finite or not positive.
    """


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
    metric: Literal["kge", "nse", "combined"] | Callable[[HydroModelResults], float],
    print_value: bool,
    base_params: Optional[dict[str, float]] = None,
    free_names: Optional[tuple[str, ...]] = None,
    all_names: Optional[tuple[str, ...]] = None,
) -> float:
    """
    Score one parameter vector with the hydrology model.

    When ``base_params``, ``free_names``, and ``all_names`` are all
    given, ``x`` holds only the free parameters' values, in
    ``free_names`` order; the full parameter vector is rebuilt by
    taking each free value and every other parameter's fixed value
    from ``base_params``.  The expansion is done here, at module
    level, so the objective stays picklable for the optimizer's
    worker pool.

    Args:
        x: The parameter vector to score: the full vector, or the
            free parameters' values when the expansion arguments are
            given.
        cls: The hydrology model class.
        forc: The forcing data.
        meas_streamflow: The measured streamflow series.
        metric: The metric to optimize: a supported name or a
            callable on the run's results.
        print_value: Whether to print the metric value.
        base_params: The fixed values of every hydrology parameter;
            None disables the free-parameter expansion.
        free_names: The names of the free parameters, in ``x``'s
            order; None disables the free-parameter expansion.
        all_names: The names of every hydrology parameter, in the
            full vector's order; None disables the expansion.

    Returns:
        The objective value (the metric, negated for the named
        metrics the optimizer minimizes), or infinity when the run
        failed.
    """
    if (
        base_params is not None
        and free_names is not None
        and all_names is not None
    ):
        free_values: dict[str, float] = dict(
            zip(free_names, x, strict=True)
        )
        x = np.array(
            [free_values.get(name, base_params[name]) for name in all_names],
            dtype=np.float64,
        )

    model = cls.hydro_from_array(x)

    try:
        results: HydroModelResults = model.run_hydro_model(
            init_state=cls.default_hydro_init_state(),
            forc=forc,
            meas_streamflow=meas_streamflow,
            verbose=False,
            # Don't persist individual optimization evaluations.
            save_results=False,
        )

        obj_val: float

        if metric == "kge":
            obj_val = -results.objective_functions["kge"]  # type: ignore
        elif metric == "nse":
            obj_val = -results.objective_functions["nse"]  # type: ignore
        elif metric == "combined":
            obj_val = (
                -results.objective_functions["kge"]  # type: ignore
                - results.objective_functions["nse"]  # type: ignore
                - results.objective_functions["log_nse"]  # type: ignore
                - results.objective_functions["log_kge"]  # type: ignore
            )  # type: ignore
        else:
            obj_val = metric(results)

        if print_value and isinstance(obj_val, str):
            print(f"{metric.upper()}: {-round(obj_val, 2)}")  # type: ignore

        return obj_val  # type: ignore
    except Exception as e:
        print(f"Failed with parameters {x}")
        print(e)
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
