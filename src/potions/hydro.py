from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional
from scipy.optimize import root_scalar
from numpy.typing import NDArray

from .common_types import HydroForcing


"""
Things to add in
- Elevation zones: precipitation gradients with elevation - lapse rates
- Routing
"""


def find_root(f: Callable[[float], float], x_0: float, tol: float = 1e-5) -> float:
    x_1: float = x_0 + 0.1

    err = abs(f(x_1))

    counter: int = 0

    while err > x_0:
        fx_0 = f(x_0)
        fx_1 = f(x_1)
        x_n = (x_0 * fx_1 - x_1 * fx_0) / (fx_1 - fx_0)
        x_0, x_1 = x_1, x_n
        err = abs(f(x_n))
        counter += 1

        if counter >= 25:
            print("Failed to find error")

    return x_1


class HydroStep:
    state: float
    forc_flux: float
    lat_flux: float
    vert_flux: float
    vap_flux: float

    def __init__(
        self,
        state: float,
        forc_flux: float,
        lat_flux: float,
        vert_flux: float,
        vap_flux: float = 0.0,
    ):
        self.state = float(state)
        self.forc_flux = float(forc_flux)
        self.lat_flux = float(lat_flux)
        self.vert_flux = float(vert_flux)
        self.vap_flux = float(vap_flux)


# ########################################################################### #
# Zone Classes
# These are regular Python classes that hold parameters
# ########################################################################### #


class HydrologicZone(ABC):
    """A generic hydrologic zone."""

    def __init__(self, name: str = "unnamed"):
        self.name = name

    def step(self, s_0: float, d: HydroForcing, dt: float) -> HydroStep:
        """Integrates the mass balance equation over a time step."""

        # def f(t: float, s: float) -> float:
        #     return self.mass_balance(s, d, q_in)

        # res = solve_ivp(f, (0, dt), y0=[float(s_0)])
        # s_new: float = res.y[0, -1]

        # Use the implicit midpoint method
        def f(s: float) -> float:
            return (s_0 - s) + dt * self.mass_balance(0.5 * (s_0 + s), d)

        res = root_scalar(f, x0=s_0, x1=s_0 + 0.1, method="secant", xtol=0.001)
        s_new: float = res.root
        # s_new = find_root(f, s_0)

        return HydroStep(
            state=s_new,
            forc_flux=self.forc_flux(s_new, d),
            vap_flux=self.vap_flux(s_new, d),
            lat_flux=self.lat_flux(s_new, d),
            vert_flux=self.vert_flux(s_new, d),
        )

    def mass_balance(self, s: float, d: HydroForcing) -> float:
        return (
            d.q_in
            + self.forc_flux(s, d)
            - self.vap_flux(s, d)
            - self.lat_flux(s, d)
            - self.vert_flux(s, d)
        )

    def forc_flux(self, s: float, d: HydroForcing) -> float:
        return 0.0

    def vap_flux(self, s: float, d: HydroForcing) -> float:
        return 0.0

    def lat_flux(self, s: float, d: HydroForcing) -> float:
        return 0.0

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        return 0.0

    @abstractmethod
    def param_list(self) -> list[float]:
        """Gets the parameters in list form for this zone"""
        return []

    def columns(self, zone_id: int) -> list[str]:
        """Gets the column names for this zone for the output DataFrame."""
        return [
            f"s_{self.name}_{zone_id}",
            f"q_forc_{self.name}_{zone_id}",
            f"q_vap_{self.name}_{zone_id}",
            f"q_lat_{self.name}_{zone_id}",
            f"q_vert_{self.name}_{zone_id}",
        ]

    @classmethod
    @abstractmethod
    def default(cls) -> HydrologicZone:
        pass

    @classmethod
    @abstractmethod
    def num_parameters(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        """
        Get a default parameter range for calibration
        """
        pass

    @classmethod
    @abstractmethod
    def base_name(cls) -> str:
        pass

    @classmethod
    def from_array(cls, arr: NDArray, name: Optional[str] = None) -> HydrologicZone:
        param_dict: dict[str, str | float] = {}
        for i, param in enumerate(cls.parameter_names()):
            param_dict[param] = arr[i]
        if name is not None:
            param_dict["name"] = name

        return cls(**param_dict)  # type: ignore

    @classmethod
    @abstractmethod
    def parameter_names(cls) -> list[str]:
        pass

    @classmethod
    def default_init_state(cls) -> float:
        """
        Return the default initial state for this type.
        """
        return 0.0


class SnowZone(HydrologicZone):
    """A zone representing a snowpack."""

    def __init__(self, tt: float = 0.0, fmax: float = 1.0, name: str = "snow"):
        super().__init__(name=name)
        self.tt: float = tt
        self.fmax: float = fmax

    def forc_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates snowmelt accumulation rate"""
        if d.temp <= self.tt:
            return d.precip
        else:
            return 0.0

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates snowmelt accumulation rate"""
        if d.temp > self.tt:
            melt: float = self.fmax * (d.temp - self.tt) + d.precip
            return min(s, melt)
        else:
            return 0.0

    def param_list(self) -> list[float]:
        return [self.tt, self.fmax]

    @classmethod
    def default(cls) -> SnowZone:
        return SnowZone(tt=0.0, fmax=1.0)

    @classmethod
    def parameter_names(cls) -> list[str]:
        return ["tt", "fmax"]

    @classmethod
    def num_parameters(cls) -> int:
        return 2

    @classmethod
    def base_name(cls) -> str:
        return "snow"

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        return {"tt": (-1, 1), "fmax": (0.5, 5.0)}


class SoilZone(HydrologicZone):
    """A zone representing a soil layer."""

    def __init__(
        self,
        fc: float = 100.0,
        lp: float = 0.5,
        beta: float = 1.0,
        k0: float = 0.1,
        thr: float = 10.0,
        name="soil",
    ):
        super().__init__(name=name)
        self.fc: float = fc  # Soil field capacity
        self.beta: float = beta  # ET nonlinearity factor
        self.k0: float = k0
        self.lp: float = lp
        self.thr: float = thr

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        return {
            "fc": (50, 1_000),
            "beta": (0.5, 5.0),
            "k0": (0, 1.0),
            "lp": (0.1, 1.0),
            "thr": (0, 100),
        }

    def forc_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates soil accumulation"""
        # if d.temp >= self.tt:
        #     return d.precip * (1 - (s / self.fc) ** self.beta)
        # else:
        #     return 0.0
        return d.precip * (1 - (s / self.fc) ** self.beta)

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates vertical percolation from the soil."""
        return d.q_in * (s / self.fc) ** self.beta

    def vap_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates actual evapotranspiration."""
        return d.pet * min(s / (self.fc * self.lp), 1.0)

    def lat_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates lateral flow from the soil."""
        return max(0.0, self.k0 * (s - self.thr))

    def param_list(self) -> list[float]:
        return [self.fc, self.lp, self.beta, self.k0, self.thr]

    @classmethod
    def parameter_names(cls) -> list[str]:
        return ["fc", "lp", "beta", "k0", "thr"]

    @classmethod
    def default(cls) -> SoilZone:
        return SoilZone(fc=100.0, lp=0.5, beta=1.0, k0=0.1, thr=10.0)

    @classmethod
    def base_name(cls) -> str:
        return "soil"

    @classmethod
    def num_parameters(cls) -> int:
        return 5

    @classmethod
    def default_init_state(cls) -> float:
        return 25.0


class GroundZone(HydrologicZone):
    """A zone representing a groundwater store."""

    def __init__(
        self, k: float = 1e-3, alpha: float = 1.0, perc: float = 1.0, name="ground"
    ):
        super().__init__(name=name)
        self.k = k
        self.alpha = alpha
        self.perc = perc

    def lat_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates lateral flow from a groundwater zone."""
        return self.k * s**self.alpha

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates lateral flow from a groundwater zone."""
        return min(s, self.perc)

    def param_list(self) -> list[float]:
        return [self.k, self.alpha, self.perc]

    @classmethod
    def default(cls) -> GroundZone:
        return GroundZone(k=0.01, alpha=1.0, perc=1.0)

    @classmethod
    def num_parameters(cls) -> int:
        return 3

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        return {
            "k": (1e-5, 0.1),
            "alpha": (0.5, 3),
            "perc": (0.1, 5),
        }

    @classmethod
    def base_name(cls) -> str:
        return "ground"

    @classmethod
    def parameter_names(cls) -> list[str]:
        return ["k", "alpha", "perc"]

    @classmethod
    def default_init_state(cls) -> float:
        return 10.0


class GroundZoneLinear(GroundZone):
    """
    A groundwater zone representing with a linear storage function for the lateral flux
    """

    def __init__(self, k: float = 1e-3, perc: float = 1.0, name="ground_linear"):
        super().__init__(k=k, alpha=1.0, perc=perc, name=name)

    @classmethod
    def num_parameters(cls) -> int:
        return 2

    def param_list(self) -> list[float]:
        return [self.k, self.perc]

    @classmethod
    def base_name(cls) -> str:
        return "ground_linear"

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        default_range = super().default_parameter_range()
        del default_range["alpha"]

        return default_range

    @classmethod
    def parameter_names(cls) -> list[str]:
        return ["k", "perc"]


class GroundZoneB(GroundZone):
    """
    A groundwater zone representing the bottom zone with a nonlinear storage function
    """

    def __init__(self, k: float = 1e-3, alpha: float = 1.0, name="bottom_ground_nl"):
        super(GroundZoneB, self).__init__(k=k, alpha=alpha, perc=0.0, name=name)

    @classmethod
    def num_parameters(cls) -> int:
        return 2

    def param_list(self) -> list[float]:
        return [self.k, self.alpha]

    @classmethod
    def base_name(cls) -> str:
        return "ground_bottom"

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        default_range = super().default_parameter_range()
        del default_range["perc"]

        return default_range

    @classmethod
    def parameter_names(cls) -> list[str]:
        return ["k", "alpha"]


class GroundZoneLinearB(GroundZone):
    """
    A groundwater zone representing the bottom zone with a linear storage function
    """

    def __init__(self, k: float = 1e-3, name="bottom_ground_l"):
        super().__init__(k=k, alpha=1.0, perc=0.0, name=name)

    @classmethod
    def num_parameters(cls) -> int:
        return 1

    def param_list(self) -> list[float]:
        return [self.k]

    @classmethod
    def base_name(cls) -> str:
        return "ground_linear_bottom"

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        default_range = super().default_parameter_range()
        del default_range["perc"]
        del default_range["alpha"]

        return default_range

    @classmethod
    def parameter_names(cls) -> list[str]:
        return ["k"]
