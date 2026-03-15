from __future__ import annotations

from abc import abstractmethod
from typing import Optional

from numpy.typing import NDArray

from .common_types_compiled import HydroForcing, HydroStep
from .math import bisect, find_root
from .utils import setup_logging

setup_logging(__file__)

"""
Things to add in
- Elevation zones: precipitation gradients with elevation - lapse rates
- Routing
"""
# ########################################################################### #
# Zone Classes
# These are regular Python classes that hold parameters
# ########################################################################### #


class HydrologicZone:
    """
    An abstract base class for a generic hydrologic computational zone.

    This class defines the common interface and behavior for all hydrologic
    zones within the model, such as snowpacks, soil layers, or groundwater
    stores. Each specific zone type must inherit from this class and implement
    its abstract methods.

    The core of the zone is the `step` method, which solves the ordinary
    differential equation for mass balance over a single time step using an
    implicit midpoint method.

    Attributes:
        name (str): A descriptive name for the zone (e.g., "snow", "soil").
    """

    __name: str

    def __init__(self, name: str = "unnamed"):
        """
        Initializes the HydrologicZoneCompiled.

        Args:
            name (str, optional): The name of the zone. Defaults to "unnamed".
        """
        self.__name = name

    def __midpoint_func(
        self, s: float, s_0: float, d: HydroForcing, dt: float
    ) -> float:
        """A wrapper function for solving using the midpoint method"""
        return (s_0 - s) + dt * self.mass_balance(0.5 * (s_0 + s), d)

    def __implicit_eulers_func(
        self, s: float, s_0: float, d: HydroForcing, dt: float
    ) -> float:
        """A wrapper for the implicit Euler's method"""
        return (s_0 - s) + dt * self.mass_balance(s, d)

    def step(self: HydrologicZone, s_0: float, d: HydroForcing, dt: float) -> HydroStep:
        """
        Advances the zone's state over a single time step.

        This method integrates the mass balance ODE for the zone using an
        implicit midpoint method with inlined Secant root finding for maximum performance.
        """

        try:
            try:
                # s_new = find_root(self.__midpoint_func, s_0, d, dt, 1e-6)
                s_new = find_root(self.__implicit_eulers_func, s_0, d, dt, 1e-6)
            except ValueError:
                s_new = bisect(
                    # self.__midpoint_func,
                    self.__implicit_eulers_func,
                    0.0,
                    1000.0,
                    s_0,
                    d,
                    dt,
                    tol=1e-6,
                    max_iters=100,
                )
        except ValueError:
            s_new = 0.0

        s_new = max(0.0, s_new)

        return HydroStep(
            state=s_new,
            forc_flux=self.forc_flux(s_new, d),
            vap_flux=self.vap_flux(s_new, d),
            lat_flux=self.lat_flux(s_new, d),
            vert_flux=self.vert_flux(s_new, d),
            q_in=d.q_in,
            lat_flux_ext=self.lat_flux_ext(s_new, d),
            vert_flux_ext=self.vert_flux_ext(s_new, d),
        )

    def mass_balance(self, s: float, d: HydroForcing) -> float:
        """
        Calculates the net rate of change of storage (ds/dt) for the zone.

        This function represents the core mass balance equation:
        ds/dt = q_in + forc_flux - vap_flux - lat_flux - vert_flux

        Args:
            s (float): The current state (storage) of the zone.
            d (HydroForcing): The hydrologic forcing data for the time step.

        Returns:
            float: The net rate of change of storage (e.g., in mm/day).
        """
        return (
            d.q_in
            + self.forc_flux(s, d)
            - self.vap_flux(s, d)
            - self.lat_flux(s, d)
            - self.vert_flux(s, d)
        )

    def forc_flux(self, s: float, d: HydroForcing) -> float:
        """
        Calculates the flux into the zone from direct forcing (e.g., precip).

        This is a placeholder implementation. Subclasses should override this
        method to define specific forcing behavior.

        Args:
            s (float): The current state (storage) of the zone.
            d (HydroForcing): The hydrologic forcing data.

        Returns:
            float: The forcing flux, which is 0.0 by default.
        """
        return 0.0

    def vap_flux(self, s: float, d: HydroForcing) -> float:
        """
        Calculates the vaporization flux out of the zone (e.g., ET).

        This is a placeholder implementation. Subclasses should override this
        method to define specific vaporization behavior.

        Args:
            s (float): The current state (storage) of the zone.
            d (HydroForcing): The hydrologic forcing data.

        Returns:
            float: The vaporization flux, which is 0.0 by default.
        """
        return 0.0

    def lat_flux(self, s: float, d: HydroForcing) -> float:
        """
        Calculates the lateral flux out of the zone.

        This is a placeholder implementation. Subclasses should override this
        method to define specific lateral flow behavior.

        Args:
            s (float): The current state (storage) of the zone.
            d (HydroForcing): The hydrologic forcing data.

        Returns:
            float: The lateral flux, which is 0.0 by default.
        """
        return 0.0

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        """
        Calculates the vertical flux out of the zone to a lower zone.

        This is a placeholder implementation. Subclasses should override this
        method to define specific vertical percolation behavior.

        Args:
            s (float): The current state (storage) of the zone.
            d (HydroForcing): The hydrologic forcing data.

        Returns:
            float: The vertical flux, which is 0.0 by default.
        """
        return 0.0

    def lat_flux_ext(self, s: float, d: HydroForcing) -> float:
        return self.lat_flux(s, d)

    def vert_flux_ext(self, s: float, d: HydroForcing) -> float:
        return self.vert_flux(s, d)

    def param_list(self) -> list[float]:
        """
        Returns a list of the zone's parameter values.

        This method is used for parameter analysis, optimization, and saving
        model states. The order of parameters in the list must be consistent.

        Returns:
            list[float]: An ordered list of the zone's parameter values.
        """
        return []

    @property
    def name(self) -> str:
        return self.__name

    def columns(self, zone_id: int) -> list[str]:
        """
        Gets the column names for this zone for the final output DataFrame.

        The names are standardized to include the state and all fluxes,
        prefixed by the zone's name and a unique ID.

        Args:
            zone_id (int): The unique integer ID of the zone within the model.

        Returns:
            list[str]: A list of column names for the zone's time series data.
        """
        return [
            f"s_{self.__name}",
            f"q_forc_{self.__name}",
            f"q_vap_{self.__name}",
            f"q_lat_{self.__name}",
            f"q_vert_{self.__name}",
            f"q_lat_ext_{self.__name}",
            f"q_vert_ext_{self.__name}",
        ]

    @classmethod
    def default(cls) -> HydrologicZone:
        """
        Creates a default instance of the hydrologic zone.

        This class method should be implemented by each subclass to provide a
        standard, default-parameterized instance of the zone.

        Returns:
            HydrologicZoneCompiled: A default instance of the specific zone class.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def num_parameters(cls) -> int:
        """
        Returns the number of parameters for this zone class.

        Returns:
            int: The total count of tunable parameters.
        """
        pass

    @classmethod
    @abstractmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        """
        Returns a default parameter range for calibration.

        This method provides a reasonable starting range (min, max) for each
        tunable parameter, which can be used by optimization algorithms.

        Returns:
            dict[str, tuple[float, float]]: A dictionary mapping parameter
                names to their (min, max) range.
        """
        pass

    @classmethod
    @abstractmethod
    def base_name(cls) -> str:
        """
        Returns the base name of the zone class.

        This should be a static, descriptive name for the zone type (e.g.,
        "snow", "soil").

        Returns:
            str: The base name of the zone.
        """
        pass

    @classmethod
    def from_array(cls, arr: NDArray, name: Optional[str] = None) -> HydrologicZone:
        """
        Creates a new zone instance from a NumPy array of parameters.

        This is a factory method used to instantiate a zone from a flat array
        of parameter values, typically during a calibration or sensitivity
        analysis run. The order of parameters in the array must match the
        order defined in `parameter_names`.

        Args:
            arr (NDArray): A 1D NumPy array of parameter values.
            name (Optional[str], optional): An optional name for the new zone
                instance. Defaults to None.

        Returns:
            HydrologicZoneCompiled: A new instance of the zone class.
        """
        param_dict: dict[str, str | float] = {}
        for i, param in enumerate(cls.parameter_names()):
            param_dict[param] = arr[i]
        if name is not None:
            param_dict["name"] = name

        return cls(**param_dict)  # type: ignore

    @classmethod
    @abstractmethod
    def parameter_names(cls) -> list[str]:
        """
        Returns an ordered list of the zone's parameter names.

        The order must be consistent with `param_list` and `from_array`.

        Returns:
            list[str]: An ordered list of parameter names.
        """
        pass

    @classmethod
    def default_init_state(cls) -> float:
        """
        Returns the default initial state (storage) for this zone type.

        Returns:
            float: The default initial storage value, which is 0.0.
        """
        return 0.0

    @classmethod
    def from_dict(cls, params: dict[str, float]) -> HydrologicZone:
        """
        Creates a new zone instance from a dictionary of parameters.

        Args:
            params (dict[str, float]): A dictionary mapping parameter names
                to their values.

        Returns:
            HydrologicZoneCompiled: A new instance of the zone class.
        """
        try:
            return cls(**params)  # type: ignore
        except TypeError:
            raise TypeError(
                f"Failed to construct Hydrologic zone with type {type(cls)} and params {params}, expected parameters named {cls.parameter_names()}"
            ) from None


class SnowZone(HydrologicZone):
    """
    A hydrologic zone representing a snowpack, based on a simple degree-day model.

    This zone accumulates precipitation as snow when the temperature is below a
    threshold and releases it as melt when the temperature is above it.

    Attributes:
        tt (float): The threshold temperature (°C) for snowmelt. Below this,
            precipitation is snow; above, it is rain and melt can occur.
        fmax (float): The maximum melt factor (degree-day factor) in mm/°C/day.
            This controls the rate of snowmelt.
    """

    tt: float
    fmax: float

    def __init__(self, tt: float = 0.0, fmax: float = 1.0, name: str = "snow"):
        """
        Initializes the SnowZone.

        Args:
            tt (float, optional): Threshold temperature for melt (°C).
                Defaults to 0.0.
            fmax (float, optional): Maximum melt factor (mm/°C/day).
                Defaults to 1.0.
            name (str, optional): Name of the zone. Defaults to "snow".
        """
        super().__init__(name=name)
        self.tt = tt
        self.fmax = fmax

    def forc_flux(self, s: float, d: HydroForcing) -> float:
        """
        Calculates snow accumulation from precipitation.

        If the temperature is at or below the threshold `tt`, all precipitation
        is added to the snowpack storage.

        Args:
            s (float): Current snow storage (mm).
            d (HydroForcing): Hydrologic forcing data.

        Returns:
            float: The rate of snow accumulation (mm/day).
        """
        if d.temp <= self.tt:
            return d.precip
        else:
            return 0.0

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        """
        Calculates the vertical flux of water out of the snowpack (snowmelt).

        Melt occurs when the temperature is above the threshold `tt`. The melt
        rate is determined by the degree-day factor `fmax`. Any incoming rain
        also contributes directly to the outflow. The total outflow cannot
        exceed the available snow storage.

        Args:
            s (float): Current snow storage (mm).
            d (HydroForcing): Hydrologic forcing data.

        Returns:
            float: The rate of snowmelt and rain pass-through (mm/day).
        """
        if d.temp > self.tt:
            melt: float = self.fmax * (d.temp - self.tt)
            return min(s, melt)
        else:
            return 0.0

    def vert_flux_ext(self, s: float, d: HydroForcing) -> float:
        if (
            d.temp > self.tt
        ):  # Temperature is above the freezing point, return the snow melt
            return self.vert_flux(s, d) + d.precip
        else:  # Temperature is below the freezing point, only return the snow melt
            return self.vert_flux(s, d)

    def param_list(self) -> list[float]:
        """
        Returns a list of the zone's parameter values.

        Returns:
            list[float]: An ordered list of parameters: [tt, fmax].
        """
        return [self.tt, self.fmax]

    @classmethod
    def default(cls) -> SnowZone:
        """
        Creates a default instance of the SnowZone.

        Returns:
            SnowZone: A SnowZone with tt=0.0 and fmax=1.0.
        """
        return SnowZone(tt=0.0, fmax=1.0)

    @classmethod
    def parameter_names(cls) -> list[str]:
        """
        Returns an ordered list of the zone's parameter names.

        Returns:
            list[str]: The list of parameter names: ['tt', 'fmax'].
        """
        return ["tt", "fmax"]

    @classmethod
    def num_parameters(cls) -> int:
        """
        Returns the number of parameters for this zone class.

        Returns:
            int: The number of parameters, which is 2.
        """
        return 2

    @classmethod
    def base_name(cls) -> str:
        """
        Returns the base name of the zone class.

        Returns:
            str: The base name, "snow".
        """
        return "snow"

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        """
        Returns a default parameter range for calibration.

        Returns:
            dict[str, tuple[float, float]]: A dictionary mapping parameter
                names to their (min, max) range.
        """
        return {"tt": (-1, 1), "fmax": (0.5, 5.0)}

    def __repr__(self) -> str:
        return f"SnowZone(tt={self.tt:.2f}, fmax={self.fmax:.2f})"


class SurfaceZone(HydrologicZone):
    """
    A hydrologic zone representing a soil layer, inspired by the HBV model.

    This zone models soil moisture dynamics, including infiltration, actual
    evapotranspiration, lateral flow (interflow), and vertical percolation.

    Attributes:
        fc (float): Field capacity of the soil (mm). Maximum storage.
        lp (float): Parameter controlling the limit for potential ET. ET is
            at the potential rate until storage `s` exceeds `lp * fc`.
        beta (float): A non-linear factor controlling infiltration and
            percolation, representing the contribution of runoff from a saturated
            area.
        k0 (float): A rate constant for lateral flow (interflow) (1/day).
        thr (float): Storage threshold (mm) for the initiation of lateral flow.
    """

    fc: float
    lp: float
    beta: float
    k0: float
    thr: float

    def __init__(
        self,
        fc: float = 100.0,
        lp: float = 0.5,
        beta: float = 1.0,
        k0: float = 0.1,
        thr: float = 10.0,
        name: str = "surface",
    ):
        """
        Initializes the SurfaceZone.

        Args:
            fc (float, optional): Field capacity (mm). Defaults to 100.0.
            lp (float, optional): ET limit parameter. Defaults to 0.5.
            beta (float, optional): Runoff non-linearity factor.
                Defaults to 1.0.
            k0 (float, optional): Lateral flow rate constant (1/day).
                Defaults to 0.1.
            thr (float, optional): Lateral flow threshold (mm).
                Defaults to 10.0.
            name (str, optional): Name of the zone. Defaults to "soil".
        """
        super().__init__(name=name)
        self.fc = fc  # Soil field capacity
        self.beta = beta  # ET nonlinearity factor
        self.k0 = k0
        self.lp = lp
        self.thr = thr
        # self.name: str = name

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        """
        Returns a default parameter range for calibration.

        Returns:
            dict[str, tuple[float, float]]: A dictionary mapping parameter
                names to their (min, max) range.
        """
        return {
            "fc": (50, 1_000),
            "lp": (0.05, 1.0),
            "beta": (0.05, 5.0),
            "k0": (0, 1.0),
            "thr": (0, 1_000),
        }

    def forc_flux(self, s: float, d: HydroForcing) -> float:
        """
        Calculates infiltration from precipitation into the soil.

        The amount of precipitation that becomes runoff (and does not enter
        the soil) increases non-linearly as the soil storage `s` approaches
        the field capacity `fc`.

        Args:
            s (float): Current soil moisture storage (mm).
            d (HydroForcing): Hydrologic forcing data.

        Returns:
            float: The rate of infiltration (mm/day).
        """
        # return d.q_in * (1 - (s / self.fc) ** self.beta)
        return 0.0

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        """
        Calculates vertical percolation from the soil to a lower zone.

        This flux is sourced from the total water input to the zone (`d.q_in`),
        which typically represents melt from an overlying snow zone. The
        percolation rate is scaled by the relative soil moisture.

        Args:
            s (float): Current soil moisture storage (mm).
            d (HydroForcing): Hydrologic forcing data.

        Returns:
            float: The rate of vertical percolation (mm/day).
        """
        return d.q_in * (s / self.fc) ** self.beta

    def vap_flux(self, s: float, d: HydroForcing) -> float:
        """
        Calculates actual evapotranspiration (AET) from the soil.

        AET is equal to the potential evapotranspiration (PET) until the soil
        moisture drops below a threshold (`lp * fc`), after which it decreases
        linearly with storage.

        Args:
            s (float): Current soil moisture storage (mm).
            d (HydroForcing): Hydrologic forcing data.

        Returns:
            float: The rate of actual evapotranspiration (mm/day).
        """
        return d.pet * min(s / (self.fc * self.lp), 1.0)

    def lat_flux(self, s: float, d: HydroForcing) -> float:
        """
        Calculates lateral flow (interflow) from the soil.

        Lateral flow is generated as a linear function of storage above a
        certain threshold `thr`.

        Args:
            s (float): Current soil moisture storage (mm).
            d (HydroForcing): Hydrologic forcing data.

        Returns:
            float: The rate of lateral flow (mm/day).
        """
        return max(0.0, self.k0 * (s - self.thr))

    def param_list(self) -> list[float]:
        """
        Returns a list of the zone's parameter values.

        Returns:
            list[float]: An ordered list of parameters: [fc, lp, beta, k0, thr].
        """
        return [self.fc, self.lp, self.beta, self.k0, self.thr]

    @classmethod
    def parameter_names(cls) -> list[str]:
        """
        Returns an ordered list of the zone's parameter names.

        Returns:
            list[str]: The list of parameter names.
        """
        return ["fc", "lp", "beta", "k0", "thr"]

    @classmethod
    def default(cls) -> SurfaceZone:
        """
        Creates a default instance of the SoilZone.

        Returns:
            SoilZone: A SoilZone with default parameter values.
        """
        return SurfaceZone(fc=100.0, lp=0.5, beta=1.0, k0=0.1, thr=10.0)

    @classmethod
    def base_name(cls) -> str:
        """
        Returns the base name of the zone class.

        Returns:
            str: The base name, "soil".
        """
        return "surface"

    @classmethod
    def num_parameters(cls) -> int:
        """
        Returns the number of parameters for this zone class.

        Returns:
            int: The number of parameters, which is 5.
        """
        return 5

    @classmethod
    def default_init_state(cls) -> float:
        """
        Returns the default initial state (storage) for this zone type.

        Returns:
            float: The default initial storage value, which is 25.0.
        """
        return 25.0

    def __repr__(self) -> str:
        return f"SurfaceZone(fc={round(self.fc)}, lp={self.lp:.2f}, beta={self.beta:.2f}, k0={self.k0:.3f}, thr={self.thr:.1f})"


class GroundZone(HydrologicZone):
    """
    A zone representing a generic groundwater store with non-linear outflow.

    This zone simulates a reservoir where lateral outflow is a power function
    of storage, and vertical outflow (percolation) is a constant fraction of
    the storage.

    Attributes:
        k (float): Rate constant for lateral groundwater flow (1/day).
        alpha (float): Exponent for the non-linear storage-outflow relationship.
        perc (float): The maximum rate of vertical percolation to a lower zone
            (mm/day).
    """

    k: float
    alpha: float
    perc: float

    def __init__(
        self,
        k: float = 1e-3,
        alpha: float = 1.0,
        perc: float = 1.0,
        name: str = "ground",
    ):
        """
        Initializes the GroundZone.

        Args:
            k (float, optional): Lateral flow rate constant (1/day).
                Defaults to 1e-3.
            alpha (float, optional): Lateral flow exponent. Defaults to 1.0.
            perc (float, optional): Maximum vertical percolation rate (mm/day).
                Defaults to 1.0.
            name (str, optional): Name of the zone. Defaults to "ground".
        """
        super().__init__(name=name)
        self.k = k
        self.alpha = alpha
        self.perc = perc
        # self.name = name

    def lat_flux(self, s: float, d: HydroForcing) -> float:
        """
        Calculates lateral flow from the groundwater zone.

        The outflow is a non-linear function of storage: Q_lat = k * s^alpha.

        Args:
            s (float): Current groundwater storage (mm).
            d (HydroForcing): Hydrologic forcing data (not used).

        Returns:
            float: The rate of lateral flow (mm/day).
        """
        try:
            if s < 1e-12:
                return 0.0
            else:
                return self.k * max(0.0, s) ** self.alpha
        except TypeError as e:
            print("Got error in `GroundZone` calculation. Here are the values:")
            print(f"Parameters: k={self.k}, alpha={self.alpha}, perc={self.perc}")
            print(f"Storage: {s}")
            print(f"Forcing data: {d}")
            raise e

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        """
        Calculates vertical percolation to a lower zone.

        The percolation is limited by the `perc` parameter and cannot exceed
        the available storage `s`.

        Args:
            s (float): Current groundwater storage (mm).
            d (HydroForcing): Hydrologic forcing data (not used).

        Returns:
            float: The rate of vertical percolation (mm/day).
        """
        return min(s, self.perc)

    def param_list(self) -> list[float]:
        """
        Returns a list of the zone's parameter values.

        Returns:
            list[float]: An ordered list of parameters: [k, alpha, perc].
        """
        return [self.k, self.alpha, self.perc]

    @classmethod
    def default(cls) -> GroundZone:
        """
        Creates a default instance of the GroundZone.

        Returns:
            GroundZone: A GroundZone with default parameter values.
        """
        return GroundZone(k=0.01, alpha=1.0, perc=1.0)

    @classmethod
    def num_parameters(cls) -> int:
        """
        Returns the number of parameters for this zone class.

        Returns:
            int: The number of parameters, which is 3.
        """
        return 3

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        """
        Returns a default parameter range for calibration.

        Returns:
            dict[str, tuple[float, float]]: A dictionary mapping parameter
                names to their (min, max) range.
        """
        return {
            "k": (1e-5, 0.1),
            "alpha": (0.5, 3.0),
            "perc": (0.0, 5.0),
        }

    @classmethod
    def base_name(cls) -> str:
        """
        Returns the base name of the zone class.

        Returns:
            str: The base name, "ground".
        """
        return "ground"

    @classmethod
    def parameter_names(cls) -> list[str]:
        """
        Returns an ordered list of the zone's parameter names.

        Returns:
            list[str]: The list of parameter names: ['k', 'alpha', 'perc'].
        """
        return ["k", "alpha", "perc"]

    @classmethod
    def default_init_state(cls) -> float:
        """
        Returns the default initial state (storage) for this zone type.

        Returns:
            float: The default initial storage value, which is 10.0.
        """
        return 10.0

    def __repr__(self) -> str:
        return (
            f"GroundZone(k={self.k:.2e}, alpha={self.alpha:.2f}, perc={self.perc:0.2f})"
        )


class GroundZoneLinear(GroundZone):
    """
    A groundwater zone with a linear storage-outflow relationship.

    This is a specialized version of `GroundZone` where the lateral flow
    exponent `alpha` is fixed to 1.0, resulting in a linear reservoir model.

    Attributes:
        k (float): Rate constant for lateral groundwater flow (1/day).
        perc (float): The maximum rate of vertical percolation to a lower zone
            (mm/day).
    """

    k: float
    perc: float

    def __init__(
        self,
        k: float = 1e-3,
        perc: float = 1.0,
        name: str = "ground_linear",
    ) -> None:
        """
        Initializes the GroundZoneLinear.

        Args:
            k (float, optional): Lateral flow rate constant (1/day).
                Defaults to 1e-3.
            perc (float, optional): Maximum vertical percolation rate (mm/day).
                Defaults to 1.0.
            name (str, optional): Name of the zone. Defaults to "ground_linear".
        """
        super().__init__(k=k, alpha=1.0, perc=perc, name=name)

    @classmethod
    def num_parameters(cls) -> int:
        """
        Returns the number of parameters for this zone class.

        Returns:
            int: The number of parameters, which is 3.
        """
        return 2

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        """
        Returns a default parameter range for calibration.

        Returns:
            dict[str, tuple[float, float]]: A dictionary mapping parameter
                names to their (min, max) range.
        """
        return {
            "k": (1e-5, 0.1),
            "perc": (0.1, 5),
        }

    def param_list(self) -> list[float]:
        return [self.k, self.perc]

    @classmethod
    def parameter_names(cls) -> list[str]:
        return ["k", "perc"]

    @classmethod
    def base_name(cls) -> str:
        return "ground_linear"

    @classmethod
    def default(cls) -> GroundZoneLinear:
        return GroundZoneLinear()


class GroundZoneB(GroundZone):
    """
    A bottom groundwater zone with a non-linear storage-outflow relationship.

    This zone is intended to be the lowest layer in a model profile. As such,
    its vertical percolation (`perc`) is fixed to 0.0.

    Attributes:
        k (float): Rate constant for lateral groundwater flow (1/day).
        alpha (float): Exponent for the non-linear storage-outflow relationship.
    """

    k: float
    alpha: float

    def __init__(
        self,
        k: float = 1e-3,
        alpha: float = 1.0,
        name: str = "deep",
    ):
        """
        Initializes the GroundZoneB.

        Args:
            k (float, optional): Lateral flow rate constant (1/day).
                Defaults to 1e-3.
            alpha (float, optional): Lateral flow exponent. Defaults to 1.0.
            name (str, optional): Name of the zone.
                Defaults to "bottom_ground_nl".
        """
        super(GroundZoneB, self).__init__(k=k, alpha=alpha, perc=0.0, name=name)
        self.k = k
        self.alpha = alpha

    @classmethod
    def num_parameters(cls) -> int:
        """
        Returns the number of parameters for this zone class.

        Returns:
            int: The number of parameters, which is 2.
        """
        return 2

    def param_list(self) -> list[float]:
        """
        Returns a list of the zone's parameter values.

        Returns:
            list[float]: An ordered list of parameters: [k, alpha].
        """
        return [self.k, self.alpha]

    @classmethod
    def base_name(cls) -> str:
        """
        Returns the base name of the zone class.

        Returns:
            str: The base name, "ground_bottom".
        """
        return "ground_bottom"

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        """
        Returns a default parameter range for calibration.

        Returns:
            dict[str, tuple[float, float]]: A dictionary mapping parameter
                names to their (min, max) range.
        """
        default_range = super().default_parameter_range()
        del default_range["perc"]

        return default_range

    @classmethod
    def parameter_names(cls) -> list[str]:
        """
        Returns an ordered list of the zone's parameter names.

        Returns:
            list[str]: The list of parameter names: ['k', 'alpha'].
        """
        return ["k", "alpha"]

    @classmethod
    def default(cls) -> GroundZone:
        """
        Creates a default instance of the GroundZoneB.

        Returns:
            GroundZone: A GroundZoneB with default parameter values.
        """
        return GroundZoneB(k=0.01, alpha=1.0)

    def __repr__(self) -> str:
        return f"GroundZoneB(k={self.k:.2e}, alpha={self.alpha:.2f})"


class GroundZoneLinearB(GroundZone):
    """
    A bottom groundwater zone with a linear storage-outflow relationship.

    This is the simplest groundwater zone, acting as a linear reservoir at the
    bottom of a model profile. Its vertical percolation is fixed to 0.0, and
    its lateral flow exponent `alpha` is fixed to 1.0.

    Attributes:
        k (float): Rate constant for lateral groundwater flow (1/day).
    """

    k: float

    def __init__(self, k: float = 1e-3, name="bottom_ground_l"):
        """
        Initializes the GroundZoneLinearB.

        Args:
            k (float, optional): Lateral flow rate constant (1/day).
                Defaults to 1e-3.
            name (str, optional): Name of the zone.
                Defaults to "bottom_ground_l".
        """
        super().__init__(k=k, alpha=1.0, perc=0.0, name=name)

    @classmethod
    def num_parameters(cls) -> int:
        """
        Returns the number of parameters for this zone class.

        Returns:
            int: The number of parameters, which is 1.
        """
        return 1

    def param_list(self) -> list[float]:
        """
        Returns a list of the zone's parameter values.

        Returns:
            list[float]: An ordered list of parameters: [k].
        """
        return [self.k]

    @classmethod
    def base_name(cls) -> str:
        """
        Returns the base name of the zone class.

        Returns:
            str: The base name, "ground_linear_bottom".
        """
        return "ground_linear_bottom"

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        """
        Returns a default parameter range for calibration.

        Returns:
            dict[str, tuple[float, float]]: A dictionary mapping parameter
                names to their (min, max) range.
        """
        default_range = super().default_parameter_range()
        del default_range["perc"]
        del default_range["alpha"]

        return default_range

    @classmethod
    def parameter_names(cls) -> list[str]:
        """
        Returns an ordered list of the zone's parameter names.

        Returns:
            list[str]: The list of parameter names: ['k'].
        """
        return ["k"]
