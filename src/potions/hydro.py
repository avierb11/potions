"""@package zones"""

from __future__ import annotations
from typing import Final, ClassVar
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from .utils import HydroForcing
from .interfaces import Zone, StepResult


@dataclass(frozen=True)
class HydroStep(StepResult[float]):
    """Holds the results of a single time step for a HydrologicZone.

    This is an immutable data structure that contains the new state and the
    calculated fluxes for a single zone over a single time step (`dt`).

    Attributes:
        state: The updated hydrologic state (e.g., storage) of the zone at the end of the time step.
        forc_flux: The flux into the zone from external forcing (e.g., precipitation).
        vap_flux: The flux out of the zone due to vaporization (e.g., evapotranspiration).
        lat_flux: The lateral flux out of the zone to adjacent zones.
        vert_flux: The vertical flux out of the zone to the zone below.
    """
    state: float
    forc_flux: float
    vap_flux: float
    lat_flux: float
    vert_flux: float


class HydrologicZone(Zone[float, HydroForcing, HydroStep]):
    """An abstract base class that defines the interface for a single computational unit.

    This class acts as a contract for all other hydrologic zone types (e.g.,
    `SnowZone`, `SoilZone`). It should not be instantiated directly. Subclasses are
    expected to be dataclasses that define specific parameters and may override
    the flux calculation methods.

    The `step` method is implemented using the template method pattern. It uses a
    numerical solver (`scipy.integrate.solve_ivp`) on the `mass_balance`
    equation. Subclasses can either rely on this default `step` implementation
    and simply override the individual flux methods (`vert_flux`, `lat_flux`,
    etc.), or they can provide a completely custom `step` method for more
    complex or analytical solutions (see `SnowZone` for an example).
    """
    name: ClassVar[str] = "unnamed"

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        """Calculate vertical flux out of the zone (e.g., deep percolation).

        Args:
            s: Current storage in the zone.
            d: Forcing data for the current time step.

        Returns:
            The calculated vertical flux rate. Defaults to 0.0.
        """
        return 0.0

    def lat_flux(self, s: float, d: HydroForcing) -> float:
        """Calculate lateral flux out of the zone (e.g., subsurface flow).

        Args:
            s: Current storage in the zone.
            d: Forcing data for the current time step.

        Returns:
            The calculated lateral flux rate. Defaults to 0.0.
        """
        return 0.0

    def vap_flux(self, s: float, d: HydroForcing) -> float:
        """Calculate vaporization flux out of the zone (e.g., evapotranspiration).

        Args:
            s: Current storage in the zone.
            d: Forcing data for the current time step.

        Returns:
            The calculated vaporization flux rate. Defaults to 0.0.
        """
        return 0.0

    def forc_flux(self, s: float, d: HydroForcing) -> float:
        """Calculate forcing flux into the zone (e.g., precipitation as liquid water).

        Args:
            s: Current storage in the zone.
            d: Forcing data for the current time step.

        Returns:
            The calculated forcing flux rate. Defaults to 0.0.
        """
        return 0.0

    def mass_balance(self, s: float, d: HydroForcing, q_in: float) -> float:
        """Calculates the net rate of change of storage in the zone.

        This method sums all incoming and outgoing fluxes to determine the
        ordinary differential equation for the zone's state.

        Args:
            s: Current storage in the zone.
            d: Forcing data for the current time step.
            q_in: Incoming flux from other connected zones.

        Returns:
            The net rate of change of storage (ds/dt).
        """
        return (
            q_in
            + self.forc_flux(s, d)
            - self.vert_flux(s, d)
            - self.lat_flux(s, d)
            - self.vap_flux(s, d)
        )

    def param_list(self) -> list[float]:
        """Return a list of the zone's parameter values.

        This method must be implemented by all subclasses. The order of
        parameters should be consistent for analysis and optimization purposes.

        Returns:
            A list of floating-point parameter values.
        """
        raise NotImplementedError  # This must be implemented by subclasses

    def step(self, s_0: float, d: HydroForcing, dt: float, q_in: float) -> HydroStep:
        """Advances the state of the zone by one time step.

        This method calculates the new state and all fluxes for the zone over
        the duration `dt`, given the initial state `s_0`, forcing data `d`, and
        any incoming flux from other zones `q_in`.

        This base implementation uses `scipy.integrate.solve_ivp` to solve the
        ordinary differential equation defined by the `mass_balance` method.
        Subclasses may override this for analytical solutions or different
        numerical schemes.

        Args:
            s_0: The initial state (storage) of the zone.
            d: The hydrologic forcing data for the current time step.
            dt: The duration of the time step in days.
            q_in: The total incoming flux from other connected zones.

        Returns:
            A `HydroStep` object containing the new state and all calculated fluxes.
        """
        def f(t: float, s: float) -> float:
            return self.mass_balance(s, d, q_in)

        res = solve_ivp(f, (0, dt), y0=[s_0])

        s_new: float = res.y[0, -1]
        return HydroStep(
            state=s_new,
            forc_flux=self.forc_flux(s_new, d),
            vap_flux=self.vap_flux(s_new, d),
            lat_flux=self.lat_flux(s_new, d),
            vert_flux=self.vert_flux(s_new, d),
        )

    def columns(self, zone_id: int) -> list[str]:
        """Gets the column names for this zone for the output DataFrame.

        Args:
            zone_id: The unique integer ID of the zone in the flattened model.

        Returns:
            A list of strings for state and flux column headers.
        """
        name: Final[str] = self.name
        return [
            f"s_{name}_{zone_id}",
            f"q_forc_{name}_{zone_id}",
            f"q_vap_{name}_{zone_id}",
            f"q_lat_{name}_{zone_id}",
            f"q_vert_{name}_{zone_id}",
        ]


@dataclass(frozen=True)
class SnowZone(HydrologicZone):
    """A simple degree-day snowpack model.

    This zone accumulates precipitation as snow when the temperature is below a
    threshold and releases it as meltwater when the temperature is above it.

    Attributes:
        tt: The threshold temperature (°C) below which precipitation is snow.
        fmax: The maximum melt rate factor (mm/day/°C).
    """
    tt: float
    fmax: float
    name: ClassVar[str] = "snow"

    def forc_flux(self, s: float, d: HydroForcing) -> float:
        """Accumulates precipitation as snow if temperature is below the threshold."""
        if d.temp < self.tt:
            return d.precip
        else:
            return 0.0

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        """Generates meltwater if temperature is above the threshold.

        The melt is treated as a vertical flux out of the snowpack to the
        zone below.
        """
        if d.temp > self.tt:
            max_melt: float = min(s, self.fmax * (d.temp - self.tt))
            return max(0.0, max_melt)
        else:
            return 0.0

    def param_list(self) -> list[float]:
        """Returns the list of parameter values for the zone."""
        return [self.tt, self.fmax]

    def step(self, s_0: float, d: HydroForcing, dt: float, q_in: float) -> HydroStep:
        """Advances the snowpack state using an analytical solution.

        This custom `step` method bypasses the ODE solver for a more stable
        explicit forward solution. This is necessary because the melt and
        accumulation fluxes do not depend on the current snow storage `s`,
        which can lead to numerical instability and negative storage with a
        standard solver.
        """
        q_in_vol: float = q_in * dt  # Total volume from incoming flux
        forc_vol: float
        vert_vol: float

        # Determine accumulation (forcing) and melt (vertical) volumes
        if d.temp > self.tt:
            forc_vol = 0.0
            # Melt is limited by available snow (s_0) and temperature
            melt_potential = dt * self.fmax * (d.temp - self.tt)
            vert_vol = min(s_0, melt_potential)
        else:
            # Accumulation
            forc_vol = dt * d.precip
            vert_vol = 0.0

        new_state: float = s_0 + forc_vol - vert_vol + q_in_vol

        # Return a HydroStep with flux *rates*, not volumes
        return HydroStep(
            state=new_state,
            forc_flux=forc_vol / dt,
            vap_flux=0,
            lat_flux=0,
            vert_flux=vert_vol / dt,
        )


@dataclass(frozen=True)
class SoilZone(HydrologicZone):
    """A soil moisture bucket model inspired by the HBV model concept.

    This zone represents the unsaturated soil layer. It receives throughfall
    (or snowmelt), and loses water through evapotranspiration, lateral flow,
    and vertical percolation.

    Attributes:
        tt: Threshold temperature (°C) for precipitation to be liquid.
        fc: Field capacity (mm). Maximum storage for evapotranspiration and
            base for percolation calculation.
        lp: Parameter for evapotranspiration calculation (currently unused).
        beta: Exponent for the non-linear percolation function.

        k0: Rate constant for linear, threshold-based lateral flow (mm/day).
        thr: Storage threshold (mm) for lateral flow to begin.
    """
    tt: float
    fc: float
    lp: float
    beta: float

    k0: float
    thr: float
    name: ClassVar[str] = "soil"

    def vap_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates evapotranspiration, linearly related to storage up to fc."""
        return d.pet * min(1, s / self.fc)

    def lat_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates lateral flow, which occurs when storage exceeds a threshold."""
        return max(0.0, self.k0 * (s - self.thr))

    def forc_flux(self, s: float, d: HydroForcing) -> float:
        """Receives liquid precipitation if the temperature is above the threshold."""
        if d.temp >= self.tt:
            return d.precip
        else:
            return 0.0

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates vertical percolation as a non-linear function of storage."""
        if d.temp >= self.tt:
            return d.precip * (s / self.fc) ** self.beta
        else:
            return 0.0

    def param_list(self) -> list[float]:
        """Returns the list of parameter values for the zone."""
        return [self.tt, self.fc, self.lp, self.beta, self.k0, self.thr]


@dataclass(frozen=True)
class GroundZone(HydrologicZone):
    """A simple groundwater reservoir model.

    This zone represents a deeper, saturated groundwater store. It receives
    percolation from above and generates lateral baseflow.

    Attributes:
        k: Rate constant for lateral (baseflow) flux.
        alpha: Exponent for the non-linear lateral flux relationship.
        perc: Maximum rate of deep percolation out of the bottom of the model (mm/day).
    """
    k: float
    alpha: float
    perc: float
    name: ClassVar[str] = "ground"

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates deep percolation, limited by a max rate and available storage."""
        return min(self.perc, s)

    def lat_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates lateral baseflow as a power-law function of storage."""
        return self.k * s**self.alpha

    def param_list(self) -> list[float]:
        """Returns the list of parameter values for the zone."""
        return [self.k, self.alpha, self.perc]
