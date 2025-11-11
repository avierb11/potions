from __future__ import annotations
from typing import TypeVar, Protocol

# 1. DEFINE GENERIC TYPE VARIABLES
# These act as placeholders for the specific types that each model will use.
StateType = TypeVar("StateType")  # e.g., float for hydro, NDArray for RT
ForcingType = TypeVar("ForcingType")  # e.g., HydroForcing or RtForcing

# We define StepResultType as a TypeVar bound to StepResult to ensure
# any implementation is a subclass of our StepResult interface.
StepResultType = TypeVar("StepResultType", bound="StepResult")


# 2. DEFINE A GENERIC INTERFACE FOR A STEP RESULT
class StepResult(Protocol[StateType]):
    """A protocol for the results of a single zone's time step.

    This defines the "shape" that a step result object must have. Dataclasses
    like `HydroStep` can conform to this protocol structurally.
    """

    state: StateType
    forc_flux: StateType
    vap_flux: StateType
    lat_flux: StateType
    vert_flux: StateType


# 3. DEFINE A GENERIC INTERFACE FOR A COMPUTATIONAL ZONE
class Zone:
    """An abstract base class that defines the generic interface for a single
    computational unit in the model (e.g., a HydrologicZone or a
    ReactiveTransportZone).
    """

    def step(self, s_0: object, d: object, dt: float, q_in: object) -> object:
        """Advances the state of the zone by one time step.

        Args:
            s_0: The initial state of the zone.
            d: The forcing data for the current time step.
            dt: The duration of the time step.
            q_in: The total incoming flux from other connected zones.

        Returns:
            A StepResult object containing the new state and all calculated fluxes.
        """
        raise NotImplementedError

    def param_list(self) -> list[float]:
        """Return a list of the zone's parameter values for analysis."""
        raise NotImplementedError

    def columns(self, zone_id: int) -> list[str]:
        """Gets the column names for this zone for the output DataFrame."""
        raise NotImplementedError
