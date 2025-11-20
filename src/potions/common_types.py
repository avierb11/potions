from typing import TypeVar
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from numpy import float64 as f64
from pandas import Series


# ==== Types ==== #
M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
Vector = np.ndarray[tuple[N], np.dtype[f64]]
Matrix = np.ndarray[tuple[M, N], np.dtype[f64]]


@dataclass(frozen=True)
class HydroForcing:
    """Contains hydrologic forcing data for a single zone at a single time step.

    Attributes:
        precip: Precipitation rate (e.g., mm/day).
        temp: Temperature (e.g., °C).
        pet: Potential evapotranspiration rate (e.g., mm/day).
        q_in: Water input from an external zone
    """

    precip: float
    temp: float
    pet: float
    q_in: float


@dataclass(frozen=True)
class ForcingData:
    """Represents the time series of meteorological forcing data for a single location.

    Attributes:
        precip: Time series of precipitation (e.g., mm/day).
        temp: Time series of temperature (e.g., °C).
        pet: Time series of potential evapotranspiration (e.g., mm/day).
    """

    precip: Series
    temp: Series
    pet: Series


@dataclass
class LapseRateParameters:
    temp_factor: float  # The slope of the temperature line
    precip_factor: float  # The slope of the precipitation line

    def scale_temperature(
        self, gauge_elevation: float, elev: float, temp_series: Series
    ) -> Series:
        """Scale the temperature time series according to the elevation"""
        return temp_series + self.temp_factor * (elev - gauge_elevation)

    def scale_precipitation(
        self, gauge_elevation: float, elev: float, precip_series: Series
    ) -> Series:
        """Scale the precipitation according to elevation zones"""
        return precip_series + self.precip_factor * (elev - gauge_elevation)

    def scale_forcing_data(
        self, gauge_elevation: float, elev: float, forcing_data: ForcingData
    ) -> ForcingData:
        """Scale the forcing data according to elevation zones"""
        return ForcingData(
            precip=self.scale_precipitation(gauge_elevation, elev, forcing_data.precip),
            temp=self.scale_temperature(gauge_elevation, elev, forcing_data.temp),
            pet=forcing_data.pet,
        )

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        return {"temp_factor": (-1, 0), "precip_factor": (0, 10)}

    @classmethod
    def from_dict(cls, params: dict) -> "LapseRateParameters":
        try:
            return cls(**params)
        except TypeError:
            raise TypeError(
                f"Invalid parameters for lapse rate: {params}, expected {list(cls.default_parameter_range().keys())}"
            )


@dataclass(frozen=True)
class ChemicalState:
    """Represents the chemical state of a zone, partitioned by species type."""

    prim_aq_conc: NDArray
    min_conc: NDArray
    sec_conc: NDArray
    exchange_conc: NDArray

    def to_primary_array(self) -> Vector:
        """Concatenates primary aqueous and mineral species into a single array."""
        return np.concatenate([self.prim_aq_conc, self.min_conc])  # type: ignore

    def to_array(self) -> NDArray:
        """Concatenates all species into a single array."""
        raise NotImplementedError()

    @property
    def aqueous_concentrations(self) -> Vector:
        """
        Get a vector of aqueous concentrations, including primary and secondary
        """
        return np.concatenate([self.prim_aq_conc, self.sec_conc])  # type: ignore


@dataclass(frozen=True)
class RtForcing:
    """Contains the hydrologic and chemical drivers for a reactive transport step."""

    conc_in: ChemicalState
    q_in: float
    q_lat_out: float
    q_vert_out: float
    hydro_forc: HydroForcing
    storage: float  # Water storage in the zone, in millimeters
    s_w: float  # Fraction of soil taken up by water, ranges from [0,1], with 1 indicating all porosity is filled
    z_w: float  # Depth of the water table

    @property
    def q_out(self) -> float:
        """The total flux of water out of this zone"""
        return self.q_lat_out + self.q_vert_out
