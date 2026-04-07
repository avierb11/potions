from dataclasses import dataclass
from typing import Any, Optional, TypeVar

import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray
from pandas import DataFrame, Series

from .core import RtStep


# ==== Types ==== #
M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
Vector = np.ndarray[tuple[N], np.dtype[f64]]
Matrix = np.ndarray[tuple[M, N], np.dtype[f64]]


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
            ) from None


@dataclass(frozen=True)
class ChemicalState:
    """Represents the chemical state of a zone, partitioned by species type."""

    prim_aq_conc: NDArray  # Primary aqueous species concentrations
    sec_conc: NDArray  # Secondary species concentrations
    min_conc: NDArray  # Mineral concentrations

    def to_primary_array(self) -> NDArray:
        """Concatenates primary aqueous and mineral species into a single array."""
        return np.concatenate([self.prim_aq_conc, self.min_conc])  # type: ignore

    def to_array(self) -> NDArray:
        """Concatenates all species into a single array."""
        raise NotImplementedError()

    @property
    def aqueous_concentrations(self) -> NDArray:
        """
        Get a vector of aqueous concentrations, including primary and secondary
        """
        return np.concatenate([self.prim_aq_conc, self.sec_conc])  # type: ignore


@dataclass
class HydroSimulation:
    forcing: list[ForcingData]
    storage: DataFrame
    forc_flux: DataFrame
    lat_flux: DataFrame
    vert_flux: DataFrame
    vap_flux: DataFrame
    q_in: DataFrame
    lat_flux_ext: DataFrame
    vert_flux_ext: DataFrame


@dataclass
class HydroModelResults:
    """A type containing the results of a hydrologic model run.

    Attributes:
        simulation (DataFrame): A DataFrame with time series of states and
            fluxes for all zones, plus simulated and measured streamflow.
        objective_functions (Series): A Series with the values of each of the objective functions as keys
    """

    simulation: DataFrame
    objective_functions: Series


@dataclass
class RtModelResults:
    simulation: DataFrame
    objective_functions: Optional[DataFrame]
    rt_forcings: np.ndarray
    steps: list[list[RtStep]]
    other: dict[str, Any]


@dataclass
class ModelResults:
    hydro: HydroModelResults
    reactive_transport: RtModelResults
