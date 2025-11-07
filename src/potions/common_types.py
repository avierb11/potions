from typing import TypeVar
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from numpy import float64 as f64

from .utils import HydroForcing

# ==== Types ==== #
M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
Vector = np.ndarray[tuple[N], np.dtype[f64]]
Matrix = np.ndarray[tuple[M, N], np.dtype[f64]]


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
