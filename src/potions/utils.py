from abc import abstractmethod
from collections import namedtuple
from typing import Generic, TypeVar
from dataclasses import dataclass
import numpy as np
from numpy import float64 as f64
from numba.experimental import jitclass

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


# @jitclass
# class HydroForcing:
#     """Contains hydrologic forcing data for a single zone at a single time step.

#     Attributes:
#         precip: Precipitation rate (e.g., mm/day).
#         temp: Temperature (e.g., °C).
#         pet: Potential evapotranspiration rate (e.g., mm/day).
#     """

#     precip: float
#     temp: float
#     pet: float

#     def __init__(self, precip: float, temp: float, pet: float) -> None:
#         self.precip = precip
#         self.temp = temp
#         self.pet = pet

HydroForcing = namedtuple("HydroForcing", ["precip", "temp", "pet"])
