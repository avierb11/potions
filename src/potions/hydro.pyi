from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class HydroForcing:
    precip: float
    temp: float
    pet: float
    q_in: float

@dataclass
class HydroStep:
    state: float
    forc_flux: float
    lat_flux: float
    vert_flux: float
    vap_flux: float
    lat_flux_ext: float
    vert_flux_ext: float

@dataclass
class HydrologicZone:
    # name: str = "unnamed"
    # name: str
    def step(self, s_0: float, d: HydroForcing, dt: float) -> HydroStep: ...
    def mass_balance(self, s: float, d: HydroForcing) -> float: ...
    def forc_flux(self, s: float, d: HydroForcing) -> float: ...
    def vap_flux(self, s: float, d: HydroForcing) -> float: ...
    def lat_flux(self, s: float, d: HydroForcing) -> float: ...
    def vert_flux(self, s: float, d: HydroForcing) -> float: ...
    def lat_flux_ext(self, s: float, d: HydroForcing) -> float: ...
    def vert_flux_ext(self, s: float, d: HydroForcing) -> float: ...
    def param_list(self) -> list[float]: ...
    @classmethod
    def num_params(cls) -> int: ...
    @classmethod
    def from_array(cls, arr: np.ndarray) -> HydrologicZone: ...
    @classmethod
    def default(cls) -> HydrologicZone: ...

@dataclass
class SnowZone(HydrologicZone):
    tt: float = 0.0
    fmax: float = 1.0
    name: str = "snow"

@dataclass
class SurfaceZone(HydrologicZone):
    fc: float = 100.0
    lp: float = 0.5
    beta: float = 1.0
    k0: float = 0.1
    thr: float = 50.0
    name: str = "surface"

@dataclass
class GroundZone(HydrologicZone):
    k: float = 1e-3
    alpha: float = 1.0
    perc: float = 1.0
    name: str = "ground"

@dataclass
class GroundZoneB(HydrologicZone):
    k: float = 1e-3
    alpha: float = 1.0
    name: str = "ground"

@dataclass
class GroundZoneLinear(HydrologicZone):
    k: float = 1e-3
    perc: float = 1.0
    name: str = "ground"

@dataclass
class GroundZoneLinearB(HydrologicZone):
    k: float = 1e-3
    name: str = "ground"
