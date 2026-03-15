from dataclasses import dataclass

@dataclass
class ForcingData:
    precip: float
    temp: float
    pet: float
    q_in: float

@dataclass
class HydroForcing:
    pass

@dataclass
class HydroStep:
    pass

@dataclass
class LapseRateParameters:
    pass

@dataclass
class RtForcing:
    pass

# Hydrology
@dataclass
class HydrologicZone:
    pass

@dataclass
class GroundZone:
    pass

@dataclass
class GroundZoneB:
    pass

@dataclass
class SnowZone:
    pass

@dataclass
class SurfaceZone:
    pass

# Kinetic Structures
@dataclass
class MonodParameters:
    pass

@dataclass
class TstParameters:
    pass

@dataclass
class EquilibriumParameters:
    pass

@dataclass
class MineralAuxParams:
    pass

@dataclass
class ZoneDimensions:
    pass

@dataclass
class MineralParameters:
    pass

@dataclass
class RtParameters:
    pass

# Reactive transport
@dataclass
class RtZone:
    pass
