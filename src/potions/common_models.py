"""
This file contains commonly-used model configurations
"""

from .model import Model
from .core import SnowZone, SurfaceZone, GroundZone, GroundZoneB

SubsurfaceZone = GroundZone
SubsurfaceZoneB = GroundZoneB


class HbvModel(Model):
    """A standard, single-column HBV-like model structure."""

    structure = [
        [SnowZone(name="snow")],
        [SurfaceZone(name="surface")],
        [SubsurfaceZone(name="shallow")],
        [SubsurfaceZoneB(name="deep")],
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        lines = [
            "HbvModel(",
            f"\ttt={round(self['snow'].tt, 2)},",  # type: ignore
            f"\tfmax={round(self['snow'].fmax, 2)},",  # type: ignore
            f"\tfc={round(self['surface'].fc, 2)},",  # type: ignore
            f"\tlp={round(self['surface'].lp, 2)},",  # type: ignore
            f"\tbeta={round(self['surface'].beta, 2)},",  # type: ignore
            f"\tk0={round(self['surface'].k0, 4)},",  # type: ignore
            f"\tthr={round(self['surface'].thr, 2)},",  # type: ignore
            f"\tk1={round(self['shallow'].k, 4)},",  # type: ignore
            f"\tshallow_alpha={
                round(self['shallow'].alpha, 2)},",  # type: ignore
            f"\tperc={round(self['shallow'].perc, 2)},",  # type: ignore
            f"\tk2={round(self['deep'].k, 4)},",  # type: ignore
            f"\tdeep_alpha={round(self['deep'].alpha, 2)},",  # type: ignore
            ")",
        ]

        return "\n".join(lines)

    @property
    def snow(self) -> SnowZone:
        return self.hydro_zones["snow"]  # type: ignore

    @property
    def surface(self) -> SurfaceZone:
        return self.hydro_zones["surface"]  # type: ignore

    @property
    def shallow(self) -> SubsurfaceZone:
        return self.hydro_zones["shallow"]  # type: ignore

    @property
    def deep(self) -> SubsurfaceZoneB:
        return self.hydro_zones["deep"]  # type: ignore


class HbvLateralModel(Model):
    """An HBV-like model with two lateral columns (e.g., hillslope/riparian)."""

    structure = [
        [SnowZone(name="snow_hs"), SnowZone(name="snow_rp")],
        [SurfaceZone(name="surface_hs"), SurfaceZone(name="surface_rp")],
        [SubsurfaceZone(name="shallow_hs"), SubsurfaceZone(name="shallow_rp")],
        [SubsurfaceZoneB(name="deep_hs"), SubsurfaceZoneB(name="deep_rp")],
    ]

    @property
    def snow_hs(self) -> SnowZone:
        return self.hydro_zones["snow_hs"]  # type: ignore

    @property
    def surface_hs(self) -> SurfaceZone:
        return self.hydro_zones["surface_hs"]  # type: ignore

    @property
    def shallow_hs(self) -> SubsurfaceZone:
        return self.hydro_zones["shallow_hs"]  # type: ignore

    @property
    def deep_hs(self) -> SubsurfaceZoneB:
        return self.hydro_zones["deep_hs"]  # type: ignore

    @property
    def snow_rp(self) -> SnowZone:
        return self.hydro_zones["snow_rp"]  # type: ignore

    @property
    def surface_rp(self) -> SurfaceZone:
        return self.hydro_zones["surface_rp"]  # type: ignore

    @property
    def shallow_rp(self) -> SubsurfaceZone:
        return self.hydro_zones["shallow_rp"]  # type: ignore

    @property
    def deep_rp(self) -> SubsurfaceZoneB:
        return self.hydro_zones["deep_rp"]  # type: ignore


class HbvNonlinearModel(Model):
    """A single-column HBV-like model with non-linear groundwater reservoirs."""

    structure = [
        [SnowZone(name="snow")],
        [SurfaceZone(name="surface")],
        [SubsurfaceZone(name="shallow")],
        [SubsurfaceZoneB(name="deep")],
    ]


class ThreeLayerModel(Model):
    """A simple three-layer model: Snow, Soil, and a single Groundwater zone."""

    structure = [
        [SnowZone(name="snow")],
        [SurfaceZone(name="surface")],
        [SubsurfaceZoneB(name="ground")],
    ]

    @property
    def snow(self) -> SnowZone:
        return self.hydro_zones["snow"]  # type: ignore

    @property
    def surface(self) -> SurfaceZone:
        return self.hydro_zones["surface"]  # type: ignore

    @property
    def ground(self) -> SubsurfaceZoneB:
        return self.hydro_zones["ground"]  # type: ignore


class LateralThreeLayerModel(Model):
    """A simple three-layer model: Snow, Soil, and a single Groundwater zone."""

    structure = [
        [SnowZone(name="snow_hs"), SnowZone(name="snow_rp")],
        [SurfaceZone(name="surface_hs"), SurfaceZone(name="surface_rp")],
        [SubsurfaceZoneB(name="ground_hs"), SubsurfaceZoneB(name="ground_rp")],
    ]

    @property
    def snow_hs(self) -> SnowZone:
        return self.hydro_zones["snow_hs"]  # type: ignore

    @property
    def surface_hs(self) -> SurfaceZone:
        return self.hydro_zones["surface_hs"]  # type: ignore

    @property
    def ground_hs(self) -> SubsurfaceZoneB:
        return self.hydro_zones["ground_hs"]  # type: ignore

    @property
    def snow_rp(self) -> SnowZone:
        return self.hydro_zones["snow_rp"]  # type: ignore

    @property
    def surface_rp(self) -> SurfaceZone:
        return self.hydro_zones["surface_rp"]  # type: ignore

    @property
    def ground_rp(self) -> SubsurfaceZoneB:
        return self.hydro_zones["ground_rp"]  # type: ignore
