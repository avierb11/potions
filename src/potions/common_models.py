"""
This file contains commonly-used model configurations
"""

from .model import Model
from .core import SnowZone, SurfaceZone, GroundZone, GroundZoneB


class HbvModel(Model):
    """A standard, single-column HBV-like model structure."""

    structure = [
        [SnowZone(name="snow")],
        [SurfaceZone(name="surface")],
        [GroundZone(name="shallow")],
        [GroundZoneB(name="deep")],
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


class HbvLateralModel(Model):
    """An HBV-like model with two lateral columns (e.g., hillslope/riparian)."""

    structure = [
        [SnowZone(name="snow_hs"), SnowZone(name="snow_rp")],
        [SurfaceZone(name="surface_hs"), SurfaceZone(name="surface_rp")],
        [GroundZone(name="shallow_hs"), GroundZone(name="shallow_rp")],
        [GroundZoneB(name="deep_hs"), GroundZoneB(name="deep_rp")],
    ]


class HbvNonlinearModel(Model):
    """A single-column HBV-like model with non-linear groundwater reservoirs."""

    structure = [
        [SnowZone(name="snow")],
        [SurfaceZone(name="surface")],
        [GroundZone(name="shallow")],
        [GroundZoneB(name="deep")],
    ]


class ThreeLayerModel(Model):
    """A simple three-layer model: Snow, Soil, and a single Groundwater zone."""

    structure = [
        [SnowZone(name="snow")],
        [SurfaceZone(name="surface")],
        [GroundZoneB(name="ground")],
    ]
