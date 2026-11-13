"""
The curated model registry of the GUI.

The GUI lets the user calibrate one of a small, hand-curated set of
hydrology model structures.  Each structure is described by a
`ModelSpec`: the model class, the reactive-transport configuration of
every zone, and the zone labels and colors the parameter panel and the
figures use.

The registry is deliberately closed: adding another selectable model
means adding one entry here, not changing the GUI.  User-authored model
configurations are out of scope.
"""

from dataclasses import dataclass
from typing import Final

import potions as pt

from potions.gui.common import ZONE_COLORS, ZONE_LABELS


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """
    One selectable hydrology model and its GUI metadata.

    Attributes:
        key: The stable identifier used in persisted state and in the
            GUI's model selector.
        display_name: The human-readable name shown in the selector and
            the window title.
        model_class: The model class a run instantiates.
        zone_configs: The reactive-transport configuration of each zone,
            keyed by zone name in model order.
        zone_labels: The display name of each zone.
        zone_colors: The figure color of each zone.
    """

    key: str
    display_name: str
    model_class: type[pt.Model]
    zone_configs: dict[str, pt.RtZoneConfiguration]
    zone_labels: dict[str, str]
    zone_colors: dict[str, str]

    @property
    def zones(self) -> list[str]:
        """
        The model's zone names, in model order.

        The order is layer-major: the zones of the first structure layer
        (the snow layer), then the second, and so on.
        """
        return [
            zone.name
            for layer in self.model_class.structure
            for zone in layer
        ]

    @property
    def snow_zones(self) -> list[str]:
        """The zones whose class is a snow zone."""
        return [
            zone
            for zone in self.zones
            if issubclass(
                self.model_class.get_zone_type(zone), pt.SnowZone
            )
        ]

    @property
    def reactive_zones(self) -> list[str]:
        """The zones whose reactive-transport config runs reactions."""
        return [
            zone
            for zone in self.zones
            if self.zone_configs[zone].do_reactions
        ]

    @property
    def speciation_zones(self) -> list[str]:
        """The zones whose reactive-transport config runs speciation."""
        return [
            zone
            for zone in self.zones
            if self.zone_configs[zone].do_speciation
        ]

    @property
    def zone_layer_index(self) -> dict[str, int]:
        """The structure-layer index of every zone."""
        return {
            zone.name: index
            for index, layer in enumerate(self.model_class.structure)
            for zone in layer
        }

    @property
    def stack_order(self) -> list[str]:
        """
        The zones deepest-layer first, model order within a layer.

        This is the draw order of the discharge-partitioning figure's
        stacked bands, so the surface layers sit on top of the deeper
        ones.
        """
        return [
            zone.name
            for layer in reversed(self.model_class.structure)
            for zone in layer
        ]


# Snow zones carry no reactive chemistry beyond water storage; every
# other zone runs both mineral reactions and aqueous speciation.
SNOW: Final[pt.RtZoneConfiguration] = pt.RtZoneConfiguration(
    do_reactions=False, do_speciation=False
)
REACTIVE: Final[pt.RtZoneConfiguration] = pt.RtZoneConfiguration(
    do_reactions=True, do_speciation=True
)


def configs(
    zones: list[str], snow_zones: list[str]
) -> dict[str, pt.RtZoneConfiguration]:
    """
    Build the reactive-transport configs of a model, in model order.

    The insertion order matters: `rt_params_from_array` iterates the
    config's items but zips them with the model's zone names, so the
    configs must be built in the same order as the model's zones.

    Args:
        zones: The model's zone names, in model order.
        snow_zones: The subset of ``zones`` that carries no chemistry.

    Returns:
        The configuration of every zone, keyed by zone name.
    """
    return {
        zone: SNOW if zone in snow_zones else REACTIVE
        for zone in zones
    }


LATERAL_THREE_LAYER: Final[ModelSpec] = ModelSpec(
    key="lateral_three_layer",
    display_name="Lateral three-layer",
    model_class=pt.LateralThreeLayerModel,
    zone_configs=configs(
        ["snow_hs", "snow_rp", "surface_hs", "surface_rp",
         "ground_hs", "ground_rp"],
        ["snow_hs", "snow_rp"],
    ),
    zone_labels=dict(ZONE_LABELS),
    zone_colors=dict(ZONE_COLORS),
)

THREE_LAYER: Final[ModelSpec] = ModelSpec(
    key="three_layer",
    display_name="Three-layer",
    model_class=pt.ThreeLayerModel,
    zone_configs=configs(
        ["snow", "surface", "ground"],
        ["snow"],
    ),
    zone_labels={
        "snow": "Snow",
        "surface": "Surface",
        "ground": "Ground",
    },
    zone_colors={
        "snow": "#598DB1",
        "surface": "#F64740",
        "ground": "#76B041",
    },
)

HBV: Final[ModelSpec] = ModelSpec(
    key="hbv",
    display_name="HBV",
    model_class=pt.HbvModel,
    zone_configs=configs(
        ["snow", "surface", "shallow", "deep"],
        ["snow"],
    ),
    zone_labels={
        "snow": "Snow",
        "surface": "Surface",
        "shallow": "Shallow subsurface",
        "deep": "Deep subsurface",
    },
    zone_colors={
        "snow": "#598DB1",
        "surface": "#F64740",
        "shallow": "#76B041",
        "deep": "#A67B5B",
    },
)

HBV_LATERAL: Final[ModelSpec] = ModelSpec(
    key="hbv_lateral",
    display_name="HBV lateral",
    model_class=pt.HbvLateralModel,
    zone_configs=configs(
        ["snow_hs", "snow_rp", "surface_hs", "surface_rp",
         "shallow_hs", "shallow_rp", "deep_hs", "deep_rp"],
        ["snow_hs", "snow_rp"],
    ),
    zone_labels={
        "snow_hs": "Snow (HS)",
        "snow_rp": "Snow (RP)",
        "surface_hs": "Surface (HS)",
        "surface_rp": "Surface (RP)",
        "shallow_hs": "Shallow (HS)",
        "shallow_rp": "Shallow (RP)",
        "deep_hs": "Deep (HS)",
        "deep_rp": "Deep (RP)",
    },
    zone_colors={
        "snow_hs": "#598DB1",
        "snow_rp": "#2E4D61",
        "surface_hs": "#F64740",
        "surface_rp": "#750906",
        "shallow_hs": "#76B041",
        "shallow_rp": "#466827",
        "deep_hs": "#A67B5B",
        "deep_rp": "#5C4033",
    },
)


MODEL_SPECS: Final[dict[str, ModelSpec]] = {
    spec.key: spec
    for spec in (
        LATERAL_THREE_LAYER,
        THREE_LAYER,
        HBV,
        HBV_LATERAL,
    )
}

# The model the window opens on, and the fallback for older saved runs
# and state files that predate the selector.
DEFAULT_MODEL_KEY: Final[str] = "lateral_three_layer"
DEFAULT_MODEL_SPEC: Final[ModelSpec] = MODEL_SPECS[DEFAULT_MODEL_KEY]


def get_model_spec(key: str) -> ModelSpec:
    """
    Return the spec registered under ``key``.

    Args:
        key: The spec's stable identifier.

    Returns:
        The registered `ModelSpec`.

    Raises:
        ValueError:
            Raised when no spec is registered under ``key``; the
            message lists the known keys.
    """
    try:
        return MODEL_SPECS[key]
    except KeyError:
        raise ValueError(
            f"Unknown model key {key!r}; expected one of "
            f"{sorted(MODEL_SPECS)}."
        ) from None


def model_key_for_class_name(name: str) -> str | None:
    """
    Map a fully qualified model class name to its spec key.

    Saved runs record their model as a dotted class path such as
    ``potions.common_models.LateralThreeLayerModel``.  Only the final
    component is matched, so the lookup survives a module move.

    Args:
        name: The recorded class name, dotted or bare.

    Returns:
        The matching spec key, or None when the class is not one of the
        curated models.
    """
    short_name: str = name.rsplit(".", 1)[-1]
    for key, spec in MODEL_SPECS.items():
        if spec.model_class.__name__ == short_name:
            return key
    return None
