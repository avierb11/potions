from dataclasses import dataclass
import operator

from functools import reduce

from .core import HydrologicZone, SubsurfaceZoneB

from typing import Any, Iterator, Optional, overload


class Layer:
    """A horizontal collection of one or more computational zones.

    A `Layer` represents a set of zones that exist at the same vertical level
    within the model structure. It acts as a container for zones that are
    laterally connected or are part of the same conceptual stratum (e.g., a
    soil layer composed of multiple hillslope positions).

    Example:
        >>> # A layer with two snow zones
        >>> snow_layer = Layer(SnowZone(name="snow_hs"), SnowZone(name="snow_rp"))
        >>> len(snow_layer)
        2

    Attributes:
        zones (list[HydrologicZone]): The list of zone objects within the layer.
    """

    @overload
    def __init__(self, *zones: HydrologicZone) -> None:
        """Initializes a Layer with a variable number of Zone objects."""
        ...

    @overload
    def __init__(self, zones: list[HydrologicZone]) -> None:
        """Initializes a Layer with a list of Zone objects."""
        ...

    def __init__(self, *args: Any) -> None:  # type: ignore
        """Initializes a Layer.

        Args:
            *args: Either a variable number of `HydrologicZone` objects or a
                single list of `HydrologicZone` objects.
        """
        if len(args) == 1 and isinstance(args[0], list):
            self.__zones: list[HydrologicZone] = args[0]
        else:
            self.__zones = list(args)

    @property
    def zones(self) -> list[HydrologicZone]:
        """The list of zones contained within this layer."""
        return self.__zones

    def __iter__(self) -> Iterator[HydrologicZone]:
        """Returns an iterator over the zones in the layer."""
        return iter(self.zones)

    def __len__(self) -> int:
        """Returns the number of zones in the layer."""
        return len(self.zones)

    def __getitem__(self, ind: int) -> Optional[HydrologicZone]:
        """Retrieves a zone by its index within the layer.

        Args:
            ind (int): The index of the zone to retrieve.

        Returns:
            Optional[HydrologicZone]: The zone at the specified index, or `None`
                if the index is out of bounds.
        """
        if 0 <= ind < len(self.zones):
            return self.__zones[ind]
        else:
            return None


@dataclass(frozen=True)
class Hillslope:
    """A vertical stack of Layers. (Deprecated)

    This class is deprecated and will be removed in a future version. The model
    structure is now defined directly as a list of lists of zones.

    Attributes:
        layers: A list of Layer objects, ordered from top to bottom.
    """

    layers: list[Layer]

    def __iter__(self) -> Iterator[Layer]:
        """Returns an iterator over the layers in the hillslope."""
        return iter(self.layers)

    def __len__(self) -> int:
        """Returns the number of zones in the hillslope."""
        return reduce(operator.add, map(len, self.layers), 0)

    def __getitem__(self, ind: int) -> Optional[Layer]:
        """Retrieves a layer by its index.

        Args:
            ind (int): The index of the layer.

        Returns:
            Optional[Layer]: The layer at the index, or `None` if out of bounds.
        """
        if 0 <= ind < len(self.layers):
            return self.layers[ind]
        else:
            return None

    def flatten(self) -> list[HydrologicZone]:
        """Flattens the hillslope structure into a single list of zones.

        The zones are ordered from top layer to bottom layer, and within each
        layer, from left to right. This sequential list is used by the model
        engine for processing.

        Returns:
            A list of all zones in the hillslope.
        """
        return reduce(operator.add, map(lambda x: x.zones, self.layers), [])


@dataclass(frozen=True)
class ZonePosition:
    """Represents the unique position of a zone within the model's structure.

    Attributes:
        model_id: The global index of the zone in the flattened model.
        zone_id: The index of the zone within its layer (laterally).
        layer_id: The index of the layer within its hillslope (vertically).
        hillslope_id: The index of the hillslope within the model.
    """

    model_id: int
    zone_id: int
    layer_id: int
    hillslope_id: int


@dataclass(frozen=True)
class AnnotatedZone:
    """A wrapper class that holds a zone and its associated metadata. (Deprecated)

    Attributes:
        zone: The hydrologic zone object.
        size: The proportion of the total catchment area this zone represents.
        pos: The `ZonePosition` of this zone in the model.
        incoming_fluxes: A list of model_ids for zones that flow into this one.
    """

    zone: HydrologicZone
    size: float
    pos: ZonePosition
    incoming_fluxes: list[int]


# ===================== Bottom-zone registry ===================== #

BOTTOM_ZONE_CLASSES: set[type] = set()
"""Registry of zone classes that may only appear in the *last* (bottom) layer
of a model structure.

A "bottom zone" is a zone that transmits no vertical flux to a zone below it
(e.g. ``SubsurfaceZoneB``, which has a hardcoded zero vertical flux). Model
structure validation enforces two rules against this registry:

1. A bottom zone must *not* appear in any layer other than the last one.
2. The last layer must consist *only* of bottom zones, so that no water is
   lost through the base of the model domain.
"""


def register_bottom_zone(cls: type) -> type:
    """Mark a zone class as a "bottom zone" (only valid in the model's bottom layer).

    Use as a class decorator on a custom zone class that, like
    ``SubsurfaceZoneB``, discharges no vertical flux to a zone below it::

        @register_bottom_zone
        class MyNoThroughflowZone(SubsurfaceZoneB):
            ...

    The decorated class is registered in ``BOTTOM_ZONE_CLASSES`` and returned
    unchanged, so the decorator is transparent to the class definition.
    Alternatively, add the class directly to ``BOTTOM_ZONE_CLASSES``.

    Args:
        cls: The zone class (typically a subclass of ``HydrologicZone``).

    Returns:
        The (unchanged) class, for use with ``@register_bottom_zone``.
    """
    BOTTOM_ZONE_CLASSES.add(cls)
    return cls


def is_bottom_zone(zone: Any) -> bool:
    """Check whether ``zone`` is an instance (or a subclass) of a registered
    bottom-zone class.

    Args:
        zone: A zone instance, or (for convenience) a zone class.

    Returns:
        True if ``zone`` belongs to a class registered via
        ``register_bottom_zone`` (or a subclass of one), else False.
    """
    ztype = zone if isinstance(zone, type) else type(zone)
    return any(issubclass(ztype, c) for c in BOTTOM_ZONE_CLASSES)


# Register the built-in bottom zone: ``SubsurfaceZoneB`` has a hardcoded zero
# vertical flux (no throughflow), making it the canonical base zone.
BOTTOM_ZONE_CLASSES.add(SubsurfaceZoneB)
