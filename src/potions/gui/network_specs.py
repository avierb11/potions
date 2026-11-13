"""
The curated reaction-network registry of the GUI.

The GUI runs one of a small, hand-curated set of reaction networks.
Each preset is described by a `NetworkSpec`: a display name and a
zero-argument builder that constructs the network from the default
database. The registry is deliberately closed: adding another
selectable network means adding one entry here, not changing the
GUI. User-authored networks are loaded from a YAML file through
`main`'s ``--network`` option or the window's network editor.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import potions as pt

from potions.reactive_transport.common_networks import (
    get_carbon_network,
    get_chloride_network,
    get_simple_carbon_adsorption_network,
    get_simple_carbon_network,
    get_soc_calcite_network,
)


@dataclass(frozen=True, slots=True)
class NetworkSpec:
    """
    One selectable reaction network and its GUI metadata.

    Attributes:
        key: The stable identifier used in the GUI's network selector.
        display_name: The human-readable name shown in the selector.
        builder: The zero-argument factory for the network, built
            from the default database.
    """

    key: str
    display_name: str
    builder: Callable[[], pt.ReactionNetwork]


SOC_CALCITE: Final[NetworkSpec] = NetworkSpec(
    key="soc_calcite",
    display_name="East River SOC / DOC / calcite",
    builder=get_soc_calcite_network,
)

CARBON: Final[NetworkSpec] = NetworkSpec(
    key="carbon",
    display_name="Carbon (SOC, no calcite)",
    builder=get_carbon_network,
)

SIMPLE_CARBON: Final[NetworkSpec] = NetworkSpec(
    key="simple_carbon",
    display_name="Simple carbon (DOC + SOC)",
    builder=get_simple_carbon_network,
)

SIMPLE_CARBON_ADSORPTION: Final[NetworkSpec] = NetworkSpec(
    key="simple_carbon_adsorption",
    display_name="Simple carbon with adsorption",
    builder=get_simple_carbon_adsorption_network,
)

CHLORIDE: Final[NetworkSpec] = NetworkSpec(
    key="chloride",
    display_name="Chloride (no speciation)",
    builder=get_chloride_network,
)


NETWORK_SPECS: Final[dict[str, NetworkSpec]] = {
    spec.key: spec
    for spec in (
        SOC_CALCITE,
        CARBON,
        SIMPLE_CARBON,
        SIMPLE_CARBON_ADSORPTION,
        CHLORIDE,
    )
}

# The network the window opens on, and the fallback for state files
# and saved runs that predate the network selector.
DEFAULT_NETWORK_KEY: Final[str] = "soc_calcite"
DEFAULT_NETWORK_SPEC: Final[NetworkSpec] = NETWORK_SPECS[DEFAULT_NETWORK_KEY]


def get_network_spec(key: str) -> NetworkSpec:
    """
    Return the network spec registered under ``key``.

    Args:
        key: The spec's stable identifier.

    Returns:
        The registered `NetworkSpec`.

    Raises:
        ValueError:
            Raised when no spec is registered under ``key``; the
            message lists the known keys.
    """
    try:
        return NETWORK_SPECS[key]
    except KeyError:
        raise ValueError(
            f"Unknown network key {key!r}; expected one of "
            f"{sorted(NETWORK_SPECS)}."
        ) from None
