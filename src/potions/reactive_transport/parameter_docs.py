"""Metadata and helpers for describing adjustable model parameters.

This module is purely Python-side: it adds human-readable units and
physical meaning to the tunable parameters already exposed by the Rust
hydrology, reactive-transport, and river zones. No Rust rebuild is
required; it only reads the public Python getters those zones already
expose (``parameter_names``/``param_list`` for hydro zones and
``dimensions``/``mineral_params`` for RT and river zones).

Two kinds of output are produced for each parameter:

* a *row* -- a dict with ``zone``, ``name``, ``value``, ``unit`` and
  ``description`` keys, suitable for assembling into a
  :class:`pandas.DataFrame`;
* a *rendered* string -- a grouped, indented, human-readable block built
  from those rows.

Reaction-network kinetics (stoichiometry, Monod/TST coefficient
matrices) are intentionally out of scope: this only covers the
per-zone tunable dimensions and mineral reaction parameters.
"""

from __future__ import annotations


PARAMETER_DOCS: dict[str, dict[str, str]] = {
    # --- Hydro-transport zone dimensions ---
    "max_water": {
        "unit": "mm",
        "description": "Maximum reactive water volume of the zone.",
    },
    "passive_water_storage": {
        "unit": "mm",
        "description": "Non-reactive water that does not transport solutes.",
    },
    # --- River control volume ---
    "bed_depth": {
        "unit": "m",
        "description": "Depth of the river-zone control volume.",
    },
    # --- Per-mineral reaction parameters ---
    "sw_threshold": {
        "unit": "",
        "description": "Soil-moisture threshold above which the reaction is enabled.",
    },
    "sw_exp": {
        "unit": "",
        "description": "Exponent shaping the response above sw_threshold.",
    },
    "q_10": {
        "unit": "1/degC",
        "description": "Arrhenius-style sensitivity of the rate to a 10 degC change.",
    },
    "k_a": {
        "unit": "1/m3/day",
        "description": (
            "Mineral dissolution rate constant (stored value is log-scaled "
            "when natural_scales=True)."
        ),
    },
}

# Dimension ordering differs between RT and river zones; each tuple lists the
# dimension fields in the order they should be reported for that zone kind.
DIMENSION_ORDER = {
    "rt": ("max_water", "passive_water_storage"),
    "river": ("bed_depth", "passive_water_storage"),
}

MINERAL_FIELDS = ("sw_threshold", "sw_exp", "q_10", "k_a")


def fmt(value: object) -> str:
    """Render a scalar parameter value the way ``structure_diagram`` does."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return str(round(value, 4))
    return str(value)


def mineral_arrays(mineral_params) -> dict[str, list]:
    """Pull the four per-mineral arrays out of a ``MineralParameters``."""
    return {
        field: list(getattr(mineral_params, getter)())
        for field, getter in (
            ("sw_threshold", "get_sw_threshold"),
            ("sw_exp", "get_sw_exp"),
            ("q_10", "get_q_10"),
            ("k_a", "get_k_a"),
        )
    }


def row(zone_name: str, name: str, value: object) -> dict:
    """Build a single parameter row, falling back to empty unit/description."""
    meta = PARAMETER_DOCS.get(name, {})
    return {
        "zone": zone_name,
        "name": name,
        "value": fmt(value),
        "unit": meta.get("unit", ""),
        "description": meta.get("description", ""),
    }


def rt_zone_rows(zone) -> list[dict]:
    """Build description rows for a reactive-transport zone.

    Emits the three zone dimensions always, plus the per-mineral reaction
    parameters when ``do_reactions`` is true and ``mineral_params`` exist.
    """
    zone_name = zone.name
    rows: list[dict] = []

    dimensions = zone.dimensions
    for field in DIMENSION_ORDER["rt"]:
        rows.append(row(zone_name, field, getattr(dimensions, field)))

    if zone.do_reactions:
        mineral_params = zone.parameters.mineral_params
        if mineral_params is not None:
            arrays = mineral_arrays(mineral_params)
            num_minerals = len(arrays["k_a"])
            species_names = list(zone.mineral_species)
            for i in range(num_minerals):
                label = species_names[i] if i < len(species_names) else f"Mineral {i}"
                for field in MINERAL_FIELDS:
                    name = f"{label}.{field}"
                    rows.append(row(zone_name, name, arrays[field][i]))

    return rows


def river_zone_rows(zone) -> list[dict]:
    """Build description rows for the river zone (mirrors :func:`rt_zone_rows`)."""
    zone_name = zone.name
    rows: list[dict] = []

    dimensions = zone.dimensions
    for field in DIMENSION_ORDER["river"]:
        rows.append(row(zone_name, field, getattr(dimensions, field)))

    if zone.do_reactions:
        mineral_params = zone.parameters.mineral_params
        if mineral_params is not None:
            arrays = mineral_arrays(mineral_params)
            num_minerals = len(arrays["k_a"])
            species_names = list(zone.mineral_species)
            for i in range(num_minerals):
                label = species_names[i] if i < len(species_names) else f"Mineral {i}"
                for field in MINERAL_FIELDS:
                    name = f"{label}.{field}"
                    rows.append(row(zone_name, name, arrays[field][i]))

    return rows


def render_describe(rows: list[dict], title: str = "Model parameters") -> str:
    """Render parameter rows into a grouped, human-readable block."""
    if not rows:
        return title

    by_zone: dict[str, list[dict]] = {}
    for row in rows:
        by_zone.setdefault(row["zone"], []).append(row)

    lines = [title, "=" * len(title)]
    for zone_name, zone_rows in by_zone.items():
        lines.append("")
        lines.append(f"{zone_name}:")
        for row in zone_rows:
            unit = f"  [{row['unit']}]" if row["unit"] else ""
            lines.append(f"    {row['name']} = {row['value']}{unit}")
            if row["description"]:
                lines.append(f"        {row['description']}")
    return "\n".join(lines)


def zone_parameter_rows(
    zone,
    kind: str,
    names: list[str] | None = None,
    values: list[float] | None = None,
) -> list[dict]:
    """Dispatch to the appropriate per-zone row builder.

    Args:
        zone: The zone object. For ``"rt"`` and ``"river"`` this is an
            ``RtZone``/``RiverZone``; for ``"hydro"`` it is a
            ``HydrologicZone``.
        kind: One of ``"rt"``, ``"river"``, or ``"hydro"``.
        names: Hydro zone parameter names (required for ``kind="hydro"``).
        values: Hydro zone parameter values (required for ``kind="hydro"``).
    """
    if kind == "rt":
        return rt_zone_rows(zone)
    if kind == "river":
        return river_zone_rows(zone)
    if kind == "hydro":
        if names is None or values is None:
            raise ValueError("Hydro rows require both `names` and `values`")
        return [row(zone.name, n, v) for n, v in zip(names, values, strict=False)]
    raise ValueError(f"Unknown zone kind: {kind!r}")
