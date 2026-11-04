import numpy as np
import pytest

from ..core import (
    HydrologicZone,
    MineralParameters,
    RiverDimensions,
    RiverParameters,
    RiverZone,
    RtParameters,
    RtZone,
    SnowZone,
    SubsurfaceZoneB,
    SurfaceZone,
    ZoneDimensions,
)
from ..model import Model
from ..reactive_transport.common_networks import get_carbon_network
from ..reactive_transport_model import ReactiveTransportModel


# ===================== Hydro-only model ===================== #
class HydroTestModel(Model):
    """A single catchment column: snow -> surface -> ground (hydro only)."""

    structure: list[list] = [
        [SnowZone(name="snow")],
        [SurfaceZone(name="surface")],
        [SubsurfaceZoneB(name="ground")],
    ]


def _hydro_zones() -> dict[str, HydrologicZone]:
    # Pass explicit zones so that the stored zone instances keep the names
    # declared in `structure` (a zone's compiled `default()` constructor may
    # use a different internal name).
    return {
        "snow": SnowZone(name="snow"),
        "surface": SurfaceZone(name="surface"),
        "ground": SubsurfaceZoneB(name="ground"),
    }


@pytest.fixture(scope="module")
def hydro_model() -> HydroTestModel:
    return HydroTestModel(scales=[1.0], zones=_hydro_zones())


def test_hydro_describe_parameters_columns(hydro_model) -> None:
    df = hydro_model.describe_parameters()
    assert list(df.columns) == ["zone", "name", "value", "unit", "description"]


def test_hydro_describe_parameters_rows(hydro_model) -> None:
    df = hydro_model.describe_parameters()
    # One row per hydro-zone parameter: 2 snow + 5 surface + 2 ground (bottom
    # zone) = 9. A single surface zone means no catchment
    # proportions are emitted.
    assert len(df) == 9
    assert set(df["zone"]) == {"snow", "surface", "ground"}
    # Row names match the zone's declared parameter names, in order.
    for zone_name in ("snow", "surface", "ground"):
        zone = hydro_model[zone_name]
        assert set(df[df["zone"] == zone_name]["name"]) == set(
            zone.parameter_names()
        )


def test_hydro_describe_values_roundtrip(hydro_model) -> None:
    df = hydro_model.describe_parameters()
    for zone_name in ("snow", "surface", "ground"):
        zone = hydro_model[zone_name]
        names = zone.parameter_names()
        values = zone.param_list()
        zone_df = df[df["zone"] == zone_name].set_index("name")
        for name, value in zip(names, values, strict=False):
            assert name in zone_df.index
            # Value is rendered by `_fmt`: a rounded string of the float.
            assert str(round(float(value), 4)) == zone_df.loc[name, "value"]


def test_hydro_describe_string(hydro_model) -> None:
    text = hydro_model.describe()
    assert "Model parameters" in text
    for zone_name in ("snow", "surface", "ground"):
        assert zone_name in text


def test_hydro_describe_unit_column(hydro_model) -> None:
    df = hydro_model.describe_parameters()
    # Hydro parameters carry no dedicated units in this feature.
    assert df["unit"].fillna("").eq("").all()


def test_hydro_describe_all_fields_present(hydro_model) -> None:
    df = hydro_model.describe_parameters()
    # Every row has a non-empty name and a scalar (string) value.
    assert df["name"].notna().all()
    assert df["value"].map(lambda v: isinstance(v, str)).all()


# ===================== Reactive-transport model ===================== #
@pytest.fixture(scope="module")
def network():
    """The native carbon network (one mineral, SOC(s))."""
    return get_carbon_network()


def _rt_zone(network, name: str, do_reactions: bool) -> RtZone:
    dim = ZoneDimensions(20.0, 10.0)
    if do_reactions:
        mp = MineralParameters(
            np.array([0.1]),
            np.array([2.0]),
            np.array([1.5]),
            np.array([10.0]),
        )
        params = RtParameters(dim, mp)
    else:
        params = RtParameters(dim, None)
    return RtZone(
        network, params, do_reactions=do_reactions, do_speciation=True, name=name
    )


def _river_zone(network, do_reactions: bool) -> RiverZone:
    dim = RiverDimensions(1.0, 5.0)
    if do_reactions:
        mp = MineralParameters(
            np.array([0.2]),
            np.array([3.0]),
            np.array([1.2]),
            np.array([50.0]),
        )
        params = RiverParameters(dim, mp)
    else:
        params = RiverParameters(dim, None)
    return RiverZone(
        network, params, do_reactions=do_reactions, do_speciation=True, name="river"
    )


class RtTestModel(ReactiveTransportModel):
    structure: list[list] = [
        [SnowZone(name="snow")],
        [SurfaceZone(name="surface")],
        [SubsurfaceZoneB(name="ground")],
    ]


def _model(network, do_reactions: bool, with_river: bool) -> RtTestModel:
    zones = {
        "snow": _rt_zone(network, "snow", do_reactions),
        "surface": _rt_zone(network, "surface", do_reactions),
        "ground": _rt_zone(network, "ground", do_reactions),
    }
    river = _river_zone(network, do_reactions) if with_river else None
    return RtTestModel(
        zones=_hydro_zones(),
        scales=[1.0],
        network=network,
        rt_zones=zones,
        river_zone=river,
    )


def test_rt_describe_columns(network) -> None:
    model = _model(network, True, False)
    df = model.describe_parameters()
    assert list(df.columns) == ["zone", "name", "value", "unit", "description"]


def test_rt_describe_contains_hydro_zones(network, hydro_model) -> None:
    model = _model(network, True, False)
    df = model.describe_parameters()
    # Hydro zone parameters are still present (RT rows reuse the same zone
    # name, so they share the group with the hydro rows).
    for zone_name in ("snow", "surface", "ground"):
        assert zone_name in set(df["zone"])
        present = set(df[df["zone"] == zone_name]["name"].values)
        assert set(hydro_model[zone_name].parameter_names()) <= present


def test_rt_describe_rt_dimensions(network) -> None:
    model = _model(network, True, False)
    df = model.describe_parameters()
    snow = df[df["zone"] == "snow"]
    dims = {"max_water", "passive_water_storage"}
    assert dims <= set(snow["name"].values)
    # Dimension values round-trip.
    ground = df[df["zone"] == "ground"].set_index("name")
    assert ground.loc["max_water", "value"] == str(round(20.0, 4))
    assert ground.loc["passive_water_storage", "value"] == str(round(10.0, 4))


def test_rt_describe_mineral_params_present(network) -> None:
    model = _model(network, True, False)
    df = model.describe_parameters()
    snow = df[df["zone"] == "snow"]
    # One mineral in the network -> four per-mineral rows.
    for field in ("sw_threshold", "sw_exp", "q_10", "k_a"):
        assert f"SOC(s).{field}" in snow["name"].values
    # Dimensions are also present.
    assert {"max_water", "passive_water_storage"} <= set(
        snow["name"].values
    )


def test_rt_describe_no_reactions_omits_mineral_rows(network) -> None:
    model = _model(network, False, False)
    df = model.describe_parameters()
    ground = df[df["zone"] == "ground"]
    # Dimensions still reported.
    assert {"max_water", "passive_water_storage"} <= set(
        ground["name"].values
    )
    # No per-mineral rows anywhere.
    assert not any("SOC(s)." in n for n in df["name"].values)


def test_rt_describe_river_zone(network) -> None:
    model = _model(network, True, True)
    assert model.has_river_zone
    df = model.describe_parameters()
    river = df[df["zone"] == "river"]
    assert "bed_depth" in river["name"].values
    assert "passive_water_storage" in river["name"].values
    assert "SOC(s).q_10" in river["name"].values
    # River zone has 2 dimensions + 4 mineral rows = 6 rows.
    assert len(river) == 6


def test_rt_describe_no_river_zone(network) -> None:
    model = _model(network, True, False)
    assert not model.has_river_zone
    df = model.describe_parameters()
    assert "river" not in set(df["zone"])


def test_rt_describe_string(network) -> None:
    model = _model(network, True, True)
    text = model.describe()
    assert "Model parameters" in text
    for zone_name in ("snow", "surface", "ground", "river"):
        assert zone_name in text
    assert "max_water" in text
    assert "bed_depth" in text


def test_rt_describe_string_no_river(network) -> None:
    model = _model(network, True, False)
    text = model.describe()
    assert "Model parameters" in text
    assert "max_water" in text
    assert "bed_depth" not in text
