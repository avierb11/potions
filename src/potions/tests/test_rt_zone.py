import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from ..reactive_transport.kinetic_structures import ZoneDimensions
from ..reactive_transport.rt_zone import (
    MiscData,
    WaterVolumeError,
    calculate_moisture_fraction,
    calculate_water_table_depth,
    get_hydro_steps,
)


def _make_zone(dimensions: ZoneDimensions) -> MagicMock:
    """A mock RtZone exposing only the attribute the rt_zone helpers read:
    zone.parameters.dimensions."""
    zone = MagicMock()
    zone.parameters.dimensions = dimensions
    return zone


@pytest.fixture
def dimensions() -> ZoneDimensions:
    # porosity*depth - passive = 0.2*100 - 5 = 15 mm max water volume.
    return ZoneDimensions(porosity=0.2, depth=100.0, passive_water_storage=5.0)


@pytest.fixture
def zones(dimensions: ZoneDimensions) -> dict:
    return {"shallow": _make_zone(dimensions)}


def _storage_series(values) -> pd.DataFrame:
    return pd.DataFrame({"s_shallow": values})


def test_max_water_volume(dimensions: ZoneDimensions) -> None:
    assert dimensions.max_water_volume == pytest.approx(15.0)


def test_calculate_moisture_fraction_formula(zones, dimensions) -> None:
    sim = _storage_series([0.0, 10.0, 14.9])
    sw = calculate_moisture_fraction(zones, sim)
    assert sw.shape == (1, 3)
    # (s + passive) / (max_water_volume + passive)
    expected = (np.array([0.0, 10.0, 14.9]) + 5.0) / (15.0 + 5.0)
    assert np.allclose(sw[0], expected)


def test_moisture_fraction_full_at_max_capacity(zones, dimensions) -> None:
    # As storage approaches the max water volume (15 mm), the fraction approaches 1.
    # (Exactly at max it raises, so use a value just below.)
    sim = _storage_series([14.99])
    sw = calculate_moisture_fraction(zones, sim)
    assert sw[0, 0] == pytest.approx(1.0, abs=1e-3)


def test_moisture_fraction_exceeds_max_raises(zones) -> None:
    sim = _storage_series([15.1])
    with pytest.raises(WaterVolumeError):
        calculate_moisture_fraction(zones, sim)


def test_calculate_water_table_depth_formula(zones, dimensions) -> None:
    sim = _storage_series([0.0, 10.0, 14.9])
    zw = calculate_water_table_depth(zones, sim)
    # zw = depth - (s + passive) / porosity
    # 100 - (s + 5) / 0.2  ->  [75.0, 25.0, 0.5]
    expected = np.array([75.0, 25.0, 0.5])
    assert np.allclose(zw[0], expected)


def test_water_table_depth_full_storage_gives_surface(zones, dimensions) -> None:
    # At max water volume (15 mm), the water table is at the surface (zw -> ~0).
    sim = _storage_series([15.0])
    zw = calculate_water_table_depth(zones, sim)
    assert zw[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_water_table_depth_full_storage_negative_raises(zones) -> None:
    sim = _storage_series([15.1])
    with pytest.raises(WaterVolumeError):
        calculate_water_table_depth(zones, sim)


def _full_simulation(storage_values) -> pd.DataFrame:
    index = pd.date_range("2000-01-01", periods=len(storage_values))
    sim = pd.DataFrame({"s_shallow": storage_values}, index=index)
    for flux in [
        "q_forc_shallow",
        "q_vap_shallow",
        "q_lat_shallow",
        "q_lat_ext_shallow",
        "q_vert_shallow",
        "q_vert_ext_shallow",
        "q_in_shallow",
    ]:
        sim[flux] = np.linspace(0.0, 1.0, len(storage_values))
    return sim


def test_get_hydro_steps_shape() -> None:
    sim = _full_simulation([0.0, 10.0, 14.9])
    steps = get_hydro_steps(sim)
    assert steps.shape == (1, 3)


def test_get_hydro_steps_carries_per_zone_values() -> None:
    sim = _full_simulation([0.0, 10.0, 14.9])
    steps = get_hydro_steps(sim)
    cell = steps[0, 1]
    assert cell.state == pytest.approx(10.0)
    # The 3 non-s_ column values at step 1 are 0.5 each.
    assert cell.q_in == pytest.approx(0.5)
    assert cell.lat_flux_ext == pytest.approx(0.5)
    assert cell.vap_flux == pytest.approx(0.5)


def test_get_hydro_steps_multiple_zones() -> None:
    index = pd.date_range("2000-01-01", periods=2)
    sim = _full_simulation([0.0, 5.0])
    # Add a second zone's state + flux columns so both zones are discovered.
    for z in ["deep"]:
        sim[f"s_{z}"] = np.array([1.0, 2.0])
        for flux in [
            "q_forc_deep",
            "q_vap_deep",
            "q_lat_deep",
            "q_lat_ext_deep",
            "q_vert_deep",
            "q_vert_ext_deep",
            "q_in_deep",
        ]:
            sim[flux] = np.array([0.2, 0.4])
    steps = get_hydro_steps(sim)
    assert steps.shape == (2, 2)


def _misc() -> MiscData:
    return MiscData(
        mineral_stoichiometry=np.array([[1.0]]),
        species_mobility=np.array([True, False]),
        mineral_molar_mass=np.array([100.0]),
        rate_const=np.array([1e-6]),
    )


def test_miscdata_equality_allclose() -> None:
    a = MiscData(
        np.array([[1.0]]), np.array([True, False]), np.array([100.0]), np.array([1e-6])
    )
    b = MiscData(
        # Values within floating tolerance round-trip as equal.
        np.array([[1.0 + 1e-13]]),
        np.array([True, False]),
        np.array([100.0]),
        np.array([1e-6]),
    )
    assert a == b


def test_miscdata_inequality_on_differing_field() -> None:
    a = _misc()
    c = MiscData(
        np.array([[2.0]]),
        np.array([True, False]),
        np.array([100.0]),
        np.array([1e-6]),
    )
    assert a != c


def test_miscdata_equality_type_mismatch_raises() -> None:
    with pytest.raises(TypeError):
        _misc() == "not a micdata"  # type: ignore[comparison-overlap]
