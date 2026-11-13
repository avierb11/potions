import numpy as np
import pytest
from pandas import DataFrame, Series
from unittest.mock import MagicMock

from ..reactive_transport.kinetic_structures import (
    MineralAuxParams,
    MineralParameters,
    PARAMETERS_PER_MINERAL,
    RtParameters,
    TstParameters,
    ZoneDimensions,
)


def make_rt_forcing(s_w: float, temp: float) -> MagicMock:
    """A stand-in exposing only the fields the MineralParameters factor functions
    read: s_w and hydro_forc.temp. A real RtForcing is a Rust object, but
    these pure-Python methods never touch anything else on it."""
    mock = MagicMock()
    mock.s_w = float(s_w)
    mock.hydro_forc.temp = float(temp)
    return mock


def test_zone_dimensions_round_trip() -> None:
    dim = ZoneDimensions(max_water=20.0, passive_water_storage=10.0)
    restored = ZoneDimensions.from_array(dim.to_array())
    assert restored == dim
    assert restored.to_array().tolist() == [20.0, 10.0]


def test_zone_dimensions_max_water() -> None:
    dim = ZoneDimensions(max_water=20.0, passive_water_storage=10.0)
    assert dim.max_water == pytest.approx(20.0)


def test_zone_dimensions_equality_cross_type_raises() -> None:
    dim = ZoneDimensions(20.0, 10.0)
    with pytest.raises(TypeError):
        assert dim == [20.0, 10.0]


def test_zone_dimensions_inequality() -> None:
    a = ZoneDimensions(20.0, 10.0)
    b = ZoneDimensions(30.0, 10.0)
    assert a != b


def test_mineral_aux_params_round_trip() -> None:
    m = MineralAuxParams(0.3, 2.0, 1.1, 40.0)
    restored = MineralAuxParams.from_array(m.to_array())
    assert restored == m


def test_mineral_parameters_to_array_layout() -> None:
    # For n minerals, to_array should have 4 * n entries.
    mp = MineralParameters(
        sw_threshold=np.array([0.1, 0.2]),
        sw_exp=np.array([2.0, 2.0]),
        q_10=np.array([1.5, 1.5]),
        k_a=np.array([10.0, 20.0]),
    )
    arr = mp.to_array()
    assert arr.size == 4 * 2
    assert np.allclose(arr, [0.1, 2.0, 1.5, 10.0, 0.2, 2.0, 1.5, 20.0])


def test_mineral_parameters_from_array_bad_size_raises() -> None:
    with pytest.raises(ValueError):
        MineralParameters.from_array(np.ones(3))


def test_mineral_parameters_from_array_default_natural_scales() -> None:
    # Slot 3 is stored on a natural log base-10 scale when natural_scales=True.
    mp = MineralParameters.from_array(np.array([0.1, 2.0, 1.5, 1.0]), natural_scales=True)
    assert np.allclose(mp.sw_threshold, [0.1])
    assert np.allclose(mp.sw_exp, [2.0])
    assert np.allclose(mp.q_10, [1.5])
    assert np.allclose(mp.k_a, [10.0 ** 1.0])


def test_mineral_parameters_from_array_linear_scale() -> None:
    mp = MineralParameters.from_array(
        np.array([0.1, 2.0, 1.5, 10.0]), natural_scales=False
    )
    assert np.allclose(mp.k_a, [10.0])


def test_mineral_parameters_from_mineral_parameters() -> None:
    agg = MineralParameters.from_mineral_parameters(
        [
            MineralAuxParams(0.1, 2.0, 1.5, 10.0),
            MineralAuxParams(0.2, 3.0, 2.0, 20.0),
        ]
    )
    assert np.allclose(agg.sw_threshold, [0.1, 0.2])
    assert np.allclose(agg.q_10, [1.5, 2.0])
    assert np.allclose(agg.k_a, [10.0, 20.0])


def test_mineral_parameters_equality_and_inequality() -> None:
    a = MineralParameters(
        np.array([0.1]), np.array([2.0]), np.array([1.5]), np.array([10.0])
    )
    b = MineralParameters(
        np.array([0.1]), np.array([2.0]), np.array([1.5]), np.array([10.0])
    )
    c = MineralParameters(
        np.array([0.5]), np.array([2.0]), np.array([1.5]), np.array([10.0])
    )
    assert a == b
    assert a != c
    with pytest.raises(TypeError):
        assert a == "not a mineral parameter"


def test_soil_water_factor_above_threshold() -> None:
    mp = MineralParameters(
        sw_threshold=np.array([0.3]),
        sw_exp=np.array([2.0]),
        q_10=np.array([1.5]),
        k_a=np.array([10.0]),
    )
    forc = make_rt_forcing(s_w=0.5, temp=30.0)
    # s_w >= threshold: ((1-s_w)/(1-thr))^exp = (0.5/0.7)^2
    assert mp.soil_water_factor(forc)[0] == pytest.approx((0.5 / 0.7) ** 2)


def test_soil_water_factor_below_threshold() -> None:
    mp = MineralParameters(
        sw_threshold=np.array([0.3]),
        sw_exp=np.array([2.0]),
        q_10=np.array([1.5]),
        k_a=np.array([10.0]),
    )
    forc = make_rt_forcing(s_w=0.1, temp=20.0)
    # s_w < threshold: (s_w/thr)^exp = (0.1/0.3)^2
    assert mp.soil_water_factor(forc)[0] == pytest.approx((0.1 / 0.3) ** 2)


def test_temperature_factor() -> None:
    mp = MineralParameters(
        sw_threshold=np.array([0.3]),
        sw_exp=np.array([2.0]),
        q_10=np.array([1.5]),
        k_a=np.array([10.0]),
    )
    # At 20C the factor is exactly 1 (the reference temperature).
    assert mp.temperature_factor(make_rt_forcing(0.5, 20.0))[0] == pytest.approx(1.0)
    # At 30C the factor is q10^((30-20)/10) = 1.5.
    assert mp.temperature_factor(make_rt_forcing(0.5, 30.0))[0] == pytest.approx(1.5)


def test_combined_factor_is_product_of_components() -> None:
    mp = MineralParameters(
        sw_threshold=np.array([0.3]),
        sw_exp=np.array([2.0]),
        q_10=np.array([1.5]),
        k_a=np.array([10.0]),
    )
    forc = make_rt_forcing(s_w=0.5, temp=30.0)
    expected = (
        mp.soil_water_factor(forc)[0]
        * mp.temperature_factor(forc)[0]
    )
    assert mp.factor(forc)[0] == pytest.approx(expected)


def test_tst_rate_matches_hand_computation() -> None:
    # Single mineral, two species.
    # IAP = 10^(1*log10(A) + 2*log10(B)); dep = 10^(log10(B))
    stoich = DataFrame([[1.0, 2.0]], index=["m1"], columns=["A", "B"])
    dep = DataFrame([[np.nan, 1.0]], index=["m1"], columns=["A", "B"])
    k = Series([10.0], index=["m1"])
    params = TstParameters(stoich, dep, k)

    conc = np.array([2.0, 3.0])
    log_iap = 1.0 * np.log10(2.0) + 2.0 * np.log10(3.0)
    log_dep = 1.0 * np.log10(3.0)
    expected = 10.0**log_dep * (1.0 - 10.0**log_iap / 10.0)

    assert params.rate(conc)[0] == pytest.approx(expected)


def test_tst_rate_ignores_nan_dependency_terms() -> None:
    # With all dependency coefficients NaN, the dependency factor is 10^0 = 1.
    stoich = DataFrame([[1.0]], index=["m1"], columns=["A"])
    dep = DataFrame([[np.nan]], index=["m1"], columns=["A"])
    k = Series([1.0], index=["m1"])
    params = TstParameters(stoich, dep, k)

    conc = np.array([2.0])
    # rate = 1 * (1 - 10^log10(2.0) / 1.0) = 1 - 2.0 = -1.0
    assert params.rate(conc)[0] == pytest.approx(-1.0)


def test_tst_parameters_equality() -> None:
    stoich = DataFrame([[1.0, 2.0]], index=["m1"], columns=["A", "B"])
    dep = DataFrame([[1.0, 1.0]], index=["m1"], columns=["A", "B"])
    k = Series([5.0], index=["m1"])
    a = TstParameters(stoich.copy(), dep.copy(), k.copy())
    b = TstParameters(stoich.copy(), dep.copy(), k.copy())
    c = TstParameters(stoich * 2, dep.copy(), k.copy())
    assert a == b
    assert a != c
    with pytest.raises(TypeError):
        assert a == "not tst"


def test_rt_parameters_dimensions_and_params_combined() -> None:
    # Construct a single-mineral dimension set and check the structure.
    dim = ZoneDimensions(20.0, 10.0)
    mp = MineralParameters(
        np.array([0.1]), np.array([2.0]), np.array([1.5]), np.array([10.0])
    )
    rp = RtParameters(dimensions=dim, mineral_params=mp)
    assert rp.dimensions == dim
    assert rp.mineral_params == mp


def test_rt_parameters_equality() -> None:
    dim = ZoneDimensions(20.0, 10.0)
    mp = MineralParameters(
        np.array([0.1]), np.array([2.0]), np.array([1.5]), np.array([10.0])
    )
    a = RtParameters(dimensions=dim, mineral_params=mp)
    b = RtParameters(
        dimensions=ZoneDimensions(20.0, 10.0),
        mineral_params=MineralParameters(
            np.array([0.1]),
            np.array([2.0]),
            np.array([1.5]),
            np.array([10.0]),
        ),
    )
    assert a == b
    with pytest.raises(TypeError):
        assert a == "not rt params"


def test_parameters_per_mineral_constant() -> None:
    # The calibration budgeting (Model.from_array, num_mineral_parameters) all
    # hinge on this constant; pin it to detect silent drift.
    assert PARAMETERS_PER_MINERAL == 4
