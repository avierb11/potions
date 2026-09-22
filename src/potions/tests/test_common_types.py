import numpy as np
import pytest
from pandas import Series

from ..common_types import (
    ForcingData,
    HydroModelStep,
    LapseRateParameters,
    RtZoneConfiguration,
)


@pytest.fixture
def lapse_rates() -> LapseRateParameters:
    return LapseRateParameters(temp_factor=-0.5, precip_factor=0.3)


def test_scale_temperature(lapse_rates: LapseRateParameters) -> None:
    # temp' = temp + temp_factor * (elev - gauge_elevation)
    out = lapse_rates.scale_temperature(
        1000.0, 1100.0, Series([10.0], dtype="float64")
    )
    # -0.5 * (1100 - 1000) = -50  =>  10 - 50 = -40
    assert out.iloc[0] == pytest.approx(-40.0)


def test_scale_temperature_lower_elevation(lapse_rates: LapseRateParameters) -> None:
    # A warmer gauge at lower elevation should yield higher (less negative) temps.
    out = lapse_rates.scale_temperature(
        1100.0, 1000.0, Series([10.0], dtype="float64")
    )
    # -0.5 * (1000 - 1100) = +50 => 10 + 50 = 60
    assert out.iloc[0] == pytest.approx(60.0)


def test_scale_precipitation(lapse_rates: LapseRateParameters) -> None:
    out = lapse_rates.scale_precipitation(
        1000.0, 1100.0, Series([100.0], dtype="float64")
    )
    # 0.3 * 100 = 30 => 100 + 30 = 130
    assert out.iloc[0] == pytest.approx(130.0)


def test_scale_forcing_data_passes_pet_through(
    lapse_rates: LapseRateParameters,
) -> None:
    fd = ForcingData(
        precip=Series([100.0], dtype="float64"),
        temp=Series([10.0], dtype="float64"),
        pet=Series([3.0], dtype="float64"),
    )
    scaled = lapse_rates.scale_forcing_data(1000.0, 1100.0, fd)
    assert scaled.precip.iloc[0] == pytest.approx(130.0)  # 100 + 0.3*100
    assert scaled.temp.iloc[0] == pytest.approx(-40.0)  # 10 - 0.5*100
    assert scaled.pet.iloc[0] == pytest.approx(3.0)  # unchanged


def test_default_parameter_range() -> None:
    rng = LapseRateParameters.default_parameter_range()
    assert set(rng.keys()) == {"temp_factor", "precip_factor"}
    # Temperature lapse rates should be negative (cooler higher up).
    assert rng["temp_factor"][0] < 0 <= rng["temp_factor"][1]


def test_from_dict_valid() -> None:
    lr = LapseRateParameters.from_dict(
        {"temp_factor": -0.5, "precip_factor": 0.3}
    )
    assert lr.temp_factor == -0.5
    assert lr.precip_factor == 0.3


def test_from_dict_invalid_keys_raises() -> None:
    with pytest.raises(TypeError):
        LapseRateParameters.from_dict({"temp_factor": -0.5, "bogus": 0.3})


def test_forcing_data_is_frozen() -> None:
    fd = ForcingData(
        precip=Series([1.0], dtype="float64"),
        temp=Series([2.0], dtype="float64"),
        pet=Series([3.0], dtype="float64"),
    )
    with pytest.raises(Exception):
        fd.precip = Series([0.0], dtype="float64")  # type: ignore[misc]


def test_hydro_model_step_construction() -> None:
    step = HydroModelStep(
        state=np.array([1.0, 2.0]),
        forc_flux=np.array([0.1, 0.2]),
        vap_flux=np.array([0.0, 0.0]),
        lat_flux=np.array([0.0, 0.0]),
        vert_flux=np.array([0.5, 0.5]),
        q_in=np.array([0.3, 0.3]),
        lat_flux_ext=np.array([0.0, 0.0]),
        vert_flux_ext=np.array([0.0, 0.0]),
    )
    assert list(step.state) == [1.0, 2.0]


def test_rt_zone_configuration() -> None:
    cfg = RtZoneConfiguration(do_reactions=True, do_speciation=False)
    assert cfg.do_reactions is True
    assert cfg.do_speciation is False
