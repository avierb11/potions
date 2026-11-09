import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from ..common_types import (
    ForcingData,
    HydroModelResults,
    HydroModelStep,
    HydroZoneResults,
    LapseRateParameters,
    RtModelResults,
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


# ==== HydroModelResults named views ==== #

# Deliberately not East River names, and underscored: the accessors must
# treat a zone name as an opaque label, never split it on underscores.
HYDRO_ZONES: list[str] = ["wet_hs", "wet_rp", "dry_rp"]


def _hydro_sim(index: pd.DatetimeIndex) -> DataFrame:
    """A flat hydrology frame with every per-zone column, each distinct."""
    prefixes: list[str] = [
        "s_",
        "q_forc_",
        "q_vap_",
        "q_lat_",
        "q_vert_",
        "q_in_",
        "q_lat_ext_",
        "q_vert_ext_",
    ]
    data: dict[str, np.ndarray] = {}
    for prefix in prefixes:
        for i, zone in enumerate(HYDRO_ZONES):
            data[f"{prefix}{zone}"] = np.full(len(index), 10 * i + len(prefix), float)
    data["streamflow_sim"] = np.arange(len(index), dtype=float)
    data["meas_streamflow_mmd"] = np.arange(len(index), dtype=float) + 100.0
    data["prop_q_wet_rp"] = np.full(len(index), 0.3, float)
    data["prop_q_dry_rp"] = np.full(len(index), 0.7, float)
    return DataFrame(data, index=index)


def _hydro_forcing(index: pd.DatetimeIndex) -> ForcingData:
    return ForcingData(
        precip=pd.Series(1.0, index=index),
        temp=pd.Series(5.0, index=index),
        pet=pd.Series(0.5, index=index),
    )


def _hydro_results(zone_names: list[str] | None = None) -> HydroModelResults:
    index = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i) for i in range(4)]
    )
    if zone_names is None:
        return HydroModelResults(
            forcing=_hydro_forcing(index),
            simulation=_hydro_sim(index),
            objective_functions=Series(dtype=float),
        )
    return HydroModelResults(
        forcing=_hydro_forcing(index),
        simulation=_hydro_sim(index),
        objective_functions=Series(dtype=float),
        zone_names=zone_names,
    )


@pytest.fixture
def hydro_results() -> HydroModelResults:
    return _hydro_results(list(HYDRO_ZONES))


def test_hydro_quantity_frame_relabels_and_orders(
    hydro_results: HydroModelResults,
) -> None:
    frame = hydro_results.lateral
    assert list(frame.columns) == HYDRO_ZONES
    for i, zone in enumerate(HYDRO_ZONES):
        np.testing.assert_array_equal(
            frame[zone].to_numpy(),
            hydro_results.simulation[f"q_lat_{zone}"].to_numpy(),
        )


@pytest.mark.parametrize(
    ("quantity", "prefix"),
    [
        ("storage", "s_"),
        ("precip_flux", "q_forc_"),
        ("evaporation", "q_vap_"),
        ("lateral", "q_lat_"),
        ("vertical", "q_vert_"),
        ("inflow", "q_in_"),
        ("external_lateral", "q_lat_ext_"),
        ("external_vertical", "q_vert_ext_"),
    ],
)
def test_hydro_all_quantity_frames(
    hydro_results: HydroModelResults, quantity: str, prefix: str
) -> None:
    frame: DataFrame = getattr(hydro_results, quantity)
    assert list(frame.columns) == HYDRO_ZONES
    for zone in HYDRO_ZONES:
        np.testing.assert_array_equal(
            frame[zone].to_numpy(),
            hydro_results.simulation[f"{prefix}{zone}"].to_numpy(),
        )


def test_hydro_streamflow_accessors(hydro_results: HydroModelResults) -> None:
    pd.testing.assert_series_equal(
        hydro_results.simulated_streamflow,
        hydro_results.simulation["streamflow_sim"],
    )
    measured = hydro_results.measured_streamflow
    assert measured is not None
    pd.testing.assert_series_equal(measured, hydro_results.simulation["meas_streamflow_mmd"])


def test_hydro_measured_streamflow_none_when_absent(
    hydro_results: HydroModelResults,
) -> None:
    hydro_results.simulation = hydro_results.simulation.drop(
        columns=["meas_streamflow_mmd"]
    )
    assert hydro_results.measured_streamflow is None


def test_hydro_river_zone_fractions_relabels(
    hydro_results: HydroModelResults,
) -> None:
    frame = hydro_results.river_zone_fractions
    assert list(frame.columns) == ["wet_rp", "dry_rp"]
    np.testing.assert_array_equal(
        frame["wet_rp"].to_numpy(),
        hydro_results.simulation["prop_q_wet_rp"].to_numpy(),
    )


def test_hydro_zone_view_matches_frame_columns(
    hydro_results: HydroModelResults,
) -> None:
    view: HydroZoneResults = hydro_results.zone("wet_rp")
    assert view.zone == "wet_rp"
    for quantity, prefix in [
        ("storage", "s_"),
        ("lateral", "q_lat_"),
        ("external_vertical", "q_vert_ext_"),
    ]:
        series: Series = getattr(view, quantity)
        np.testing.assert_array_equal(
            series.to_numpy(),
            hydro_results.simulation[f"{prefix}wet_rp"].to_numpy(),
        )


def test_hydro_zones_dict_covers_all_zones(
    hydro_results: HydroModelResults,
) -> None:
    assert list(hydro_results.zones) == HYDRO_ZONES
    for zone in HYDRO_ZONES:
        assert hydro_results.zones[zone].zone == zone


def test_hydro_to_dataframe_is_stored_frame(
    hydro_results: HydroModelResults,
) -> None:
    assert hydro_results.to_dataframe() is hydro_results.simulation


def test_hydro_zone_names_parsed_when_unset() -> None:
    # The frame is self-describing through its `s_` storage columns, so a
    # results object built without zone_names still partitions correctly.
    results = _hydro_results()
    assert results.zone_names == []
    assert list(results.lateral.columns) == HYDRO_ZONES


def test_hydro_zone_names_unset_and_no_s_columns_raises() -> None:
    index = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i) for i in range(2)]
    )
    frame = DataFrame({"other_col": np.ones(len(index))}, index=index)
    bare = HydroModelResults(
        forcing=_hydro_forcing(index),
        simulation=frame,
        objective_functions=Series(dtype=float),
    )
    with pytest.raises(ValueError, match="zone names"):
        _ = bare.lateral


def test_hydro_recorded_zone_missing_column_raises(
    hydro_results: HydroModelResults,
) -> None:
    bad = HydroModelResults(
        forcing=hydro_results.forcing,
        simulation=hydro_results.simulation,
        objective_functions=Series(dtype=float),
        zone_names=HYDRO_ZONES + ["ghost_zone"],
    )
    with pytest.raises(ValueError, match="ghost_zone"):
        _ = bad.lateral


def test_hydro_unknown_zone_raises(hydro_results: HydroModelResults) -> None:
    with pytest.raises(ValueError, match="not a zone of these results"):
        hydro_results.zone("no_such_zone")


# ==== RtModelResults named views ==== #

# Every species appears in every zone (an RT model declares an RT zone
# for every model zone); DOC and DOC2 prove the per-species accessor
# builds exact columns rather than prefix-matching.
RT_ZONES: list[str] = ["wet_hs", "wet_rp", "dry_rp"]
RT_SPECIES: list[str] = ["DOC", "DOC2", "Ca++"]
RT_MINERALS: list[str] = ["Calcite(s)"]
RT_REACTIVE: list[str] = ["wet_rp", "dry_rp"]  # wet_hs does not react


def _rt_sim(index: pd.DatetimeIndex) -> DataFrame:
    """A flat reactive-transport frame with every kind of column."""
    data: dict[str, np.ndarray] = {}
    for zone in RT_ZONES:
        for species in RT_SPECIES:
            data[f"{species}_{zone}"] = np.full(len(index), 1.0, float)
        if zone in RT_REACTIVE:
            for mineral in RT_MINERALS:
                data[f"{mineral}_rate_{zone}"] = np.full(len(index), 2.0, float)
    for species in RT_SPECIES:
        data[f"{species}_riv"] = np.full(len(index), 3.0, float)
    data["DIC_riv"] = np.full(len(index), 4.0, float)  # a derived measurement
    return DataFrame(data, index=index)


def _rt_results(zone_names: list[str] | None = None) -> RtModelResults:
    index = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i) for i in range(3)]
    )
    if zone_names is None:
        return RtModelResults(
            simulation=_rt_sim(index),
            objective_functions=None,
        )
    return RtModelResults(
        simulation=_rt_sim(index),
        objective_functions=None,
        zone_names=zone_names,
    )


def test_rt_river_frame_is_all_riv_columns() -> None:
    frame = _rt_results(zone_names=list(RT_ZONES)).river
    assert list(frame.columns) == ["DOC_riv", "DOC2_riv", "Ca++_riv", "DIC_riv"]


def test_rt_mineral_rates_frame_is_all_rate_columns() -> None:
    frame = _rt_results(zone_names=list(RT_ZONES)).mineral_rates
    assert list(frame.columns) == [
        f"{m}_rate_{z}" for z in RT_REACTIVE for m in RT_MINERALS
    ]


def test_rt_species_frame_relabels_across_zones() -> None:
    frame = _rt_results(zone_names=list(RT_ZONES)).species("DOC")
    assert list(frame.columns) == RT_ZONES
    np.testing.assert_array_equal(
        frame["dry_rp"].to_numpy(),
        _rt_results(list(RT_ZONES)).simulation["DOC_dry_rp"].to_numpy(),
    )


def test_rt_zone_view_species_and_mineral_rates() -> None:
    results = _rt_results(zone_names=list(RT_ZONES))
    view = results.zone("wet_rp")
    assert view.zone == "wet_rp"
    assert list(view.species.columns) == RT_SPECIES
    assert view.mineral_rates is not None
    assert list(view.mineral_rates.columns) == RT_MINERALS
    # The non-reactive zone has no rate columns: the view says so; it
    # does not fabricate zeros.
    assert results.zone("wet_hs").mineral_rates is None


def test_rt_zones_dict_covers_recorded_zones() -> None:
    results = _rt_results(zone_names=list(RT_ZONES))
    assert list(results.zones) == RT_ZONES


def test_rt_to_dataframe_is_stored_frame() -> None:
    results = _rt_results(zone_names=list(RT_ZONES))
    assert results.to_dataframe() is results.simulation


def test_rt_zone_accessors_require_zone_names() -> None:
    results = _rt_results()  # default: zone_names == []
    with pytest.raises(ValueError, match="zone names"):
        _ = results.species("DOC")
    with pytest.raises(ValueError, match="zone names"):
        _ = results.zone("wet_rp")
    with pytest.raises(ValueError, match="zone names"):
        _ = results.zones


def test_rt_unknown_zone_raises() -> None:
    with pytest.raises(ValueError, match="not a zone of these results"):
        _rt_results(zone_names=list(RT_ZONES)).zone("no_such_zone")


def test_rt_recorded_zone_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="DOC_ghost"):
        _rt_results(zone_names=RT_ZONES + ["ghost"]).species("DOC")


def test_rt_setstate_defaults_missing_zone_names() -> None:
    # A pickle written before zone_names existed: its __dict__ lacks
    # the key. Replaying the unpickle contract on such a state must not
    # raise, and must leave the accessor in the "not recorded" state.
    results = _rt_results(zone_names=list(RT_ZONES))
    state = dict(results.__dict__)
    del state["zone_names"]
    loaded = RtModelResults.__new__(RtModelResults)
    RtModelResults.__setstate__(loaded, state)
    assert loaded.zone_names == []
    pd.testing.assert_frame_equal(loaded.simulation, results.simulation)
