import datetime as dt
from typing import Any

import numpy as np
import pytest
from pandas import DataFrame, Series

from ..calibration import build_rt_bounds, make_objective
from ..common_models import HbvModel
from ..common_types import (
    CalibrationOptions,
    ForcingData,
    RtCalibrationData,
    RtZoneConfiguration,
)
from ..objective_functions import (
    DEFAULT_RT_OBJECTIVE_SPEC,
    flow_weighted_kge,
    flow_weighted_log_nse,
    flow_weighted_nse,
    nse as plain_nse,
)
from ..reactive_transport.common_networks import get_simple_carbon_network

# ---------------------------------------------------------------------------
# ObjectiveSpec structure
# ---------------------------------------------------------------------------


def test_default_spec_has_nine_components() -> None:
    assert len(DEFAULT_RT_OBJECTIVE_SPEC.components) == 9


def test_default_spec_targets() -> None:
    targets = [c.target for c in DEFAULT_RT_OBJECTIVE_SPEC.components]
    assert targets.count("q") == 3
    assert targets.count("doc") == 3
    assert targets.count("doc_load") == 3


def test_default_spec_flow_weight_positions() -> None:
    flow_idxs = [
        i for i, c in enumerate(DEFAULT_RT_OBJECTIVE_SPEC.components) if c.use_flow_weight
    ]
    assert flow_idxs == [2, 5, 8]


def test_default_spec_all_unit_power() -> None:
    for c in DEFAULT_RT_OBJECTIVE_SPEC.components:
        assert c.power == 1.0


# ---------------------------------------------------------------------------
# Flow-weighted metrics
# ---------------------------------------------------------------------------


@pytest.fixture
def q_meas_fixture() -> Series:
    return Series(np.linspace(0.5, 10.0, 50), dtype="float64")


@pytest.fixture
def q_sim_fixture() -> Series:
    return Series(np.linspace(0.6, 11.0, 50), dtype="float64")


def test_flow_weighted_nse_perfect_fit(q_meas_fixture: Series) -> None:
    val = flow_weighted_nse(q_meas_fixture, q_meas_fixture.copy(), q_meas_fixture)
    assert val == pytest.approx(1.0)


def test_flow_weighted_nse_differs_from_unweighted(
    q_meas_fixture: Series, q_sim_fixture: Series
) -> None:
    fw = flow_weighted_nse(q_meas_fixture, q_sim_fixture, q_meas_fixture)
    unweighted = plain_nse(q_meas_fixture, q_sim_fixture)
    assert fw != unweighted


def test_flow_weighted_log_nse_perfect_fit(q_meas_fixture: Series) -> None:
    val = flow_weighted_log_nse(q_meas_fixture, q_meas_fixture.copy(), q_meas_fixture)
    assert val == pytest.approx(1.0)


def test_flow_weighted_kge_perfect_fit(q_meas_fixture: Series) -> None:
    # KGE with uniform weights approximates the standard KGE; a perfect fit
    # scores very high (the flow-weighted form has a small (n-1)/n std bias,
    # so we assert a loose upper bound rather than exact 1.0).
    uniform_w = Series(1.0, index=q_meas_fixture.index)
    val = flow_weighted_kge(q_meas_fixture, q_meas_fixture.copy(), uniform_w)
    assert val > 0.95


# ---------------------------------------------------------------------------
# build_rt_bounds
# ---------------------------------------------------------------------------


def zone_configs() -> dict[str, RtZoneConfiguration]:
    cfg = RtZoneConfiguration(do_reactions=True, do_speciation=True)
    return {
        "surface": cfg,
        "shallow": cfg,
        "deep": RtZoneConfiguration(do_reactions=False, do_speciation=True),
    }


def make_network() -> Any:
    return get_simple_carbon_network()


def test_build_bounds_hbv_has_hydro_params() -> None:
    network = make_network()
    bounds = build_rt_bounds(HbvModel, network, zone_configs())
    for name in HbvModel.default_parameter_ranges():
        assert name in bounds, f"Missing hydro param {name!r} in bounds"


def test_build_bounds_hbv_has_zone_dims() -> None:
    network = make_network()
    bounds = build_rt_bounds(HbvModel, network, zone_configs())
    assert "surface.max_water" in bounds
    assert "surface.passive_water_storage" in bounds
    assert "shallow.max_water" in bounds
    assert "deep.passive_water_storage" in bounds


def test_build_bounds_hbv_reactive_zone_mineral_params() -> None:
    network = make_network()
    num_min = network.num_minerals
    bounds = build_rt_bounds(HbvModel, network, zone_configs())
    for mid in range(num_min):
        assert f"surface.sw_threshold[{mid}]" in bounds
        assert f"surface.k_a[{mid}]" in bounds
    for mid in range(num_min):
        assert f"deep.sw_threshold[{mid}]" not in bounds


def test_build_bounds_key_order_starts_with_hydro() -> None:
    network = make_network()
    bounds = build_rt_bounds(HbvModel, network, zone_configs())
    hydro_names = list(HbvModel.default_parameter_ranges())
    all_keys = list(bounds.keys())
    for i, hname in enumerate(hydro_names):
        assert all_keys[i] == hname, f"Key order mismatch: {all_keys[i]!r} != {hname!r}"


def test_build_bounds_no_river_zone() -> None:
    network = make_network()
    bounds = build_rt_bounds(HbvModel, network, zone_configs())
    assert all("river." not in k for k in bounds)


def test_build_bounds_mineral_bounds_values() -> None:
    network = make_network()
    bounds = build_rt_bounds(HbvModel, network, zone_configs())
    # Surface zone: free soil-moisture / temperature / rate bounds.
    assert bounds["surface.sw_threshold[0]"] == (0.0, 1.0)
    assert bounds["surface.sw_exp[0]"] == (0.5, 2.5)
    assert bounds["surface.q_10[0]"] == (1.0, 5.0)
    assert bounds["surface.k_a[0]"] == (-15.0, -5.0)
    # Subsurface zone: dependencies fixed (degenerate bounds), only k_a varies.
    assert bounds["shallow.sw_threshold[0]"] == (0.0, 0.0)
    assert bounds["shallow.sw_exp[0]"] == (0.0, 0.0)
    assert bounds["shallow.q_10[0]"] == (1.0, 1.0)
    assert bounds["shallow.k_a[0]"] == (-15.0, -5.0)


def test_build_bounds_custom_hydro_bounds() -> None:
    network = make_network()
    custom = dict(HbvModel.default_parameter_ranges())
    custom["surface.k0"] = (0.01, 0.5)
    bounds = build_rt_bounds(HbvModel, network, zone_configs(), hydro_bounds=custom)
    assert bounds["surface.k0"] == (0.01, 0.5)


# ---------------------------------------------------------------------------
# CalibrationOptions defaults
# ---------------------------------------------------------------------------


def test_calibration_options_defaults() -> None:
    opts = CalibrationOptions()
    assert opts.maxiter == 10
    assert opts.polish is False
    assert opts.strategy == "best1bin"
    assert opts.updating == "deferred"
    assert opts.workers == -1


# ---------------------------------------------------------------------------
# make_objective — smoke test
# ---------------------------------------------------------------------------


def test_make_objective_returns_callable() -> None:
    t_idx = [dt.datetime(2020, 1, 1) + dt.timedelta(days=i) for i in range(4)]
    fd = ForcingData(
        precip=Series([1.0, 1.0, 1.0, 1.0], index=t_idx),
        temp=Series([5.0, 5.0, 5.0, 5.0], index=t_idx),
        pet=Series([0.5, 0.5, 0.5, 0.5], index=t_idx),
    )
    q_meas = Series([1.0, 2.0, 3.0, 4.0], index=t_idx)
    conc_df = DataFrame({"DOC": [1.0, 2.0, 3.0, 4.0]}, index=t_idx)

    data = RtCalibrationData(
        model_type=HbvModel,
        network=make_network(),
        zone_configs=zone_configs(),
        forc=fd,
        precip_conc=np.zeros((1, 4, 1)),
        q_meas=q_meas,
        conc_df=conc_df,
        target_species="DOC",
    )
    obj_fn = make_objective(data, DEFAULT_RT_OBJECTIVE_SPEC)
    assert callable(obj_fn)
