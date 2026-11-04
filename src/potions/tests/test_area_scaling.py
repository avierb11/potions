"""Tests for area-weighted lateral transfer and outlet discharge.

These tests pin down the convention that area factors live on the *boundaries*
of the connectivity graph:

- forcing is distributed to each zone by its area fraction (existing behaviour);
- a lateral edge from donor ``d`` to receiver ``r`` carries the area ratio
  ``s_d / s_r`` (mass-conserving concentration of upslope runoff onto a
  smaller downstream footprint);
- the outlet (river) row carries the receiver scale ``s_r`` so the summed
  discharge is a catchment-mean depth (mm/d).

They also add a *system* water-balance check and a scale-invariance check that
together guard against the parameters silently absorbing zone size.
"""

import numpy as np
import pandas as pd
import pytest

from ..common_types import ForcingData
from ..core import SubsurfaceZoneB, SurfaceZone
from ..hydro_model import HydrologicalModel


class TwoColumnModel(HydrologicalModel):
    """Two columns, each with a surface and a ground zone.

    Within each layer the zones form a lateral chain: surface ``a`` drains
    into surface ``b`` (the river zone), and ground ``g1`` drains into ``g2``
    (the river zone). This gives a donor/receiver lateral edge in each layer.
    """

    structure = [
        [SurfaceZone(name="a"), SurfaceZone(name="b")],
        [SubsurfaceZoneB(name="g1"), SubsurfaceZoneB(name="g2")],
    ]


class AggregatingModel(HydrologicalModel):
    """Two surface columns aggregating into a single bottom zone."""

    structure = [
        [SurfaceZone(name="a"), SurfaceZone(name="b")],
        [SubsurfaceZoneB(name="g")],
    ]


# ===================== Per-zone scales ===================== #
def test_zone_scales_propagate_down_columns() -> None:
    m = TwoColumnModel(scales=[0.75, 0.25])
    # Column a carries 0.75 down its surface + ground zones; column b carries
    # 0.25. Order matches the order of evaluation.
    np.testing.assert_allclose(m.zone_scales, [0.75, 0.25, 0.75, 0.25])


def test_zone_scales_sum_to_one_per_layer() -> None:
    m = TwoColumnModel(scales=[0.6, 0.4])
    # The two surface zones sum to 1, and so do the two ground zones.
    assert m.zone_scales[[0, 1]].sum() == pytest.approx(1.0)
    assert m.zone_scales[[2, 3]].sum() == pytest.approx(1.0)


# ===================== Scaled lateral matrix ===================== #
def test_lateral_edge_carries_donor_receiver_ratio() -> None:
    m = TwoColumnModel(scales=[0.75, 0.25])
    lat = m.lat_mat_scaled
    # Edge a -> b is a lateral transfer in the surface layer. The receiver is
    # surface b (zone index 1), the donor surface a (zone index 0). The matrix
    # is stored receiver-by-donor, so lat[r=1, d=0] = s_a / s_b = 0.75 / 0.25.
    assert lat[1, 0] == pytest.approx(0.75 / 0.25)
    # Edge g1 -> g2 in the ground layer: lat[r=3, d=2] = 0.75 / 0.25.
    assert lat[3, 2] == pytest.approx(0.75 / 0.25)


def test_river_row_carries_receiver_scale() -> None:
    m = TwoColumnModel(scales=[0.75, 0.25])
    lat = m.lat_mat_scaled
    n = m.num_zones
    # River zones are the last zone of each layer: surface b (index 1) and
    # ground g2 (index 3). The river row weights each by its area fraction.
    assert lat[n, 1] == pytest.approx(0.25)
    assert lat[n, 3] == pytest.approx(0.25)
    # Non-river zones contribute nothing to the river row.
    assert lat[n, 0] == pytest.approx(0.0)
    assert lat[n, 2] == pytest.approx(0.0)


def test_unweighted_lat_mat_unchanged() -> None:
    # The plain adjacency matrix remains 0/1 (used by the RT coupling, which
    # mixes *concentrations* using water fluxes that are already area-weighted).
    m = TwoColumnModel(scales=[0.75, 0.25])
    lat = m.lat_mat
    assert lat[1, 0] == pytest.approx(1.0)
    assert lat[3, 2] == pytest.approx(1.0)
    assert lat[m.num_zones, 1] == pytest.approx(1.0)


def test_equal_scales_leave_lateral_edges_unity() -> None:
    # With equal areas, the donor/receiver ratio is 1, so lateral edges keep
    # their adjacency value and the outlet row is 0.5 per river zone.
    m = TwoColumnModel(scales=[0.5, 0.5])
    lat = m.lat_mat_scaled
    assert lat[1, 0] == pytest.approx(1.0)
    assert lat[3, 2] == pytest.approx(1.0)
    assert lat[m.num_zones, 1] == pytest.approx(0.5)
    assert lat[m.num_zones, 3] == pytest.approx(0.5)


# ===================== Scaled vertical matrix (aggregation) ===================== #
def test_vertical_edges_unity_when_columns_match() -> None:
    # Matched columns: each vertical edge joins zones of equal area, so the
    # scaled vertical matrix equals the plain adjacency matrix.
    m = TwoColumnModel(scales=[0.7, 0.3])
    np.testing.assert_allclose(m.vert_mat_scaled, m.vert_mat)


def test_vertical_edges_weighted_when_aggregating() -> None:
    # Two surface columns (0.7, 0.3) aggregate into one bottom zone that
    # represents the full catchment area (scale 1.0). Each vertical edge must
    # carry the donor/receiver ratio, or the donor's intensive flux would be
    # applied as if it covered the whole catchment.
    m = AggregatingModel(scales=[0.7, 0.3])
    np.testing.assert_allclose(m.zone_scales, [0.7, 0.3, 1.0])
    vert = m.vert_mat_scaled
    assert vert[2, 0] == pytest.approx(0.7)  # a -> g
    assert vert[2, 1] == pytest.approx(0.3)  # b -> g


def test_aggregation_inflow_matches_weighted_donors() -> None:
    # The bottom zone's incoming vertical flux must equal the area-weighted
    # sum of the donors' vertical fluxes.
    m = AggregatingModel(scales=[0.7, 0.3])
    res = m.run_hydro_model(_forcing(20), save_results=False)
    sim = res.simulation
    expected = 0.7 * sim["q_vert_a"] + 0.3 * sim["q_vert_b"]
    np.testing.assert_allclose(sim["q_in_g"].to_numpy(), expected.to_numpy())


# ===================== Outlet discharge ===================== #
def _forcing(steps: int) -> ForcingData:
    dates = pd.date_range("2020-01-01", periods=steps, freq="D")
    return ForcingData(
        precip=pd.Series(np.full(steps, 3.0), index=dates),
        temp=pd.Series(np.full(steps, 6.0), index=dates),
        pet=pd.Series(np.full(steps, 0.5), index=dates),
    )


def test_outlet_is_area_weighted_sum() -> None:
    # Two identical columns with unequal areas: the reported streamflow must be
    # the area-weighted sum of the raw q_lat fluxes at the river zones.
    m = TwoColumnModel(scales=[0.75, 0.25])
    res = m.run_hydro_model(_forcing(20), save_results=False)
    sim = res.simulation
    names = m.get_zone_names()
    river_names = [names[i] for i in m.get_river_zone_ids()]
    scales = m.zone_scales[m.get_river_zone_ids()]

    expected = sum(
        sim[f"q_lat_{n}"] * float(s) for n, s in zip(river_names, scales)
    )
    np.testing.assert_allclose(sim["streamflow_sim"].to_numpy(), expected.to_numpy())


def test_outlet_prop_columns_are_area_weighted() -> None:
    m = TwoColumnModel(scales=[0.75, 0.25])
    res = m.run_hydro_model(_forcing(20), save_results=False)
    sim = res.simulation
    # The area-weighted source proportions sum to 1 at each step.
    prop_cols = [c for c in sim.columns if c.startswith("prop_q_")]
    assert prop_cols
    np.testing.assert_allclose(
        sim[prop_cols].sum(axis=1).to_numpy(), np.ones(len(sim)), atol=1e-8
    )


# ===================== System water balance ===================== #
def test_system_water_balance_invariant_to_area_split() -> None:
    # Total water entering the domain is the same regardless of how the
    # catchment is partitioned into zones (same forcing, same total area). The
    # area-weighted outlet discharge plus vapour loss plus storage change must
    # therefore reconstruct the *same* total input for every split. Because the
    # precipitation term for a surface zone is applied inside the zone (not
    # surfaced as a flux column), we compare the reconstructed input across
    # splits rather than against an analytic precipitation total. Equality here
    # is the guardrail proving the boundary area factors conserve mass.
    steps = 40
    forc = _forcing(steps)

    def recovered_input(model: HydrologicalModel) -> float:
        res = model.run_hydro_model(forc, save_results=False)
        sim = res.simulation
        names = model.get_zone_names()
        zone_scales = model.zone_scales
        storage = np.array([sim[f"s_{n}"].to_numpy() for n in names])
        vap = np.array([sim[f"q_vap_{n}"].to_numpy() for n in names])
        sys_storage = (zone_scales[:, None] * storage).sum(axis=0)
        sys_vap = (zone_scales[:, None] * vap).sum(axis=0)
        return (
            sim["streamflow_sim"].to_numpy().sum()
            + sys_vap.sum()
            + sys_storage[-1]  # start from zero storage
        )

    # Matched-column structure: vary the split.
    matched = [
        recovered_input(TwoColumnModel(scales=s))
        for s in ([0.5, 0.5], [0.9, 0.1], [0.1, 0.9])
    ]
    np.testing.assert_allclose(matched, matched[0], atol=1e-6)

    # Aggregating structure (N zones -> 1 bottom): vary the split.
    aggregated = [
        recovered_input(AggregatingModel(scales=s))
        for s in ([0.5, 0.5], [0.7, 0.3], [0.3, 0.7])
    ]
    np.testing.assert_allclose(aggregated, aggregated[0], atol=1e-6)


# ===================== Scale invariance ===================== #
def test_donor_intensive_flux_is_split_independent() -> None:
    # The upstream (donor) zone's *intensive* lateral flux depends only on the
    # depth of water it receives and its own parameters -- not on the area
    # split between the two columns. This is the property that stops
    # calibration from absorbing zone size: the donor's per-unit-area
    # behaviour is scale-independent.
    #
    # Column a receives `scale_a * P_a` as an intensive depth (the model
    # applies the area fraction to the forcing). To hold that input fixed
    # across splits, we rescale column a's own forcing series so that the
    # model's `scale_a * P_a` product is the same in every split.
    steps = 30
    dates = pd.date_range("2020-01-01", periods=steps, freq="D")
    target_depth = 1.5  # the fixed intensive precipitation depth for column a

    donor_fluxes = []
    for scale_a in (0.5, 0.9, 0.1):
        scale_b = 1.0 - scale_a
        forc_a = ForcingData(
            precip=pd.Series(np.full(steps, target_depth / scale_a), index=dates),
            temp=pd.Series(np.full(steps, 6.0), index=dates),
            # PET is distributed by the same area factors as precipitation, so
            # rescale it too to keep column a's intensive PET depth fixed.
            pet=pd.Series(np.full(steps, 0.5 / scale_a), index=dates),
        )
        forc_b = ForcingData(
            precip=pd.Series(np.full(steps, target_depth / scale_b), index=dates),
            temp=pd.Series(np.full(steps, 6.0), index=dates),
            pet=pd.Series(np.full(steps, 0.5 / scale_b), index=dates),
        )
        m = TwoColumnModel(scales=[scale_a, scale_b])
        res = m.run_hydro_model([forc_a, forc_b], save_results=False)
        donor_fluxes.append(res.simulation["q_lat_a"].to_numpy())

    for flux in donor_fluxes[1:]:
        np.testing.assert_allclose(flux, donor_fluxes[0], atol=1e-6)


def test_run_with_check_water_balance() -> None:
    # The conservative model still passes its internal per-zone balance check
    # with the area-weighted lateral transfer enabled.
    m = TwoColumnModel(scales=[0.6, 0.4])
    m.run_hydro_model(_forcing(20), check_water_balance=True, save_results=False)
