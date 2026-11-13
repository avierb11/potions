import numpy as np
import pytest
from networkx import DiGraph
from types import SimpleNamespace

from ..core import SnowZone, SubsurfaceZoneB, SurfaceZone
from ..hydro_model import HydrologicalModel


class ColumnModel(HydrologicalModel):
    """A single catchment column: snow -> surface -> ground."""

    structure = [
        [SnowZone(name="snow")],
        [SurfaceZone(name="surface")],
        [SubsurfaceZoneB(name="ground")],
    ]


class TwoColumnModel(HydrologicalModel):
    """Two side-by-side catchments, each with a surface and a ground zone."""

    structure = [
        [SurfaceZone(name="a"), SurfaceZone(name="b")],
        [SubsurfaceZoneB(name="g1"), SubsurfaceZoneB(name="g2")],
    ]


@pytest.fixture(scope="module")
def column() -> ColumnModel:
    return ColumnModel(scales=[1.0])


@pytest.fixture(scope="module")
def two_col() -> TwoColumnModel:
    return TwoColumnModel()


def test_zone_names_and_indices(column: ColumnModel) -> None:
    assert column.zone_names == ["snow", "surface", "ground"]
    assert column.zone_indices == {"snow": 0, "surface": 1, "ground": 2}
    assert column.num_zones == 3


def test_surface_zone_ids(column: ColumnModel) -> None:
    assert column.surface_zone_ids == [0]
    assert column.num_surface_zones == 1


def test_num_hydro_parameters(column: ColumnModel) -> None:
    # 2 snow + 5 surface + 2 ground (bottom zone) = 9 default parameters.
    assert column.get_num_hydro_parameters() == 9


def test_default_init_state(column: ColumnModel) -> None:
    state = column.default_hydro_init_state()
    # One value per zone.
    assert state.shape == (3,)
    # Snow storage starts at 0.
    assert state[column.zone_indices["snow"]] == pytest.approx(0.0)


def test_to_dict_has_all_zone_params(column: ColumnModel) -> None:
    d = column.to_dict()
    for name in column.zone_names:
        # Each zone name appears as a prefix in at least one key.
        assert any(k.startswith(f"{name}.") for k in d), name


def test_getitem_zone(column: ColumnModel) -> None:
    zone = column["surface"]
    assert zone.name == "surface"


def test_getitem_missing_zone_raises(column: ColumnModel) -> None:
    with pytest.raises(ValueError):
        column["does_not_exist"]


def test_graph_nodes_and_edges(column: ColumnModel) -> None:
    g = column.graph
    assert isinstance(g, DiGraph)
    # One node per (layer, column); a 3-layer, 1-column model has 3 nodes.
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2


def test_construct_hydrologic_graph_returns_digraph(column: ColumnModel) -> None:
    g = column.construct_hydrologic_graph()
    assert isinstance(g, DiGraph)
    # Edges flow downward: (0,0) -> (1,0) -> (2,0)
    assert ((0, 0), (1, 0)) in g.edges
    assert ((1, 0), (2, 0)) in g.edges


def test_size_mat(two_col: TwoColumnModel) -> None:
    # One row per zone, one column per surface zone. Zone i gets size 1 from the
    # surface zone it sits beneath (identity over the two catchments, propagated
    # downward).
    size = two_col.get_size_mat()
    assert size.shape == (4, 2)
    assert np.allclose(size, np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))


def test_lat_mat_connectivity(two_col: TwoColumnModel) -> None:
    # One row per zone (4) plus an appended river row -> 5 rows.
    lat = two_col.lat_mat
    assert lat.shape == (5, 4)
    # Surface zone a (row 1) drains laterally into the river row (last row).
    assert lat[1, 0] == pytest.approx(1.0)
    # Surface zone b (row 2) drains laterally into the river row as well.
    assert lat[4, 1] == pytest.approx(1.0)


def test_vert_mat_connectivity(two_col: TwoColumnModel) -> None:
    # One row per zone, no appended river row.
    vert = two_col.vert_mat
    assert vert.shape == (4, 4)
    # Surface a -> ground g1, surface b -> ground g2.
    assert vert[2, 0] == pytest.approx(1.0)
    assert vert[3, 1] == pytest.approx(1.0)
    assert vert[2, 1] == pytest.approx(0.0)
    assert vert[3, 0] == pytest.approx(0.0)


def test_precip_mat_scales(two_col: TwoColumnModel) -> None:
    # Default equal scales of 1/2 per surface zone.
    precip = two_col.precip_mat
    assert precip.shape == (4, 2)
    # Snow rows split evenly across the two catchments.
    assert np.allclose(precip[0, :], [0.5, 0.0])
    assert np.allclose(precip[1, :], [0.0, 0.5])


def test_pet_mat_matches_precip(two_col: TwoColumnModel) -> None:
    # PET is distributed over surface zones the same way as precipitation.
    np.testing.assert_allclose(two_col.pet_mat, two_col.precip_mat)


def test_temp_mat_is_identity_over_surface(two_col: TwoColumnModel) -> None:
    # Each column's temperature is tied to its own surface zone.
    temp = two_col.temp_mat
    assert temp.shape == (4, 2)
    assert np.allclose(temp[0, :], [1.0, 0.0])
    assert np.allclose(temp[1, :], [0.0, 1.0])


def test_get_forc_mat_relative_sums_to_one(two_col: TwoColumnModel) -> None:
    # relative=True normalizes each ROW to sum to 1 (area-weighted average), so
    # with equal scales the rows become pure unit vectors over the source.
    rel = two_col.get_forc_mat([0.5, 0.5], relative=True)
    assert rel.shape == (4, 2)
    assert np.allclose(rel.sum(axis=1), [1.0, 1.0, 1.0, 1.0])


def test_river_zone_ids(two_col: TwoColumnModel) -> None:
    # The bottom (ground) zones feed the river: the row after the last layer.
    river_ids = two_col.get_river_zone_ids()
    assert sorted(river_ids) == [1, 3]


def test_structure_diagram_layout(two_col: TwoColumnModel) -> None:
    d = two_col.structure_diagram()
    lines = d.split("\n")
    # One border per band plus a closing border.
    assert lines[0][0] == "+"
    assert lines[-1][0] == "+"
    assert all(len(r) == len(lines[0]) for r in lines)
    # Box borders use '-' for zones and '~' for the river channel.
    assert "-" in lines[0] and "~" in lines[0]
    # Each zone name and its parameters are shown.
    assert "|" in lines[1]
    for name in ("a", "b", "g1", "g2"):
        assert name in d, name
    assert "fc =" in d
    assert "k =" in d
    # The river column is labeled.
    assert "River" in d
    # Every content row is a full-width row of boxed cells.
    for r in (lines[1], lines[-2]):
        assert r.startswith("|") and r.endswith("|")


def test_structure_diagram_single_column(column: ColumnModel) -> None:
    d = column.structure_diagram()
    lines = d.split("\n")
    # One border line per band plus a closing border.
    border_rows = [r for r in lines if r.startswith("+")]
    assert len(border_rows) == len(column) + 1
    # All rows share the same (constant) width.
    assert all(len(r) == len(lines[0]) for r in lines)
    for name in column.zone_names:
        assert name in d, name
    assert "River" in d


def test_zones_override() -> None:
    # A custom-parametrized zone replaces the default when key matches.
    custom = SurfaceZone(name="surface", fc=80.0)
    m = ColumnModel(scales=[1.0])
    # Re-access via the constructed dict by overriding in a fresh model.
    class OverrideModel(HydrologicalModel):
        structure = [
            [SurfaceZone(name="surface")],
            [SubsurfaceZoneB(name="deep")],  # bottom layer must be bottom zones
        ]

    om = OverrideModel(zones={"surface": custom})
    # The provided zone is stored by reference, so the custom parameters are used.
    assert om.hydro_zones["surface"] is custom


def test_len_is_num_layers(column: ColumnModel) -> None:
    # Number of layers (rows in structure).
    assert len(column) == 3


def test_simple_calibration_freezes_base_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A partial range searches only the free parameters; the objective
    receives the pinned values through ``base_params`` and the best
    point is expanded back into the full vector.
    """
    all_names: tuple[str, ...] = tuple(
        ColumnModel.default_parameter_ranges()
    )
    free_name: str = all_names[0]
    base_params: dict[str, float] = {
        name: 0.1 + 0.01 * index
        for index, name in enumerate(all_names)
    }
    free_high: float = 0.42
    seen: list[np.ndarray] = []
    de_records: list[dict[str, object]] = []
    sentinel: object = object()

    class StubModel:
        def run_hydro_model(self, **_kwargs: object) -> object:
            return sentinel

    def fake_from_array(x: np.ndarray) -> StubModel:
        seen.append(np.array(x))
        return StubModel()

    def fake_de(func: object, bounds: object, **kwargs: object) -> object:
        de_records.append({"func": func, "bounds": bounds, **kwargs})
        return SimpleNamespace(x=np.array([free_high]), fun=-0.9, nfev=11)

    monkeypatch.setattr(
        ColumnModel, "hydro_from_array", staticmethod(fake_from_array)
    )
    monkeypatch.setattr(
        "potions.hydro_model.opt.differential_evolution", fake_de
    )
    callback_calls: list[tuple[np.ndarray, float]] = []

    params, results, opt_res = ColumnModel.simple_calibration(
        forc=object(),  # type: ignore[arg-type]
        meas_streamflow=object(),
        metric="kge",
        maxiter=4,
        popsize=17,
        param_ranges={free_name: (0.0, 1.0)},
        base_params=base_params,
        callback=lambda x, convergence: callback_calls.append(
            (np.array(x), convergence)
        ),
    )

    assert params == {free_name: free_high}
    assert results is sentinel
    assert opt_res.nfev == 11
    assert len(seen) == 1
    expected: np.ndarray = np.array(
        [free_high if name == free_name else base_params[name] for name in all_names],
        dtype=np.float64,
    )
    assert np.array_equal(seen[0], expected)
    assert len(de_records) == 1
    record = de_records[0]
    assert record["bounds"] == [(0.0, 1.0)]
    assert record["maxiter"] == 4
    assert record["popsize"] == 17
    assert record["callback"] is not None
    args = record["args"]
    assert isinstance(args, tuple)
    assert args[0] is ColumnModel
    assert args[5] == base_params
    assert args[6] == (free_name,)
    assert args[7] == all_names


def test_simple_calibration_rejects_partial_range_without_base() -> None:
    """A partial range must name the pinned values, or it is refused."""
    all_names: tuple[str, ...] = tuple(
        ColumnModel.default_parameter_ranges()
    )
    with pytest.raises(ValueError) as excinfo:
        ColumnModel.simple_calibration(
            forc=object(),  # type: ignore[arg-type]
            meas_streamflow=object(),
            metric="kge",
            param_ranges={all_names[0]: (0.0, 1.0)},
        )
    assert "base_params" in str(excinfo.value)


def test_simple_calibration_rejects_unknown_names() -> None:
    """The free and fixed names must be the model's hydrology parameters."""
    all_names: tuple[str, ...] = tuple(
        ColumnModel.default_parameter_ranges()
    )
    with pytest.raises(ValueError) as excinfo:
        ColumnModel.simple_calibration(
            forc=object(),  # type: ignore[arg-type]
            meas_streamflow=object(),
            metric="kge",
            param_ranges={"bogus": (0.0, 1.0)},
            base_params={name: 0.2 for name in all_names},
        )
    assert "unknown=['bogus']" in str(excinfo.value)
