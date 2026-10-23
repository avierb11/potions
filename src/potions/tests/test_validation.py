"""Tests for the pre-run validation layer.

Covers:
- Bottom-zone placement rules (R1/R2) and other structural rules, raised
  (with descriptive messages) at model construction time.
- The bottom-zone registry helpers (``register_bottom_zone`` /
  ``is_bottom_zone``).
- Runtime input validation (forcing series, measured streamflow, initial
  state), raised before the first model step is taken.
- The error-class hierarchy.
"""

import numpy as np
import pandas as pd
import pytest

from ..common_types import ForcingData
from ..core import (
    GroundZone,
    GroundZoneB,
    MineralParameters,
    RtParameters,
    RtZone,
    SnowZone,
    SurfaceZone,
    ZoneDimensions,
)
from ..hydro_model import HydrologicalModel
from ..model_components import (
    BOTTOM_ZONE_CLASSES,
    is_bottom_zone,
    register_bottom_zone,
)
from ..reactive_transport.common_networks import get_carbon_network
from ..reactive_transport_model import ReactiveTransportModel
from ..utils import (
    HydrologyError,
    HydrologyNumericalError,
    ModelInputError,
    ModelStructureError,
)


# ===================== Model structures ===================== #
class ValidModel(HydrologicalModel):
    """A legal configuration: a surface zone over a bottom zone."""

    structure = [
        [SurfaceZone(name="surface")],
        [GroundZoneB(name="ground")],
    ]


class TwoSurfaceModel(HydrologicalModel):
    """Two independent columns, each bottomed by a bottom zone."""

    structure = [
        [SurfaceZone(name="a"), SurfaceZone(name="b")],
        [GroundZoneB(name="ga"), GroundZoneB(name="gb")],
    ]


class BottomNotLastLayerModel(HydrologicalModel):
    """Violates R1: a bottom zone with another zone below it."""

    structure = [
        [SurfaceZone(name="s1")],
        [GroundZoneB(name="b")],
        [SurfaceZone(name="s2")],
    ]


class NonBottomLastLayerModel(HydrologicalModel):
    """Violates R2: the last layer is not a bottom zone."""

    structure = [
        [SurfaceZone(name="s1")],
        [SurfaceZone(name="s2")],
    ]


class EmptyStructureModel(HydrologicalModel):
    structure: list = []


class BadLayerTypeModel(HydrologicalModel):
    structure: list = [42]  # type: ignore[misc]


# ===================== Fixtures / helpers ===================== #
@pytest.fixture(scope="module")
def model() -> ValidModel:
    return ValidModel(scales=[1.0])


@pytest.fixture(scope="module")
def dates() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=10, freq="D")


def _make_forcing(dates: pd.DatetimeIndex, **overrides) -> ForcingData:
    n: int = len(dates)
    data: dict = {
        "precip": pd.Series(np.linspace(0.1, 2.0, n), index=dates),
        "temp": pd.Series(np.linspace(-2.0, 8.0, n), index=dates),
        "pet": pd.Series(np.linspace(0.1, 1.0, n), index=dates),
    }
    data.update(overrides)
    return ForcingData(**data)


def _with_bad_value(series: pd.Series, idx: int, value: float) -> pd.Series:
    out = series.copy()
    out.iloc[idx] = value
    return out


# ===================== Structural checks (construction time) ===================== #
def test_empty_structure_raises() -> None:
    with pytest.raises(ModelStructureError, match="structure is empty"):
        EmptyStructureModel(scales=[1.0])


def test_bottom_zone_not_in_last_layer_raises() -> None:
    # R1: nothing may be placed below a bottom zone.
    with pytest.raises(
        ModelStructureError, match=r"is not allowed.*below"
    ):
        BottomNotLastLayerModel(scales=[1.0])


def test_non_bottom_zone_in_last_layer_raises() -> None:
    # R2: the last layer must consist of bottom zones.
    with pytest.raises(
        ModelStructureError, match="is not a bottom zone"
    ):
        NonBottomLastLayerModel(scales=[1.0])


def test_override_declared_bottom_with_non_bottom_raises() -> None:
    # The declared structure has a bottom zone, but the zone that actually
    # runs is not one -- the check must catch it via the instantiated zone.
    with pytest.raises(ModelStructureError, match="is not a bottom zone"):
        ValidModel(
            zones={
                "ground": GroundZone(
                    k=0.1, alpha=0.5, perc=0.1
                )
            },
            scales=[1.0],
        )


def test_nan_zone_parameter_raises() -> None:
    bad = SurfaceZone(name="surface", fc=float("nan"))
    with pytest.raises(
        ModelStructureError, match=r"non-finite parameter: fc"
    ):
        ValidModel(zones={"surface": bad}, scales=[1.0])


@pytest.mark.parametrize("scales", [[-0.25], [float("nan")]])
def test_invalid_scales_raise(scales: list[float]) -> None:
    with pytest.raises(ModelStructureError, match="strictly positive"):
        ValidModel(scales=scales)


def test_zero_scale_raises() -> None:
    # Every zone must occupy a non-zero area so the area-weighted lateral
    # transfer (donor/receiver ratio) is well defined.
    with pytest.raises(ModelStructureError, match="strictly positive"):
        TwoSurfaceModel(scales=[0.0, 1.0])


def test_scales_must_sum_to_one() -> None:
    with pytest.raises(ModelStructureError, match="add up to 1"):
        TwoSurfaceModel(scales=[0.3, 0.4])


def test_invalid_layer_type_raises() -> None:
    with pytest.raises(TypeError):
        BadLayerTypeModel(scales=[1.0])


# ===================== Bottom-zone registry ===================== #
def test_is_bottom_zone_recognizes_registered_classes() -> None:
    assert is_bottom_zone(GroundZoneB())
    assert not is_bottom_zone(SurfaceZone(name="s"))
    assert not is_bottom_zone(SnowZone(name="s"))


def test_register_bottom_zone_decorator() -> None:
    @register_bottom_zone
    class _DecoratedZone:
        pass

    class _ChildZone(_DecoratedZone):
        pass

    assert _DecoratedZone in BOTTOM_ZONE_CLASSES
    assert is_bottom_zone(_DecoratedZone())
    # Subclasses of a registered zone are bottom zones too.
    assert is_bottom_zone(_ChildZone())
    # Unregistered objects are not.
    assert not is_bottom_zone(object())


# ===================== Error hierarchy ===================== #
def test_error_class_hierarchy() -> None:
    # Structure/input errors are both ValueError and HydrologyError, so
    # existing `except ValueError` handlers keep working.
    assert issubclass(ModelStructureError, ValueError)
    assert issubclass(ModelStructureError, HydrologyError)
    assert issubclass(ModelInputError, ValueError)
    assert issubclass(ModelInputError, HydrologyError)
    # Numerical errors carry a distinct identity: they are HydrologyErrors
    # but are NOT ValueErrors.
    assert issubclass(HydrologyNumericalError, HydrologyError)
    assert not issubclass(HydrologyNumericalError, ValueError)


# ===================== Runtime input validation ===================== #
def test_nan_precip_raises(model: ValidModel, dates: pd.DatetimeIndex) -> None:
    fd = _make_forcing(
        dates, precip=_with_bad_value(pd.Series(0.5, index=dates), 3, float("nan"))
    )
    with pytest.raises(
        ModelInputError, match=r"the 'precip' series contains non-finite"
    ):
        model.run_hydro_model(fd)


def test_negative_precip_raises(
    model: ValidModel, dates: pd.DatetimeIndex
) -> None:
    fd = _make_forcing(
        dates,
        precip=_with_bad_value(
            pd.Series(np.linspace(0.1, 2.0, len(dates)), index=dates), 2, -1.0
        ),
    )
    with pytest.raises(
        ModelInputError, match="precipitation contains negative"
    ):
        model.run_hydro_model(fd)


def test_negative_pet_raises(model: ValidModel, dates: pd.DatetimeIndex) -> None:
    fd = _make_forcing(
        dates, pet=_with_bad_value(pd.Series(0.5, index=dates), 2, -0.5)
    )
    with pytest.raises(ModelInputError, match="PET contains negative"):
        model.run_hydro_model(fd)


def test_non_numeric_series_raises(
    model: ValidModel, dates: pd.DatetimeIndex
) -> None:
    fd = _make_forcing(
        dates, precip=pd.Series(list("abcdefghij"), index=dates)
    )
    with pytest.raises(
        ModelInputError, match="must contain only numeric values"
    ):
        model.run_hydro_model(fd)


def test_mismatched_series_lengths_raise(
    model: ValidModel, dates: pd.DatetimeIndex
) -> None:
    fd = _make_forcing(
        dates,
        temp=pd.Series(np.linspace(-2.0, 8.0, len(dates) - 2), index=dates[2:]),
    )
    with pytest.raises(
        ModelInputError, match="must all have the same length"
    ):
        model.run_hydro_model(fd)


def test_empty_forcing_raises(model: ValidModel, dates: pd.DatetimeIndex) -> None:
    with pytest.raises(ModelInputError, match=r"series are empty"):
        model.run_hydro_model(_make_forcing(dates[0:0]))


def test_wrong_number_of_forcing_data_raises(
    model: ValidModel, dates: pd.DatetimeIndex
) -> None:
    m = TwoSurfaceModel(scales=[0.5, 0.5])
    fd = _make_forcing(dates)
    with pytest.raises(ModelInputError, match="one per surface zone"):
        m.run_hydro_model([fd, fd, fd])


def test_meas_streamflow_wrong_length_raises(
    model: ValidModel, dates: pd.DatetimeIndex
) -> None:
    stream = pd.Series(0.1, index=dates[2:])
    with pytest.raises(ModelInputError, match="must have the same length"):
        model.run_hydro_model(_make_forcing(dates), meas_streamflow=stream)


def test_meas_streamflow_negative_raises(
    model: ValidModel, dates: pd.DatetimeIndex
) -> None:
    stream = pd.Series(0.1, index=dates)
    stream.iloc[0] = -0.3
    with pytest.raises(
        ModelInputError, match="Measured streamflow must be >= 0"
    ):
        model.run_hydro_model(_make_forcing(dates), meas_streamflow=stream)


def test_meas_streamflow_nan_raises(
    model: ValidModel, dates: pd.DatetimeIndex
) -> None:
    stream = pd.Series(0.1, index=dates)
    stream.iloc[1] = float("nan")
    with pytest.raises(
        ModelInputError, match="meas_streamflow contains non-finite"
    ):
        model.run_hydro_model(_make_forcing(dates), meas_streamflow=stream)


def test_init_state_wrong_length_raises(
    model: ValidModel, dates: pd.DatetimeIndex
) -> None:
    # The model has 2 zones (surface, ground).
    with pytest.raises(
        ModelInputError, match="One initial storage value per zone"
    ):
        model.run_hydro_model(_make_forcing(dates), init_state=np.zeros(3))


def test_init_state_negative_raises(
    model: ValidModel, dates: pd.DatetimeIndex
) -> None:
    init = np.zeros(model.num_zones)
    init[0] = -1.0
    with pytest.raises(ModelInputError, match="init_state contains negative"):
        model.run_hydro_model(_make_forcing(dates), init_state=init)


def test_init_state_nan_raises(
    model: ValidModel, dates: pd.DatetimeIndex
) -> None:
    init = np.zeros(model.num_zones)
    init[1] = float("nan")
    with pytest.raises(
        ModelInputError, match="init_state contains non-finite"
    ):
        model.run_hydro_model(_make_forcing(dates), init_state=init)


# ===================== Happy paths ===================== #
def test_run_happy_path(model: ValidModel, dates: pd.DatetimeIndex) -> None:
    stream = pd.Series(np.linspace(0.0, 0.5, len(dates)), index=dates)
    res = model.run_hydro_model(_make_forcing(dates), meas_streamflow=stream)
    numeric = res.simulation.select_dtypes("number")
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()


def test_check_water_balance_happy_path(
    model: ValidModel, dates: pd.DatetimeIndex
) -> None:
    # The conservative model passes its own water-balance check (opt-in).
    model.run_hydro_model(_make_forcing(dates), check_water_balance=True)


def test_negative_temperatures_are_allowed(
    model: ValidModel, dates: pd.DatetimeIndex
) -> None:
    # Temperature legitimately goes below zero: only finiteness is checked.
    fd = _make_forcing(dates, temp=pd.Series(-5.0, index=dates))
    model.run_hydro_model(fd)


# ===================== Reactive-transport input validation ===================== #
class RtValidationModel(ReactiveTransportModel):
    structure = [
        [SnowZone(name="snow")],
        [SurfaceZone(name="surface")],
        [GroundZoneB(name="ground")],
    ]


@pytest.fixture(scope="module")
def rt_model() -> RtValidationModel:
    network = get_carbon_network()
    mp = MineralParameters(
        np.array([0.1]), np.array([2.0]), np.array([1.5]), np.array([10.0])
    )
    dim = ZoneDimensions(20.0, 10.0)
    rt_zones = {
        name: RtZone(
            network,
            RtParameters(dim, mp),
            do_reactions=True,
            do_speciation=True,
            name=name,
        )
        for name in ("snow", "surface", "ground")
    }
    return RtValidationModel(
        zones={
            "snow": SnowZone(name="snow"),
            "surface": SurfaceZone(name="surface"),
            "ground": GroundZoneB(name="ground"),
        },
        scales=[1.0],
        network=network,
        rt_zones=rt_zones,
    )


def _hydro_sim_df(n: int) -> pd.DataFrame:
    """A minimal simulated-hydrology table as produced by run_hydro_model."""
    df = pd.DataFrame(index=pd.date_range("2020-01-01", periods=n, freq="D"))
    for z in ("snow", "surface", "ground"):
        df[f"s_{z}"] = 5.0
        df[f"q_forc_{z}"] = 0.1
        df[f"q_vap_{z}"] = 0.05
        df[f"q_lat_{z}"] = 0.0
        df[f"q_lat_ext_{z}"] = 0.0
        df[f"q_vert_{z}"] = 0.05
        df[f"q_vert_ext_{z}"] = 0.0
        df[f"q_in_{z}"] = 0.05
    return df


def test_rt_precip_conc_nan_raises(
    rt_model: RtValidationModel, dates: pd.DatetimeIndex
) -> None:
    pc = np.ones((1, len(dates), 2))
    pc[0, 2, 1] = np.nan
    with pytest.raises(
        ModelInputError, match="non-finite \\(NaN/Inf\\) concentration"
    ):
        rt_model.construct_rt_forcing_matrix(
            _make_forcing(dates), _hydro_sim_df(len(dates)), pc, rt_model.rt_zones
        )


def test_rt_precip_conc_wrong_surface_dim(
    rt_model: RtValidationModel, dates: pd.DatetimeIndex
) -> None:
    pc = np.ones((2, len(dates), 2))  # 2 != 1 surface zone
    with pytest.raises(ModelInputError, match="number of surface zones"):
        rt_model.construct_rt_forcing_matrix(
            _make_forcing(dates), _hydro_sim_df(len(dates)), pc, rt_model.rt_zones
        )


def test_rt_precip_conc_wrong_steps_dim(
    rt_model: RtValidationModel, dates: pd.DatetimeIndex
) -> None:
    pc = np.ones((1, 3, 2))  # 3 steps != len(dates)
    with pytest.raises(
        ModelInputError, match="number of steps in the hydro simulation"
    ):
        rt_model.construct_rt_forcing_matrix(
            _make_forcing(dates), _hydro_sim_df(len(dates)), pc, rt_model.rt_zones
        )
