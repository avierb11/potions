import time
import datetime
import numpy as np
import pandas as pd  # type: ignore
from pandas import DataFrame, Series
from numpy.typing import NDArray
from numpy import float64 as f64
from ..hydro import (
    GroundZone,
    HydroForcing,
    HydrologicZone,
    SnowZone,
    SoilZone,
)
from ..model import (
    Layer,
    Model,
    ModelStep,
    ForcingData,
    run_hydro_model,
    HbvModel,
)
from .utils import approx_eq


def test_Model_connection_matrices_simple_3_box() -> None:
    class TestModel(Model):
        structure: list[list[HydrologicZone]] = [
            [SnowZone(name="z0")],
            [SoilZone(name="z1")],
            [GroundZone(name="z2")],
        ]

    scales = [1.0]
    model = TestModel(scales=scales)

    vert_mat: NDArray[f64] = model.vert_mat

    act_vert_mat: NDArray[f64] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )

    assert vert_mat.shape == act_vert_mat.shape
    assert approx_eq(vert_mat, act_vert_mat)

    # Lateral matrix
    assert abs(model.lat_mat[:-1, :]).max() < 1e-8

    # Forcing matrix
    act_forc_mat: NDArray[f64] = np.array([[1.0], [1.0], [1.0]])
    assert approx_eq(act_forc_mat, model.precip_mat)


def test_Model_connection_matrix_mixed_sizes() -> None:
    zone: GroundZone = GroundZone(0.0, 0.0, 0.0)
    layer: Layer = Layer([zone, zone])

    scales = [[0.6, 0.4]]

    class TestModel(Model):
        structure: list[list[HydrologicZone]] = [
            [GroundZone(name="z0"), GroundZone(name="z1")],
            [GroundZone(name="z2"), GroundZone(name="z3")],
            [GroundZone(name="z4")],
        ]

    model = TestModel(scales=scales[0])

    # Lateral matrix
    act_lat_mat: NDArray[f64] = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert approx_eq(act_lat_mat, model.lat_mat[:-1, :])

    # Vertical matrix
    vert_mat: NDArray[f64] = model.vert_mat

    act_vert_mat: NDArray[f64] = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
        ]
    )

    assert vert_mat.shape == act_vert_mat.shape
    assert approx_eq(vert_mat, act_vert_mat)

    # Forcing matrix
    act_forc_mat: NDArray[f64] = np.array(
        [
            [0.6, 0.0],
            [0.0, 0.4],
            [0.6, 0.0],
            [0.0, 0.4],
            [0.6, 0.4],
        ]
    )
    assert approx_eq(act_forc_mat, model.precip_mat)


def test_Model_connection_matrices_3_by_2() -> None:
    zone: GroundZone = GroundZone(0.0, 0.0, 0.0)
    layer: Layer = Layer([zone, zone])

    class TestModel(Model):
        structure: list[list[HydrologicZone]] = [
            [GroundZone(name="z0"), GroundZone(name="z1")],
            [GroundZone(name="z2"), GroundZone(name="z3")],
            [GroundZone(name="z4"), GroundZone(name="z5")],
        ]

    scales = [0.6, 0.4]
    model = TestModel(scales=scales)

    # Lateral matrix
    act_lat_mat: NDArray[f64] = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )

    assert approx_eq(act_lat_mat, model.lat_mat[:-1, :])

    # Vertical matrix
    vert_mat: NDArray[f64] = model.vert_mat

    act_vert_mat: NDArray[f64] = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    assert vert_mat.shape == act_vert_mat.shape
    assert approx_eq(vert_mat, act_vert_mat)

    # Forcing matrix
    act_forc_mat: NDArray[f64] = np.array(
        [
            [0.6, 0.0],
            [0.0, 0.4],
            [0.6, 0.0],
            [0.0, 0.4],
            [0.6, 0.0],
            [0.0, 0.4],
        ]
    )
    assert approx_eq(act_forc_mat, model.precip_mat)


def test_3_box_simple_model_steady_state() -> None:
    model = HbvModel()

    const_forcing = HydroForcing(precip=1.0, temp=25.0, pet=1.0, q_in=0.0)
    forcing: list[HydroForcing] = [const_forcing] * len(model.flat_model)

    state: NDArray[f64] = model.default_init_state()
    error: float = 1e10

    num_steps: int = 1_000
    new_state: NDArray[f64] = np.array([])

    start = time.time()
    for i in range(num_steps):
        model_step: ModelStep = model.step(np.array(state), forcing, 1.0)
        new_state = np.array(model_step.state)
        new_error = abs(new_state - state).sum()
        state = new_state
        assert new_error < error
        error = new_error
    finish = time.time()
    dur: float = finish - start
    rate: float = num_steps / dur


def test_3_box_simple_model_steady_state_v2() -> None:
    model = HbvModel()

    num_steps: int = 2_000  # Number of simulation steps

    # Prepare dates
    start_date = datetime.date(2000, 1, 1)
    date_list: list[datetime.datetime] = [
        start_date + datetime.timedelta(days=i) for i in range(num_steps)
    ]
    dates_series: Series = pd.Series(date_list)

    # Prepare ForcingData
    precip_series = pd.Series([1.0] * num_steps, index=dates_series, dtype=f64)
    temp_series = pd.Series([25.0] * num_steps, index=dates_series, dtype=f64)
    pet_series = pd.Series([1.0] * num_steps, index=dates_series, dtype=f64)

    forcing_data_item = ForcingData(
        precip=precip_series, temp=temp_series, pet=pet_series
    )
    forc_arg: list[ForcingData] = [forcing_data_item]

    # Initial state
    init_state: NDArray[f64] = model.default_init_state()

    start_time = time.time()
    output_df: DataFrame = run_hydro_model(model, init_state, forc_arg, dates_series)
    end_time = time.time()
    dur = end_time - start_time
    rate = num_steps / dur if dur > 0 else float("inf")

    # Assertions
    assert output_df.shape == (num_steps, len(model.flat_model) * 5), (
        f"Output DataFrame shape mismatch. Expected: ({num_steps}, {len(model.flat_model) * 5}), Got: {output_df.shape}"
    )

    # Check for NaNs in the last row of state variables
    state_cols = [col for col in output_df.columns if col.startswith("s_")]
    assert not output_df[state_cols].iloc[-1].isnull().any(), (
        "NaNs found in final states."
    )

    # Check for convergence (state change near the end is small)
    convergence_check_period = 20
    if num_steps > convergence_check_period:
        for state_col in state_cols:
            s_end: f64 = output_df[state_col].iloc[-1]
            s_prev: f64 = output_df[state_col].iloc[-1 - convergence_check_period]
            assert abs(s_end - s_prev) < 1e-3, (
                f"State {state_col} did not converge. End: {s_end:.4f}, Prev: {s_prev:.4f}"
            )
