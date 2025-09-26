import time
import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy.typing import NDArray
from numpy import float64 as f64
from ..hydro import GroundZone, SnowZone, SoilZone, HydroForcing
from ..model import Layer, Hillslope, HydrologicModel, HydroModelStep, ForcingData, run_hydro_model
from .utils import approx_eq


def test_HydrologicModel_connection_matrices_simple_3_box() -> None:
    layer: Layer = Layer([GroundZone(0.0, 0.0, 0.0)])

    scales = [[1.0]]
    model: HydrologicModel = HydrologicModel(
        hillslopes=[Hillslope([layer, layer, layer])], scales=scales
    )

    vert_mat: NDArray[f64] = model.get_vert_mat()

    act_vert_mat: NDArray[f64] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )

    assert vert_mat.shape == act_vert_mat.shape
    assert abs((vert_mat - act_vert_mat).sum()) < 1e-8

    # Lateral matrix
    assert abs(model.get_lat_mat()).max() < 1e-8

    # Forcing matrix
    act_forc_mat: NDArray[f64] = np.array([[1.0], [1.0], [1.0]])
    assert approx_eq(act_forc_mat, model.get_forc_mat(scales[0]))


def test_HydrologicModel_connection_matrix_mixed_sizes() -> None:
    zone: GroundZone = GroundZone(0.0, 0.0, 0.0)
    layer: Layer = Layer([zone, zone])

    scales = [[0.6, 0.4]]
    model: HydrologicModel = HydrologicModel(
        hillslopes=[Hillslope([layer, layer, Layer([zone])])], scales=scales
    )

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

    assert approx_eq(act_lat_mat, model.get_lat_mat())

    # Vertical matrix
    vert_mat: NDArray[f64] = model.get_vert_mat()

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
    assert approx_eq(act_forc_mat, model.get_forc_mat(scales[0]))


def test_HydrologicModel_connection_matrices_3_by_2() -> None:
    zone: GroundZone = GroundZone(0.0, 0.0, 0.0)
    layer: Layer = Layer([zone, zone])

    scales = [[0.6, 0.4]]
    model: HydrologicModel = HydrologicModel(
        hillslopes=[Hillslope([layer, layer, layer])], scales=scales
    )

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

    assert approx_eq(act_lat_mat, model.get_lat_mat())

    # Vertical matrix
    vert_mat: NDArray[f64] = model.get_vert_mat()

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
    assert approx_eq(act_forc_mat, model.get_forc_mat(scales[0]))


def test_3_box_simple_model_steady_state() -> None:
    print("Starting full model test...")

    snow: SnowZone = SnowZone(0, 1)
    soil: SoilZone = SoilZone(0, 100, 0.5, 1, 0.1, 10)
    ground: GroundZone = GroundZone(0.01, 1.0, 0.0)

    hillslope: Hillslope = Hillslope(
        layers=[Layer([snow]), Layer([soil]), Layer([ground])]
    )

    model = HydrologicModel(hillslopes=[hillslope], scales=[[1.0]])

    const_forcing = HydroForcing(1.0, 25.0, 1.0)
    forcing: list[HydroForcing] = [const_forcing] * 3

    state: NDArray[f64] = np.ones(3, dtype=float)
    error: float = 1e10

    num_steps: int = 1_000
    new_state: NDArray[f64] = np.array([])

    start = time.time()
    for i in range(num_steps):
        model_step: HydroModelStep = model.step(state, forcing, 1.0)
        new_state = model_step.state
        new_error = abs(new_state - state).sum()
        state = new_state
        assert new_error < error
        error = new_error
    finish = time.time()
    dur: float = finish - start
    rate: float = num_steps / dur

    print(f"Error: {error}")
    print(f"New state: {new_state}")
    print(f"Vaporization flux: {model_step.vap_flux.round(2)}") # type: ignore
    print(f"Forcing flux: {model_step.forc_flux.round(2)}") # type: ignore
    print(f"Lateral flux: {model_step.lat_flux.round(2)}") # type: ignore
    print(f"Vertical flux: {model_step.vert_flux.round(2)}") # type: ignore
    print(f"Total time: {round(dur, 2)} seconds, {round(rate)} iterations per second")


def test_3_box_simple_model_steady_state_v2() -> None:
    print("Starting full model test with run_hydro_model_v2...")

    snow: SnowZone = SnowZone(0, 1)
    soil: SoilZone = SoilZone(0, 100, 0.5, 1, 0.1, 10) # tt, fc, lp, beta, k0, thr
    ground: GroundZone = GroundZone(0.01, 1.0, 0.0) # k, alpha, perc

    hillslope: Hillslope = Hillslope(
        layers=[Layer([snow]), Layer([soil]), Layer([ground])]
    )

    model = HydrologicModel(hillslopes=[hillslope], scales=[[1.0]])

    num_steps: int = 2_000 # Number of simulation steps
    dt: float = 1.0      # Time step

    # Prepare dates
    start_date = datetime.date(2000, 1, 1)
    date_list: list[datetime.datetime] = [start_date + datetime.timedelta(days=i) for i in range(num_steps)] # type: ignore
    dates_series: pd.Series[datetime.datetime] = pd.Series(date_list) # type: ignore

    # Prepare ForcingData
    precip_series = pd.Series([1.0] * num_steps, index=dates_series, dtype=f64)
    temp_series = pd.Series([25.0] * num_steps, index=dates_series, dtype=f64)
    pet_series = pd.Series([1.0] * num_steps, index=dates_series, dtype=f64)
    
    forcing_data_item = ForcingData(precip=precip_series, temp=temp_series, pet=pet_series)
    forc_arg: list[ForcingData] = [forcing_data_item]

    # Initial state
    init_state: NDArray[f64] = np.ones(len(model), dtype=f64)

    start_time = time.time()
    output_df: DataFrame = run_hydro_model(model, init_state, forc_arg, dates_series, dt) # type: ignore
    end_time = time.time()
    dur = end_time - start_time
    rate = num_steps / dur if dur > 0 else float('inf')

    print(f"Simulation finished in {dur:.2f} seconds ({rate:.0f} steps/sec).")

    # Assertions
    assert output_df.shape == (num_steps, len(model) * 5), \
        f"Output DataFrame shape mismatch. Expected: ({num_steps}, {len(model)*5}), Got: {output_df.shape}"

    # Check for NaNs in the last row of state variables
    state_cols = [col for col in output_df.columns if col.startswith('s_')]
    assert not output_df[state_cols].iloc[-1].isnull().any(), "NaNs found in final states."

    # Check for convergence (state change near the end is small)
    convergence_check_period = 20
    if num_steps > convergence_check_period:
        for state_col in state_cols:
            s_end: f64 = output_df[state_col].iloc[-1] # type: ignore
            s_prev: f64 = output_df[state_col].iloc[-1 - convergence_check_period] # type: ignore
            assert abs(s_end - s_prev) < 1e-3, \
                f"State {state_col} did not converge. End: {s_end:.4f}, Prev: {s_prev:.4f}" # type: ignore

    print(f"Final states ({state_cols}): {output_df[state_cols].iloc[-1].values.round(4)}") # type: ignore
    # Example: print specific flux if needed for debugging
    # print(f"Final soil vertical flux: {output_df['q_vert_soil_1'].iloc[-1]:.4f}")
