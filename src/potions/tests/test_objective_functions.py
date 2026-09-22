import numpy as np
import pytest
from pandas import Series

from ..objective_functions import (
    bias,
    kge,
    log_kge,
    log_nse,
    nse,
    objective_high_flow,
    objective_low_flow,
    r_squared,
    spearman_rho,
)


@pytest.fixture
def obs() -> Series:
    return Series(np.arange(1.0, 11.0), dtype="float64")


@pytest.fixture
def obs_with_noise() -> Series:
    return Series(
        [1.1, 1.9, 2.8, 4.1, 4.9, 6.2, 6.8, 8.1, 8.9, 10.2], dtype="float64"
    )


def test_kge_perfect_fit(obs) -> None:
    assert kge(obs, obs) == pytest.approx(1.0)


def test_kge_no_perfect_fit(obs, obs_with_noise) -> None:
    assert kge(obs, Series(obs_with_noise)) < 1.0
    assert kge(obs, Series(obs_with_noise)) > 0.95  # close fit but not perfect


def test_nse_perfect_fit(obs) -> None:
    assert nse(obs, obs) == pytest.approx(1.0)


def test_nse_no_worse_than_mean(obs) -> None:
    # Fit to the mean of the observations is NSE = 0 by definition.
    assert nse(obs, Series([obs.mean()] * len(obs))) == pytest.approx(0.0)


def test_nse_worse_than_mean(obs) -> None:
    # A wildly wrong prediction scores below zero.
    assert nse(obs, Series([100.0] * len(obs))) < 0.0


def test_log_metrics_perfect_fit(obs) -> None:
    assert log_kge(obs, obs) == pytest.approx(1.0)
    assert log_nse(obs, obs) == pytest.approx(1.0)


def test_bias_zero_for_perfect_fit(obs) -> None:
    assert bias(obs, obs) == pytest.approx(0.0)


def test_bias_positive_when_overpredicted(obs) -> None:
    assert bias(obs, Series(obs + 1.0)) > 0.0


def test_bias_negative_when_underpredicted(obs) -> None:
    assert bias(obs, Series(obs - 1.0)) < 0.0


def test_r_squared_perfect_fit(obs) -> None:
    assert r_squared(obs, obs) == pytest.approx(1.0)


def test_r_squared_independent_series_is_low() -> None:
    a = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype="float64")
    b = Series([6.0, 1.0, 5.0, 2.0, 4.0, 3.0], dtype="float64")
    # r^2 is a squared correlation, so it must be in [0, 1].
    r2 = r_squared(a, b)
    assert 0.0 <= r2 <= 1.0


def test_spearman_perfectly_monotone(obs) -> None:
    assert spearman_rho(obs, obs) == pytest.approx(1.0)


def _mean_abs_of_selected(fn, pctl, meas) -> float:
    """Run fn over the selected timesteps and return the mean absolute value of
    the selected observations, which is unaffected by ties."""

    selected: list[float] = []

    def recording(m: Series, s: Series) -> float:
        selected.extend(m.tolist())
        return 0.0

    assert fn(recording, pctl, meas, meas) == 0.0
    return float(np.mean(np.abs(selected)))


def test_low_flow_selects_lowest_values() -> None:
    obs = Series(np.arange(1.0, 11.0), dtype="float64")
    # At the 50th percentile, the bottom half (values 1-5) are selected.
    assert _mean_abs_of_selected(objective_low_flow, 0.5, obs) == pytest.approx(3.0)


def test_low_flow_full_percentile_keeps_all() -> None:
    obs = Series(np.arange(1.0, 11.0), dtype="float64")
    assert _mean_abs_of_selected(objective_low_flow, 1.0, obs) == pytest.approx(5.5)


def test_high_flow_selects_highest_values() -> None:
    obs = Series(np.arange(1.0, 11.0), dtype="float64")
    # At the 50th percentile, the top half (values 6-10) are selected.
    assert _mean_abs_of_selected(objective_high_flow, 0.5, obs) == pytest.approx(8.0)


def test_low_flow_invalid_percentile_raises() -> None:
    obs = Series([1.0, 2.0, 3.0], dtype="float64")
    with pytest.raises(ValueError):
        objective_low_flow(nse, 0.0, obs, obs)
    with pytest.raises(ValueError):
        objective_low_flow(nse, 1.5, obs, obs)


def test_high_flow_invalid_percentile_raises() -> None:
    obs = Series([1.0, 2.0, 3.0], dtype="float64")
    with pytest.raises(ValueError):
        objective_high_flow(nse, 0.0, obs, obs)
    with pytest.raises(ValueError):
        objective_high_flow(nse, 1.5, obs, obs)
