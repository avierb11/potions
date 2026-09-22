import numpy as np
import pytest
from unittest.mock import MagicMock

from ..utils import (
    HydrologyNumericalError,
    HydrologyError,
    PotionsError,
    ReactiveTransportError,
    RtNumericalError,
    find_root,
    log_prior,
    rt_minerals_to_array,
    setup_logging,
)


def test_log_prior_in_bounds() -> None:
    assert log_prior(np.array([0.5, 2.0]), {"a": (0, 1), "b": (0, 10)}) == 0.0


def test_log_prior_out_of_bounds_low() -> None:
    assert log_prior(np.array([-0.5, 2.0]), {"a": (0, 1), "b": (0, 10)}) == -np.inf


def test_log_prior_out_of_bounds_high() -> None:
    assert log_prior(np.array([0.5, 11.0]), {"a": (0, 1), "b": (0, 10)}) == -np.inf


def test_log_prior_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        log_prior(np.array([1.0, 2.0]), {"a": (0, 1)})


def test_single_arg_find_root() -> None:
    # Root of x^2 - 2 is sqrt(2).
    assert find_root(lambda x: x**2 - 2, 1.0) == pytest.approx(np.sqrt(2), abs=1e-6)


def test_single_arg_find_root_no_convergence_raises() -> None:
    # x^2 + 1 has no real root.
    with pytest.raises(Exception):
        find_root(lambda x: x**2 + 1, 0.5, tol=1e-12)


def test_rt_minerals_to_array_empty_returns_none() -> None:
    assert rt_minerals_to_array([], ["m1"], ["z1"]) is None


def test_rt_minerals_to_array_array_like() -> None:
    out = rt_minerals_to_array([[1.0, 2.0], [3.0, 4.0]], ["m1", "m2"], ["z1", "z2"])
    assert out is not None
    assert np.allclose(out, [[1.0, 2.0], [3.0, 4.0]])


def test_rt_minerals_to_array_row_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        rt_minerals_to_array([[1.0]], ["m1", "m2"], ["z1", "z2"])


def test_rt_minerals_to_array_dict_of_dict() -> None:
    out = rt_minerals_to_array(
        {"z1": {"m1": 1.0}, "z2": {"m1": 2.0}}, ["m1"], ["z1", "z2"]
    )
    assert out is not None
    # Rows are ordered by zone_order.
    assert np.allclose(out, [[1.0], [2.0]])


def test_rt_minerals_to_array_dict_of_array() -> None:
    out = rt_minerals_to_array(
        {"z1": [1.0, 2.0], "z2": [3.0, 4.0]}, ["m1", "m2"], ["z1", "z2"]
    )
    assert out is not None
    assert np.allclose(out, [[1.0, 2.0], [3.0, 4.0]])


def test_rt_minerals_to_array_dict_missing_zone_raises() -> None:
    with pytest.raises(ValueError):
        rt_minerals_to_array({"z1": {"m1": 1.0}}, ["m1"], ["z1", "z2"])


def test_rt_minerals_to_array_dict_missing_mineral_raises() -> None:
    with pytest.raises(ValueError):
        rt_minerals_to_array({"z1": {"m1": 1.0}}, ["m1", "m2"], ["z1"])


def test_error_hierarchy() -> None:
    assert issubclass(HydrologyError, PotionsError)
    assert issubclass(HydrologyNumericalError, HydrologyError)
    assert issubclass(ReactiveTransportError, PotionsError)
    assert issubclass(RtNumericalError, ReactiveTransportError)


def test_hydrology_numerical_error_stores_attributes() -> None:
    zone = MagicMock()
    params = np.array([0.1, 0.2])
    forcing = MagicMock()
    err = HydrologyNumericalError(
        model_type=MagicMock(),
        zone=zone,
        parameters=params,
        state=5.0,
        hydro_forcing=forcing,
    )
    assert err.zone is zone
    assert err.state == 5.0
    assert err.parameters is params


def test_rt_numerical_error_stores_attributes() -> None:
    zone = MagicMock()
    params = np.array([0.1, 0.2])
    state = np.array([1.0, 2.0])
    forc = MagicMock()
    math_err = MagicMock()
    err = RtNumericalError(
        error_type="iteration",
        model_type=MagicMock(),
        zone=zone,
        parameters=params,
        state=state,
        rt_forcing=forc,
        math_err=math_err,
    )
    assert err.error_type == "iteration"
    assert err.zone is zone
    assert err.math_err is math_err


def test_setup_logging_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("POTIONSLOGGING", raising=False)
    import potions.utils as u

    monkeypatch.setattr(u, "DO_LOGGING", False)
    # With logging disabled, setup_logging returns without touching the logger.
    assert u.setup_logging("test_module.py") is None
