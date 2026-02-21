import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from ..common_types_compiled import HydroForcing
from ..math import bisect, find_root, ode_is_stable, sign


def test_sign():
    assert sign(-1) == -1
    assert sign(0) == 0
    assert sign(1) == 1
    assert sign(-0.5) == -1
    assert sign(0.5) == 1


def test_bisect():
    # Define a test function
    def test_function(x, x_init, d, dt):
        return x**2 - 2  # Root at approximately sqrt(2)``

    # Create a mock HydroForcing object
    mock_hydro_forcing = MagicMock()

    # Define test parameters
    x_l = 1.0
    x_r = 2.0
    x_init = 0.0
    d = mock_hydro_forcing
    dt = 1.0
    tol = 1e-6

    # Call the bisect function
    root = bisect(test_function, x_l, x_r, x_init, d, dt, tol)

    # Assert that the root is close to the expected value
    assert abs(root - np.sqrt(2)) < tol

    # Test case where the root is exactly at x_l
    def test_function_exact(x, x_init, d, dt):
        return x - 1.0

    root = bisect(test_function_exact, 1.0, 2.0, x_init, d, dt, tol)
    assert abs(root - 1.0) < tol

    # Test case where the root is exactly at x_r
    def test_function_exact_r(x, x_init, d, dt):
        return x - 2.0

    root = bisect(test_function_exact_r, 1.0, 2.0, x_init, d, dt, tol)
    assert abs(root - 2.0) < tol

    # Test exception when no root is within the bounds
    def test_function_no_root(x, x_init, d, dt):
        return x + 5.0

    with pytest.raises(ValueError):
        bisect(test_function_no_root, 1.0, 2.0, x_init, d, dt, tol)

    # Test maximum iteration exception
    def test_function_slow_convergence(x, x_init, d, dt):
        return x**2 - 2.00000000000000000001

    with pytest.raises(ValueError):
        bisect(
            test_function_slow_convergence, 1.0, 2.0, x_init, d, dt, tol, max_iters=2
        )


def test_find_root():
    # Define a test function
    def test_function(x, *args):
        return x**2 - 2  # Root at approximately sqrt(2)

    # Test parameters
    x_guess = 1.0
    mock_hydro_forcing = MagicMock()
    tol = 1e-6
    max_iters = 50

    # Call the find_root function
    root = find_root(
        test_function,
        x_guess,
        mock_hydro_forcing,
        dt=1.0,
        tol=tol,
    )

    # Assert that the root is close to the expected value
    assert abs(root - np.sqrt(2)) < tol

    # Test case with different function
    def test_function2(x, *args):
        return math.sin(x)

    root = find_root(test_function2, 3.0, mock_hydro_forcing, dt=1.0, tol=tol)
    assert abs(root - math.pi) < tol

    # Test case where derivative is zero
    def test_function_no_root(x, *args):
        return x**2 + 1

    # Newton's method will fail
    with pytest.raises(ValueError):
        find_root(test_function_no_root, 0.5, mock_hydro_forcing, 1.0, tol=tol)


def test_ode_is_stable():
    # Test function with negative lambda (stable)
    def stable_func(x, d):
        return -x  # lambda = -1

    # Test function with positive lambda (unstable)
    def unstable_func(x, d):
        return x  # lambda = 1

    # Test function with zero lambda (boundary case)
    def zero_lambda_func(x, d):
        return 0 * x  # lambda = 0

    d = HydroForcing(0, 0, 0, 0)

    # Test stable case
    assert ode_is_stable(stable_func, 1.0, d, 1.0) == True
    assert ode_is_stable(stable_func, 1.0, d, 2.0) == True
    assert ode_is_stable(stable_func, 1.0, d, 0.5) == True

    # Test unstable case
    assert ode_is_stable(unstable_func, 1.0, d, 1.0) == False
    assert ode_is_stable(unstable_func, 1.0, d, 2.0) == False

    # Test boundary case (lambda = 0)
    assert ode_is_stable(zero_lambda_func, 1.0, d, 1.0) == False
    assert ode_is_stable(zero_lambda_func, 1.0, d, 2.0) == False

    # Test with a more complex function
    def quadratic_func(x, d):
        return x**2

    # For x = 1, f(x) = 1, f'(x) = 2*1 = 2
    assert (
        ode_is_stable(quadratic_func, 1.0, d, 0.5) == False
    )  # dt = 0.5, 2 / |2| = 1, 0.5 < 1
    assert (
        ode_is_stable(quadratic_func, 1.0, d, 2.0) == True
    )  # dt = 2, 2 / |2| = 1, 2 > 1

    # Test with a linear function with negative slope
    def linear_negative(x, d):
        return -2 * x  # lambda = -2

    assert (
        ode_is_stable(linear_negative, 1.0, d, 0.25) == True
    )  # dt = 0.25, 2 / |-2| = 1, 0.25 < 1
    assert (
        ode_is_stable(linear_negative, 1.0, d, 2.0) == True
    )  # dt = 2, 2 / |-2| = 1, 2 > 1

    # Test with a linear function with positive slope
    def linear_positive(x, d):
        return 2 * x  # lambda = 2

    assert (
        ode_is_stable(linear_positive, 1.0, d, 0.25) == False
    )  # dt = 0.25, 2 / |2| = 1, 0.25 < 1
    assert (
        ode_is_stable(linear_positive, 1.0, d, 2.0) == True
    )  # dt = 2, 2 / |2| = 1, 2 > 1
