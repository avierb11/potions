import numpy as np

from ..calibration import parallel_numerical_gradient


def _quadratic(x: np.ndarray) -> float:
    # f = x0^2 + 2*x1^2 + 3*x2^2  =>  grad = [2*x0, 4*x1, 6*x2]
    return float(x[0] ** 2 + 2.0 * x[1] ** 2 + 3.0 * x[2] ** 2)


def _quad2(x: np.ndarray) -> float:
    # f = x0^2 + x1^2  =>  grad = [2*x0, 2*x1]
    return float(x[0] ** 2 + x[1] ** 2)


def _linear1(x: np.ndarray) -> float:
    # f = 2*x0  =>  grad = [2]
    return float(2.0 * x[0])


def test_parallel_numerical_gradient_matches_analytic() -> None:
    x = np.array([1.0, 2.0, 3.0])
    grad = parallel_numerical_gradient(_quadratic, x, num_threads=2)
    assert np.allclose(grad, np.array([2.0, 8.0, 18.0]), atol=1e-3)


def test_parallel_numerical_gradient_zero_axis() -> None:
    # At x0 = 0, the step size falls back to the absolute `dx` rather than the
    # relative one; the gradient should still be correct.
    x = np.array([0.0, 5.0])
    grad = parallel_numerical_gradient(_quad2, x, num_threads=2)
    assert np.allclose(grad, np.array([0.0, 10.0]), atol=1e-3)


def test_parallel_numerical_gradient_single_axis() -> None:
    # A one-dimensional problem.
    x = np.array([3.0])
    grad = parallel_numerical_gradient(_linear1, x, num_threads=1)
    assert grad.shape == (1,)
    assert np.allclose(grad, np.array([2.0]))
