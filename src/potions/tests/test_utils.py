import pytest
from potions.utils import find_root


def test_find_root_simple_linear():
    """
    Test find_root with a simple linear function f(x) = x - 5.
    The root is 5.
    """
    func = lambda x: x - 5
    root = find_root(func, x_0=1.0, tol=1e-6)
    assert abs(root - 5.0) < 1e-6


def test_find_root_quadratic():
    """
    Test find_root with a quadratic function f(x) = x^2 - 4.
    We'll find the positive root, which is 2.
    """
    func = lambda x: x**2 - 4
    root = find_root(func, x_0=1.0, tol=1e-6)
    assert abs(root - 2.0) < 1e-6


def test_find_root_with_negative_root():
    """
    Test find_root with a function where the root is negative.
    f(x) = x + 10. Root is -10.
    """
    func = lambda x: x + 10
    root = find_root(func, x_0=-5.0, tol=1e-6)
    assert abs(root - (-10.0)) < 1e-6


def test_find_root_fails_to_converge():
    """Test that find_root raises an error if it doesn't converge."""
    with pytest.raises(RuntimeError):
        find_root(lambda x: x**2 + 1, x_0=0)  # No real root
