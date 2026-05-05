from typing import Callable
from scipy.optimize import approx_fprime
import numpy as np
from potions.core import HydroForcing


def ode_is_stable(
    f: Callable[[float, HydroForcing], float],
    x: float,
    d: HydroForcing,
    dt: float,
) -> bool:
    dt_diff: float = 1e-3
    lambda_val = (f(x + dt_diff, d) - f(x - dt_diff, d)) / (2 * dt_diff)
    if lambda_val < 0 and dt < 2 / (abs(lambda_val)):
        return True
    else:
        return False


def midpoint_method(
    mass_balance_func: Callable[[float, HydroForcing], float],
    x_init: float,
    d: HydroForcing,
    dt: float,
) -> float:
    x_mid = x_init + 0.5 * dt * mass_balance_func(x_init, d)
    return x_init + dt * mass_balance_func(x_mid, d)


def find_root(
    f: Callable,
    x_init: float,
    d: HydroForcing,
    dt: float,
    tol: float = 1e-6,
) -> float:
    """
    Optimized Secant method root finder.
    """
    x_0 = x_init

    x_1 = x_0 + 0.1

    # Pre-calculate initial function values to avoid re-evaluating inside the loop
    fx_0 = f(x_0, x_init, d, dt)
    fx_1 = f(x_1, x_init, d, dt)

    err = abs(fx_1)
    counter = 0
    max_iter = 50

    while err > tol:
        if abs(fx_1 - fx_0) < 1e-12:
            break

        # Standard Secant Formula
        # x_n = (x_0 * fx_1 - x_1 * fx_0) / (fx_1 - fx_0)
        x_n = x_1 - fx_1 * (x_1 - x_0) / (fx_1 - fx_0)

        # Shift values for next iteration
        x_0 = x_1
        fx_0 = fx_1

        x_1 = x_n
        # Only ONE python function call per iteration now
        fx_1 = float(f(x_n, x_init, d, dt))  # type: ignore

        err = abs(fx_1)
        counter += 1

        if counter >= max_iter:
            # We raise a standard Python error here, which is fine as it's the failure case
            raise ValueError(
                f"Root finding failed to converge after {max_iter} iterations"
            )

    return x_1


def sign(x: float) -> int:
    """Check the sign of a number. Negative means negative, positive means the number is positive, and zero means the number is zero"""
    if x > 0.0:
        return 1
    elif x < 0.0:
        return -1
    else:
        return 0


def bisect(
    f: Callable,
    x_l: float,
    x_r: float,
    x_init: float,
    d: HydroForcing,
    dt: float,
    tol: float = 1e-6,
    max_iters: int = 50,
) -> float:
    """
    Bisection method root solver
    """
    x_m = 0.5 * (x_l + x_r)  # Midpoint is halfway between
    fx_l = f(x_l, x_init, d, dt)
    fx_r = f(x_r, x_init, d, dt)
    fx_m = f(x_m, x_init, d, dt)
    err = abs(fx_m)

    if abs(fx_l) < 1e-12:  # Point is on the left boundary
        return x_l
    if abs(fx_r) < 1e-12:  # Point is on the right boundary
        return x_r

    if sign(fx_l) == sign(fx_r):
        raise ValueError(
            f"Bisection method failed. There is no value contained in [{x_l},{x_r}]"
        )

    cur_iter = 0

    while err > tol and cur_iter < max_iters:
        if fx_l * fx_m > 0:  # The point is to the right of the left edge
            x_l = x_m
            fx_l = fx_m
        else:  # The point is to the left of the right edge, move the right edge inwards
            x_r = x_m
            fx_r = fx_m

        x_m = 0.5 * (x_l + x_r)
        fx_m = f(x_m, x_init, d, dt)
        err = abs(fx_m)

        cur_iter += 1

    if cur_iter == max_iters:
        raise ValueError(
            f"Bisection method failed with too many iterations, final error after {max_iters} iterations was {err}"
        )

    return x_m


def find_root_multi(
    f: Callable[[np.ndarray], np.ndarray],
    x_0: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    debug: bool = False,
) -> np.ndarray:
    x: np.ndarray = x_0.copy()
    f_x: np.ndarray = f(x)
    err: float = (f_x**2).mean()

    if debug:
        print(f"Initial f(x): {f_x}")
        print(f"Initial error: {err}")

    i: int
    for i in range(max_iter):
        if err <= tol:
            return x

        jac_x: np.ndarray = approx_fprime(x, f)  # type: ignore
        step: np.ndarray = np.linalg.solve(jac_x, f_x)
        x_new: np.ndarray = x - step
        x = x_new
        f_x = f(x)
        err = (f_x**2).mean()
        if np.isnan(err):
            # print("Rootfinding error: the calculated error is NaN. Values:")
            # print(f"{x_0=}")
            # print(f"{x=}")
            # print(f"{f_x=}")
            # print(f"{jac_x=}")
            # print(f"{step=}")
            raise ValueError("NaN value encountered in rootfinding")

        if debug:
            print("-" * 10)
            print(f"Step {i}")
            print(f"f(x): {f_x}")
            print(f"Jacobian matrix: \n{jac_x}")
            print(f"Error: {err}")
            print()

    raise ValueError(f"Failed to find root starting at {x_0=} with final error {err}")


def fast_matvec(
    matrix: np.ndarray, vector: np.ndarray, out: np.ndarray  # type: ignore
) -> None:  # type: ignore
    """
    Performs out = matrix * vector.
    Inputs must be memoryviews (e.g., NumPy arrays).
    """

    # Declare C types for all local variables
    i: int
    j: int
    m: int = matrix.shape[0]  # type: ignore
    n: int = matrix.shape[1]  # type: ignore
    dot_product: float

    # 'prange' enables OpenMP parallelism and releases the GIL
    # 'nogil=True' ensures no Python objects are accessed inside the loop
    for i in range(m):
        dot_product = 0.0
        for j in range(n):
            # Accessing memoryviews directly avoids Python attribute lookups
            dot_product += matrix[i, j] * vector[j]  # type: ignore
        out[i] = dot_product  # type: ignore
