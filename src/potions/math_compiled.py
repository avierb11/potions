# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False
# cython: linetrace=False

import cython  # type: ignore
from typing import Callable

from potions.common_types_compiled import HydroForcing


@cython.ccall
@cython.locals(h_diff=cython.double, lambda_val=cython.double)
def ode_is_stable(
    f: Callable[[cython.double, HydroForcing], cython.double],
    x: cython.double,
    d: HydroForcing,
    dt: cython.double,
) -> bool:
    dt_diff: cython.double = 1e-3
    lambda_val = (f(x + dt_diff, d) - f(x - dt_diff, d)) / (2 * dt_diff)
    if lambda_val < 0 and dt < 2 / (abs(lambda_val)):
        return True
    else:
        return False


@cython.ccall
@cython.locals(x_mid=cython.double)
def midpoint_method(
    mass_balance_func: Callable[[cython.double, HydroForcing], cython.double],
    x_init: cython.double,
    d: HydroForcing,
    dt: float,
) -> cython.double:
    x_mid = x_init + 0.5 * dt * mass_balance_func(x_init, d)
    return x_init + dt * mass_balance_func(x_mid, d)


@cython.ccall
@cython.locals(
    x_0=cython.double,
    x_1=cython.double,
    tol=cython.double,
    err=cython.double,
    fx_0=cython.double,
    fx_1=cython.double,
    x_n=cython.double,
    counter=cython.int,
    max_iter=cython.int,
)
def find_root(
    f: Callable,
    x_init: cython.double,
    d: HydroForcing,
    dt: float,
    tol: cython.double = 1e-6,
) -> cython.double:
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


@cython.ccall
def sign(x: cython.double) -> cython.int:
    """Check the sign of a number. Negative means negative, positive means the number is positive, and zero means the number is zero"""
    if x > 0.0:
        return 1
    elif x < 0.0:
        return -1
    else:
        return 0


@cython.ccall
@cython.locals(
    x_m=cython.double,
    fx_l=cython.double,
    fx_r=cython.double,
    fx_m=cython.double,
    err=cython.double,
    cur_iter=cython.int,
)
def bisect(
    f: Callable,
    x_l: cython.double,
    x_r: cython.double,
    x_init: cython.double,
    d: HydroForcing,
    dt: cython.double,
    tol: cython.double = 1e-6,
    max_iters: cython.int = 50,
) -> cython.double:
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
