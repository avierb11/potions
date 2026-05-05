import os
from typing import Callable, Optional
from multiprocessing import Pool
import numpy as np
from numpy.typing import NDArray


def _run_twice(
    f: Callable[[NDArray], float], x1: NDArray, x2: NDArray
) -> tuple[float, float]:
    return (f(x1), f(x2))


def parallel_numerical_gradient(
    f: Callable[[NDArray], float],
    x: NDArray,
    num_threads: Optional[int] = None,
    rel_dx: float = 1e-3,
    dx: float = 1e-8,
) -> NDArray:
    """Calculate the gradient of the function in parallel"""
    dxs_list: list[float] = []
    args: list[tuple[Callable, NDArray, NDArray]] = []

    for i, x_i in enumerate(x):
        dx_i: float
        if abs(x_i) <= 1e-12:
            dx_i = dx
        else:
            dx_i = abs(rel_dx * x_i)

        dxs_list.append(dx_i)
        xs_dn: NDArray = x.copy()
        xs_dn[i] -= dx_i
        xs_up: NDArray = x.copy()
        xs_up[i] += dx_i

        args.append((f, xs_dn, xs_up))

    if num_threads is None:
        num_threads = os.cpu_count()

    with Pool(num_threads) as pool:
        res: list[tuple[float, float]] = pool.starmap(_run_twice, args)

    fx_dn: NDArray = np.array([x[0] for x in res])
    fx_up: NDArray = np.array([x[1] for x in res])
    dxs: NDArray = np.array(dxs_list)

    return (fx_up - fx_dn) / (2.0 * dxs)
