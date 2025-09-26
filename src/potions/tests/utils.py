from numpy.typing import NDArray
from numpy import float64 as f64

def approx_eq(x: NDArray[f64], y: NDArray[f64], tol: float = 1e-8) -> bool:
    return ((x - y) ** 2).sum() < tol