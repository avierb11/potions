from typing import Callable
from numpy.typing import NDArray
from .utils import HydroForcing

def create_forcing_function(forc_mat: NDArray) -> Callable[[float], NDArray]:
    """
    Take a list of forcing data and produce a function `d(t)` that 
    returns some `data` for the time `t`. This is necessary to be able to
    apply this function to scipy.

    """

    def func(t: float) -> NDArray:
        return forc_mat[int(t)]

    return func