from __future__ import annotations
from typing import Callable, Iterable, Optional, TypeVar

from pandas import DataFrame, Series
import numpy as np
from .model import ForcingData, Model
from .interfaces import Zone

ZoneType = TypeVar("ZoneType", bound=Zone)


def calibrate(
    model: Model[ZoneType],
    forcing_data: list[ForcingData],
    calibration_data: DataFrame,
    param_ranges: Iterable[tuple[float, float]],
    objective_function: Callable[[DataFrame], float],
    init_state: np.ndarray,
    dates: Series,
    dt: float,
    save_threshold: Optional[float] = None,
) -> DataFrame:
    """(Not Implemented) Run calibration of a model structure over a range of parameters.

    Args:
        model: The model structure being tested.
        forcing_data: A list of `ForcingData` objects for the simulation.
        calibration_data: A DataFrame containing the observed data for comparison.
        param_ranges: The minimum and maximum values of the parameters to test.
        objective_function: The objective function to quantify the measure of fit.
        init_state: The initial state of the model's zones.
        dates: The time series of dates for the simulation.
        dt: The time step duration in days.
        save_threshold: Above this value, the model will save the model result.
            If `None`, then all runs will be saved. Defaults to None.

    Returns:
        DataFrame: A dataframe of all parameter sets if no threshold, or all above the threshold if it is set
    """
    raise NotImplementedError()
