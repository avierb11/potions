from typing import Callable
from pandas import Series


def kge(y_pred: Series, y_obs: Series) -> float:
    """Kling-Gupta efficiency"""
    r: float = y_pred.corr(y_obs, method="pearson")
    alpha: float = float(y_pred.std() / y_obs.std())
    beta: float = float(y_pred.mean() / y_obs.mean())

    return 1.0 - ((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) ** 0.5


def nse(y_pred: Series, y_obs: Series) -> float:
    """Nash-Sutcliffe Efficiency"""
    numer: float = ((y_pred - y_obs) ** 2).sum()
    denom: float = ((y_obs - y_obs.mean()) ** 2).sum()

    return 1 - numer / denom


def objective_low_flow(
    obj_func: Callable[[Series, Series], float],
    low_flow_pctl: float,
    meas: Series,
    sim: Series,
) -> float:
    """
    Evaluate the objective function on only the bottom `low_flow_pctl` percentile of the data.
    """
    if not (0 < low_flow_pctl <= 1.0):
        raise ValueError(f"Percentile must be between 0 and 1, not {low_flow_pctl}")

    thresh: float = meas.quantile(low_flow_pctl)

    mask: Series = meas <= thresh

    sub_meas: Series = meas[mask]
    sub_sim: Series = sim[mask]

    return obj_func(sub_meas, sub_sim)


def objective_high_flow(
    obj_func: Callable[[Series, Series], float],
    high_flow_pctl: float,
    meas: Series,
    sim: Series,
) -> float:
    """
    Evaluate the objective function on only the top `high_flow_pctl` percentile of the data.
    """
    if not (0 < high_flow_pctl <= 1.0):
        raise ValueError(f"Percentile must be between 0 and 1, not {high_flow_pctl}")

    thresh: float = meas.quantile(high_flow_pctl)

    mask: Series = meas >= thresh

    sub_meas: Series = meas[mask]
    sub_sim: Series = sim[mask]

    return obj_func(sub_meas, sub_sim)
