from pandas import Series


def kge(y_pred: Series, y_obs: Series) -> float:
    """Kling-Gupta efficiency"""
    r: float = y_pred.corr(y_obs, method="pearson")
    alpha: float = float(y_pred.std() / y_obs.std())
    beta: float = float(y_pred.mean() / y_obs.mean())

    return 1.0 - ((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) ** 0.5


def nse(y_pred: Series, y_obs: Series) -> float:
    """Kling-Gupta efficiency"""
    numer: float = ((y_pred - y_obs) ** 2).sum()
    denom: float = ((y_obs - y_obs.mean()) ** 2).sum()

    return 1 - numer / denom
