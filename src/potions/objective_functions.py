from typing import Callable, Optional

import numpy as np
from pandas import Series

from .common_types import ObjectiveComponent, ObjectiveSpec


def kge(y_obs: Series, y_pred: Series) -> float:
    """Kling-Gupta efficiency"""
    r: float = y_pred.corr(y_obs, method="pearson")
    alpha: float = float(y_pred.std() / y_obs.std())
    beta: float = float(y_pred.mean() / y_obs.mean())

    return 1.0 - ((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) ** 0.5


def log_kge(y_obs: Series, y_pred: Series) -> float:
    """Kling-Gupta efficiency of the log-transformed data"""
    y_obs = y_obs.map(lambda x: np.log10(x + 0.1))
    y_pred = y_pred.map(lambda x: np.log10(x + 0.1))

    r: float = y_pred.corr(y_obs, method="pearson")
    alpha: float = float(y_pred.std() / y_obs.std())
    beta: float = float(y_pred.mean() / y_obs.mean())

    return 1.0 - ((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) ** 0.5


def nse(y_obs: Series, y_pred: Series) -> float:
    """Nash-Sutcliffe Efficiency"""
    numer: float = ((y_pred - y_obs) ** 2).sum()
    denom: float = ((y_obs - y_obs.mean()) ** 2).sum()

    return 1 - numer / denom


def log_nse(y_obs: Series, y_pred: Series) -> float:
    """Nash-Sutcliffe Efficiency of the log-transformed data"""
    y_obs = y_obs.map(lambda x: np.log10(x + 0.1))
    y_pred = y_pred.map(lambda x: np.log10(x + 0.1))
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


def bias(y_obs: Series, y_pred: Series) -> float:
    """Calculate the bias of the model"""
    return (y_pred - y_obs).mean() / y_obs.mean()


def r_squared(y_obs: Series, y_pred: Series) -> float:
    return y_obs.corr(y_pred, method="pearson") ** 2


def spearman_rho(y_obs: Series, y_pred: Series) -> float:
    return y_obs.corr(y_pred, method="spearman")


DEFAULT_OBJECTIVE_FUNCTIONS: list[tuple[str, Callable[[Series, Series], float]]] = [
    ("kge", kge),
    ("log_kge", log_kge),
    ("nse", nse),
    ("log_nse", log_nse),
    ("bias", bias),
    ("spearman_rho", spearman_rho),
    ("r_squared", r_squared),
]


METRIC_REGISTRY: dict[str, Callable[[Series, Series], float]] = {
    "kge": kge,
    "log_kge": log_kge,
    "nse": nse,
    "log_nse": log_nse,
    "bias": bias,
    "spearman_rho": spearman_rho,
    "r_squared": r_squared,
}


def flow_weighted_nse(meas_series: Series, sim_series: Series, weights: Series) -> float:
    """Nash-Sutcliffe efficiency with each time step's squared error weighted
    by ``weights``. Passing the discharge series as the weights yields a
    flow-weighted NSE that emphasizes high-flow time steps.
    """
    numer: float = float((weights * (sim_series - meas_series).pow(2)).sum())
    denom: float = float((weights * (meas_series - meas_series.mean()).pow(2)).sum())
    return 1.0 - numer / denom


def flow_weighted_log_nse(meas_series: Series, sim_series: Series, weights: Series) -> float:
    """Flow-weighted Nash-Sutcliffe efficiency on the base-10 log-transformed
    data (with a small offset so that zeros do not log to -inf)."""
    m: Series = meas_series.map(lambda x: np.log10(x + 0.1))
    s: Series = sim_series.map(lambda x: np.log10(x + 0.1))
    return flow_weighted_nse(m, s, weights)


def flow_weighted_kge(meas_series: Series, sim_series: Series, weights: Series) -> float:
    """Flow-weighted Kling-Gupta efficiency. Each residual / ratio term is
    weighted by the supplied series; the overall metric is 1 minus the
    Euclidean norm of the weighted errors, mirroring ``kge``.
    """
    r: float = float((weights * ((sim_series - sim_series.mean()) / (sim_series.std() + 1e-12)) * ((meas_series - meas_series.mean()) / (meas_series.std() + 1e-12))).sum() / max(float((weights).sum()), 1e-12))
    alpha: float = float(np.sqrt((weights * (sim_series.std() ** 2)).sum() / max(float((weights).sum()), 1e-12)) / (meas_series.std() + 1e-12))
    beta: float = float((weights * sim_series).sum() / max(float((weights * meas_series).sum()), 1e-12))
    return 1.0 - ((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) ** 0.5


def _component_value(
    meas: Series, sim: Series, weights: Series, comp: "ObjectiveComponent"
) -> float:
    """Evaluate one :class:`ObjectiveComponent` against the (meas, sim) series.

    When :attr:`comp.use_flow_weight` is True, a flow-weighted variant of the
    metric is used, with ``weights`` supplied as the per-time-step weights.
    """
    if comp.use_flow_weight:
        if comp.metric == "nse":
            return flow_weighted_nse(meas, sim, weights)
        if comp.metric == "log_nse":
            return flow_weighted_log_nse(meas, sim, weights)
        if comp.metric == "kge":
            return flow_weighted_kge(meas, sim, weights)

    if comp.metric not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric name: {comp.metric!r}")

    return METRIC_REGISTRY[comp.metric](meas, sim)


def evaluate_components(
    spec: "ObjectiveSpec",
    q_meas: Series,
    q_sim: Series,
    doc_meas: Optional[Series] = None,
    doc_sim: Optional[Series] = None,
    target_species: str = "DOC",
) -> float:
    """Evaluate the composite objective defined by ``spec`` against a
    measured / simulated streamflow pair and (optionally) a measured /
    simulated solute-concentration pair.

    Each :class:`ObjectiveComponent` selects a series from these inputs:

    * ``target == "q"`` uses ``(q_meas, q_sim)``;
    * ``target == "doc"`` uses ``(doc_meas, doc_sim)``;
    * ``target == "doc_load"`` uses ``(doc_meas * q_meas, doc_sim * q_sim)``.

    The sum of ``(1 - value) ** power`` over every component is returned.
    A perfect fit yields zero.

    Args:
        spec: The specification of the objective terms to combine.
        q_meas: The observed streamflow.
        q_sim: The simulated streamflow, aligned with ``q_meas``.
        doc_meas: The observed solute concentration (required by any
            ``"doc"`` or ``"doc_load"`` component).
        doc_sim: The simulated solute concentration, aligned with
            ``doc_meas``.
        target_species: The name of the solute species, used only to
            populate the error message when a doc series is missing.
    """
    weights: Series = q_meas
    total: float = 0.0
    for comp in spec.components:
        if comp.target == "q":
            meas: Series = q_meas
            sim: Series = q_sim
        elif comp.target == "doc":
            if doc_meas is None or doc_sim is None:
                raise ValueError(
                    f"Component {comp.metric!r} requires a {target_species!r} "
                    "concentration series; provide `doc_meas` / `doc_sim`."
                )
            meas = doc_meas
            sim = doc_sim
        elif comp.target == "doc_load":
            if doc_meas is None or doc_sim is None:
                raise ValueError(
                    f"Component {comp.metric!r} requires a {target_species!r} "
                    "concentration series; provide `doc_meas` / `doc_sim`."
                )
            meas = doc_meas * q_meas
            sim = doc_sim * q_sim
        else:
            raise ValueError(f"Unknown target: {comp.target!r}")

        val: float = _component_value(meas, sim, weights, comp)
        total += (1.0 - val) ** comp.power

    return total


# The showcase's 9-term preset: the composite objective that has been used
# for the East River calibration. Streamflow (q), the target solute
# concentration (doc), and the target solute load (doc_load) are each
# scored with nse, log_nse, and a flow-weighted nse, giving nine terms.
DEFAULT_RT_OBJECTIVE_SPEC: ObjectiveSpec = ObjectiveSpec(
    components=(
        ObjectiveComponent(metric="nse", target="q"),
        ObjectiveComponent(metric="log_nse", target="q"),
        ObjectiveComponent(metric="nse", target="q", use_flow_weight=True),
        ObjectiveComponent(metric="nse", target="doc"),
        ObjectiveComponent(metric="log_nse", target="doc"),
        ObjectiveComponent(metric="nse", target="doc", use_flow_weight=True),
        ObjectiveComponent(metric="nse", target="doc_load"),
        ObjectiveComponent(metric="log_nse", target="doc_load"),
        ObjectiveComponent(metric="nse", target="doc_load", use_flow_weight=True),
    )
)
