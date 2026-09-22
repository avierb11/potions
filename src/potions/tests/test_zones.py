import numpy as np
from numpy import float64 as f64

from ..core import (
    GroundZone,
    HydroForcing,
    HydroStep,
    SnowZone,
    SurfaceZone,
)


def _assert_step_mass_balance(
    zone, s_0: float, d: HydroForcing, dt: float, tol: float = 1e-6
) -> None:
    """For every zone, a single step must satisfy the storage update identity
    s_new = s_0 + dt*(q_in + forc_flux - vap_flux - lat_flux - vert_flux).
    This is the invariant the Python steering logic relies on, and it does not
    depend on the exact (Rust) flux formulas.
    """
    new_state: HydroStep = zone.step(s_0, d, dt)
    rhs: f64 = s_0 + dt * (
        d.q_in + new_state.forc_flux - new_state.vap_flux - new_state.lat_flux - new_state.vert_flux
    )
    assert abs(new_state.state - rhs) < tol, (
        f"Mass balance failed for zone {type(zone).__name__}: "
        f"state={new_state.state}, rhs={rhs}"
    )


def test_SnowZone() -> None:
    d: HydroForcing = HydroForcing(precip=1.0, temp=-1.0, pet=1.0, q_in=0.0)
    s_0: float = 0.0
    d_t: float = 1.0

    sz: SnowZone = SnowZone(0.0, 1.0)

    new_state: HydroStep = sz.step(s_0, d, d_t)
    # Below freezing, precipitation accumulates in the snowpack.
    assert abs(new_state.state - 1.0) < 1e-7
    assert abs(new_state.forc_flux - 1.0) < 1e-7

    _assert_step_mass_balance(sz, s_0, d, d_t)

    # Base case, no snowfall -> no new state.
    d_base = HydroForcing(precip=0.0, temp=0.0, pet=0.0, q_in=0.0)
    sz = SnowZone(0.0, 0.0)
    new_state = sz.step(s_0, d_base, d_t)
    assert abs(new_state.state) < 1e-7
    assert abs(new_state.forc_flux) < 1e-7


def test_SurfaceZone() -> None:
    d: HydroForcing = HydroForcing(precip=1.0, temp=5.0, pet=2.0, q_in=1.0)
    s_0: float = 50.0
    d_t: float = 1.0

    sz: SurfaceZone = SurfaceZone(fc=100.0, lp=1.0, beta=1.0, k0=0.1, thr=50.0)

    # Parameter round-trip must be exact.
    assert sz.param_list() == [100.0, 1.0, 1.0, 0.1, 50.0]

    # Surface zone has no lateral outflow by default.
    assert abs(sz.lat_flux(s_0, d)) < 1e-9

    # The step must conserve mass.
    new_state: HydroStep = sz.step(s_0, d, d_t)
    _assert_step_mass_balance(sz, s_0, d, d_t)

    # Evapotranspiration cannot exceed the incoming water budget.
    assert new_state.state >= -1e-9
    assert 0.0 <= new_state.vap_flux


def test_GroundZone() -> None:
    d: HydroForcing = HydroForcing(precip=1.0, temp=5.0, pet=2.0, q_in=0.0)

    gz: GroundZone = GroundZone(1e-3, 1.0, 1.0)

    assert gz.param_list() == [1e-3, 1.0, 1.0]

    # Vertical flux is capped at the percolation rate `perc` and otherwise
    # limited by the available storage: vert_flux = min(s, perc).
    assert abs(gz.vert_flux(50.0, d) - 1.0) < 1e-9
    assert abs(gz.vert_flux(0.2, d) - 0.2) < 1e-9
    assert abs(gz.vert_flux(0.3, d) - 0.3) < 1e-9

    # A different percolation cap should bind instead.
    gz_low_perc: GroundZone = GroundZone(1e-3, 1.0, 0.3)
    assert abs(gz_low_perc.vert_flux(50.0, d) - 0.3) < 1e-9
    assert abs(gz_low_perc.vert_flux(0.2, d) - 0.2) < 1e-9

    # Lateral (baseflow) flux scales with storage: lat_flux = k * s.
    assert abs(gz.lat_flux(50.0, d) - 1e-3 * 50.0) < 1e-9

    s_0: float = 50.0
    d_t: float = 1.0
    _assert_step_mass_balance(gz, s_0, d, d_t)


def test_GroundZone_default() -> None:
    gz: GroundZone = GroundZone.default()
    assert isinstance(gz, GroundZone)
    # Parameter round-trip is exact.
    assert gz.param_list() == [1e-2, 1.0, 1.0]
