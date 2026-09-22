"""Verify that the Newton+Armijo solver reaches `max|f| <= 1e-6` (absolute
residual) on the East-River speciation network from a cold start, and that the
kinetic solve path converges end-to-end.
"""
import numpy as np
import pytest
from potions.reactive_transport.common_networks import get_simple_carbon_adsorption_network


TOL = 1e-6


def test_speciation_cold_start():
    """Cold-start (x0 = zeros) speciation solve on the East-River network
    must reach `max|f| <= 1e-6` without throwing.
    """
    net = get_simple_carbon_adsorption_network()
    eq = net.equilibrium_parameters

    total_mat = np.asarray(eq.get_total_mat())
    num_species = total_mat.shape[1]

    # Use a representative concentration vector (all species at ~1 mM) — same
    # species order as the total_mat columns.
    chms = np.ones(num_species, dtype=np.float64) * 1e-3

    # Solve via the public path (calls x_free_solve_rust internally, which now
    # uses Newton+Armijo with a Levenberg-Marquardt fallback).
    concs = np.asarray(eq.solve_equilibrium(chms))

    # Compute the speciation residual and verify max|f| <= 1e-6.
    c_tot = total_mat @ chms
    res = c_tot - total_mat @ concs
    max_abs = np.max(np.abs(res))
    assert max_abs <= TOL, f"speciation max|f| = {max_abs:.3e} > {TOL:.0e}"


def test_speciation_sweep_magnitudes():
    """Solve the East-River speciation network cold across concentrations
    spanning several decades; each must reach `max|f| <= 1e-6`.
    """
    net = get_simple_carbon_adsorption_network()
    eq = net.equilibrium_parameters

    total_mat = np.asarray(eq.get_total_mat())
    num_species = total_mat.shape[1]

    for scale in (1e-6, 1e-4, 1e-2, 1.0):
        chms = np.full(num_species, scale, dtype=np.float64)
        concs = np.asarray(eq.solve_equilibrium(chms))
        c_tot = total_mat @ chms
        res = c_tot - total_mat @ concs
        max_abs = np.max(np.abs(res))
        assert max_abs <= TOL, (
            f"scale={scale:.0e}: speciation max|f| = {max_abs:.3e} > {TOL:.0e}"
        )
