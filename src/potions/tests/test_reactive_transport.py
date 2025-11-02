import numpy as np
import pytest
from potions.reactive_transport import EquilibriumParameters, ChemicalState

def test_solve_equilibrium_simple():
    # System: A <=> B with K = 2
    # Reaction: B - A = 0
    stoich = np.array([[-1, 1]])
    equilibrium = np.array([2.0])
    total = np.array([[1, 1]])

    params = EquilibriumParameters(
        stoich=stoich,
        equilibrium=equilibrium,
        total=total
    )

    # Initial state: [A]=1.0, [B]=0.0
    initial_state = ChemicalState(
        prim_aq_conc=np.array([1.0]), # A
        sec_conc=np.array([0.0]),     # B
        min_conc=np.array([])
    )

    new_state = params.solve_equilibrium(initial_state)

    expected_prim_conc = np.array([1/3])
    expected_sec_conc = np.array([2/3])

    assert np.allclose(new_state.prim_aq_conc, expected_prim_conc)
    assert np.allclose(new_state.sec_conc, expected_sec_conc)

def test_solve_equilibrium_complex():
    # System: 2A <=> C with K = 4
    # Reaction: C - 2A = 0
    stoich = np.array([[-2, 1]])
    equilibrium = np.array([4.0])
    total = np.array([[1, 2]])

    params = EquilibriumParameters(
        stoich=stoich,
        equilibrium=equilibrium,
        total=total
    )

    # Initial state: [A]=1.0, [C]=0.0
    initial_state = ChemicalState(
        prim_aq_conc=np.array([1.0]), # A
        sec_conc=np.array([0.0]),     # C
        min_conc=np.array([])
    )

    new_state = params.solve_equilibrium(initial_state)

    expected_A = (-1 + np.sqrt(33)) / 16
    expected_C = (17 - np.sqrt(33)) / 32
    
    expected_prim_conc = np.array([expected_A])
    expected_sec_conc = np.array([expected_C])

    assert np.allclose(new_state.prim_aq_conc, expected_prim_conc)
    assert np.allclose(new_state.sec_conc, expected_sec_conc)

def test_solve_equilibrium_multiple_reactions():
    # System:
    # 1. A + B <=> C, K1 = 2
    # 2. C <=> D, K2 = 3
    # Reactions:
    # 1. C - A - B = 0
    # 2. D - C = 0
    stoich = np.array([
        [-1, -1, 1, 0],
        [0, 0, -1, 1]
    ])
    equilibrium = np.array([2.0, 3.0])
    total = np.array([
        [1, 0, 1, 1],
        [0, 1, 1, 1]
    ])

    params = EquilibriumParameters(
        stoich=stoich,
        equilibrium=equilibrium,
        total=total
    )

    # Initial state: [A]=1.0, [B]=1.0, [C]=0.0, [D]=0.0
    initial_state = ChemicalState(
        prim_aq_conc=np.array([1.0, 1.0]), # A, B
        sec_conc=np.array([0.0, 0.0]),     # C, D
        min_conc=np.array([])
    )

    new_state = params.solve_equilibrium(initial_state)

    expected_C = (17 - np.sqrt(33)) / 64
    expected_A = (-1 + np.sqrt(33)) / 16
    expected_B = expected_A
    expected_D = 3 * expected_C

    expected_prim_conc = np.array([expected_A, expected_B])
    expected_sec_conc = np.array([expected_C, expected_D])

    assert np.allclose(new_state.prim_aq_conc, expected_prim_conc)
    assert np.allclose(new_state.sec_conc, expected_sec_conc)
