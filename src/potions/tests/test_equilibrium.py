import numpy as np
import pandas as pd
import pytest
from potions.reactive_transport import EquilibriumParameters, ChemicalState


def test_solve_equilibrium_simple_pandas():
    # System: A <=> B with K = 2
    # Reaction: B - A = 0  => log(B) - log(A) = log(2)
    species = ["A", "B"]
    # primary_species = ["A"]
    secondary_species = ["B"]

    stoich_df = pd.DataFrame(
        [
            [-1, 1],
        ],
        index=secondary_species,
        columns=species,
    )

    equilibrium_series = pd.Series([2.0], index=secondary_species)
    log_k = np.log10(equilibrium_series)

    # Total A = [A] + [B]
    total_df = pd.DataFrame(
        [
            [1, 1],
        ],
        columns=species,
    )

    params = EquilibriumParameters(
        stoich=stoich_df, equilibrium=equilibrium_series, total=total_df
    )

    # Initial state: [A]=1.0, [B]=0.0
    initial_state = ChemicalState(
        prim_aq_conc=np.array([1.0]),  # A
        sec_conc=np.array([0.0]),  # B
        min_conc=np.array([]),
        exchange_conc=np.array([]),
    )

    initial_all_conc = np.concatenate(
        [initial_state.prim_aq_conc, initial_state.sec_conc]
    )
    c_tot = params.total.to_numpy() @ initial_all_conc

    new_state = params.solve_equilibrium(initial_state)
    new_all_conc = np.concatenate([new_state.prim_aq_conc, new_state.sec_conc])

    # Check mass balance
    final_c_tot = params.total.to_numpy() @ new_all_conc
    assert np.allclose(c_tot, final_c_tot)

    # Check equilibrium
    log_new_conc = np.log10(new_all_conc)
    iap = params.stoich.to_numpy() @ log_new_conc
    assert np.allclose(iap, log_k)

    # Check analytical solution
    # [B]/[A] = 2, [A]+[B]=1 => A=1/3, B=2/3
    assert np.allclose(new_state.prim_aq_conc, [1 / 3])
    assert np.allclose(new_state.sec_conc, [2 / 3])


def test_solve_equilibrium_carbonate_system():
    # Simplified carbonate system
    # Primary species: H+, HCO3-
    # Secondary species: OH-, H2CO3, CO3-2
    # Reactions:
    # 1. H+ + OH- <=> H2O (logK = -14)
    # 2. H2CO3 <=> H+ + HCO3- (logK = -6.3)
    # 3. HCO3- <=> H+ + CO3-2 (logK = -10.33)

    # Express secondary species in terms of primary species:
    # log(OH-) = -14 - log(H+)
    # log(H2CO3) = log(H+) + log(HCO3-) + 6.3
    # log(CO3-2) = log(HCO3-) - log(H+) - 10.33

    # Reactions for stoich matrix: sum(nu_i * log(C_i)) = logK
    # log(OH-) + log(H+) = -14
    # log(H2CO3) - log(H+) - log(HCO3-) = 6.3
    # log(CO3-2) + log(H+) - log(HCO3-) = -10.33

    species = ["H+", "HCO3-", "OH-", "H2CO3", "CO3-2"]
    secondary_species = ["OH-", "H2CO3", "CO3-2"]

    stoich_df = pd.DataFrame(
        [
            [1, 0, 1, 0, 0],  # OH-
            [-1, -1, 0, 1, 0],  # H2CO3
            [1, -1, 0, 0, 1],  # CO3-2
        ],
        index=secondary_species,
        columns=species,
    )

    log_k = pd.Series([-14, 6.3, -10.33], index=secondary_species)
    equilibrium_series = 10**log_k

    # Total concentration matrix
    # Rows are conserved quantities: Charge, Total Carbonate
    charges = pd.Series({"H+": 1, "HCO3-": -1, "OH-": -1, "H2CO3": 0, "CO3-2": -2})
    total_df = pd.DataFrame(
        [
            charges[species].values,
            [0, 1, 0, 1, 1],  # Tot_C: sum of species containing Carbon
        ],
        columns=species,
    )

    params = EquilibriumParameters(
        stoich=stoich_df, equilibrium=equilibrium_series, total=total_df
    )

    # Initial state
    initial_state = ChemicalState(
        prim_aq_conc=np.array([1e-7, 1e-3]),  # [H+], [HCO3-]
        sec_conc=np.array([1e-7, 1e-6, 1e-8]),  # [OH-], [H2CO3], [CO3-2]
        min_conc=np.array([]),
        exchange_conc=np.array([]),
    )

    initial_all_conc = np.concatenate(
        [initial_state.prim_aq_conc, initial_state.sec_conc]
    )
    c_tot = params.total.to_numpy() @ initial_all_conc

    new_state = params.solve_equilibrium(initial_state)

    new_all_conc = np.concatenate([new_state.prim_aq_conc, new_state.sec_conc])

    # 1. Check for mass balance
    final_c_tot = params.total.to_numpy() @ new_all_conc
    assert np.allclose(c_tot, final_c_tot)

    # 2. Check for equilibrium condition
    # Add a small epsilon to avoid log(0)
    log_new_conc = np.log10(new_all_conc + 1e-30)
    iap = params.stoich.to_numpy() @ log_new_conc
    assert np.allclose(iap, log_k, atol=1e-6)
