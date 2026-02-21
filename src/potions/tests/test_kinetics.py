import numpy as np
from pandas import DataFrame
from ..reaction_network import MonodParameters


def test_monod_rate_simple_case() -> None:
    """
    Test the Monod rate function with a simple case.
    Reaction: A (mineral) + B (aqueous) -> products
    Monod matrix: [[1.0], [np.nan]] (for species B and A with mineral A)
    Inhibition matrix: [[np.nan], [np.nan]]
    Concentrations: [2.0] (aqueous B) and [1.0] (mineral A)
    Expected rate: (2.0 / (1.0 + 2.0)) = 0.666...
    """
    minerals = ["A"]  # Mineral species
    aqueous_species = ["B"]  # Aqueous species
    species = aqueous_species + minerals  # Total species list: ["B", "A"]

    # Monod matrix: shape (2 species x 1 mineral)
    # Row for "B" (aqueous): 1.0 (Monod parameter for species B affecting mineral A)
    # Row for "A" (mineral): np.nan (minerals don't have Monod parameters)
    monod_arr = np.array([[1.0, np.nan]])

    # Inhibition matrix: shape (2 species x 1 mineral)
    # Row for "B" (aqueous): np.nan (no inhibition)
    # Row for "A" (mineral): np.nan (minerals don't have inhibition parameters)
    inhib_arr = np.array([[np.nan, np.nan]])

    monod_params = MonodParameters(
        monod_mat=DataFrame(monod_arr, index=minerals, columns=species),
        inhib_mat=DataFrame(inhib_arr, index=minerals, columns=species),
    )

    # Concentrations: [aqueous B, mineral A]
    conc = np.array([2.0, 1.0])  # [aqueous, minerals] as required

    rate = monod_params.rate(conc)

    assert np.isclose(
        rate[0], 0.666666666666666666666
    ), f"Expected rate of 0.666..., got {rate[0]}"


def test_monod_rate_with_inhibition() -> None:
    """
    Test the Monod rate function with inhibition.
    Reaction: A (mineral) + B (aqueous) -> products
    Monod matrix: [[1.0], [np.nan]] (for species B and A with mineral A)
    Inhibition matrix: [[2.0], [np.nan]] (for species B and A with mineral A)
    Concentrations: [2.0] (aqueous B) and [1.0] (mineral A)
    Expected rate: (2.0 / (1.0 + 2.0)) * (2.0 / (2.0 + 2.0)) = 0.666... * 0.5 = 0.333...
    """
    minerals = ["A"]  # Mineral species
    aqueous_species = ["B"]  # Aqueous species
    species = aqueous_species + minerals  # Total species list: ["B", "A"]

    # Monod matrix: shape (2 species x 1 mineral)
    # For species "B" (aqueous) and mineral "A": 1.0 (this is the Monod parameter)
    # For species "A" (mineral) and mineral "A": np.nan (minerals don't have Monod parameters)
    monod_arr = np.array([[1.0, np.nan]])

    # Inhibition matrix: shape (2 species x 1 mineral)
    # For species "B" (aqueous) and mineral "A": 2.0 (this is the inhibition parameter)
    # For species "A" (mineral) and mineral "A": np.nan (minerals don't have inhibition parameters)
    inhib_arr = np.array([[2.0, np.nan]])

    monod_params = MonodParameters(
        monod_mat=DataFrame(monod_arr, index=minerals, columns=species),
        inhib_mat=DataFrame(inhib_arr, index=minerals, columns=species),
    )

    # Concentrations: [aqueous B, mineral A]
    conc = np.array([2.0, 1.0])  # [aqueous, minerals] as required

    rate = monod_params.rate(conc)

    assert np.isclose(
        rate[0], 0.3333333333333333
    ), f"Expected rate of 0.333..., got {rate[0]}"


def test_monod_rate_multiple_minerals() -> None:
    """
    Test the Monod rate function with multiple minerals.
    Reaction: A (mineral) + B (mineral) + C (aqueous) -> products
    Monod matrix: [[1.0, 1.0], [np.nan, np.nan], [np.nan, np.nan]] (for species C, A, B with minerals A, B)
    Inhibition matrix: [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]] (no inhibition)
    Concentrations: [2.0] (aqueous C), [1.0] (mineral A), [1.0] (mineral B)
    Expected rate: (2.0 / (1.0 + 2.0)) = 0.666...
    """
    minerals = ["A", "B"]  # Mineral species
    aqueous_species = ["C"]  # Aqueous species
    species = aqueous_species + minerals  # Total species list: ["C", "A", "B"]

    # Monod matrix: shape (3 species x 2 minerals)
    # For species "C" (aqueous) and minerals "A" and "B": 1.0 (this is the Monod parameter)
    # For species "A" and "B" (minerals) and minerals "A" and "B": np.nan (minerals don't have Monod parameters)
    monod_arr = np.array([[1.0, np.nan, np.nan], [np.nan, 0.5, np.nan]])

    # Inhibition matrix: shape (3 species x 2 minerals)
    # For species "C", "A", "B" and minerals "A" and "B": np.nan (no inhibition)
    inhib_arr = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])

    monod_params = MonodParameters(
        monod_mat=DataFrame(monod_arr, index=minerals, columns=species),
        inhib_mat=DataFrame(inhib_arr, index=minerals, columns=species),
    )

    # Concentrations: [aqueous C, mineral A, mineral B]
    conc = np.array([2.0, 1.0, 1.0])  # [aqueous, minerals] as required

    rate = monod_params.rate(conc)

    assert np.isclose(
        rate[0], 0.6666666666666666
    ), f"Expected rate of 0.666..., got {rate[0]}"
