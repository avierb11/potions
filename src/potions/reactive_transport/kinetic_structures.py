from __future__ import annotations
from functools import partial
import logging
from typing import Final, Callable, Iterable
from dataclasses import dataclass
from pandas import DataFrame, Series
import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
from scipy.optimize import fsolve

from potions.core import RtForcing
from ..math import find_root_multi
from ..utils import DO_LOGGING, setup_logging

setup_logging("kinetic_structures.py")

# ==== Constants ==== #
PARAMETERS_PER_MINERAL: Final[int] = 5
# =================== #


class MonodParameters:
    """
    A class containing the Monod parameters for a reaction network in a reactive transport model.

    This class holds two matrices: `monod_mat` and `inhib_mat`, which are used to calculate the
    reaction rate using Monod kinetics. These matrices define how the water chemistry alters the
    reaction rate of solids in the model.

    The matrices contain values for all species involved in the reaction network, including primary,
    secondary (controlled by equilibrium), and mineral species. This allows the model to track all
    chemical states in a single vector, rather than managing them separately.

    The columns of both matrices represent all species in the reaction network in the order:
    Primary species, Secondary species (controlled by equilibrium), and Mineral species.

    Important Notes:
        - Values in both `monod_mat` and `inhib_mat` are only non-NaN if they affect the reaction rate.
          This ensures that only relevant species are considered in the calculation.
        - None of the values in either matrix can be zero. A zero value would cause numerical instability
          during the calculation of the reaction rate, leading to incorrect or undefined results.

    Attributes:
        monod_mat (DataFrame): A matrix of shape (number of minerals x number of total species)
                               representing the Monod kinetic parameters for each mineral and species.
                               The columns represent all species in the reaction network in the order:
                               Primary species, Secondary species (controlled by equilibrium),
                               and Mineral species.

        inhib_mat (DataFrame): A matrix of shape (number of minerals x number of total species)
                               representing the inhibition parameters for each mineral and species.
                               The columns follow the same order as `monod_mat`.
    """

    monod_mat: DataFrame
    inhib_mat: DataFrame
    monod_mat_np: np.ndarray
    inhib_mat_np: np.ndarray

    def __init__(self, monod_mat: DataFrame, inhib_mat: DataFrame):
        self.monod_mat = monod_mat
        self.inhib_mat = inhib_mat
        self.monod_mat_np = monod_mat.to_numpy()  # type: ignore
        self.inhib_mat_np = inhib_mat.to_numpy()  # type: ignore

    def rate(self, chms: np.ndarray) -> np.ndarray:
        """
        Calculate the rate of reaction using Monod kinetics based on the concentration of species.

        This method computes the reaction rate for each mineral in the network, taking into account
        the concentrations of all species involved in the reaction. The rate is a value between 0 and 1,
        which modulates the maximum rate of the reaction. This function is essential for modeling how
        the chemical state of the system influences the kinetics of mineral reactions.

        Parameters:
            conc (NDArray): A 1D numpy array of size equal to the number of species in the reaction
                            network. The order of species in this array must match the columns of
                            `monod_mat` and `inhib_mat`. Each entry in the array represents the
                            concentration of a species in the system.

        Returns:
            NDArray: A 1D numpy array of size equal to the number of minerals in the network.
                     Each entry corresponds to the reaction rate for a specific mineral, which is
                     modulated by the chemical state of the system.

        Example:
            >>> import numpy as np
            >>> params = MonodParameters()
            >>> conc = np.array([1.0, 2.0, 0.5])  # Concentrations of primary, secondary, and mineral species
            >>> rates = params.rate(conc)
            >>> print(rates)
            [0.6, 0.4]  # Example output: reaction rates for two minerals
        """

        n_minerals: int = self.monod_mat_np.shape[0]  # type: ignore
        n_species: int = self.monod_mat_np.shape[1]  # type: ignore
        monod = np.zeros(n_minerals, dtype=np.float64)  # type: ignore
        inhib = np.zeros(n_minerals, dtype=np.float64)  # type: ignore

        # Memoryviews
        monod_mv: np.ndarray = monod  # type: ignore
        inhib_mv: np.ndarray = inhib  # type: ignore
        chms_mv: np.ndarray = chms  # type: ignore

        i: int
        j: int
        val: float
        prod: float

        # --- Calculate Monod term ---
        for i in range(n_minerals):
            prod = 1.0
            for j in range(n_species):
                val = self.monod_mat_np[i, j]  # type: ignore
                if np.isfinite(val):
                    prod *= chms_mv[j] / (val + chms_mv[j])  # type: ignore
            monod_mv[i] = prod  # type: ignore

        # --- Calculate Inhibition term ---
        for i in range(n_minerals):
            prod = 1.0
            for j in range(n_species):
                val = self.inhib_mat_np[i, j]  # type: ignore
                if np.isfinite(val):
                    prod *= val / (val + chms_mv[j])  # type: ignore
            inhib_mv[i] = prod  # type: ignore

        return monod * inhib

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MonodParameters):
            raise TypeError(f"Cannot compare MonodParameters with '{type(other)}'")
        else:
            monod_arr_1 = self.monod_mat.to_numpy().flatten()
            monod_arr_2 = other.monod_mat.to_numpy().flatten()
            inhib_arr_1 = self.inhib_mat.to_numpy().flatten()
            inhib_arr_2 = other.inhib_mat.to_numpy().flatten()

            monod_mask_1 = np.isfinite(monod_arr_1)
            monod_mask_2 = np.isfinite(monod_arr_2)
            inhib_mask_1 = np.isfinite(inhib_arr_1)
            inhib_mask_2 = np.isfinite(inhib_arr_2)

            if not all(monod_mask_1 == monod_mask_2):
                return False

            if not all(inhib_mask_1 == inhib_mask_2):
                return False

            monod_are_equal = np.allclose(
                monod_arr_1[monod_mask_1], monod_arr_2[monod_mask_2]
            )
            inhib_are_equal = np.allclose(
                inhib_arr_1[inhib_mask_1], inhib_arr_2[inhib_mask_2]
            )

            return bool(monod_are_equal and inhib_are_equal)


class TstParameters:
    """
    The Transition-State Theory (TST) parameters, containing two matrices of shape
    (number of minerals x number of total species in the simulation).

    This class is used to define the parameters for a reaction network based on Transition-State Theory
    (TST), which is commonly used to model the dissolution kinetics of minerals in reactive transport models.
    The class contains matrices that define the stoichiometry and dependencies of the reactions, as well
    as equilibrium constants for the minerals involved.

    The matrices contain values for all species involved in the reaction network, including primary,
    secondary (controlled by equilibrium), and mineral species. This allows the model to track all
    chemical states in a single vector, rather than managing them separately.

    The columns of both matrices represent all species in the reaction network in the order:
    Primary species, Secondary species (controlled by equilibrium), and Mineral species.

    Important Notes:
        - Values in both `stoich` and `dep` matrices are only non-NaN if they affect the reaction rate.
          This ensures that only relevant species are considered in the calculation.
        - The `min_eq_const` vector contains equilibrium constants for the minerals, which are used
          to calculate the saturation index and determine the direction of the reaction.

    Attributes:
        stoich (DataFrame): A matrix of shape (number of minerals x number of total species)
                           representing the stoichiometric coefficients for each mineral and species.
                           The columns represent all species in the reaction network in the order:
                           Primary species, Secondary species (controlled by equilibrium),
                           and Mineral species.

        dep (DataFrame): A matrix of shape (number of minerals x number of total species)
                        representing the dependency coefficients for each mineral and species.
                        The columns follow the same order as `stoich`.

        min_eq_const (Series): A 1D vector of equilibrium constants for each mineral in the network.
                              These constants are used to determine the saturation index and the
                              direction of the reaction.
    """

    # Python types
    stoich: DataFrame  # Stoichiometry of the mineral reactions
    dep: DataFrame  # Dependence of the rate law on other species concentrations (typically pH through H+ or OH- dependence)
    min_eq_const: Series  # Vector of equilibrium constants

    # Cython types
    stoich_np: np.ndarray
    dep_np: np.ndarray
    min_eq_const_np: np.ndarray

    def __init__(self, stoich: DataFrame, dep: DataFrame, min_eq_const: Series):
        # Python types
        self.stoich = stoich
        self.dep = dep
        self.min_eq_const = min_eq_const

        # Cython types
        self.stoich_np = stoich.to_numpy()  # type: ignore
        self.dep_np = dep.to_numpy()  # type: ignore
        self.min_eq_const_np = min_eq_const.to_numpy()  # type: ignore

    def rate(self, chms_in: np.ndarray) -> np.ndarray:
        """
        Calculate the rate of reaction using Transition-State Theory (TST) kinetics.

        This method computes the reaction rate for each mineral in the network, based on the
        concentrations of all species involved in the reaction. The rate is determined by the
        difference between the actual activity product (IAP) and the equilibrium constant (K),
        and is used to model dissolution-only reactions for minerals.

        Parameters:
            all_species_conc (NDArray): A 1D numpy array of size equal to the number of species
                                        in the reaction network. The order of species in this array
                                        must match the columns of `stoich` and `dep`. Each entry
                                        represents the concentration of a species in the system.

        Returns:
            NDArray: A 1D numpy array of size equal to the number of minerals in the network.
                        Each entry corresponds to the reaction rate for a specific mineral, based on
                        the difference between the actual activity product and the equilibrium constant.

        Example:
            >>> import numpy as np
            >>> params = TstParameters()
            >>> conc = np.array([1.0, 2.0, 0.5])  # Concentrations of primary, secondary, and mineral species
            >>> rates = params.rate(conc)
            >>> print(rates)
            [0.2, 0.1]  # Example output: reaction rates for two minerals
        """
        n_minerals: int = self.stoich_np.shape[0]  # type: ignore
        n_species: int = self.stoich_np.shape[1]  # type: ignore
        i: int
        j: int

        # --- 2. Allocate output arrays and create memoryviews for fast access ---
        rate_out: np.ndarray = np.empty(n_minerals, dtype=np.float64)
        log_conc_arr: np.ndarray = np.empty(n_species, dtype=np.float64)

        chms_mv: np.ndarray = chms_in  # type: ignore

        # --- 3. Pre-calculate log10 of concentrations in a C loop ---
        # Replaces: log_conc: NDArray = np.log10(chms)
        for j in range(n_species):
            log_conc_arr[j] = np.log10(chms_mv[j])  # type: ignore

        # --- 4. Main loop over each mineral ---
        log_dep_i: float
        log_iap_i: float
        dep_val: float
        stoich_val: float

        for i in range(n_minerals):
            log_dep_i = 0.0
            log_iap_i = 0.0

            # --- 5. Inner loop to calculate dot products (matrix-vector multiply) ---
            # Replaces: log_dep = self.dep_np @ log_conc and log_iap = self.stoich_np @ log_conc
            for j in range(n_species):
                dep_val = self.dep_np[i, j]  # type: ignore
                if np.isfinite(dep_val):
                    log_dep_i += dep_val * log_conc_arr[j]  # type: ignore

                stoich_val = self.stoich_np[i, j]  # type: ignore
                if np.isfinite(stoich_val):
                    log_iap_i += stoich_val * log_conc_arr[j]  # type: ignore

            # --- 6. Calculate the final rate for the current mineral ---
            # Replaces: dep = 10**log_dep, iap = 10**log_iap, and the final calculation
            rate_out[i] = pow(10.0, log_dep_i) * (  # type: ignore
                1.0 - pow(10.0, log_iap_i) / self.min_eq_const_np[i]  # type: ignore
            )

        return rate_out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TstParameters):
            raise TypeError(f"Cannot compare TstParameters with '{type(other)}'")
        else:
            stoich_diff = self.stoich - other.stoich
            dep_diff = self.dep - other.dep
            min_eq_diff = self.min_eq_const - other.min_eq_const

            stoich_max_diff: float = stoich_diff.abs().max().max()
            dep_max_diff: float = dep_diff.abs().max().max()
            eq_max_diff: float = min_eq_diff.abs().max()

            return bool(
                stoich_max_diff < 1e-12 and dep_max_diff < 1e-12 and eq_max_diff < 1e-12
            )


class EquilibriumParameters:
    """
    Container class for equilibrium parameters used in aqueous chemical equilibrium calculations.

    This class stores all necessary parameters for solving equilibrium concentrations
    in aqueous chemical reaction networks. It handles stoichiometric relationships,
    equilibrium constants, and mass/charge balance constraints for chemical species.

    The equilibrium parameters are used to calculate steady-state concentrations
    of chemical species in aqueous solutions, taking into account thermodynamic
    properties and reaction stoichiometry.

    Attributes
    ----------
    stoich : pandas.DataFrame
        Matrix describing the stoichiometry of the secondary species. Each row
        represents a constraint (mass balance, charge balance) and each column
        represents a secondary species. The matrix defines the relationship
        between species concentrations in the equilibrium system.
    log_eq_consts : pandas.Series
        Vector of the equilibrium constants for the secondary species in log10 scale.
        These constants determine the ratio of product concentrations to reactant
        concentrations at equilibrium, expressed in base-10 logarithmic form for
        numerical stability.
    total : pandas.DataFrame
        Matrix describing the mass and charge balance of the species. This matrix
        defines how the total concentrations of primary species relate to the
        concentrations of secondary species in the system.

    Examples
    --------
    >>> # Create equilibrium parameters from species data
    >>> params = EquilibriumParameters.from_species(
    ...     species=species_series,
    ...     primary=primary_species_list,
    ...     secondary=secondary_species_list,
    ...     exchange=exchange_reactions_list
    ... )

    >>> # Solve equilibrium concentrations
    >>> concentrations = params.solve_equilibrium(concentrations_vector)

    Notes
    -----
    The EquilibriumParameters class is designed for aqueous chemical systems where
    the equilibrium is determined by:

    1. Stoichiometric relationships between species
    2. Equilibrium constants for reactions
    3. Mass and charge conservation constraints

    The logarithmic representation of equilibrium constants (log10 scale) provides
    numerical stability for calculations involving very large or very small constants.

    The class uses linear algebra operations to solve the constrained equilibrium
    problem, where:

    - The stoichiometry matrix defines the relationship between species
    - The null space of stoichiometry provides the degrees of freedom
    - The particular solution gives the equilibrium concentrations for a reference
    - Total concentrations are used to solve for the actual equilibrium distribution
    """

    # Pandas types
    stoich: DataFrame  # Matrix describing the stoichiometry of the secondary species
    log_eq_consts: Series  # Vector of the equilibrium constants for the secondary species in log10 scale
    total: DataFrame  # Matrix describing the mass and charge balance of the species
    stoich_null_space: NDArray
    log10_k_w: NDArray
    x_particular: NDArray

    # C types
    stoich_np: np.ndarray
    log_eq_consts_np: np.ndarray
    total_np: np.ndarray
    stoich_null_space_np: np.ndarray
    log10_k_w_np: np.ndarray
    x_particular_np: np.ndarray

    def __init__(self, stoich: DataFrame, log_eq_consts: Series, total: DataFrame):
        # Solve the python versions
        self.stoich = stoich
        self.log_eq_consts = log_eq_consts
        self.total = total
        self.stoich_null_space = la.null_space(self.stoich.values)
        self.log10_k_w = self.log_eq_consts.to_numpy()
        self.x_particular = la.pinv(self.stoich) @ self.log10_k_w

        # Create the stoichiometry matrices
        self.stoich_np = stoich.to_numpy()  # type: ignore
        self.log_eq_consts_np = log_eq_consts.to_numpy()  # type: ignore
        self.total_np = total.to_numpy()  # type: ignore
        self.stoich_null_space_np = self.stoich_null_space  # type: ignore
        self.log10_k_w_np = self.log10_k_w  # type: ignore
        self.x_particular_np = self.x_particular  # type: ignore

    def conc_func(self, x_free: np.ndarray) -> np.ndarray:
        return 10.0 ** (self.stoich_null_space_np @ x_free + self.x_particular_np)

    def residual_func(self, x_free: np.ndarray, c_tot: np.ndarray) -> np.ndarray:
        return c_tot - self.total_np @ self.conc_func(x_free)

    def solve_equilibrium(self, chms: np.ndarray, verbose: bool = False) -> NDArray:
        """
        Solve for the equilibrium concentrations of all of the species.

        This method calculates the equilibrium concentrations of all chemical
        species in the system given the total concentrations of primary species.

        Parameters
        ----------
        chms : numpy.ndarray
            Vector of total concentrations of primary species in the system.
            These represent the initial conditions for the equilibrium calculation.
        debug : bool, optional
            If True, print debug information including the total concentrations
            and intermediate calculation steps (default: False)

        Returns
        -------
        numpy.ndarray
            Vector of equilibrium concentrations for all secondary species in
            logarithmic scale (base-10). Each element represents the log10 of the
            equilibrium concentration of the corresponding species.

        Notes
        -----
        The equilibrium calculation uses a root-finding algorithm to solve the
        system of equations that represent the mass balance constraints and
        equilibrium relationships. The method:

        1. Computes total concentrations from primary species concentrations
        2. Defines a function that relates total concentrations to equilibrium
           concentrations
        3. Uses a multi-dimensional root solver to find the equilibrium solution
        4. Returns the logarithmic equilibrium concentrations

        The solution is based on the relationship:
        c_eq = 10^(null_space @ tot_conc + x_particular)
        where c_eq represents equilibrium concentrations and tot_conc represents
        total concentrations.
        """
        c_tot: np.ndarray = self.total_np @ chms

        # Set the charge balance value to zero (in case there is an error)
        if "Charge" in self.total.index:
            charge_ind: int = self.total.index.tolist().index("Charge")  # type: ignore
            c_tot[charge_ind] = 0.0

        if verbose:
            print(f"{c_tot=}")

        # Create a callable for the root-finder with c_tot "frozen"
        f_to_solve = partial(self.residual_func, c_tot=c_tot)

        # After some testing, it seems like having a standard initial guess of zero is generally
        # good across a range of concentrations for faster convergence.
        initial_guess = np.zeros_like(c_tot)

        try:
            sol: np.ndarray = find_root_multi(f_to_solve, initial_guess)
        except Exception as _e:
            if DO_LOGGING:
                logging.warning(
                    f"Equilibrium speciation error (find_root_multi): {','.join(chms.astype(str))}"
                )
            try:
                sol = fsolve(f_to_solve, initial_guess)
            except Exception as e2:
                if DO_LOGGING:
                    logging.warning(
                        f"Equilibrium speciation error (fsolve): {','.join(chms.astype(str))}"
                    )
                raise e2

        new_conc: np.ndarray = self.conc_func(sol)

        return new_conc

    def get_equilibrium_residual_function(
        self, chms: NDArray
    ) -> Callable[[NDArray], NDArray]:
        """
        Get the function that describes the residual for this zone
        """
        c_tot: Final[NDArray] = (
            self.total.values @ chms
        )  # Total concentrations in the system

        # Set the charge balance value to zero (in case there is an error)
        if "Charge" in self.total.index:
            charge_ind: int = self.total.index.tolist().index("Charge")  # type: ignore
            c_tot[charge_ind] = 0.0

        def conc(tot_conc: NDArray) -> NDArray:
            """
            Return a vector of the aqueous concentrations in base-10 logarithm.

            Parameters
            ----------
            tot_conc : numpy.ndarray
                Vector of total concentrations

            Returns
            -------
            numpy.ndarray
                Vector of logarithmic aqueous concentrations
            """
            return 10 ** (self.stoich_null_space @ tot_conc + self.x_particular)

        def f(tot_conc: NDArray) -> NDArray:
            return c_tot - self.total.values @ conc(tot_conc)

        return f

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EquilibriumParameters):
            raise TypeError(
                f"Cannot compare EquilibriumParameters with '{type(other)}'"
            )
        else:
            stoich_diff = self.stoich - other.stoich
            total_diff = self.total - other.total
            eq_diff = self.log_eq_consts - other.log_eq_consts

            stoich_max_diff: float = stoich_diff.abs().max().max()
            total_max_diff: float = total_diff.abs().max().max()
            eq_max_diff: float = eq_diff.abs().max()

            return bool(
                stoich_max_diff < 1e-12
                and total_max_diff < 1e-12
                and eq_max_diff < 1e-12
            )


@dataclass(frozen=True)
class MineralAuxParams:
    sw_threshold: float
    sw_exp: float
    n_alpha: float
    q_10: float
    ssa: float  # Specific surface area in units of g/mol

    def to_array(self) -> NDArray:
        return np.array(
            [
                self.sw_threshold,
                self.sw_exp,
                self.n_alpha,
                self.q_10,
                self.ssa,
            ]
        )

    @staticmethod
    def from_array(arr: NDArray) -> MineralAuxParams:
        return MineralAuxParams(
            sw_threshold=arr[0],
            sw_exp=arr[1],
            n_alpha=arr[2],
            q_10=arr[3],
            ssa=arr[4],
        )


@dataclass(frozen=True)
class ZoneDimensions:
    porosity: float
    depth: float
    passive_water_storage: float

    def to_array(self) -> NDArray:
        return np.array([self.porosity, self.depth, self.passive_water_storage])

    @staticmethod
    def from_array(arr: NDArray) -> ZoneDimensions:
        """Conver this object from an array into a ZoneDimensions object"""
        return ZoneDimensions(arr[0], arr[1], arr[2])

    @property
    def max_water_volume(self) -> float:
        """The maximum water storage that can exist in this zone"""
        return self.porosity * self.depth - self.passive_water_storage

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZoneDimensions):
            raise TypeError(f"Cannot compare ZoneDimensions with {type(other)}")
        else:
            return np.allclose(self.to_array(), other.to_array())

    def __ne__(self, other: object) -> bool:
        return not (self == other)


class MineralParameters:
    sw_threshold: NDArray  # The soil water threshold
    sw_exp: NDArray  # The soil water exponent
    n_alpha: NDArray  # The water table depth factor
    q_10: NDArray  # The temperature factor range
    ssa: NDArray  # The specific surface area in units of g/mol

    sw_threshold_cy: np.ndarray
    sw_exp_cy: np.ndarray
    n_alpha_cy: np.ndarray
    q_10_cy: np.ndarray
    ssa_cy: np.ndarray

    def __init__(
        self,
        sw_threshold: np.ndarray,
        sw_exp: np.ndarray,
        n_alpha: np.ndarray,
        q_10: np.ndarray,
        ssa: np.ndarray,
    ):
        self.sw_threshold = sw_threshold
        self.sw_exp = sw_exp
        self.n_alpha = n_alpha
        self.q_10 = q_10
        self.ssa = ssa

        # Cython types
        self.sw_threshold_cy = sw_threshold  # type: ignore
        self.sw_exp_cy = sw_exp  # type: ignore
        self.n_alpha_cy = n_alpha  # type: ignore
        self.q_10_cy = q_10  # type: ignore
        self.ssa_cy = ssa  # type: ignore

    def to_array(self) -> NDArray:
        combined_arr = np.vstack(
            [
                self.sw_threshold,
                self.sw_exp,
                self.n_alpha,
                self.q_10,
                self.ssa,
            ]
        )

        return combined_arr.T.flatten()

    @staticmethod
    def from_array(arr: NDArray) -> MineralParameters:
        if arr.size % PARAMETERS_PER_MINERAL != 0:
            raise ValueError(
                f"Array passed to `MineralParameters.from_array` must have size that is as multiple of 6, not {arr.size}"
            )
        num_minerals = arr.size // PARAMETERS_PER_MINERAL

        param_arr = arr.reshape((num_minerals, PARAMETERS_PER_MINERAL)).T

        return MineralParameters(
            sw_threshold=param_arr[0],
            sw_exp=param_arr[1],
            n_alpha=param_arr[2],
            q_10=param_arr[3],
            ssa=param_arr[4],
        )

    def soil_water_factor(self, forc: RtForcing) -> np.ndarray:
        """
        Calculate the dependence on the soil moisture
        """
        n_minerals: int = self.sw_threshold_cy.shape[0]  # type: ignore
        arr_out: np.ndarray = np.empty(n_minerals, dtype=np.float64)
        arr_mv: np.ndarray = arr_out  # type: ignore
        i: int

        # This loop replaces the NumPy boolean masking for better performance
        for i in range(n_minerals):
            if forc.s_w >= self.sw_threshold_cy[i]:  # type: ignore
                arr_mv[i] = pow(  # type: ignore
                    (1.0 - forc.s_w) / (1.0 - self.sw_threshold_cy[i]),  # type: ignore
                    self.sw_exp_cy[i],  # type: ignore
                )
            else:
                arr_mv[i] = pow(forc.s_w / self.sw_threshold_cy[i], self.sw_exp_cy[i])  # type: ignore
        return arr_out

    def temperature_factor(self, forc: RtForcing) -> np.ndarray:
        """
        Calculate the dependence on temperature
        """
        n_minerals: int = self.q_10_cy.shape[0]  # type: ignore
        arr_out: np.ndarray = np.empty(n_minerals, dtype=np.float64)
        arr_mv: np.ndarray = arr_out  # type: ignore
        i: int

        # This loop replaces the vectorized NumPy power operation
        for i in range(n_minerals):
            arr_mv[i] = pow(self.q_10_cy[i], (forc.hydro_forc.temp - 20.0) / 10.0)  # type: ignore
        return arr_out

    def water_table_factor(self, forc: RtForcing) -> np.ndarray:
        """
        Calculate the dependence on the water table
        """
        n_minerals: int = self.n_alpha_cy.shape[0]  # type: ignore
        arr_out: np.ndarray = np.ones(n_minerals, dtype=np.float64)  # type: ignore
        arr_mv: np.ndarray = arr_out  # type: ignore
        i: int
        n_alpha_i: float

        # This loop is the C-optimized version of your original loop
        for i in range(n_minerals):
            n_alpha_i = self.n_alpha_cy[i]  # type: ignore
            if n_alpha_i != 0.0:
                arr_mv[i] = np.exp(  # type: ignore
                    -abs(n_alpha_i) * pow(forc.z_w, (n_alpha_i / abs(n_alpha_i)))  # type: ignore
                )
        return arr_out

    def factor(self, forc: RtForcing) -> np.ndarray:
        """
        Calculate the auxiliary factor for each function.

        This optimized version combines the logic from soil_water_factor,
        temperature_factor, and water_table_factor into a single loop
        to avoid creating intermediate arrays.
        """
        n_minerals: int = self.ssa_cy.shape[0]  # type: ignore
        factor_out: np.ndarray = np.empty(n_minerals, dtype=np.float64)
        factor_mv: np.ndarray = factor_out  # type: ignore

        i: int
        sw_factor: float
        temp_factor: float
        wt_factor: float
        n_alpha_i: float

        for i in range(n_minerals):
            # --- 1. soil_water_factor logic ---
            if forc.s_w >= self.sw_threshold_cy[i]:  # type: ignore
                sw_factor = pow(
                    (1.0 - forc.s_w) / (1.0 - self.sw_threshold_cy[i]),  # type: ignore
                    self.sw_exp_cy[i],  # type: ignore
                )
            else:
                sw_factor = pow(
                    forc.s_w / self.sw_threshold_cy[i], self.sw_exp_cy[i]  # type: ignore
                )

            # --- 2. temperature_factor logic ---
            temp_factor = pow(
                self.q_10_cy[i], (forc.hydro_forc.temp - 20.0) / 10.0  # type: ignore
            )

            # --- 3. water_table_factor logic ---
            n_alpha_i = self.n_alpha_cy[i]  # type: ignore
            if n_alpha_i == 0.0:
                wt_factor = 1.0
            else:
                wt_factor = np.exp(
                    -abs(n_alpha_i) * pow(forc.z_w, n_alpha_i / abs(n_alpha_i))
                )

            # --- 4. Combine all factors ---
            factor_mv[i] = sw_factor * temp_factor * wt_factor  # type: ignore

        return factor_out

    @staticmethod
    def from_mineral_parameters(
        minerals: Iterable[MineralAuxParams],
    ) -> MineralParameters:
        """Take a list of mineral parameters and convert them into an aggregation"""
        sw_thrs: list[float] = []
        sw_exps: list[float] = []
        n_alphas: list[float] = []
        q_10s: list[float] = []
        ssas: list[float] = []

        for m in minerals:
            sw_thrs.append(m.sw_threshold)
            sw_exps.append(m.sw_exp)
            n_alphas.append(m.n_alpha)
            q_10s.append(m.q_10)
            ssas.append(m.ssa)
        return MineralParameters(
            sw_threshold=np.array(sw_thrs, dtype=np.float64),
            sw_exp=np.array(sw_exps, dtype=np.float64),
            n_alpha=np.array(n_alphas, dtype=np.float64),
            q_10=np.array(q_10s, dtype=np.float64),
            ssa=np.array(ssas, dtype=np.float64),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MineralParameters):
            raise TypeError(f"Cannot compare MineralParameters with {type(other)}")
        else:
            vals = [
                np.allclose(self.sw_threshold, other.sw_threshold),
                np.allclose(self.sw_exp, other.sw_exp),
                np.allclose(self.n_alpha, other.n_alpha),
                np.allclose(self.q_10, other.q_10),
                np.allclose(self.ssa, other.ssa),
            ]
            return all(vals)

    def __ne__(self, other: object) -> bool:
        return not (self == other)


@dataclass(frozen=True)
class RtParameters:
    dimensions: ZoneDimensions
    mineral_params: MineralParameters

    def to_array(self) -> NDArray:
        """Convert this object to an array"""
        params_list: list[NDArray] = [self.dimensions.to_array()] + [
            self.mineral_params.to_array()
        ]

        return np.concat(params_list)

    @staticmethod
    def from_array(arr: NDArray) -> RtParameters:
        """Convert this object from a numpy array into a set of parameters"""
        size_params: NDArray
        mineral_params: NDArray
        size_params, mineral_params = arr[0:3], arr[3:]

        return RtParameters(
            dimensions=ZoneDimensions.from_array(size_params),
            mineral_params=MineralParameters.from_array(mineral_params),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RtParameters):
            raise TypeError(f"Cannot compare RtParameters with {type(other)}")
        else:
            return (
                self.dimensions == other.dimensions
                and self.mineral_params == other.mineral_params
            )
