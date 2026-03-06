from __future__ import annotations
import os
from typing import Optional, Final, Callable, Iterable
from dataclasses import dataclass
from pandas import DataFrame, Series
import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
from scipy.optimize import fsolve


from ..common_types import RtForcing, Vector
from .database import PrimaryAqueousSpecies, SecondarySpecies, ExchangeReaction
from ..utils import find_root_multi


# ==== Constants ==== #
PARAMETERS_PER_MINERAL: Final[int] = 5
# =================== #


@dataclass(frozen=True)
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

    def rate(self, chms: NDArray) -> NDArray:
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
        monod = np.zeros(self.monod_mat.shape[0], dtype=np.float64)
        inhib = np.zeros(self.inhib_mat.shape[0], dtype=np.float64)

        for i, row in enumerate(self.monod_mat.values):
            monod[i] = np.nanprod(chms / (row + chms))

        for i, row in enumerate(self.inhib_mat.values):
            inhib[i] = np.nanprod(row / (row + chms))

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


@dataclass(frozen=True)
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

    stoich: DataFrame  # Stoichiometry of the mineral reactions
    dep: DataFrame  # Dependence of the rate law on other species concentrations (typically pH through H+ or OH- dependence)
    min_eq_const: Series  # Vector of equilibrium constants

    def rate(self, chms: NDArray) -> NDArray:
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
        log_conc: NDArray = np.log10(chms)

        log_dep: NDArray = self.dep.values @ log_conc
        log_iap: NDArray = self.stoich.values @ log_conc

        dep: NDArray = 10**log_dep
        iap: NDArray = 10**log_iap

        return dep * (1.0 - iap / self.min_eq_const)

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


@dataclass(frozen=True)
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

    stoich: DataFrame  # Matrix describing the stoichiometry of the secondary species
    log_eq_consts: Series  # Vector of the equilibrium constants for the secondary species in log10 scale
    total: DataFrame  # Matrix describing the mass and charge balance of the species

    @staticmethod
    def from_species(
        species: Series,
        primary: list[PrimaryAqueousSpecies],
        secondary: list[SecondarySpecies],
        exchange: list[ExchangeReaction],
    ) -> EquilibriumParameters:
        """
        Construct the equilibrium parameters from the database species information.

        This static method creates an EquilibriumParameters object by processing
        the database information about primary and secondary species, as well as
        exchange reactions to build the stoichiometric matrices and equilibrium
        constant vectors needed for equilibrium calculations.

        Parameters
        ----------
        species : pandas.Series
            Series containing species information from the database
        primary : list of PrimaryAqueousSpecies
            List of primary aqueous species in the system
        secondary : list of SecondarySpecies
            List of secondary species formed from primary species
        exchange : list of ExchangeReaction
            List of exchange reactions that define the relationships between
            species in the system

        Returns
        -------
        EquilibriumParameters
            Initialized EquilibriumParameters object with stoichiometry and
            equilibrium constants set up for equilibrium calculations

        Raises
        ------
        NotImplementedError
            This method is not yet implemented and raises an exception when called

        Notes
        -----
        This method serves as the factory constructor for the EquilibriumParameters
        class. It processes the database information to build the mathematical
        framework needed for solving aqueous chemical equilibrium problems.
        """
        raise NotImplementedError()

    @property
    def stoich_null_space(self) -> NDArray:
        """
        Return the null space of the stoichiometry matrix.

        The null space represents the degrees of freedom in the equilibrium system.
        Any vector in the null space corresponds to a valid set of concentrations
        that satisfy the stoichiometric constraints.

        Returns
        -------
        numpy.ndarray
            Matrix whose columns form a basis for the null space of the stoichiometry
            matrix. Each column represents an independent degree of freedom in the
            equilibrium system.

        Notes
        -----
        The null space is computed using linear algebra operations and provides
        the mathematical foundation for solving the equilibrium problem. It
        represents the space of all possible concentration distributions that
        satisfy the stoichiometric constraints.
        """
        return la.null_space(self.stoich)

    @property
    def log10_k_w(self) -> NDArray:
        """
        Return a vector of the equilibrium constants in base-10 logarithm.

        This property provides the equilibrium constants in logarithmic scale,
        which is more numerically stable for calculations involving very large
        or very small equilibrium constants.

        Returns
        -------
        numpy.ndarray
            Vector of equilibrium constants in base-10 logarithmic scale,
            corresponding to the secondary species in the system.

        Notes
        -----
        The logarithmic representation is used throughout the equilibrium
        calculations to avoid numerical overflow or underflow issues that
        could occur with direct exponential representations of equilibrium
        constants.
        """
        return self.log_eq_consts.to_numpy()

    @property
    def x_particular(self) -> NDArray:
        """
        Return a vector of the particular solution of the null space of the stoichiometry.

        This particular solution represents a specific equilibrium concentration
        distribution that satisfies the stoichiometric constraints. It is used
        as a reference point in the equilibrium calculation.

        Returns
        -------
        numpy.ndarray
            Vector representing a particular solution to the equilibrium problem,
            derived from the pseudo-inverse of the stoichiometry matrix and the
            logarithmic equilibrium constants.

        Notes
        -----
        The particular solution is computed as: x_particular = pinv(stoich) @ log10_k_w
        This provides a reference equilibrium state that can be combined with
        the null space to find all possible equilibrium solutions.
        """
        return la.pinv(self.stoich) @ self.log10_k_w

    def solve_equilibrium(
        self, chms: NDArray, debug: bool = False, failed_dir: Optional[str] = None
    ) -> NDArray:
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
        c_tot: Final[NDArray] = (
            self.total.values @ chms
        )  # Total concentrations in the system

        # Set the charge balance value to zero (in case there is an error)
        if "Charge" in self.total.index:
            charge_ind: int = self.total.index.tolist().index("Charge")  # type: ignore
            c_tot[charge_ind] = 0.0

        if debug:
            print(f"{c_tot=}")

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

        try:
            sol = find_root_multi(f, c_tot)
        except Exception as _e:
            print("Failed to find root with `find_root_multi`, trying my `fsolve`")
            if failed_dir is not None:
                os.makedirs(failed_dir, exist_ok=True)
                my_function_path: str = os.path.join(
                    failed_dir, "my_function_failed.txt"
                )
                if os.path.exists(my_function_path):
                    file = open(my_function_path, "a")
                else:
                    file = open(my_function_path, "a+")
                file.write(",".join(map(str, chms.tolist())) + "\n")
            try:
                sol = fsolve(f, c_tot)
            except Exception as e2:
                print("Failed to find root with `fsolve` :(")
                if failed_dir is not None:
                    os.makedirs(failed_dir, exist_ok=True)
                    fsolve_path: str = os.path.join(failed_dir, "fsolve_failed.txt")
                    if os.path.exists(fsolve_path):
                        file = open(fsolve_path, "a")
                    else:
                        file = open(fsolve_path, "a+")
                    file.write(",".join(map(str, chms.tolist())) + "\n")
                raise e2

        new_conc: NDArray = conc(sol)

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


@dataclass(frozen=True)
class MineralParameters:
    sw_threshold: NDArray  # The soil water threshold
    sw_exp: NDArray  # The soil water exponent
    n_alpha: NDArray  # The water table depth factor
    q_10: NDArray  # The temperature factor range
    ssa: NDArray  # The specific surface area in units of g/mol

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

    def soil_water_factor(self, forc: RtForcing) -> Vector:
        """
        Calculate the dependence on the soil moisture
        """
        greater_mask: NDArray = forc.s_w >= self.sw_threshold
        less_mask: NDArray = ~greater_mask

        less: NDArray = (forc.s_w / self.sw_threshold) ** self.sw_exp
        greater: NDArray = ((1 - forc.s_w) / (1 - self.sw_threshold)) ** self.sw_exp

        arr: NDArray = greater_mask * greater + less_mask * less

        return arr

    def temperature_factor(self, forc: RtForcing) -> Vector:
        """
        Calculate the dependence on temperature
        """
        return self.q_10 ** ((forc.hydro_forc.temp - 20.0) / 10.0)  # type: ignore

    def water_table_factor(self, forc: RtForcing) -> Vector:
        """
        Calculate the dependence on the water table
        """
        fzw_mask: NDArray = self.n_alpha == 0

        n_alpha_vals: NDArray = np.ones_like(fzw_mask, dtype=np.float64)
        # second_term: NDArray = np.exp(
        #     -abs(self.n_alpha) * forc.z_w ** (self.n_alpha / abs(self.n_alpha))
        # )

        # n_alpha_vals[~fzw_mask] = second_term
        for i, n_alpha_i in enumerate(self.n_alpha):
            if n_alpha_i != 0:
                n_alpha_vals[i] = np.exp(
                    -abs(n_alpha_i) * forc.z_w ** (n_alpha_i / abs(n_alpha_i))
                )

        return n_alpha_vals

    def factor(self, forc: RtForcing) -> Vector:
        """
        Calculate the auxiliary factor for each function
        """
        return self.soil_water_factor(forc) * self.temperature_factor(forc) * self.water_table_factor(forc)  # type: ignore

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
            sw_threshold=np.array(sw_thrs),
            sw_exp=np.array(sw_exps),
            n_alpha=np.array(n_alphas),
            q_10=np.array(q_10s),
            ssa=np.array(ssas),
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
