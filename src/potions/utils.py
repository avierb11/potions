from abc import abstractmethod
from typing import Generic, TypeVar
from dataclasses import dataclass
import math
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
Vector = np.ndarray[tuple[N], np.dtype[f64]]
Matrix = np.ndarray[tuple[M,N], np.dtype[f64]]
NumTot = TypeVar("NumTot", bound=int) # Number of total species
NumPrim = TypeVar("NumPrim", bound=int) # Number of primary aqueous species
NumMin = TypeVar("NumMin", bound=int) # Number of mineral species
NumSec = TypeVar("NumSec", bound=int) # Number of secondary aqueous species
NumSpec = TypeVar("NumSpec", bound=int) # Number of species in the model - mineral and aqueous

@dataclass(frozen=True)
class HydroForcing:
    """Contains hydrologic forcing data for a single zone at a single time step.

    Attributes:
        precip: Precipitation rate (e.g., mm/day).
        temp: Temperature (e.g., °C).
        pet: Potential evapotranspiration rate (e.g., mm/day).
    """

    precip: float
    temp: float
    pet: float


# ==== Reactive Transport ==== #
@dataclass(frozen=True)
class ChemicalState(Generic[NumMin, NumPrim, NumTot, NumSec]):
    """Represents the chemical state of a zone, partitioned by species type.

    This is a container for concentrations of different types of chemical species
    involved in reactive transport calculations.

    Attributes:
        prim_aq_conc: Concentrations of primary aqueous species in units of moles per liter
        min_conc: Concentrations of mineral species in units of ???
        sec_conc: Concentrations of secondary aqueous species in units of moles per liter
    """

    prim_aq_conc: Vector[NumPrim]
    min_conc: Vector[NumMin]
    sec_conc: Vector[NumSec]

    def to_primary_array(self) -> Vector[NumTot]:
        """Concatenates primary aqueous and mineral species into a single array.

        This is often used for the state variables in the ODE solver where only
        the total concentrations of primary components are tracked.

        Returns:
            A 1D NumPy array containing primary aqueous and mineral concentrations.
        """
        return np.concatenate([self.prim_aq_conc, self.min_conc]) # type: ignore
    
    @property
    def primary_concentrations(self) -> Vector[NumTot]:
        """
        Get the vector of primary concentrations in the format [mineral concentrations, primary aqueous concentrations]
        """
        return self.to_primary_array()

    def to_array(self) -> Vector[NumSpec]:
        """Concatenates all species concentrations into a single array.

        Returns:
            A 1D NumPy array containing all species concentrations.
        """
        return np.concatenate([self.prim_aq_conc, self.min_conc, self.sec_conc]) # type: ignore


@dataclass(frozen=True)
class RtForcing:
    """
    Contains the hydrologic and chemical drivers for a reactive transport step.

    Attributes:
        conc_in: The chemical state of the incoming water.
        q_in: The flow rate of incoming water (e.g., mm/day).
        hydro_forc: The local hydrologic forcing (P, T, PET).
        s_w: The soil moisture fraction (dimensionless, [0, 1]).
        z_w: A factor related to the water table depth.
    """
    conc_in: ChemicalState  # Concentration entering this zone
    q_in: float  # Flow rate entering this zone
    hydro_forc: HydroForcing  # Hydrologic forcing, may be useful later on
    s_w: float # Soil moisture fraction [0,1]
    z_w: float # Water table depth factor


@dataclass(frozen=True)
class KineticParams(Generic[NumMin, NumPrim]):
    """An abstract base class for kinetic reaction rate parameters.

    This class defines the interface and common environmental modifier functions
    (temperature, soil moisture, water table depth) for kinetic reactions.
    Subclasses must implement the `conc_func` and `rate` methods.

    Attributes:
        ssa: The specific surface area for this solid
        q_10: The Q10 temperature coefficient for the reaction rate.
        sw_threshold: The soil water content threshold for the moisture function.
        sw_exponent: The exponent for the soil moisture function.
        n_alpha: The exponent for the water table depth function.
    """
    ssa: Vector[NumMin] # Specific surface area
    rate_const: Vector[NumMin] # The rate constant
    q_10: Vector[NumMin] # The base for the temperature rate
    sw_threshold: Vector[NumMin] # The soil water threshold
    sw_exponent: Vector[NumMin] # The soil water exponent
    n_alpha: Vector[NumMin] # The water table depth exponent

    def temp_func(self, temp: float) -> Vector[NumMin]:
        """Calculates the temperature-dependent rate modifier (Q10 formulation).

        Args:
            temp: The current temperature in Celsius.

        Returns:
            A dimensionless factor that scales the reaction rate based on temperature.
        """
        return self.q_10 ** ((temp - 20.0) / 10.0) # type: ignore
    
    def soil_moisture_func(self, sw: float) -> Vector[NumMin]:
        """Calculates the soil moisture-dependent rate modifier.

        This function can represent, for example, limitations under very dry or
        very wet (anoxic) conditions.

        Args:
            sw: The current soil water content as a fraction [0, 1].

        Returns:
            A dimensionless factor that scales the reaction rate.
        """
        if sw < self.sw_threshold:
            return (sw / self.sw_threshold) ** self.sw_exponent # type: ignore
        else:
            return ((1.0 - sw) / (1.0 - self.sw_threshold)) ** self.sw_exponent # type: ignore
        
    def water_table_func(self, zw: float) -> Vector[NumMin]:
        """Calculates the water table depth-dependent rate modifier.

        This can represent how reactant availability changes with depth or
        distance from the water table.

        Args:
            zw: A factor related to water table depth.

        Returns:
            A dimensionless factor that scales the reaction rate.
        """
        if abs(self.n_alpha) == 0.0:
            return np.ones_like(self.n_alpha) # type: ignore
        else:
            return np.exp(-abs(self.n_alpha) * zw ** (self.n_alpha / abs(self.n_alpha))) # type: ignore
    
    @abstractmethod
    def conc_func(self, chem_state: ChemicalState) -> Vector[NumMin]:
        """Calculates the concentration-dependent part of the reaction rate.

        This "characteristic equation" defines the core rate law (e.g., Monod, TST)
        based on the concentrations of aqueous species.

        Args:
            chem_state: An array of current chemical concentrations.

        Returns:
            The concentration-dependent portion of the rate.
        """
        raise NotImplementedError()
    
    def rate(self, chem_state: ChemicalState, d: RtForcing) -> Vector[NumMin]:
        """Calculates the overall kinetic reaction rate.

        This method combines the concentration-dependent rate (`conc_func`) with
        all environmental modifiers (temperature, moisture, etc.).

        Args:
            chem_state: An array of current chemical concentrations.
            d: The reactive transport forcing data for the current step.

        Returns:
            The final, environmentally-modified reaction rate.
        """
        return self.ssa * self.conc_func(chem_state) * self.temp_func(d.hydro_forc.temp) * self.soil_moisture_func(d.s_w) * self.water_table_func(d.z_w) # type: ignore


@dataclass(frozen=True)
class MonodRateParams(KineticParams, Generic[NumMin, NumTot]):
    """Kinetic parameters for a Monod-type reaction rate law.

    This is commonly used for microbially-mediated or enzyme-catalyzed reactions.

    Attributes:
        max_rate: The maximum potential rate of the reaction.
        monod_const: Array of half-saturation constants for promoting species.
        inhib_const: Array of inhibition constants for inhibiting species.
    """
    max_rate: Vector[NumMin]  # Vector of maximum rates
    monod_const: Matrix[
        NumMin, NumTot
    ]  # Matrix of half-saturation constants - 2 dimensions with shape (num primary species x num ????)
    inhib_const: Matrix[NumMin, NumTot]  # Matrix of inhibition constants - 2 dimensions

    def conc_func(self, chem_state: ChemicalState) -> Vector[NumMin]:
        """Calculates the overall rate for a Monod-type reaction.

        Args:
            chem_state: An array of current chemical concentrations.
            d: The reactive transport forcing data for the current step.

        Returns:
            The final, environmentally-modified reaction rate.
        """
        tot_conc: Vector = chem_state.primary_concentrations # type: ignore
        monod_terms: Matrix[NumMin, NumTot] = tot_conc / (self.monod_const + tot_conc) # type: ignore
        inhib_terms: Matrix[NumMin, NumTot] = self.inhib_const / (self.inhib_const + tot_conc) # type: ignore

        monod_rate: Vector[NumMin] = monod_terms.prod(axis=1)
        inhib_rate: Vector[NumMin] = inhib_terms.prod(axis=1)

        return self.max_rate * monod_rate * inhib_rate # type: ignore


@dataclass(frozen=True)
class TstRateParams(KineticParams, Generic[NumMin, NumTot]):
    """Kinetic parameters for a Transition-State Theory (TST) rate law.

    This is commonly used for abiotic mineral dissolution and precipitation reactions.

    Attributes:
        rate_const: Array of intrinsic rate constants for the reactions.
        stoich: Stoichiometric matrix for the mineral reactions.
        k_eq: Array of equilibrium constants for the mineral reactions.
        dependencies: Matrix defining catalytic/inhibitory dependencies.
    """
    rate_const: Vector[NumMin]  # Rate constant for this reaction
    stoich: Matrix[NumTot, NumMin]  # Vector describing the stoichiometry of this individual reaction
    k_eq: Vector[NumMin]  # Vector of equilibrium constants
    dependencies: Matrix[NumTot, NumMin]  # Matrix of dependencies for each reaction

    def conc_func(self, chem_state: ChemicalState) -> Vector[NumMin]:
        """Calculates the overall rate for a TST-type reaction.

        Args:
            chem_state: An array of current chemical concentrations.
        """
        log_aq_conc_vec: NDArray[f64] = np.log(chem_state.prim_aq_conc)
        log_dep: NDArray[f64] = self.dependencies @ log_aq_conc_vec
        dep_arr: NDArray[f64] = np.exp(log_dep)
        log_stoich_mat: NDArray[f64] = self.stoich @ log_aq_conc_vec


        raise NotImplementedError()
