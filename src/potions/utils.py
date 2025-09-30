from dataclasses import dataclass
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray


@dataclass(frozen=True)
class HydroForcing:
    precip: float
    temp: float
    pet: float


# ==== Reactive Transport ==== #
@dataclass(frozen=True)
class ChemicalState:
    prim_aq_conc: NDArray[f64]
    min_conc: NDArray[f64]
    sec_conc: NDArray[f64]

    def to_prim_array(self) -> NDArray[f64]:
        """Convert this data structure to a numpy array of only the primary species.

        Returns:
            NDArray: Array of the primary species
        """
        return np.concatenate([self.prim_aq_conc, self.min_conc])

    def to_array(self) -> NDArray[f64]:
        """Convert this data structure to a numpy array of all species.

        Returns:
            NDArray: Array of all species
        """
        return np.concatenate([self.prim_aq_conc, self.min_conc, self.sec_conc])


@dataclass(frozen=True)
class RtForcing:
    conc_in: ChemicalState  # Concentration entering this zone
    q_in: float  # Flow rate entering this zone
    hydro_forc: HydroForcing  # Hydrologic forcing, may be useful later on


@dataclass(frozen=True)
class KineticParams:
    def rate(self, state: NDArray[f64], d: RtForcing) -> NDArray[f64]:
        raise NotImplementedError()


@dataclass(frozen=True)
class MonodRateParams(KineticParams):
    max_rate: NDArray[f64]  # Vector of maximum rates
    monod_const: NDArray[
        f64
    ]  # Matrix of half-saturation constants - 2 dimensions with shape (num primary species x num ????)
    inhib_const: NDArray[f64]  # Matrix of inhibition constants - 2 dimensions

    def rate(self, state: NDArray[f64], d: RtForcing) -> NDArray[f64]:
        """
        Calculate the rate of reaction for a Monod rate law when provided the state concentrations as a vector
        """
        monod_terms: NDArray[f64] = state / (self.monod_const + state)
        inhib_terms: NDArray[f64] = self.inhib_const / (self.inhib_const + state)

        monod_rate: NDArray[f64] = monod_terms.prod(axis=1)
        inhib_rate: NDArray[f64] = inhib_terms.prod(axis=1)

        return self.max_rate * monod_rate * inhib_rate


@dataclass(frozen=True)
class TstRateParams(KineticParams):
    rate_const: NDArray[f64]  # Vector of rate constants
    stoich: NDArray[f64]  # Matrix describing stoichiometry of solids reactions
    k_eq: NDArray[f64]  # Vector of equilibrium constants
    dependencies: NDArray[f64]  # Matrix of dependencies for each reaction

    def rate(self, state: NDArray[f64], d: RtForcing) -> NDArray[f64]:
        raise NotImplementedError
