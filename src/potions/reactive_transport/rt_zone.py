from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy.optimize import fsolve

from ..utils import DO_LOGGING, ZERO_CONC, setup_logging
from ..math import find_root_multi

from ..common_types_compiled import HydroStep


from ..common_types_compiled import RtForcing
from .kinetic_structures import (
    MineralParameters,
    EquilibriumParameters,
    MonodParameters,
    TstParameters,
    RtParameters,
)

from .reaction_network import (
    ReactionNetwork,
)

setup_logging("rt_zone.py")


@dataclass(frozen=True)
class MiscData:
    mineral_stoichiometry: NDArray  # Matrix describing the stoichiometry of the mineral dissolution reactions
    species_mobility: NDArray  # Boolean vector describing whether each species is mobile or not. All aqueous species are mobile, and all mineral species are immobile.
    mineral_molar_mass: NDArray  # The molar mass of each of the minerals
    rate_const: NDArray  # The rate constants for the mineral reactions. Note that these are _not_ parameters. They are constants.

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MiscData):
            raise TypeError(f"Cannot compare MiscData with '{type(other)}'")
        else:
            vals: list[bool] = [
                np.allclose(self.mineral_stoichiometry, other.mineral_stoichiometry),
                np.allclose(self.species_mobility, other.species_mobility),
                np.allclose(self.mineral_molar_mass, other.mineral_molar_mass),
                np.allclose(self.rate_const, other.rate_const),
            ]

            return all(vals)


@dataclass(frozen=True)
class RtStep:
    """Holds the results of a single time step for a ReactiveTransportZone."""

    state: NDArray
    conc_in: NDArray
    mass_in: NDArray
    lat_conc: NDArray
    vert_conc: NDArray
    lat_mass: NDArray
    vert_mass: NDArray
    mineral_rates: NDArray  # Rate of mineral reactions


class RtZone:
    """
    A zone that solves reactive transport processes in hydrologic systems.

    This class acts as an ODE solver and orchestrator for reactive transport
    calculations in hydrologic zones. It is initialized with functions and
    parameters that define the specific transport and kinetic reaction logic,
    allowing for flexible definition of biogeochemical processes without
    hard-coding them into the zone itself.

    The class implements operator splitting to handle both transport and
    chemical reaction processes separately, then combines them to solve
    for the complete reactive transport behavior in the zone.

    Attributes
    ----------
    name : str
        Identifier name for the reactive transport zone
    monod : MonodParameters
        Parameters for Monod-type reaction kinetics
    tst : TstParameters
        Parameters for TST-type (Two-Stage Transport) reaction kinetics
    eq : EquilibriumParameters
        Parameters for equilibrium calculations of chemical species
    aux : AuxiliaryParameters
        Auxiliary parameters for environmental factors (temperature, moisture, etc.)
    min : MineralParameters
        Parameters for mineral reaction kinetics and stoichiometry
    misc : MiscData
        Miscellaneous data and configuration parameters

    Examples
    --------
    >>> # Initialize a reactive transport zone
    >>> zone = ReactiveTransportZone(
    ...     monod=monod_params,
    ...     tst=tst_params,
    ...     eq=equilibrium_params,
    ...     aux=aux_params,
    ...     min=mineral_params,
    ...     misc=misc_data,
    ...     name="zone_1"
    ... )

    >>> # Advance the chemical state by one time step
    >>> result = zone.step(
    ...     c_0=initial_concentrations,
    ...     d=forcing_data,
    ...     dt=time_step,
    ...     q_in=water_inflow_rate
    ... )

    Notes
    -----
    The ReactiveTransportZone implements operator splitting to handle the
    complex reactive transport problem:

    1. Transport and kinetic reactions are solved first using ODE integration
    2. Chemical equilibrium is then computed for the resulting concentrations
    3. Output fluxes are calculated for mass balance and reporting

    This approach allows for efficient computation while maintaining accuracy
    in both transport and reaction processes.
    """

    def __init__(
        self,
        network: ReactionNetwork,
        params: RtParameters,
        do_reactions: bool = True,
        do_speciation: bool = True,
        name: str = "unnamed",
    ) -> None:
        """
        Initializes the zone with specific parameters.

        Parameters
        ----------
        monod : MonodParameters
            Parameters for Monod-type reaction kinetics
        tst : TstParameters
            Parameters for TST-type (Two-Stage Transport) reaction kinetics
        eq : EquilibriumParameters
            Parameters for equilibrium calculations of chemical species
        aux : AuxiliaryParameters
            Auxiliary parameters for environmental factors (temperature, moisture, etc.)
        min : MineralParameters
            Parameters for mineral reaction kinetics and stoichiometry
        misc : MiscData
            Miscellaneous data and configuration parameters
        name : str, optional
            Identifier name for the reactive transport zone (default: "unnamed")

        Notes
        -----
        The initialization sets up all the necessary components for solving
        reactive transport problems in this specific hydrologic zone. Each
        parameter object contains the specific information needed for the
        corresponding type of process.
        """
        self.__network: ReactionNetwork = network
        self.do_reactions: bool = do_reactions
        self.do_speciation: bool = do_speciation

        monod: MonodParameters = network.monod_params
        tst: TstParameters = network.tst_params
        eq: EquilibriumParameters = network.equilibrium_parameters

        minerals: MineralParameters = params.mineral_params

        stoich = network.mineral_stoichiometry
        mobility = network.transport_mask
        min_molar_mass = network.mineral_molar_masses

        misc = MiscData(
            mineral_stoichiometry=stoich.values,
            species_mobility=mobility,
            mineral_molar_mass=min_molar_mass,
            rate_const=network.rate_consts,
        )

        self.name: str = name
        self.monod: MonodParameters = monod
        self.tst: TstParameters = tst
        self.eq: EquilibriumParameters = eq
        self.aux: MineralParameters = minerals
        self.misc: MiscData = misc
        self.parameters: RtParameters = params

    @property
    def network(self) -> ReactionNetwork:
        return self.__network

    def mass_balance_ode(self, chms: NDArray, d: RtForcing) -> NDArray:
        """
        Calculates the net rate of change of concentration (dC/dt) for the mass balance ODE.

        This method implements the reactive transport equation, where the change in mass
        comes from both reaction processes and transport processes:
        dm/dt = (dm/dt)_reaction + (dm/dt)_transport

        Parameters
        ----------
        chms : numpy.ndarray
            Current concentrations of all chemical species in the zone
        d : RtForcing
            Forcing data containing hydrological and environmental conditions

        Returns
        -------
        numpy.ndarray
            Vector of net rates of change of concentration for all species
            (dC/dt) representing the total mass balance rate

        Notes
        -----
        The mass balance ODE combines:
        - Reaction rates from Monod and TST kinetics
        - Transport rates from hydrological forcing
        - Mineral reaction rates from the mineral parameters

        This combined rate represents the total change in species concentrations
        due to all processes occurring in the zone.
        """
        reaction_rate_vec: NDArray
        transport_rate_vec: NDArray = self.transport_rate(
            chms, d
        )  # Rate of transport of of each of the mobile primary species

        if self.do_reactions:
            reaction_rate_vec = self.reaction_rate(
                chms, d
            )  # Rate of production or consumption of each of the mobile primary species (not minerals). Positive is production, negative is consumption

        else:
            reaction_rate_vec = np.zeros_like(transport_rate_vec)
        return reaction_rate_vec + transport_rate_vec

    def reaction_rate(
        self, chms: NDArray, d: RtForcing, minerals_only: bool = False
    ) -> NDArray:
        """
        Calculate the rate of reaction for this time step - the rate that primary species are produced or consumed.

        This method computes the net reaction rate for all primary species based on:
        - Monod kinetics for biological reactions
        - TST kinetics for transport-type reactions
        - Auxiliary factors (temperature, moisture, etc.)
        - Mineral reaction rates and stoichiometry

        Parameters
        ----------
        chms : numpy.ndarray
            Current concentrations of all chemical species in the zone
        d : RtForcing
            Forcing data containing environmental conditions

        Returns
        -------
        numpy.ndarray
            Vector of reaction rates for primary species (positive = production, negative = consumption)

        Notes
        -----
        The reaction rate calculation involves:
        1. Computing Monod reaction rates for biological processes
        2. Computing TST reaction rates for transport processes
        3. Applying auxiliary factor modulation (temperature, moisture, etc.)
        4. Calculating mineral reaction rates based on surface area and rate constants
        5. Converting mineral rates to primary species rates using stoichiometry

        The final result represents the net rate of change for all primary species
        due to all reaction processes occurring in the zone.
        """
        monod_rate: NDArray = self.monod.rate(
            chms
        )  # Reaction rate of each species from Monod Reactions
        tst_rate: NDArray = self.tst.rate(
            chms
        )  # Reaction rate of each species from TST-type reactions
        aux_rate: NDArray = self.aux.factor(
            d
        )  # Reaction rate modulation from soil moisture, temperature, and water table factors

        num_min: int = self.monod.inhib_mat.shape[0]
        min_conc: NDArray = chms[-num_min:]  # Get just the mineral concentrations

        mineral_rates: NDArray = (
            86_400
            * self.misc.rate_const
            * self.aux.ssa
            * self.misc.mineral_molar_mass
            * min_conc
            * aux_rate
            * (monod_rate + tst_rate)
        )

        if minerals_only:
            arr = np.array(mineral_rates)
            return arr
        else:
            all_species_rates = (
                self.misc.mineral_stoichiometry @ mineral_rates
            )  # Rate of reaction in mol/s
            # Need to convert to mol/day by multiplying by 86,400 seconds per day
            all_species_rates[-num_min:] = (
                0.0  # Make sure there are no minerals being consumed
            )

            return all_species_rates

    def transport_rate(self, chms: NDArray, d: RtForcing) -> NDArray:
        """
        Calculate the rate of transport for this time step for each aqueous species.

        This method computes the transport rate for each species based on:
        - Water inflow and outflow rates
        - Concentrations of species in inflow and current state
        - Mobility status of each species (mobile vs. immobile)

        Parameters
        ----------
        chms : numpy.ndarray
            Current concentrations of all chemical species in the zone
        d : RtForcing
            Forcing data containing hydrological conditions

        Returns
        -------
        numpy.ndarray
            Vector of transport rates for each species (positive = inflow, negative = outflow)

        Notes
        -----
        Transport calculations include:
        1. Mass inflow from external sources (q_in * conc_in)
        2. Mass outflow from the zone (q_out * chms)
        3. Setting immobile species (minerals) to zero transport rate
        4. Accounting for water table and hydrological conditions

        The transport rate represents the net change in mass due to hydrological
        transport processes in the zone.
        """
        hs: HydroStep = d.hydro_step

        q_ext: float = (
            hs.lat_flux_ext + hs.vert_flux_ext - hs.lat_flux - hs.vert_flux
        )  # External fluxes passing through the zone
        q_int: float = hs.lat_flux + hs.vert_flux
        mass_in: NDArray = d.hydro_forc.q_in * d.conc_in  # type: ignore
        # mass_out: NDArray = (d.q_in / d.storage) * (d.conc_in - chms)
        mass_out_internal: NDArray = q_int * chms
        mass_out_external: NDArray = q_ext * d.conc_in  # type: ignore
        mass_out: NDArray = mass_out_internal + mass_out_external
        # Note that d.q_out = q_int + q_ext

        mass_change: NDArray = mass_in - mass_out
        mass_change[~self.misc.species_mobility] = (
            0.0  # Set all immobile (mineral) species mass change to zero
        )

        return mass_change

    def step(
        self,
        c_0: NDArray,
        d: RtForcing,
        dt_days: float,
        verbose: bool = False,
        failed_dir: Optional[str] = None,
    ) -> RtStep:
        """
        Advances the chemical state by one time step using operator splitting.

        This method implements the complete reactive transport solution using
        operator splitting where:
        1. Transport and kinetic reactions are solved first using ODE integration
        2. Chemical equilibrium is computed for the resulting concentrations

        Parameters
        ----------
        c_0 : numpy.ndarray
            Initial concentrations of all chemical species in the zone
        d : RtForcing
            Forcing data containing hydrological and environmental conditions
        dt : float
            Time step size for the calculation
        q_in : float
            Water inflow rate to the zone
        debug : bool, optional
            If True, print debug information including intermediate results
            (default: False)

        Returns
        -------
        RtStep
            Result containing the final state and fluxes for the time step

        Notes
        -----
        The step method implements the following sequence:
        1. Solve ODE for transport and kinetic reactions using solve_ivp
        2. Apply chemical equilibrium calculations to the resulting concentrations
        3. Calculate output fluxes (latitudinal and vertical) based on final concentrations
        4. Return complete time step results including state and fluxes

        This operator splitting approach allows for efficient computation while
        maintaining accuracy in both transport and reaction processes.
        """
        if verbose:
            print(f"Incoming concentrations to zone `{self.name}`: \n{c_0=}")

        # 1. Solve the ODE for kinetic reactions and transport
        def residual(c: NDArray) -> NDArray:
            # return (c_0 - c) + dt_days * self.mass_balance_ode(0.5 * (c_0 + c), d)
            return (c_0 - c) + dt_days * self.mass_balance_ode(c, d)

        # def ode(_t: float, c: NDArray) -> NDArray:
        #     return self.mass_balance_ode(c, d)

        try:
            opt_res = find_root_multi(residual, c_0)
        except (LinAlgError, ValueError) as e:
            if DO_LOGGING:
                logging.warning(
                    f"RtZone ODE Step Error (find_root_multi): {','.join(c_0.astype(str))}"
                )
            opt_res_full = fsolve(residual, c_0, full_output=True)
            opt_res = opt_res_full[0]  # type: ignore

            if opt_res_full[2] != 0:
                if DO_LOGGING:
                    logging.warning(
                        f"RtZone ODE Step Error (fsolve): {','.join(c_0.astype(str))}"
                    )
                raise ValueError("Both solvers failed to find root") from e

        # ivp_res = solve_ivp(ode, t_span=[0, dt_days], y0=c_0)
        # opt_res = ivp_res.y[:, -1]

        # Solve for the next concentration using some rootfinding method
        c_after_rt: NDArray = opt_res

        if any(c_after_rt < 0):
            print("Got negative concentrations after RT")
            print(f"{d=}")
            print(f"{dt_days=}")
            print(f"{c_0=}")
            print(f"{c_after_rt=}")
            raise ValueError("Failed in equilibrium")

        if verbose:
            print(f"C after Reaction+Transport and before equilibrium: \n{c_after_rt}")

        c_after_eq: NDArray
        # After finding the new concentrations that are not at equilibrium, solve for the equilibrium concentrations
        if self.do_speciation:
            c_after_eq = self.eq.solve_equilibrium(c_after_rt)
        else:
            c_after_eq = c_after_rt

        # 2.5 Correct the outputs so that there is nothing smaller than our threshold here
        c_after_eq[c_after_eq < ZERO_CONC] = ZERO_CONC

        # 3. Calculate output fluxes for the time step
        if verbose:
            print(f"C after equilibrium: \n{c_after_eq}")
            tot_after_rt = self.eq.total @ c_after_rt
            tot_after_eq = self.eq.total @ c_after_eq
            print(f"Total before equilibrium: {tot_after_rt}")
            print(f"Total after equilibrium:  {tot_after_eq}")

        hs: HydroStep = d.hydro_step
        q_int: float = hs.lat_flux + hs.vert_flux
        q_ext: float = hs.vert_flux_ext + hs.lat_flux_ext - q_int
        total_q_out_water: float = q_int + q_ext

        if q_int + q_ext > 1e-9:
            total_mass_out_flux = q_int * c_after_eq + q_ext * d.conc_in
            lat_conc = total_mass_out_flux * (
                d.hydro_step.lat_flux_ext / total_q_out_water
            )
            vert_conc = total_mass_out_flux * (
                d.hydro_step.vert_flux_ext / total_q_out_water
            )
        else:
            lat_conc = np.full_like(c_after_eq, ZERO_CONC)
            vert_conc = np.full_like(c_after_eq, ZERO_CONC)

        # Calculate the mineral rates using the updated concentrations
        mineral_rates: NDArray = self.reaction_rate(c_after_eq, d, minerals_only=True)

        return RtStep(
            state=c_after_eq,
            conc_in=d.conc_in,  # type: ignore
            mass_in=d.conc_in * d.hydro_step.q_in,  # type: ignore
            lat_conc=lat_conc,
            vert_conc=vert_conc,
            lat_mass=lat_conc * d.hydro_step.lat_flux_ext,
            vert_mass=vert_conc * d.hydro_step.vert_flux_ext,
            mineral_rates=mineral_rates,
        )

    @property
    def all_species(self) -> list[str]:
        """The list of all species involved in the mdoel"""
        return self.__network.species_names

    @property
    def mineral_species(self) -> list[str]:
        """The list of all species involved in the mdoel"""
        return self.__network.mineral_species_names

    @property
    def num_species(self) -> int:
        """The number of species involved in this zone"""
        return len(self.__network.species_names)

    def to_array(self) -> NDArray:
        return self.parameters.to_array()

    @staticmethod
    def from_array(
        arr: NDArray,
        network: ReactionNetwork,
        do_reactions: bool = True,
        do_speciation: bool = True,
        name: str = "unnamed",
    ) -> RtZone:
        parameters: RtParameters = RtParameters.from_array(arr)
        return RtZone(
            params=parameters,
            network=network,
            do_reactions=do_reactions,
            do_speciation=do_speciation,
            name=name,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RtZone):
            return False
        else:
            vals: list[bool] = [
                self.parameters == other.parameters,
                self.monod == other.monod,
                self.tst == other.tst,
                self.eq == other.eq,
                self.aux == other.aux,
                self.misc == other.misc,
            ]

            return all(vals)


# ==== Functions ==== #
def calculate_moisture_fraction(
    zone_params: dict[str, RtZone], sim_res: DataFrame
) -> NDArray:
    sw_vals: NDArray = np.zeros((len(zone_params), len(sim_res)))
    zone_names: list[str] = list(zone_params.keys())
    zone_name: str
    zone: RtZone
    for i, (zone_name, zone) in enumerate(zone_params.items()):  # type: ignore
        col_name: str = f"s_{zone_name}"
        max_water_volume = zone.parameters.dimensions.max_water_volume
        storage: Series = sim_res[col_name]

        if storage.max() >= max_water_volume:
            raise ValueError(
                f"Water volume error in zone '{zone_names[i]}': Maximum volume is {round(max_water_volume, 1)} mm but maximum simulated storage is {round(storage.max(),1)} mm. Increase porosity or volume"
            )

        sw_vals[i] = (storage + zone.parameters.dimensions.passive_water_storage) / (
            max_water_volume + zone.parameters.dimensions.passive_water_storage
        )

    return sw_vals


def calculate_water_table_depth(
    zone_params: dict[str, RtZone], sim_res: DataFrame
) -> NDArray:
    # zw_vals: dict[HbvZone, Series] = {}
    zw_vals: NDArray = np.zeros((len(zone_params), len(sim_res)), dtype=np.float64)
    zone_name: str
    zone: RtZone
    for i, (zone_name, zone) in enumerate(zone_params.items()):  # type: ignore
        col_name: str = f"s_{zone_name}"
        storage: Series = sim_res[col_name]
        zw_vals[i] = (
            zone.parameters.dimensions.depth
            - (storage + zone.parameters.dimensions.passive_water_storage)
            / zone.parameters.dimensions.porosity
        )

    return zw_vals


def get_hydro_steps(sim_res: DataFrame) -> NDArray:
    zone_names: list[str] = [
        c.replace("s_", "") for c in sim_res.columns if c.startswith("s_")
    ]
    hydro_steps: NDArray = np.empty((len(zone_names), len(sim_res)), dtype=object)

    for i, z in enumerate(zone_names):
        s: Series = sim_res[f"s_{z}"]
        q_forc: Series = sim_res[f"q_forc_{z}"]
        q_vap: Series = sim_res[f"q_vap_{z}"]
        q_lat: Series = sim_res[f"q_lat_{z}"]
        q_lat_ext: Series = sim_res[f"q_lat_ext_{z}"]
        q_vert: Series = sim_res[f"q_vert_{z}"]
        q_vert_ext: Series = sim_res[f"q_vert_ext_{z}"]
        q_in: Series = sim_res[f"q_in_{z}"]

        for j, s_j in enumerate(s):
            hs = HydroStep(
                state=s_j,
                forc_flux=q_forc.iloc[j],
                vap_flux=q_vap.iloc[j],
                lat_flux=q_lat.iloc[j],
                lat_flux_ext=q_lat_ext.iloc[j],
                vert_flux=q_vert.iloc[j],
                vert_flux_ext=q_vert_ext.iloc[j],
                q_in=q_in.iloc[j],
            )

            hydro_steps[i, j] = hs

    return hydro_steps


# =================== #
