from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .common_types import RtForcing
from .reaction_network import (
    MineralAuxParams,
    MineralParameters,
    EquilibriumParameters,
    MonodParameters,
    ReactionNetwork,
    TstParameters,
)


@dataclass(frozen=True)
class MiscData:
    mineral_stoichiometry: NDArray  # Matrix describing the stoichiometry of the mineral dissolution reactions
    species_mobility: NDArray  # Boolean vector describing whether each species is mobile or not. All aqueous species are mobile, and all mineral species are immobile.
    mineral_molar_mass: NDArray  # The molar mass of each of the minerals


@dataclass(frozen=True)
class RtStep:
    """Holds the results of a single time step for a ReactiveTransportZone."""

    state: NDArray
    forc_flux: NDArray
    vap_flux: NDArray
    lat_flux: NDArray
    vert_flux: NDArray


class ReactiveTransportZone:
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
        monod: MonodParameters,
        tst: TstParameters,
        eq: EquilibriumParameters,
        aux: MineralParameters,
        misc: MiscData,
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
        self.name: str = name
        self.monod: MonodParameters = monod
        self.tst: TstParameters = tst
        self.eq: EquilibriumParameters = eq
        self.aux: MineralParameters = aux
        self.misc: MiscData = misc

    @staticmethod
    def from_network(
        network: ReactionNetwork,
        params: dict[str, MineralAuxParams],
        name: str = "unnamed",
    ) -> ReactiveTransportZone:
        """Convert this object into a reactive transport zone from the parameters and reaction network"""
        monod: MonodParameters = network.monod_params
        tst: TstParameters = network.tst_params
        eq: EquilibriumParameters = network.equilibrium_parameters
        min_params: list[MineralAuxParams] = []
        for m in network.mineral:
            min_params.append(params[m.name])
        minerals: MineralParameters = MineralParameters.from_mineral_parameters(
            min_params
        )

        stoich = network.mineral_stoichiometry
        mobility = network.transport_mask
        min_molar_mass = network.mineral_molar_masses

        misc = MiscData(
            mineral_stoichiometry=stoich.values,
            species_mobility=mobility,
            mineral_molar_mass=min_molar_mass,
        )

        return ReactiveTransportZone(
            monod=monod, tst=tst, eq=eq, aux=minerals, misc=misc, name=name
        )

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
        reaction_rate_vec: NDArray = self.reaction_rate(
            chms, d
        )  # Rate of production or consumption of each of the mobile primary species (not minerals). Positive is production, negative is consumption
        transport_rate_vec: NDArray = self.transport_rate(
            chms, d
        )  # Rate of transport of of each of the mobile primary species
        return reaction_rate_vec + transport_rate_vec

    def reaction_rate(self, chms: NDArray, d: RtForcing) -> NDArray:
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
            self.aux.rate_const
            * self.aux.ssa
            * self.misc.mineral_molar_mass
            * min_conc
            * aux_rate
            * (monod_rate + tst_rate)
        )

        all_species_rates = self.misc.mineral_stoichiometry @ mineral_rates

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
        mass_in: NDArray = d.hydro_forc.q_in * d.conc_in
        # mass_out: NDArray = (d.q_in / d.storage) * (d.conc_in - chms)
        mass_out: NDArray = d.q_out * chms

        mass_change: NDArray = mass_in - mass_out
        mass_change[~self.misc.species_mobility] = (
            0.0  # Set all immobile (mineral) species mass change to zero
        )

        return mass_change

    def step(
        self, c_0: NDArray, d: RtForcing, dt: float, q_in: float, debug: bool = False
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

        # 1. Solve the ODE for kinetic reactions and transport
        def ode(_t: float, c: NDArray) -> NDArray:
            return self.mass_balance_ode(c_0, d)

        # def residual(c: NDArray) -> NDArray:
        #     return (c_0 - c) + dt * self.mass_balance_ode(c, d)

        ode_res = solve_ivp(ode, [0, dt], y0=c_0)
        # opt_res = fsolve(residual, c_0)

        # Solve for the next concentration using some rootfinding method
        c_after_rt: NDArray = ode_res.y[:, -1]
        # c_after_rt: NDArray = opt_res

        if debug:
            print(f"C after Reaction+Transport: \n{c_after_rt}")

        # After finding the new concentrations that are not at equilibrium, solve for the equilibrium concentrations
        c_after_eq = self.eq.solve_equilibrium(c_after_rt)

        # 3. Calculate output fluxes for the time step
        forc_flux = q_in * d.conc_in
        vap_flux = np.zeros_like(c_after_eq)

        if debug:
            print(f"C after equilibrium: \n{c_after_eq}")
            tot_after_rt = self.eq.total.values @ c_after_rt
            tot_after_eq = self.eq.total.values @ c_after_eq
            print(f"Total before equilibrium: {tot_after_rt}")
            print(f"Total after equilibrium:  {tot_after_eq}")

        total_q_out_water = d.q_out
        if total_q_out_water > 1e-9:
            total_mass_out_flux = total_q_out_water * c_after_eq
            lat_flux = total_mass_out_flux * (
                d.hydro_step.lat_flux_ext / total_q_out_water
            )
            vert_flux = total_mass_out_flux * (
                d.hydro_step.vert_flux_ext / total_q_out_water
            )
        else:
            lat_flux = np.zeros_like(c_after_eq)
            vert_flux = np.zeros_like(c_after_eq)

        return RtStep(
            state=c_after_eq,
            forc_flux=forc_flux,
            vap_flux=vap_flux,
            lat_flux=lat_flux,
            vert_flux=vert_flux,
        )

    def columns(self, zone_id: int) -> list[str]:
        """
        Gets the column names for this zone for the output DataFrame.

        Parameters
        ----------
        zone_id : int
            Identifier for the specific zone to generate column names for

        Returns
        -------
        list[str]
            List of column names for output DataFrame including:
            - Concentration columns
            - Flux columns (forcing, vapor, lateral, vertical)

        Notes
        -----
        The column names are generated based on the zone identifier and
        follow a consistent naming convention for easy data handling and
        analysis. The naming convention includes the zone name and ID to
        distinguish between different zones in multi-zone simulations.
        """
        # This will need to be updated based on the number of species.
        name = f"{self.name}_{zone_id}"
        return [
            f"c_{name}",
            f"j_forc_{name}",
            f"j_vap_{name}",
            f"j_lat_{name}",
            f"j_vert_{name}",
        ]
