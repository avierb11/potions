from __future__ import annotations
from typing import Callable, Iterable, Optional

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, DatetimeIndex, Series, TimedeltaIndex

from potions.objective_functions import DEFAULT_OBJECTIVE_FUNCTIONS
from potions.reactive_transport.rt_zone import (
    calculate_moisture_fraction,
    calculate_water_table_depth,
    get_hydro_steps,
)
from potions.utils import DO_LOGGING, RtNumericalError

from .common_types import (
    ChemicalState,
    ForcingData,
    HydroModelResults,
    RtModelResults,
    RtZoneConfiguration,
)

from .core import (
    HydrologicZone,
    OptimizationError,
    ReactionNetwork,
    RtForcing,
    RtParameters,
    RiverZone,
    RtStep,
    RtZone,
)

from .hydro_model import HydrologicalModel
from potions.reactive_transport.kinetic_structures import PARAMETERS_PER_MINERAL


class ReactiveTransportModel(HydrologicalModel):

    def __init__(
        self,
        # Hydrology
        zones: Optional[dict[str, HydrologicZone]] = None,
        scales: Optional[list[float]] = None,
        # Reactive transport
        network: Optional[ReactionNetwork] = None,
        rt_zones: Optional[dict[str, RtZone]] = None,
        # River reactions
        river_zone: Optional[RiverZone] = None,
        # Other things
        verbose: bool = False,
    ) -> None:
        super().__init__(zones, scales, verbose=verbose)

        # ==== Initialize reactive transport ==== #
        # Make sure that all arguments are present
        if (network is not None) == (rt_zones is not None):
            if verbose:
                print("Using reactive transport model")
        else:
            raise ValueError(
                "When doing reactive transport simulation, you must pass both `network` and `rt_zones` parameters, not just one of them"
            )

        if (network is not None) and (rt_zones is not None):
            self._rt_zones: dict[str, RtZone] = rt_zones  # type: ignore
            rt_params: dict[str, RtParameters] = {
                zone_name: zone.parameters for zone_name, zone in rt_zones.items()  # type: ignore
            }

            self._has_rt = True
            self._network: ReactionNetwork = network
            self._rt_params: dict[str, RtParameters] = rt_params
            # Make sure that all of the required zones are in the reactive transport zones
            model_zone_names = set(self.zone_names)
            rt_zone_names = set(rt_params.keys())

            if model_zone_names != rt_zone_names:
                raise ValueError(
                    f"Incorrect structure for `rt_params`. Expected dictionary with keys {
                        list(model_zone_names)
                    }, got keys {list(rt_zone_names)}"
                )

            rt_configuration: dict[str, RtZoneConfiguration] = {
                zone_name: RtZoneConfiguration(
                    do_reactions=zone.do_reactions, do_speciation=zone.do_speciation
                )
                for zone_name, zone in rt_zones.items()  # type: ignore
            }

            # Get the configurations
            cfg: dict[str, RtZoneConfiguration] = {}
            for zone_name in self.zone_names:
                if zone_name in rt_configuration:
                    cfg[zone_name] = rt_configuration[zone_name]
                else:
                    cfg[zone_name] = RtZoneConfiguration(
                        do_reactions=True, do_speciation=True
                    )

            self._zone_configs: dict[str, RtZoneConfiguration] = cfg

        if river_zone is not None:
            self._river_zone: RiverZone = river_zone
        else:
            self._river_zone = None  # type: ignore

        return

    @property
    def reaction_network(self) -> Optional[ReactionNetwork]:
        if self._has_rt:
            return self._network
        else:
            print("This model does not have reactive transport capabilities")
            return None

    @property
    def network(self) -> ReactionNetwork:
        if self._has_rt:
            return self._network
        else:
            raise ValueError("Model does not have reactive-transport capabilities")

    @property
    def rt_zones(self) -> dict[str, RtZone]:
        return self._rt_zones

    @property
    def has_river_zone(self) -> bool:
        return self._river_zone is not None

    @property
    def river_zone(self) -> RiverZone:
        if self._river_zone is not None:
            return self._river_zone
        else:
            raise ValueError("No river zone in this model")

    @classmethod
    def get_num_rt_parameters(
        cls: type[ReactiveTransportModel], config: dict[str, RtZoneConfiguration]
    ) -> int:
        """
        Get the number of parameters for a zone given the configuration of the reactive transport setup
        """
        num_dim_params: int = 3 * cls.get_num_zones()
        num_min_params: int = 0
        for _zone_name, zone in config.items():
            num_min_params += PARAMETERS_PER_MINERAL * zone.do_reactions

        return num_dim_params + num_min_params

    def rt_to_array(self) -> NDArray:
        """Convert the reactive transport structures into an array"""
        components: list[NDArray] = []

        # The reactive transport zones
        for zone_name in self.zone_names:
            components.append(self.rt_zones[zone_name].to_array())

        # The river zone
        if self.has_river_zone:
            components.append(self.river_zone.to_array())

        return np.concatenate(components)

    @classmethod
    def rt_params_from_array(
        cls: type[ReactiveTransportModel],
        arr: NDArray,
        network: ReactionNetwork,
        configs: dict[str, RtZoneConfiguration],
        verbose: bool = False,
        natural_scales: bool = True,
    ) -> dict[str, RtParameters]:
        # num_zones: int = cls.get_num_zones()
        zone_names: list[str] = cls.get_zone_names()
        # do_reactions: list[bool] = [z.do_reactions for z in configs.values()]
        num_minerals: int = network.num_minerals
        params_per_zone: list[int] = [
            3 + z.do_reactions * num_minerals * PARAMETERS_PER_MINERAL
            for zone_name, z in configs.items()
            if zone_name != "river"
        ]

        if len(arr) != sum(params_per_zone):
            raise ValueError(
                f"Incorrect number of parameters passed to `Model.rt_params_from_array`, expected {sum(params_per_zone)}, received {len(arr)}"
            )

        params_list: list[NDArray] = []
        for np_i in params_per_zone:
            params_list.append(arr[:np_i])
            arr = arr[np_i:]

        zone_params: dict[str, RtParameters] = {}
        for row_i, zone_name_i in zip(params_list, zone_names, strict=True):
            zone_params[zone_name_i] = RtParameters.from_array(
                row_i, natural_scales=natural_scales
            )

        return zone_params

    @classmethod
    def rt_zones_from_array(
        cls: type[ReactiveTransportModel],
        arr: NDArray,
        network: ReactionNetwork,
        config: list[RtZoneConfiguration],
    ) -> dict[str, RtZone]:
        # params_per_zone: int = 3 + network.num_mineral_parameters
        params_per_zone: list[int] = [
            3 * z.do_reactions * network.num_minerals * PARAMETERS_PER_MINERAL
            for z in config
        ]

        # params = arr.reshape((num_zones, params_per_zone))
        params_list: list[NDArray] = []
        for np_i in params_per_zone:
            zone_params = arr[:np_i]
            arr = arr[np_i:]
            params_list.append(zone_params)

        rt_zones: dict[str, RtZone] = {
            name_i: RtZone.from_array(
                row,
                network,
                do_reactions=config_i.do_reactions,
                do_speciation=config_i.do_speciation,
                name=name_i,  # type: ignore
            )
            for row, config_i, name_i in zip(
                params_list,
                config,
                cls.get_zone_names(),
                strict=True,  # type: ignore
            )
        }

        return rt_zones

    @property
    def zone_configurations(self) -> dict[str, RtZoneConfiguration]:
        return self._zone_configs

    def construct_rt_forcing_matrix(
        self,
        forc: ForcingData | Iterable[ForcingData],
        hydro_sim_df: DataFrame,
        precip_conc: NDArray,
        model_rt_params: dict[str, RtZone],
    ) -> NDArray:
        """
        Construct the reactive transport forcing data

        precip_conc: The precipitation concentrations at each step. Note that this has shape: (number of surface zones, number of steps)
        """
        sw_vals = calculate_moisture_fraction(model_rt_params, hydro_sim_df)
        zw_vals = calculate_water_table_depth(model_rt_params, hydro_sim_df)
        hydro_steps = get_hydro_steps(hydro_sim_df)
        hydro_forcing: NDArray = self.construct_hydro_forcing_matrix(forc).T

        num_zones: int = hydro_forcing.shape[0]
        num_steps: int = hydro_forcing.shape[1]

        rt_forcings: NDArray = np.empty((num_zones, num_steps), dtype=object)

        for i in range(num_zones):
            for j in range(num_steps):
                d_ij = RtForcing(
                    conc_in=np.zeros(self.reaction_network.num_species),  # type: ignore
                    hydro_step=hydro_steps[i, j],
                    hydro_forc=hydro_forcing[i, j],
                    s_w=sw_vals[i, j],
                    z_w=zw_vals[i, j],
                )
                rt_forcings[i, j] = d_ij

        # Now, insert the precipitation chemistry for each zone
        if precip_conc.shape[0] != len(self.surface_zone_ids):  # type: ignore
            raise ValueError(
                f"The first dimension of precipitation chemistry must be the same as the the number of surface zones: {
                    self.num_surface_zones
                }, not {precip_conc.shape[0]}"  # type: ignore
            )

        if precip_conc.shape[1] != len(hydro_sim_df):
            raise ValueError(
                f"The second dimension of precipitation chemistry must be the same as the the number of surface zones: {
                    len(hydro_sim_df)
                }, not {precip_conc.shape[1]}"
            )

        for i, zone_data in enumerate(precip_conc):
            for j, species_conc in enumerate(zone_data):
                rt_forcings[i, j].conc_in = species_conc.copy()

        return rt_forcings

    def to_array(self) -> NDArray:
        """Convert this model and turn it into a numpy array"""
        return np.concat([self.hydro_to_array(), self.rt_to_array()])

    def step_rt_model(
        self,
        model: dict[str, RtZone],
        state: NDArray,
        ds: Iterable[RtForcing],
        dt_days: float,
        verbose: bool = False,
        failed_dir: Optional[str] = None,
    ) -> list[RtStep]:
        num_zones: int = len(model)
        zones: list[RtZone] = list(model.values())

        num_species: int = zones[0].num_species

        lat_mass: NDArray = np.zeros(
            (num_zones, num_species), dtype=np.float64
        )  # Lateral mass transfered by a zone
        vert_mass: NDArray = lat_mass.copy()  # Mass transferred vertically by a zone

        steps: list[RtStep] = []

        i: int
        zone: RtZone
        s_i: NDArray
        d_i: RtForcing
        for i, (zone, s_i, d_i) in enumerate(  # type: ignore
            zip(model.values(), state.copy(), ds, strict=True)
        ):
            d_i = d_i.copy()

            if verbose:
                print(f"Starting solution for zone on step '{zone.name}'")
            tot_water_in: float = d_i.hydro_step.q_in + d_i.hydro_step.forc_flux
            if i not in self.surface_zone_ids:
                lat_dep_row = self.lat_mat[i]
                vert_dep_row = self.vert_mat[i]

                lat_mass_in = lat_mass.T @ lat_dep_row
                vert_mass_in = vert_mass.T @ vert_dep_row

                mass_in = lat_mass_in + vert_mass_in

                if tot_water_in > 1e-6:
                    conc_in = mass_in / tot_water_in
                else:
                    conc_in = np.zeros_like(mass_in)
                d_i.conc_in = conc_in  # type: ignore

                if verbose:
                    print(f"Mass entering zone '{zone.name}': \n{mass_in=}")
                    print(f"Lateral component entering: {lat_mass_in}")
                    print(f"Vertical component entering: {vert_mass_in}")

            else:  # For surface zones, the incoming precipitation provides the mass flux
                mass_in = d_i.conc_in * tot_water_in

            if verbose:
                print(f"Incoming concentration to zone {i}: {d_i.conc_in}")

            try:
                step: RtStep = zone.step(s_i, d_i, dt_days, verbose=verbose)
            except OptimizationError as e:
                raise RtNumericalError(
                    "other",
                    model_type=type(self),
                    zone=zone,
                    parameters=self.to_array(),
                    state=s_i,
                    rt_forcing=d_i.copy(),
                    math_err=e,
                ) from e
            except Exception as e:
                print(f"Failed on zone {i} with values:")
                print(f"{s_i=}")
                print(f"{d_i=}")
                print(f"{dt_days=}")
                raise e
            lat_mass[i] = step.lat_mass
            vert_mass[i] = step.vert_mass
            steps.append(step)
            if verbose:
                print(f"Lateral mass transfer outwards: {step.lat_mass}")
                print(f"Vertical mass transfer outwards: {step.vert_mass}")
                print(
                    f"Finished zone '{zone.name}', final concentration: {
                        np.array2string(
                            step.state, formatter={'all': lambda x: f'{x:.2e}'}
                        )
                    }"
                )

                print("\n\n")
        if verbose:
            print("=" * 25 + "\n")

        return steps

    def run_rt_model(
        self,
        # Forcing data
        hydro_res: HydroModelResults,
        precip_conc: NDArray,
        # Model states and parameters
        initial_state: dict[
            str, ChemicalState
        ],  # Initial chemical state for each species
        # Model performance-related things
        meas_river_conc: Optional[DataFrame] = None,
        objective_functions: list[
            tuple[str, Callable[[Series, Series], float]]
        ] = DEFAULT_OBJECTIVE_FUNCTIONS,
        # Optional diagnostic data
        verbose: bool = False,
        return_partial: bool = False,
        failed_dir: Optional[str] = None,
    ) -> RtModelResults:
        """Run the reactive transport simulation forwards
        `mineral_conc` is a dictionary containing the mineral volume fractions (0,porosity) of the minerals
        of each of the zones
        """
        # ==== Set up the forcing ==== #
        zones_list: list[RtZone] = list(self.rt_zones.values())

        hydro_sim_df: DataFrame = hydro_res.simulation
        forc: ForcingData | list[ForcingData] = hydro_res.forcing

        rt_forcing = self.construct_rt_forcing_matrix(
            hydro_sim_df=hydro_sim_df,
            forc=forc,
            precip_conc=precip_conc,
            model_rt_params=self.rt_zones,
        )

        dates: DatetimeIndex = hydro_sim_df.index  # type: ignore
        num_steps: int = dates.size
        time_deltas: TimedeltaIndex = (
            dates[1:] - dates[:-1]
        )  # Time deltas are calculated in nanoseconds
        time_delta_seconds: NDArray[np.float64] = (
            time_deltas.total_seconds().to_numpy()
        )  # Need the time step in terms of seconds
        time_step_days: list[float] = [1.0] + (time_delta_seconds / 86_400.0).astype(  # type: ignore
            np.float64
        ).tolist()  # Now, need the time delta int terms of days

        # Construct the initial concentrations
        contains_minerals: bool = self.network.num_minerals > 0
        zone_initial_states: list[np.ndarray] = []
        for zone_name in self.zone_names:
            do_reactions: bool = self.rt_zones[zone_name].do_reactions
            do_speciation: bool = self.rt_zones[zone_name].do_speciation
            if zone_name in initial_state and do_reactions and contains_minerals:
                if zone_name not in initial_state and contains_minerals:
                    raise ValueError(
                        f"You must pass mineral concentrations to your simulation for zone {zone_name}"
                    )

            zone_conc: ChemicalState
            if zone_name in initial_state:
                zone_conc = initial_state[zone_name]
            else:
                zone_conc = ChemicalState()

            zone_state: dict[str, float] = dict()

            # Primary aqueous species
            for prim_aq in self.network.primary_aqueous:
                if prim_aq.name in zone_conc.primary:
                    zone_state[prim_aq.name] = zone_conc.primary[prim_aq.name]
                else:
                    zone_state[prim_aq.name] = 1e-3  # default value

            # Secondary species
            for sec in self.network.secondary:
                if sec.name in zone_conc.secondary:
                    zone_state[sec.name] = zone_conc.secondary[sec.name]
                else:
                    zone_state[sec.name] = 1e-3  # default value

            # Mineral species
            for min_val in self.network.mineral:
                if min_val.name in zone_conc.mineral:
                    zone_state[min_val.name] = zone_conc.mineral[min_val.name]
                else:
                    if do_reactions and contains_minerals:
                        raise ValueError(
                            f"Missing initial concentration of mineral {min_val.name} for reactive transport simulation"
                        )
                    else:
                        # Zone is not doing reactions, so it does not need minerals
                        zone_state[min_val.name] = 1e-20

            # Equilibrium adsorption species
            if self.network.has_exchange:
                # Make sure that the exchange site concentration is set
                if "X-" not in zone_conc.exchange and do_speciation:
                    raise ValueError(
                        f"You must specify the initial exchange site concentraitons ('X-') for zone {zone_name} or turn speciation off"
                    )

                if do_speciation:
                    zone_state["X-"] = zone_conc.exchange["X-"]
                else:
                    zone_state["X-"] = 1e-20

                for exch in self.network.exchange_species:
                    if exch.name in zone_conc.exchange:
                        zone_state[exch.name] = zone_conc.exchange[exch.name]
                    else:
                        zone_state[exch.name] = 1e-6  # default value

            zone_init_conc_list: list[float] = []
            for spec_name in self.network.species_order:
                zone_init_conc_list.append(zone_state[spec_name])

            zone_initial_states.append(np.array(zone_init_conc_list, dtype=np.float64))

        state: np.ndarray = np.array(zone_initial_states)
        if verbose:
            print("Initial state: ")
            for zone_name, zone_conc in zip(self.zone_names, state):
                print(f"{zone_name}: {zone_conc}")
        # ============================ #

        # ==== Run the model forwards ==== #
        steps: list[list[RtStep]] = []
        old_states: list[NDArray] = []
        new_states: list[NDArray] = []

        final_completed: int = len(dates)
        try:
            for i in range(len(hydro_sim_df)):
                # print(f"Old state: {state[:, 0]}")
                # ds_for_step: list[RtForcing] = [x.copy() for x in rt_forcing[:, i]]
                ds_for_step: list[RtForcing] = rt_forcing[:, i].tolist()

                dt_days: float = time_step_days[i]
                if dt_days < 1e-6:
                    raise ValueError(f"Time step on step {i} is too small: {dt_days}")

                old_states.append(state)

                # print(f"Surface state before: {state[1]}")

                if verbose:
                    print(f"Starting step {i}")
                step: list[RtStep] = self.step_rt_model(
                    self.rt_zones,
                    state,
                    ds_for_step,
                    time_step_days[i],
                    verbose=verbose,
                    failed_dir=failed_dir,
                )
                steps.append(step)
                new_state = np.array([x.state.copy() for x in step]).copy()
                # print(f"Surface state after: {new_state[1]}")

                new_states.append(new_state[:])
                state[:] = new_state[:]
                # print(f"New state: {state[:, 0]}")

                if verbose:
                    print(f"Finished step {i}")

        except Exception as e:
            if DO_LOGGING:
                print(f"Failed on step {i} with error: {e}")  # type: ignore
            if return_partial:
                final_completed = i  # type: ignore
            else:
                # print(f"Failed on step {i}")  # type: ignore
                raise e

        # ================================ #

        # ==== Construct outputs ==== #
        species_names: list[str] = zones_list[0].all_species
        mineral_names: list[str] = zones_list[0].mineral_species
        zone_names: list[str] = self.zone_names
        data_cols: dict[str, NDArray] = {}

        # Get the concentrations of each species
        for zone_id, zone_name in enumerate(zone_names):
            for species_id, species_name in enumerate(species_names):
                conc: NDArray = np.empty(num_steps, dtype=np.float64)
                mass_in: NDArray = np.empty(num_steps, dtype=np.float64)
                lat_mass: NDArray = np.empty(num_steps, dtype=np.float64)
                lat_conc: NDArray = np.empty(num_steps, dtype=np.float64)
                vert_mass: NDArray = np.empty(num_steps, dtype=np.float64)
                vert_conc: NDArray = np.empty(num_steps, dtype=np.float64)
                tot_moles: NDArray = np.empty(num_steps, dtype=np.float64)
                for i in range(final_completed):
                    single_step: RtStep = steps[i][zone_id]
                    conc[i] = single_step.state[species_id]
                    mass_in[i] = single_step.mass_in[species_id]
                    lat_mass[i] = single_step.lat_mass[species_id]
                    lat_conc[i] = single_step.lat_conc[species_id]
                    vert_mass[i] = single_step.vert_mass[species_id]
                    vert_conc[i] = single_step.vert_conc[species_id]
                    tot_moles[i] = single_step.total_moles[species_id]

                col_name: str = f"{species_name}_{zone_name}"
                data_cols[col_name] = conc
                data_cols[f"{species_name}_{zone_name}_mass_in"] = mass_in
                data_cols[f"{species_name}_{zone_name}_lat_mass"] = lat_mass
                data_cols[f"{species_name}_{zone_name}_lat_conc"] = lat_conc
                data_cols[f"{species_name}_{zone_name}_vert_mass"] = vert_mass
                data_cols[f"{species_name}_{zone_name}_vert_conc"] = vert_conc
                data_cols[f"{species_name}_{zone_name}_total_moles"] = tot_moles

            # Get the mineral species
            for species_id, species_name in enumerate(mineral_names):
                conc = np.empty(num_steps, dtype=np.float64)
                for i in range(final_completed):
                    single_step = steps[i][zone_id]
                    conc[i] = single_step.mineral_rates[species_id]

                col_name = f"{species_name}_rate_{zone_name}"
                data_cols[col_name] = conc

        res_df: DataFrame = DataFrame(data=data_cols, index=dates)

        # Quantify the river concentrations
        river_zone_ids = self.get_river_zone_ids()
        river_zone_names: list[str] = [self.zone_names[i] for i in river_zone_ids]
        q_components: dict[str, Series] = {
            n: hydro_sim_df[f"q_lat_{n}"] for n in river_zone_names
        }
        q_sim: Series = hydro_sim_df["sim_streamflow_mmd"]
        flow_fractions: dict[str, Series] = {
            name: val / q_sim for name, val in q_components.items()
        }

        for spec_name in self.network.species_names:
            comp_concs: dict[str, Series] = {}
            for n in river_zone_names:
                comp_concs[n] = res_df[f"{spec_name}_{n}"]

            conc_comps: list[Series] = [
                flow_fractions[n] * comp_concs[n] for n in river_zone_names
            ]

            weighted_comps: Series = 0.0  # type: ignore
            for cc in conc_comps:
                weighted_comps += cc

            res_df[f"{spec_name}_riv"] = weighted_comps

        # Calculate the objective functions
        if meas_river_conc is not None:
            test_species: list[str] = list(meas_river_conc.columns)  # type: ignore
            obj_df_cols: list[str] = [x[0] for x in objective_functions]
            obj_df_index: list[str] = test_species
            data = np.full((len(obj_df_index), len(obj_df_cols)), fill_value=np.nan)

            for j, (_obj_name, obj_func) in enumerate(objective_functions):
                for i, spec in enumerate(test_species):
                    sim_col_name: str = f"{spec}_riv"
                    c_meas: Series = meas_river_conc[spec]
                    c_sim: Series = res_df[sim_col_name]
                    data[i, j] = obj_func(c_meas, c_sim)

            obj_df = DataFrame(columns=obj_df_cols, index=obj_df_index, data=data)
        else:
            obj_df = None

        rt_sim_res: RtModelResults = RtModelResults(
            simulation=res_df,
            objective_functions=obj_df,
            rt_forcings=rt_forcing,
            steps=steps,
            other={"old_states": old_states, "new_states": new_states},
        )
        # =========================== #

        return rt_sim_res  # type: ignore
