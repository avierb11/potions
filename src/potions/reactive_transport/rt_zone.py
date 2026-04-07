from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

from ..utils import HydrologyError, setup_logging

# from ..common_types_compiled import HydroStep

from potions.core import HydroStep, RtZone


setup_logging("rt_zone.py")


class WaterVolumeError(HydrologyError):
    """Error representing when the calculated water volume exceeds the maximum
    value allowed by the zone geometry"""

    pass


@dataclass(frozen=True)
class MiscData:
    # Matrix describing the stoichiometry of the mineral dissolution reactions
    mineral_stoichiometry: NDArray
    # Boolean vector describing whether each species is mobile or not. All aqueous species are mobile, and all mineral species are immobile.
    species_mobility: NDArray
    mineral_molar_mass: NDArray  # The molar mass of each of the minerals
    # The rate constants for the mineral reactions. Note that these are _not_ parameters. They are constants.
    rate_const: NDArray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MiscData):
            raise TypeError(f"Cannot compare MiscData with '{type(other)}'")
        else:
            vals: list[bool] = [
                np.allclose(self.mineral_stoichiometry,
                            other.mineral_stoichiometry),
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


# ==== Functions ==== #
def calculate_moisture_fraction(
    zone_params: dict[str, RtZone], sim_res: DataFrame, raise_err: bool = True
) -> NDArray:
    """Calculate the soil moisture values"""
    sw_vals: NDArray = np.zeros((len(zone_params), len(sim_res)))
    zone_names: list[str] = list(zone_params.keys())
    zone_name: str
    zone: RtZone
    for i, (zone_name, zone) in enumerate(zone_params.items()):  # type: ignore
        col_name: str = f"s_{zone_name}"
        max_water_volume = zone.parameters.dimensions.max_water_volume
        storage: Series = sim_res[col_name]

        if storage.max() >= max_water_volume:
            raise WaterVolumeError(
                f"Water volume error in zone '{zone_names[i]}': Maximum volume is {
                    round(max_water_volume, 1)
                } mm but maximum simulated storage is {
                    round(storage.max(), 1)
                } mm. Increase porosity or volume"
            )

        sw_vals[i] = (storage + zone.parameters.dimensions.passive_water_storage) / (
            max_water_volume + zone.parameters.dimensions.passive_water_storage
        )

    return sw_vals


def calculate_water_table_depth(
    zone_params: dict[str, RtZone], sim_res: DataFrame
) -> NDArray:
    # zw_vals: dict[HbvZone, Series] = {}
    zw_vals: NDArray = np.zeros(
        (len(zone_params), len(sim_res)), dtype=np.float64)
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

        if zw_vals[i].min() < 0:
            raise WaterVolumeError("Illegal water volume")

    return zw_vals


def get_hydro_steps(sim_res: DataFrame) -> NDArray:
    zone_names: list[str] = [
        c.replace("s_", "") for c in sim_res.columns if c.startswith("s_")
    ]
    hydro_steps: NDArray = np.empty(
        (len(zone_names), len(sim_res)), dtype=object)

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
