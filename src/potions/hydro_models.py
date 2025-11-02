from __future__ import annotations
from dataclasses import dataclass
from turtle import st
import numpy as np
from pandas import DataFrame, Series
from numpy.typing import NDArray

from potions.hydro import GroundZone, SnowZone, SoilZone
from .model import Zone, Layer, Hillslope, Model, ForcingData, run_hydro_model

@dataclass
class HbvParameters:
    tt: float
    fmax: float
    fc: float
    lp: float
    beta: float
    k0: float
    thr: float
    k1: float
    k2: float
    perc: float

    def to_array(self) -> NDArray:
        """
        Convert this set of arrays to a numpy array
        """
        return np.array([
            # Snow zone
            self.tt,
            self.fmax,
            # Soil parameters
            self.fc,
            self.lp,
            self.beta,
            self.k0,
            self.thr,
            # Shallow zone
            self.k1,
            self.perc,
            # Deep zone
            self.k2,
        ], dtype=float)
    
    @staticmethod
    def default() -> HbvParameters:
        """
        Return the default set of parameters
        """
        return HbvParameters(
            tt=0.0,
            fmax=1.0,
            fc=100.0,
            lp=0.5,
            beta=1.0,
            k0=0.1,
            thr=10.0,
            k1=0.01,
            perc=1.0,
            k2=1e-3,
        )
    
    @staticmethod
    def parameter_names() -> list[str]:
        """
        Return the names of the parameters
        """
        return [
            "tt",
            "fmax",
            "fc",
            "lp",
            "beta",
            "k0",
            "thr",
            "k1",
            "perc",
            "k2",
        ]

    def run(self, init_state: NDArray, forc: ForcingData, dates: Series, verbose: bool = False) -> DataFrame:
        """
        Run the model with the given parameters
        """
        """
        Run a model simulating the HBV model structure
        """
        snow_zone: SnowZone = SnowZone(tt=self.tt, fmax=self.fmax)
        soil_zone: SoilZone = SoilZone(tt=self.tt, fc=self.fc, lp=self.lp, beta=self.beta, k0=self.k0, thr=self.thr)
        shallow_zone: GroundZone = GroundZone(k=self.k1, alpha=0.0, perc=self.perc)
        deep_zone: GroundZone = GroundZone(k=self.k2, alpha=0.0, perc=0.0)


        hs: Hillslope = Hillslope([
            Layer([snow_zone]),
            Layer([soil_zone]),
            Layer([shallow_zone]),
            Layer([deep_zone])
        ]) # type: ignore

        model = Model([hs], scales=[[1.0]], verbose=verbose)

        if verbose:
            print(f"Zone lateral connection matrix: \n{model.lat_mat}")
            print(f"Zone vertical connection matrix: \n{model.vert_mat}")
            print(f"Precipitation matrix: \n{model.precip_mat}")
            print(f"PET matrix: \n{model.pet_mat}")
            print(f"Temperature matrix: \n{model.temp_mat}")
            print(f"Model scales: {model.scales}")

        return run_hydro_model(
            model=model,
            init_state=init_state,
            forc=[forc],
            dates=dates,
            dt=1.0
        )


def run_hbv_model(
    params: HbvParameters,
    init_state: NDArray,
    forc: ForcingData,
    dates: Series,
    verbose: bool = False
) -> DataFrame:
    """
    Run a model simulating the HBV model structure
    """
    snow_zone: SnowZone = SnowZone(tt=params.tt, fmax=params.fmax)
    soil_zone: SoilZone = SoilZone(tt=params.tt, fc=params.fc, lp=params.lp, beta=params.beta, k0=params.k0, thr=params.thr)
    shallow_zone: GroundZone = GroundZone(k=params.k1, alpha=0.0, perc=params.perc)
    deep_zone: GroundZone = GroundZone(k=params.k2, alpha=0.0, perc=0.0)


    hs: Hillslope = Hillslope([
        Layer([snow_zone]),
        Layer([soil_zone]),
        Layer([shallow_zone]),
        Layer([deep_zone])
    ]) # type: ignore

    model = Model([hs], scales=[[1.0]], verbose=verbose)

    if verbose:
        print(f"Zone lateral connection matrix: \n{model.lat_mat}")
        print(f"Zone vertical connection matrix: \n{model.vert_mat}")
        print(f"Precipitation matrix: \n{model.precip_mat}")
        print(f"PET matrix: \n{model.pet_mat}")
        print(f"Temperature matrix: \n{model.temp_mat}")
        print(f"Model scales: {model.scales}")

    return run_hydro_model(
        model=model,
        init_state=init_state,
        forc=[forc],
        dates=dates,
        dt=1.0
    )
