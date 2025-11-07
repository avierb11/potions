from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from pandas import DataFrame, Series
from numpy.typing import NDArray

from potions.hydro import GroundZone, SnowZone, SoilZone
from .model import Zone, Layer, Hillslope, Model, ForcingData, run_hydro_model


class HydrologicModel(ABC):
    @abstractmethod
    def to_array(self) -> NDArray:
        """
        Convert the parameters in this model to a numpy array for calibration
        """
        pass

    @classmethod
    def from_array(cls, arr: NDArray) -> HydrologicModel:
        """Take a numpy array and convert it to a HydrologicModel"""
        raise NotImplementedError()

    @classmethod
    def parameter_names(cls) -> list[str]:
        """
        Return the names of the parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def run(
        self,
        init_state: NDArray,
        forc: ForcingData,
        dates: Series,
        verbose: bool = False,
    ) -> DataFrame:
        """
        Run the model with the given parameters
        """
        pass

    @classmethod
    def default_parameter_range(cls) -> list[tuple[float, float]]:
        """
        Return a list of ranges for each parameter
        """
        raise NotImplementedError()


@dataclass
class HbvModel(HydrologicModel):
    tt: float
    fmax: float
    fc: float
    lp: float
    beta: float
    k0: float
    thr: float
    k1: float
    perc: float
    k2: float

    @classmethod
    def default_parameter_range(cls) -> list[tuple[float, float]]:
        """
        Return a list of ranges for each parameter
        """
        return [
            (-1, 1),  # TT
            (0.1, 5.0),  # FMAX
            (50, 1000),  # FC
            (0.1, 0.95),  # LP
            (0.5, 2.5),  # BETA
            (0.01, 0.25),  # K0
            (0, 50),  # Soil threshold
            (1e-3, 0.1),  # K1
            (0.1, 2.5),  # PERC
            (1e-5, 1e-2),  # K2
        ]

    def to_array(self) -> NDArray:
        """
        Convert this set of arrays to a numpy array
        """
        return np.array(
            [
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
            ],
            dtype=float,
        )

    @classmethod
    def from_array(cls, arr: NDArray) -> HydrologicModel:
        tt: float = arr[0]
        fmax: float = arr[1]
        fc: float = arr[2]
        lp: float = arr[3]
        beta: float = arr[4]
        k0: float = arr[5]
        thr: float = arr[6]
        k1: float = arr[7]
        perc: float = arr[8]
        k2: float = arr[9]

        return HbvModel(
            tt=tt,
            fmax=fmax,
            fc=fc,
            lp=lp,
            beta=beta,
            k0=k0,
            thr=thr,
            k1=k1,
            perc=perc,
            k2=k2,
        )

    @staticmethod
    def default() -> HbvModel:
        """
        Return the default set of parameters
        """
        return HbvModel(
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

    @classmethod
    def parameter_names(cls) -> list[str]:
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

    def run(
        self,
        init_state: NDArray,
        forc: ForcingData,
        dates: Series,
        verbose: bool = False,
    ) -> DataFrame:
        """
        Run the model with the given parameters
        """
        """
        Run a model simulating the HBV model structure
        """
        snow_zone: SnowZone = SnowZone(tt=self.tt, fmax=self.fmax)
        soil_zone: SoilZone = SoilZone(
            tt=self.tt, fc=self.fc, lp=self.lp, beta=self.beta, k0=self.k0, thr=self.thr
        )
        shallow_zone: GroundZone = GroundZone(k=self.k1, alpha=0.0, perc=self.perc)
        deep_zone: GroundZone = GroundZone(k=self.k2, alpha=0.0, perc=0.0)

        hs: Hillslope = Hillslope(
            [
                Layer([snow_zone]),
                Layer([soil_zone]),
                Layer([shallow_zone]),
                Layer([deep_zone]),
            ]
        )  # type: ignore

        model = Model([hs], scales=[[1.0]], verbose=verbose)

        if verbose:
            print(f"Zone lateral connection matrix: \n{model.lat_mat}")
            print(f"Zone vertical connection matrix: \n{model.vert_mat}")
            print(f"Precipitation matrix: \n{model.precip_mat}")
            print(f"PET matrix: \n{model.pet_mat}")
            print(f"Temperature matrix: \n{model.temp_mat}")
            print(f"Model scales: {model.scales}")

        model_res: DataFrame = run_hydro_model(
            model=model, init_state=init_state, forc=[forc], dates=dates, dt=1.0
        )

        model_res["discharge"] = (
            model_res["q_lat_snow_0"]
            + model_res["q_lat_soil_1"]
            + model_res["q_lat_ground_2"]
            + model_res["q_lat_ground_3"]
        )

        return model_res


def run_hbv_model(
    params: HbvModel,
    init_state: NDArray,
    forc: ForcingData,
    dates: Series,
    verbose: bool = False,
) -> DataFrame:
    """
    Run a model simulating the HBV model structure
    """
    snow_zone: SnowZone = SnowZone(tt=params.tt, fmax=params.fmax)
    soil_zone: SoilZone = SoilZone(
        tt=params.tt,
        fc=params.fc,
        lp=params.lp,
        beta=params.beta,
        k0=params.k0,
        thr=params.thr,
    )
    shallow_zone: GroundZone = GroundZone(k=params.k1, alpha=0.0, perc=params.perc)
    deep_zone: GroundZone = GroundZone(k=params.k2, alpha=0.0, perc=0.0)

    hs: Hillslope = Hillslope(
        [
            Layer([snow_zone]),
            Layer([soil_zone]),
            Layer([shallow_zone]),
            Layer([deep_zone]),
        ]
    )  # type: ignore

    model = Model([hs], scales=[[1.0]], verbose=verbose)

    if verbose:
        print(f"Zone lateral connection matrix: \n{model.lat_mat}")
        print(f"Zone vertical connection matrix: \n{model.vert_mat}")
        print(f"Precipitation matrix: \n{model.precip_mat}")
        print(f"PET matrix: \n{model.pet_mat}")
        print(f"Temperature matrix: \n{model.temp_mat}")
        print(f"Model scales: {model.scales}")

    return run_hydro_model(
        model=model, init_state=init_state, forc=[forc], dates=dates, dt=1.0
    )
