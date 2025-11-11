from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Literal
import numpy as np
from pandas import DataFrame, Series
from numpy.typing import NDArray
import scipy.optimize as opt

from .hydro import GroundZone, SnowZone, SoilZone
from .objective_functions import kge, nse
from .model import Layer, Hillslope, Model, ForcingData, run_hydro_model


@dataclass
class HydroModelResults:
    simulation: DataFrame
    kge: float
    nse: float
    bias: float
    r_squared: float
    spearman_rho: float


def objective_function(
    x: NDArray, cls, forc, meas_streamflow, metric, print_value: bool
) -> float:
    model = cls.from_array(x)
    results: HydroModelResults = model.run(
        init_state=cls.default_init_state(),
        forc=forc,
        streamflow=meas_streamflow,
        verbose=False,
    )

    obj_val: float

    if metric == "kge":
        obj_val = -results.kge
    elif metric == "nse":
        obj_val = -results.nse
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if print_value:
        print(f"{metric.upper()}: {-round(obj_val, 2)}")

    return obj_val


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
        forc: ForcingData | Iterable[ForcingData],
        streamflow: Series,
        verbose: bool = False,
    ) -> HydroModelResults:
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

    @classmethod
    def default_init_state(cls) -> NDArray:
        """
        Return the default initial state for this type.
        """
        raise NotImplementedError()

    @classmethod
    def num_zones(cls) -> int:
        """
        Return the number of zones in this model
        """
        raise NotImplementedError()

    @classmethod
    def num_parameters(cls) -> int:
        return 0

    @classmethod
    def simple_calibration(
        cls,
        forc: ForcingData | Iterable[ForcingData],
        meas_streamflow: Series,
        metric: Literal["kge", "nse"],
        num_threads: int = -1,
        polish: bool = False,
        maxiter=10,
        print_values: bool = False,
    ) -> tuple[dict[str, float], HydroModelResults]:
        """
        Run a simple model calibration to get an optimum model fit, NOT for sensitivity analysis
        """

        args = (cls, forc, meas_streamflow, metric, print_values)

        opt_res = opt.differential_evolution(
            func=objective_function,
            bounds=cls.default_parameter_range(),
            maxiter=maxiter,
            tol=0.1,
            rng=0,
            polish=polish,
            workers=num_threads,
            args=args,
        )

        # Now, run the model and return the optimum results
        model = cls.from_array(opt_res.x)
        best_results = model.run(
            forc=forc,
            init_state=cls.default_init_state(),
            streamflow=meas_streamflow,
            verbose=False,
        )

        opt_params = dict(zip(cls.parameter_names(), opt_res.x))

        return opt_params, best_results


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
    def default_init_state(cls) -> NDArray:
        return np.array([0.0, 10.0, 15.0, 25.0])

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

    def run(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        init_state: NDArray,
        forc: ForcingData,
        streamflow: Series,
        verbose: bool = False,
    ) -> HydroModelResults:
        """
        Run the model with the given parameters
        """
        """
        Run a model simulating the HBV model structure
        """

        dates = forc.precip.index
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

        sim_streamflow = model_res["discharge"]

        # Calculate objective functions
        kge_val: float = kge(sim_streamflow, streamflow)
        nse_val: float = nse(sim_streamflow, streamflow)
        pbias_val: float = (sim_streamflow - streamflow).mean() / streamflow.mean()
        r_squared_val: float = streamflow.corr(sim_streamflow) ** 2
        spearman_rho_val: float = streamflow.corr(sim_streamflow, method="spearman")

        return HydroModelResults(
            simulation=model_res,
            kge=kge_val,
            nse=nse_val,
            bias=pbias_val,
            r_squared=r_squared_val,
            spearman_rho=spearman_rho_val,
        )

    @classmethod
    def num_parameters(cls) -> int:
        return 10


@dataclass
class HbvLateralModel(HydrologicModel):
    # Size parameters
    riparian_prop: float

    # Hillslope parameters
    tt_hs: float
    fmax_hs: float
    fc_hs: float
    lp_hs: float
    beta_hs: float
    k0_hs: float
    thr_hs: float
    k1_hs: float
    perc_hs: float
    k2_hs: float

    # Riparian parameters
    tt_rp: float
    fmax_rp: float
    fc_rp: float
    lp_rp: float
    beta_rp: float
    k0_rp: float
    thr_rp: float
    k1_rp: float
    perc_rp: float
    k2_rp: float

    @property
    def hillslope_prop(self) -> float:
        """
        Proportion of the catchment marked as hillslope
        """
        return 1 - self.riparian_prop

    @classmethod
    def default_init_state(cls) -> NDArray:
        return np.array([0.0, 0.0, 10.0, 10.0, 15.0, 15.0, 25.0, 25.0])

    @classmethod
    def default_parameter_range(cls) -> list[tuple[float, float]]:
        """
        Return a list of ranges for each parameter
        """
        return [
            # Size
            (0.1, 0.9),
            # Hillslope
            (-1, 1),  # TT (hillslope)
            (0.1, 5.0),  # FMAX (hillslope)
            (50, 1000),  # FC (hill)
            (0.1, 0.95),  # LP (hill)
            (0.5, 2.5),  # BETA (hill)
            (0.01, 0.25),  # K0 (hill)
            (0, 50),  # Soil threshold (hill)
            (1e-3, 0.1),  # K1 (hil)
            (0.1, 2.5),  # PERC (hil)
            (1e-5, 1e-2),  # K2 (hill)
            # Riparian
            (-1, 1),  # TT (riparian)
            (0.1, 5.0),  # FMAX (riparian)
            (50, 1000),  # FC (rip)
            (0.1, 0.95),  # LP (rip)
            (0.5, 2.5),  # BETA (rip)
            (0.01, 0.25),  # K0 (rip)
            (0, 50),  # Soil threshold (rip)
            (1e-3, 0.1),  # K1 (rip)
            (0.1, 2.5),  # PERC (rip)
            (1e-5, 1e-2),  # K2 (rip)
        ]

    def to_array(self) -> NDArray:
        """
        Convert this set of arrays to a numpy array
        """
        return np.array(
            [
                # Hillslope
                self.tt_hs,
                self.fmax_hs,
                self.fc_hs,
                self.lp_hs,
                self.beta_hs,
                self.k0_hs,
                self.thr_hs,
                self.k1_hs,
                self.perc_hs,
                self.k2_hs,
                # Riparian
                self.tt_rp,
                self.fmax_rp,
                self.fc_rp,
                self.lp_rp,
                self.beta_rp,
                self.k0_rp,
                self.thr_rp,
                self.k1_rp,
                self.perc_rp,
                self.k2_rp,
            ],
            dtype=float,
        )

    @classmethod
    def from_array(cls, arr: NDArray) -> HydrologicModel:
        riparian_prop: float = arr[0]

        hs_params = arr[1:11]
        tt_hs: float = hs_params[0]
        fmax_hs: float = hs_params[1]
        fc_hs: float = hs_params[2]
        lp_hs: float = hs_params[3]
        beta_hs: float = hs_params[4]
        k0_hs: float = hs_params[5]
        thr_hs: float = hs_params[6]
        k1_hs: float = hs_params[7]
        perc_hs: float = hs_params[8]
        k2_hs: float = hs_params[9]

        rp_params = arr[11:]
        tt_rp: float = rp_params[0]
        fmax_rp: float = rp_params[1]
        fc_rp: float = rp_params[2]
        lp_rp: float = rp_params[3]
        beta_rp: float = rp_params[4]
        k0_rp: float = rp_params[5]
        thr_rp: float = rp_params[6]
        k1_rp: float = rp_params[7]
        perc_rp: float = rp_params[8]
        k2_rp: float = rp_params[9]

        return HbvLateralModel(
            # Size
            riparian_prop=riparian_prop,
            # Hillslope
            tt_hs=tt_hs,
            fmax_hs=fmax_hs,
            fc_hs=fc_hs,
            lp_hs=lp_hs,
            beta_hs=beta_hs,
            k0_hs=k0_hs,
            thr_hs=thr_hs,
            k1_hs=k1_hs,
            perc_hs=perc_hs,
            k2_hs=k2_hs,
            # Riparian
            tt_rp=tt_rp,
            fmax_rp=fmax_rp,
            fc_rp=fc_rp,
            lp_rp=lp_rp,
            beta_rp=beta_rp,
            k0_rp=k0_rp,
            thr_rp=thr_rp,
            k1_rp=k1_rp,
            perc_rp=perc_rp,
            k2_rp=k2_rp,
        )

    @staticmethod
    def default() -> HbvLateralModel:
        """
        Return the default set of parameters
        """
        return HbvLateralModel(
            # Size
            riparian_prop=0.5,
            # Hillslope
            tt_hs=0.0,
            fmax_hs=1.0,
            fc_hs=100.0,
            lp_hs=0.5,
            beta_hs=1.0,
            k0_hs=0.1,
            thr_hs=10.0,
            k1_hs=0.01,
            perc_hs=1.0,
            k2_hs=1e-3,
            # Riparian
            tt_rp=0.0,
            fmax_rp=1.0,
            fc_rp=100.0,
            lp_rp=0.5,
            beta_rp=1.0,
            k0_rp=0.1,
            thr_rp=10.0,
            k1_rp=0.01,
            perc_rp=1.0,
            k2_rp=1e-3,
        )

    @classmethod
    def parameter_names(cls) -> list[str]:
        """
        Return the names of the parameters
        """
        return [
            # Size
            "riparian_prop",
            # Hillslope
            "tt_hs",
            "fmax_hs",
            "fc_hs",
            "lp_hs",
            "beta_hs",
            "k0_hs",
            "thr_hs",
            "k1_hs",
            "perc_hs",
            "k2_hs",
            # Riparian
            "tt_rp",
            "fmax_rp",
            "fc_rp",
            "lp_rp",
            "beta_rp",
            "k0_rp",
            "thr_rp",
            "k1_rp",
            "perc_rp",
            "k2_rp",
        ]

    def run(
        self,
        init_state: NDArray,
        forc: ForcingData | Iterable[ForcingData],
        streamflow: Series,
        verbose: bool = False,
    ) -> HydroModelResults:
        """
        Run the model with the given parameters
        """
        forcing: tuple[ForcingData, ForcingData]
        if isinstance(forc, ForcingData):
            forcing = (forc, forc)
        else:
            forcing = tuple(forc)  # type: ignore

        dates = forcing[0].precip.index

        snow_zone_hs: SnowZone = SnowZone(tt=self.tt_hs, fmax=self.fmax_hs)
        soil_zone_hs: SoilZone = SoilZone(
            tt=self.tt_hs,
            fc=self.fc_hs,
            lp=self.lp_hs,
            beta=self.beta_hs,
            k0=self.k0_hs,
            thr=self.thr_hs,
        )
        shallow_zone_hs: GroundZone = GroundZone(
            k=self.k1_hs, alpha=0.0, perc=self.perc_hs
        )
        deep_zone_hs: GroundZone = GroundZone(k=self.k2_hs, alpha=0.0, perc=0.0)

        snow_zone_rp: SnowZone = SnowZone(tt=self.tt_rp, fmax=self.fmax_rp)
        soil_zone_rp: SoilZone = SoilZone(
            tt=self.tt_rp,
            fc=self.fc_rp,
            lp=self.lp_rp,
            beta=self.beta_rp,
            k0=self.k0_rp,
            thr=self.thr_rp,
        )
        shallow_zone_rp: GroundZone = GroundZone(
            k=self.k1_rp, alpha=0.0, perc=self.perc_rp
        )
        deep_zone_rp: GroundZone = GroundZone(k=self.k2_rp, alpha=0.0, perc=0.0)

        hs: Hillslope = Hillslope(
            [
                Layer([snow_zone_hs, snow_zone_rp]),
                Layer([soil_zone_hs, soil_zone_rp]),
                Layer([shallow_zone_hs, shallow_zone_rp]),
                Layer([deep_zone_hs, deep_zone_rp]),
            ]
        )  # type: ignore

        model = Model(
            [hs], scales=[[self.hillslope_prop, self.riparian_prop]], verbose=verbose
        )

        if verbose:
            print(f"Zone lateral connection matrix: \n{model.lat_mat}")
            print(f"Zone vertical connection matrix: \n{model.vert_mat}")
            print(f"Precipitation matrix: \n{model.precip_mat}")
            print(f"PET matrix: \n{model.pet_mat}")
            print(f"Temperature matrix: \n{model.temp_mat}")
            print(f"Model scales: {model.scales}")

        model_res: DataFrame = run_hydro_model(
            model=model, init_state=init_state, forc=list(forcing), dates=dates, dt=1.0
        )

        model_res["discharge"] = (
            model_res["q_lat_snow_1"]
            + model_res["q_lat_soil_3"]
            + model_res["q_lat_ground_5"]
            + model_res["q_lat_ground_7"]
        )

        sim_streamflow = model_res["discharge"]

        # Calculate objective functions
        kge_val: float = kge(sim_streamflow, streamflow)
        nse_val: float = nse(sim_streamflow, streamflow)
        pbias_val: float = (sim_streamflow - streamflow).mean() / streamflow.mean()
        r_squared_val: float = streamflow.corr(sim_streamflow) ** 2
        spearman_rho_val: float = streamflow.corr(sim_streamflow, method="spearman")

        return HydroModelResults(
            simulation=model_res,
            kge=kge_val,
            nse=nse_val,
            bias=pbias_val,
            r_squared=r_squared_val,
            spearman_rho=spearman_rho_val,
        )

    @classmethod
    def num_parameters(cls) -> int:
        return 21  # 1 size, 10 for Hillslope HBV zone, 10 for riparian HBV zone


def run_hbv_model(
    params: HbvModel,
    init_state: NDArray,
    forc: ForcingData,
    dates: Series,
    verbose: bool = False,
) -> DataFrame:
    """
    Run a model simulating the HBV model structure using a daily time step
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
