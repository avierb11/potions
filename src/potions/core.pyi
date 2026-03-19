from dataclasses import dataclass
import polars as pl
import numpy as np

@dataclass
class HydroForcing:
    precip: float
    temp: float
    pet: float
    q_in: float

@dataclass
class HydroStep:
    state: float
    forc_flux: float
    lat_flux: float
    vert_flux: float
    vap_flux: float
    q_in: float
    lat_flux_ext: float
    vert_flux_ext: float

@dataclass
class LapseRateParameters:
    temp_factor: float
    precip_factor: float

@dataclass
class RtForcing:
    conc_in: np.ndarray
    hydro_step: HydroStep
    hydro_forc: HydroForcing
    s_w: float
    z_w: float

# Hydrology
@dataclass
class HydrologicZone:
    name: str = "unnamed"

    def step(self, s_0: float, d: HydroForcing, dt: float) -> HydroStep: ...
    def mass_balance(self, s: float, d: HydroForcing) -> float: ...
    def forc_flux(self, s: float, d: HydroForcing) -> float: ...
    def vap_flux(self, s: float, d: HydroForcing) -> float: ...
    def lat_flux(self, s: float, d: HydroForcing) -> float: ...
    def vert_flux(self, s: float, d: HydroForcing) -> float: ...
    def lat_flux_ext(self, s: float, d: HydroForcing) -> float: ...
    def vert_flux_ext(self, s: float, d: HydroForcing) -> float: ...
    def param_list(self) -> list[float]: ...
    @classmethod
    def default(cls: type[HydrologicZone]) -> HydrologicZone: ...
    @classmethod
    def num_parameters(cls: type[HydrologicZone]) -> int: ...
    @classmethod
    def default_parameter_range(
        cls: type[HydrologicZone],
    ) -> dict[str, tuple[float, float]]: ...
    @classmethod
    def base_name(cls: type) -> str: ...
    @classmethod
    def parameter_names(cls) -> list[str]: ...
    @classmethod
    def default_init_state(cls) -> float: ...

@dataclass
class GroundZone(HydrologicZone):
    pass

@dataclass
class GroundZoneB(HydrologicZone):
    pass

@dataclass
class SnowZone(HydrologicZone):
    pass

@dataclass
class SurfaceZone(HydrologicZone):
    pass

# Kinetic Structures
class MonodParameters:
    monod_mat: pl.DataFrame
    inhib_mat: pl.DataFrame

    def __init__(self, monod_mat: pl.DataFrame, inhib_mat: pl.DataFrame) -> None: ...
    def rate(self, chms: np.ndarray) -> np.ndarray: ...

class TstParameters:
    stoich: pl.DataFrame
    dep: pl.DataFrame
    min_eq_const: pl.Series

    def __init__(
        self, stoich: pl.DataFrame, dep: pl.DataFrame, min_eq_const: pl.Series
    ) -> None: ...
    def rate(self, chms: np.ndarray) -> np.ndarray: ...

@dataclass
class EquilibriumParameters:
    stoich: pl.DataFrame
    log_eq_consts: pl.Series
    total: pl.DataFrame

    def __init__(
        self, stoich: pl.DataFrame, log_eq_consts: pl.Series, total: pl.DataFrame
    ): ...
    def solve_equilibrium(self, chms: np.ndarray) -> np.ndarray: ...
    def conc_func(self, x_free: np.ndarray) -> np.ndarray: ...
    def residual(self, x_free: np.ndarray, c_tot: np.ndarray) -> np.ndarray: ...
    def total_mat_shape(self) -> None: ...
    def get_total_mat(self) -> np.ndarray: ...
    def get_stoich_null_space(self) -> np.ndarray: ...
    def get_x_particular(self) -> np.ndarray: ...
    def stoich_null_space_shape(self) -> None: ...
    def log10_k_w_shape(self) -> None: ...
    def x_particular_shape(self) -> None: ...

@dataclass
class MineralAuxParams:
    sw_threshold: float
    sw_exp: float
    n_alpha: float
    q_10: float
    ssa: float

    def to_array(self) -> np.ndarray: ...
    @staticmethod
    def from_array(arr: np.ndarray) -> MineralAuxParams: ...

@dataclass
class ZoneDimensions:
    porosity: float
    depth: float
    passive_water_storage: float

    def to_array(self) -> np.ndarray: ...
    @staticmethod
    def from_array(arr: np.ndarray) -> ZoneDimensions: ...
    @property
    def max_water_volume(self) -> float: ...

@dataclass
class MineralParameters:
    sw_threshold: np.ndarray
    sw_exp: np.ndarray
    n_alpha: np.ndarray
    q_10: np.ndarray
    ssa: np.ndarray

    def to_array(self) -> np.ndarray: ...
    @staticmethod
    def from_array(arr: np.ndarray) -> MineralParameters: ...
    def soil_water_factor(self, forc: RtForcing) -> np.ndarray: ...
    def temperature_factor(self, forc: RtForcing) -> np.ndarray: ...
    def water_table_factor(self, forc: RtForcing) -> np.ndarray: ...
    def factor(self, forc: RtForcing) -> np.ndarray: ...
    @staticmethod
    def from_mineral_parameters(
        minerals: list[MineralAuxParams],
    ) -> MineralParameters: ...

@dataclass
class RtParameters:
    dimensions: ZoneDimensions
    mineral_params: MineralParameters

    def to_array(self) -> np.ndarray: ...
    @staticmethod
    def from_array(arr: np.ndarray) -> RtParameters: ...

@dataclass
class MiscData:
    mineral_stoichiometry: np.ndarray
    species_mobility: np.ndarray
    mineral_molar_mass: np.ndarray
    rate_const: np.ndarray

    def get_mineral_stoichiometry(self) -> np.ndarray: ...
    def get_species_mobility(self) -> np.ndarray: ...
    def get_mineral_molar_mass(self) -> np.ndarray: ...
    def get_rate_const(self) -> np.ndarray: ...

# Reactive transport
class RtZone:
    network: ReactionNetwork
    parameters: RtParameters
    do_reactions: bool
    do_speciation: bool
    name: str
    monod: MonodParameters
    tst: TstParameters
    eq: EquilibriumParameters
    aux: MineralParameters
    misc: MiscData

    def __init__(
        self,
        network: ReactionNetwork,
        params: RtParameters,
        do_reactions: bool = True,
        do_speciation: bool = True,
        name: str = "unnamed",
    ) -> None: ...
    def mass_balance_ode(self, chms: np.ndarray, d: RtForcing) -> np.ndarray: ...
    def reaction_rate(
        self, chms: np.ndarray, d: RtForcing, minerals_only: bool = False
    ) -> np.ndarray: ...
    def transport_rate(self, chms: np.ndarray, d: RtForcing) -> np.ndarray: ...
    def step(self, c_0: np.ndarray, d: RtForcing, dt_days: float) -> RtStep: ...
    @property
    def all_species(self) -> list[str]: ...
    @property
    def mineral_species(self) -> list[str]: ...
    @property
    def dimensions(self) -> ZoneDimensions: ...
    @property
    def num_species(self) -> int: ...
    def to_array(self) -> np.ndarray: ...
    @staticmethod
    def from_array(
        arr: np.ndarray,
        network: ReactionNetwork,
        do_reactions: bool,
        do_speciation: bool,
    ) -> RtZone: ...

@dataclass
class RtStep:
    state: np.ndarray
    conc_in: np.ndarray
    mass_in: np.ndarray
    lat_conc: np.ndarray
    vert_conc: np.ndarray
    lat_mass: np.ndarray
    vert_mass: np.ndarray
    mineral_rates: np.ndarray

class ReactionNetwork:
    primary_aqueous: list[PrimaryAqueousSpecies]
    mineral: list[MineralSpecies]
    secondary: list[SecondarySpecies]
    mineral_kinetics: MineralKineticData
    species: pl.DataFrame

    def __init__(
        self,
        primary_aqueous: list[PrimaryAqueousSpecies],
        mineral: list[MineralSpecies],
        secondary: list[SecondarySpecies],
        mineral_kinetics: MineralKineticData,
    ) -> None: ...
    @property
    def species_order(self) -> list[str]: ...
    @property
    def has_exchange(self) -> bool: ...
    @property
    def charges(self) -> pl.Series: ...
    @property
    def equilibrium_species(self) -> pl.DataFrame: ...
    @property
    def kinetic_species(self) -> pl.DataFrame: ...
    @property
    def equilibrium_parameters(self) -> EquilibriumParameters: ...
    @property
    def tst_params(self) -> TstParameters: ...
    @property
    def monod_params(self) -> MonodParameters: ...
    @property
    def species_names(self) -> list[str]: ...
    @property
    def mineral_species_names(self) -> list[str]: ...
    @property
    def mineral_stoichiometry(self) -> pl.DataFrame: ...
    @property
    def transport_mask(self) -> np.ndarray: ...
    @property
    def mineral_molar_masses(self) -> np.ndarray: ...
    @property
    def rate_consts(self) -> np.ndarray: ...
    @property
    def num_minerals(self) -> int: ...
    @property
    def num_species(self) -> int: ...
    @property
    def num_secondary(self) -> int: ...
    @property
    def num_mineral_parameters(self) -> int: ...
    @property
    def species_types(self) -> dict[str, str]: ...
    @property
    def species_indices(self) -> dict[str, int]: ...
    @property
    def num_aqueous_species(self) -> int: ...
    @property
    def num_total_species(self) -> int: ...
    @property
    def total_species(self) -> list[str]: ...
    def get_default_aqueous_initial_state(
        self, init_conc: float = 1e-6
    ) -> np.ndarray: ...
    def monod_rate(self, chms: np.ndarray) -> np.ndarray: ...
    def tst_rate(self, chms: np.ndarray) -> np.ndarray: ...
    def aux_factor(self, d: RtForcing) -> np.ndarray: ...

# Database and stuff
@dataclass
class PrimaryAqueousSpecies:
    pass

@dataclass
class SecondarySpecies:
    pass

@dataclass
class MineralSpecies:
    pass

@dataclass
class TstReaction:
    pass

@dataclass
class MonodReaction:
    pass

MineralKineticReaction = TstReaction | MonodReaction

@dataclass
class MineralKineticData:
    pass

@dataclass
class ExchangeReaction:
    pass

@dataclass
class ChemicalDatabase:
    primary_species: dict[str, PrimaryAqueousSpecies]
    secondary_species: dict[str, SecondarySpecies]
    mineral_species: dict[str, MineralSpecies]
    exchange_reactions: dict[str, ExchangeReaction]
    tst_reactions: dict[str, dict[str, TstReaction]]
    monod_reactions: dict[str, dict[str, MonodReaction]]

    @staticmethod
    def load_default() -> ChemicalDatabase: ...
    @staticmethod
    def from_file(file_path: str) -> ChemicalDatabase: ...
    def get_primary_aqueous_species(
        self, primary_names: list[str]
    ) -> list[PrimaryAqueousSpecies]: ...
    def get_secondary_species(
        self, secondary_names: list[str]
    ) -> list[SecondarySpecies]: ...
    def get_mineral_species(self, mineral_names: list[str]) -> list[MineralSpecies]: ...
    def get_single_mineral_reaction(
        self, mineral: str, label: str
    ) -> list[tuple[str, MineralKineticReaction]]: ...
    def get_mineral_reactions(
        self, mineral_names: list[str], labels: list[str]
    ) -> MineralKineticData: ...

# ==== Exceptions ==== #
class ScalarRootFindingError(Exception):
    pass

class MatMulError(Exception):
    pass

class IterationError(Exception):
    pass

class LinearSystemError(Exception):
    pass

class OtherError(Exception):
    pass
