from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

from .common_types import HydroForcing


class HydroStep:
    state: float
    forc_flux: float
    lat_flux: float
    vert_flux: float
    vap_flux: float

    def __init__(
        self,
        state: float,
        forc_flux: float,
        lat_flux: float,
        vert_flux: float,
        vap_flux: float = 0.0,
    ):
        self.state = float(state)
        self.forc_flux = float(forc_flux)
        self.lat_flux = float(lat_flux)
        self.vert_flux = float(vert_flux)
        self.vap_flux = float(vap_flux)


# ########################################################################### #
# Zone Classes
# These are regular Python classes that hold parameters
# ########################################################################### #


class HydrologicZone:
    """A generic hydrologic zone."""

    name: str = "unnamed"

    def __init__(
        self,
    ):
        pass

    def step(self, s_0: float, d: HydroForcing, dt: float, q_in: float) -> HydroStep:
        """Integrates the mass balance equation over a time step."""

        # def f(t: float, s: float) -> float:
        #     return self.mass_balance(s, d, q_in)

        # res = solve_ivp(f, (0, dt), y0=[float(s_0)])
        # s_new: float = res.y[0, -1]

        # Use the implicit midpoint method
        def f(s: float) -> float:
            return (s_0 - s) + dt * self.mass_balance(0.5 * (s_0 + s), d, q_in)

        res = root_scalar(f, x0=s_0, x1=s_0 + 0.1, method="secant", xtol=0.001)
        s_new: float = res.root

        return HydroStep(
            state=s_new,
            forc_flux=self.forc_flux(s_new, d),
            vap_flux=self.vap_flux(s_new, d),
            lat_flux=self.lat_flux(s_new, d),
            vert_flux=self.vert_flux(s_new, d),
        )

    def mass_balance(self, s: float, d: HydroForcing, q_in: float) -> float:
        return (
            q_in
            + self.forc_flux(s, d)
            - self.vap_flux(s, d)
            - self.lat_flux(s, d)
            - self.vert_flux(s, d)
        )

    def forc_flux(self, s: float, d: HydroForcing) -> float:
        return 0.0

    def vap_flux(self, s: float, d: HydroForcing) -> float:
        return 0.0

    def lat_flux(self, s: float, d: HydroForcing) -> float:
        return 0.0

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        return 0.0

    def param_list(self) -> list[float]:
        """Gets the parameters in list form for this zone"""
        return []

    def columns(self, zone_id: int) -> list[str]:
        """Gets the column names for this zone for the output DataFrame."""
        return [
            f"s_{self.name}_{zone_id}",
            f"q_forc_{self.name}_{zone_id}",
            f"q_vap_{self.name}_{zone_id}",
            f"q_lat_{self.name}_{zone_id}",
            f"q_vert+{self.name}_{zone_id}",
        ]


class SnowZone(HydrologicZone):
    """A zone representing a snowpack."""

    name: str = "snow"

    def __init__(self, tt: float, fmax: float):
        super().__init__()
        self.tt: float = tt
        self.fmax: float = fmax

    def forc_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates snowmelt accumulation rate"""
        if d.temp <= self.tt:
            return d.precip
        else:
            return 0.0

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates snowmelt accumulation rate"""
        if d.temp > self.tt:
            melt: float = self.fmax * (d.temp - self.tt)
            return min(s, melt)
        else:
            return 0.0

    def param_list(self) -> list[float]:
        return [self.tt, self.fmax]


class SoilZone(HydrologicZone):
    """A zone representing a soil layer."""

    name: str = "soil"

    def __init__(
        self, tt: float, fc: float, lp: float, beta: float, k0: float, thr: float
    ):
        super().__init__()
        self.fc: float = fc  # Soil field capacity
        self.beta: float = beta  # ET nonlinearity factor
        self.k0: float = k0
        self.lp: float = lp
        self.thr: float = thr
        self.tt: float = tt

    def forc_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates soil accumulation"""
        if d.temp >= self.tt:
            return d.precip * (1 - (s / self.fc) ** self.beta)
        else:
            return 0.0

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates vertical percolation from the soil."""
        if d.temp >= self.tt:
            return d.precip * (s / self.fc) ** self.beta
        else:
            return 0.0

    def vap_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates actual evapotranspiration."""
        return d.pet * min(s / (self.fc * self.lp), 1.0)

    def lat_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates lateral flow from the soil."""
        return max(0.0, self.k0 * (s - self.thr))

    def param_list(self) -> list[float]:
        return [self.tt, self.fc, self.lp, self.beta, self.k0, self.thr]


class GroundZone(HydrologicZone):
    """A zone representing a groundwater store."""

    name: str = "ground"

    def __init__(self, k: float, alpha: float, perc: float):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.perc = perc

    def lat_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates lateral flow from a groundwater zone."""
        return self.k * s**self.alpha

    def vert_flux(self, s: float, d: HydroForcing) -> float:
        """Calculates lateral flow from a groundwater zone."""
        return min(s, self.perc)

    def param_list(self) -> list[float]:
        return [self.k, self.alpha, self.perc]
