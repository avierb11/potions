import cython  # type: ignore


@cython.cclass
class HydroForcing:
    """Contains hydrologic forcing data for a single zone at a single time step.

    Attributes:
        precip: Precipitation rate (e.g., mm/day).
        temp: Temperature (e.g., °C).
        pet: Potential evapotranspiration rate (e.g., mm/day).
        q_in: Water input from an external zone
    """

    precip: cython.double
    temp: cython.double
    pet: cython.double
    q_in: cython.double

    def __init__(
        self,
        precip: cython.double,
        temp: cython.double,
        pet: cython.double,
        q_in: cython.double,
    ):
        self.precip: cython.double = precip
        self.temp: cython.double = temp
        self.pet: cython.double = pet
        self.q_in: cython.double = q_in

    def __repr__(self) -> str:
        return f"HydroForcing(precip={self.precip}, temp={self.temp}, pet={self.pet}, q_in={self.q_in})"
