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


@cython.cclass
class HydroStep:
    """
    Represents the results of a single time step for a hydrologic zone.

    This class is a data container that holds the new state and the calculated
    fluxes for a single zone after a time step.

    Attributes:
        state (cython.double): The updated storage or state of the zone (e.g., in mm).
        forc_flux (cython.double): The flux from external forcing (e.g., precipitation)
            into the zone during the time step (e.g., in mm/day).
        lat_flux (cython.double): The lateral flux out of the zone (e.g., in mm/day).
        vert_flux (cython.double): The vertical flux out of the zone (e.g., in mm/day).
        vap_flux (cython.double): The vaporization flux (e.g., evapotranspiration) out
            of the zone (e.g., in mm/day).
    """

    state = cython.declare(cython.double, visibility="public")
    forc_flux = cython.declare(cython.double, visibility="public")
    lat_flux = cython.declare(cython.double, visibility="public")
    vert_flux = cython.declare(cython.double, visibility="public")
    vap_flux = cython.declare(cython.double, visibility="public")
    q_in = cython.declare(cython.double, visibility="public")
    lat_flux_ext = cython.declare(cython.double, visibility="public")
    vert_flux_ext = cython.declare(cython.double, visibility="public")

    def __init__(
        self,
        state: cython.double,
        forc_flux: cython.double,
        lat_flux: cython.double,
        vert_flux: cython.double,
        q_in: cython.double,
        lat_flux_ext: cython.double,
        vert_flux_ext: cython.double,
        vap_flux: cython.double = 0.0,
    ):
        """
        Initializes a HydroStep object.

        Args:
            state (cython.double): The new state of the zone.
            forc_flux (cython.double): The forcing flux for the time step.
            lat_flux (cython.double): The lateral flux for the time step.
            vert_flux (cython.double): The vertical flux for the time step.
            vap_flux (cython.double, optional): The vaporization flux for the time
                step. Defaults to 0.0.

        Example:
            >>> step_result = HydroStep(
            ...     state=100.5, forc_flux=10.0, lat_flux=2.5,
            ...     vert_flux=1.0, vap_flux=0.5
            ... )
            >>> print(step_result.state)
            100.5
        """
        self.state = cython.cast(cython.double, state)
        self.forc_flux = cython.cast(cython.double, forc_flux)
        self.lat_flux = cython.cast(cython.double, lat_flux)
        self.vert_flux = cython.cast(cython.double, vert_flux)
        self.vap_flux = cython.cast(cython.double, vap_flux)
        self.q_in = cython.cast(cython.double, q_in)
        self.lat_flux_ext = cython.cast(cython.double, lat_flux_ext)
        self.vert_flux_ext = cython.cast(cython.double, vert_flux_ext)

    def __repr__(self) -> str:
        return (
            f"HydroStep(\n"
            f"\tstate={self.state:.2f},\n"
            f"\tforc_flux={self.forc_flux:.2f},\n"
            f"\tvap_flux={self.vap_flux:.2f},\n"
            f"\tlat_flux={self.lat_flux:.2f},\n"
            f"\tvert_flux={self.vert_flux:.2f},\n"
            f"\tq_in={self.q_in:.2f},\n"
            f"\tlat_flux_ext={self.lat_flux_ext:.2f},\n"
            f"\tvert_flux_ext={self.vert_flux_ext:.2f},\n"
            ")"
        )
