from ..hydro import GroundZone, HydroForcing, HydroStep, SnowZone, SoilZone

# CONST_FORCING: HydroForcing


def test_SnowZone() -> None:
    d: HydroForcing = HydroForcing(1, -1, 1)
    s_0: float = 0.0
    d_t: float = 1.0
    q_in: float = 0.0

    sz: SnowZone = SnowZone(0.0, 1.0)

    new_state: HydroStep = sz.step(s_0, d, d_t, q_in)
    assert abs(new_state.state - 1.0) < 1e-7

    # Base case, no snow
    d_base = HydroForcing(precip=0.0, temp=0.0, pet=0.0)
    sz = SnowZone(0.0, 0.0)
    new_state = sz.step(s_0, d_base, d_t, q_in)
    assert abs(new_state.state) < 1e-7


def test_SoilZone() -> None:
    d: HydroForcing = HydroForcing(precip=1.0, temp=5.0, pet=2.0)
    s_0: float = 50.0
    d_t: float = 1.0
    q_in: float = 0.0

    sz: SoilZone = SoilZone(0.0, 100.0, 1.0, 1.0, 0.1, 50.0)

    assert abs(sz.forc_flux(s_0, d) - 1.0) < 1e-7
    assert abs(sz.vert_flux(s_0, d) - 0.5) < 1e-7
    assert abs(sz.lat_flux(s_0, d) - 0.0) < 1e-7
    assert abs(sz.vap_flux(s_0, d) - 1.0) < 1e-7
    assert sz.param_list() == [0.0, 100.0, 1.0, 1.0, 0.1, 50.0]

    new_state: HydroStep = sz.step(s_0, d, d_t, q_in)
    # assert abs(new_state.state - 50.0) < 1e-7
    assert abs(new_state.forc_flux - 1.0) < 1e-7
    assert abs(new_state.vert_flux - 0.495) < 1e-3
    assert abs(new_state.lat_flux - 0.0) < 1e-7
    assert abs(new_state.vap_flux - 0.990) < 1e-3


def test_GroundZone() -> None:
    d: HydroForcing = HydroForcing(1.0, 5.0, 2.0)
    s_0: float = 50.0
    d_t: float = 1.0

    gz: GroundZone = GroundZone(1e-3, 1.0, 1.0)

    assert abs(gz.forc_flux(s_0, d) - 0.0) < 1e-7
    assert abs(gz.vert_flux(s_0, d) - 1.0) < 1e-7
    assert abs(gz.lat_flux(s_0, d) - 0.05) < 1e-7
    assert abs(gz.vap_flux(s_0, d) - 0.0) < 1e-7
    assert gz.param_list() == [1e-3, 1.0, 1.0]

    new_state: HydroStep = gz.step(s_0, d, d_t, 0.0)
    # assert abs(new_state.state - 48.95) < 1e-4
    assert abs(new_state.forc_flux - 0.0) < 1e-7
    assert abs(new_state.vert_flux - 1.0) < 1e-7
    assert abs(new_state.lat_flux) < 0.05
    assert abs(new_state.vap_flux - 0.0) < 1e-7


def test_SnowZone_step() -> None:
    d: HydroForcing = HydroForcing(1, -1, 1)
    s_0: float = 0.0
    d_t: float = 1.0
    q_in: float = 0.0

    sz: SnowZone = SnowZone(0.0, 1.0)

    new_state: HydroStep = sz.step(s_0, d, d_t, q_in)
    assert abs(new_state.state - 1.0) < 1e-7
    assert abs(new_state.forc_flux - 1.0) < 1e-7
    assert abs(new_state.vert_flux - 0.0) < 1e-7
    assert abs(new_state.lat_flux - 0.0) < 1e-7
    assert abs(new_state.vap_flux - 0.0) < 1e-7

    # Base case, no snow
    d_base = HydroForcing(0.0, 0.0, 0.0)
    sz = SnowZone(0.0, 0.0)
    new_state = sz.step(s_0, d_base, d_t, 0.0)
    assert abs(new_state.state) < 1e-7
    assert abs(new_state.forc_flux) < 1e-7
    assert abs(new_state.vert_flux) < 1e-7
    assert abs(new_state.lat_flux) < 1e-7
    assert abs(new_state.vap_flux) < 1e-7
