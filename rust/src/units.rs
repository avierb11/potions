//! Unit conventions and `uom` re-exports for the typed core.
//!
//! ## Scope
//!
//! Only function-body locals are annotated as `uom` quantities. Python-visible
//! struct fields stay raw `f64` (see the raw aliases below, and the doc
//! comments on `common_types` structs). Conversions happen once at the
//! boundary and once back before returning; every conversion is a single
//! power-of-10 multiply/divide.
//!
//! ## `uom` 0.38 API (verified by the contract tests in `rust/tests/uom_spike.rs`)
//!
//! * Construction: `Q::new::<unit>(v)`; extraction: `q.get::<unit>()`.
//!   There are no `new_<unit>()` methods in 0.38.
//! * Public field `q.value` stores the BASE SI-unit value (m, s, K, mole,
//!   m^3, mol/m^3, m/s); for `Ratio` it is the raw dimensionless number.
//!   Never rely on it for non-base extractions (e.g. mm) — use `get`.
//! * `autoconvert` (a default feature) is ON: cross-unit and cross-dimension
//!   arithmetic (L/T, V*T, C*V, TT +/- Interval, ...) all compile and are
//!   exact within a power-of-10 re-rounding.
//! * `TT - TT` is NOT implemented; temperature *differences* are
//!   `TemperatureInterval` (1 °C delta == 1 K delta exactly). Absolute °C
//!   values and TT inputs stay raw `f64` (0.38 has no °C unit).
//! * Integer powers only (`q.powi(typenum::P2)`); runtime-exponent math
//!   (`powf`) stays raw `f64` with a local doc comment.
//! * Composed Kinds: `Amount/Volume` does NOT accept
//!   `get::<mole_per_liter>()` (Kind mismatch) — extract `.value`
//!   (base mol/m^3) and scale if needed. `Velocity/Length` is 1/Time —
//!   `.value` is base s^-1.
//! * `unit!` places the unit (and its `f32`/`f64` helper scope) in the
//!   *calling* scope — one `unit!` per scope. `millimeter_per_day` therefore
//!   lives in this module: `crate::units::millimeter_per_day`.
//! * mm -> m -> mm round trip measured EXACT (0.0 drift) in the spike on
//!   sampled values — within the approved <= 2 ulp budget.
//!
//! ## Quantity / unit conventions used by the core
//!
//! | physics              | type                  | unit                | raw boundary alias |
//! |----------------------|-----------------------|---------------------|--------------------|
//! | water depth          | `Length`              | `millimeter`        | `Depth`            |
//! | water flux           | `Velocity`            | `millimeter_per_day`| `Flux`             |
//! | time step            | `Time`                | `day`               | `Days`             |
//! | temperature interval | `TemperatureInterval` | `kelvin`/`degree_celsius` | `TempDiff` |
//! | water concentration  | `MolarConcentration`  | `mole_per_liter`    | `Conc`             |
//! | water volume         | `Volume`              | `liter`             | `WaterVol`         |
//! | amount of substance  | `AmountOfSubstance`   | `mole`              | `MolesQty`         |
//!
//! ## Raw-`f64` fallbacks (documented at the use site)
//!
//! * Runtime exponents (`s.powf(alpha)`, `(s/fc).powf(beta)`,
//!   `q10.powf((t-20)/10)`, `10.powf(...)`): no float power in 0.38.
//! * Monod/TST coefficient scalings (`s_w`, `k * 86400`, ...) where the raw
//!   combination is a rate parameter, not a physical quantity.
//! * Absolute °C (Kelvin-only TT; intervals cover deltas).

// The re-exports below are the single import point for the core modules and
// are consumed as the typing proceeds (hydro.rs, kinetic_structures.rs,
// rt_zone.rs, river_zone.rs).
#![allow(unused_imports)]

// Water flux in millimeters per day (velocity). Defined in this module
// because `unit!` may only appear once per scope (it also generates private
// `__system`/`__quantity`/`Conversion`/`Unit` names and `f32`/`f64` helper
// mods in that scope).
uom::unit! {
    system: uom::si;
    quantity: uom::si::velocity;
    @millimeter_per_day: (1.0 / 86400.0) * 1.0e-3; "mm/d", "millimeter per day", "millimeters per day";
}

pub(crate) use uom::si::amount_of_substance::mole;
pub(crate) use uom::si::f64::{
    AmountOfSubstance, Length, MolarConcentration, Ratio, TemperatureInterval,
    ThermodynamicTemperature, Time, Velocity, Volume,
};
pub(crate) use uom::si::length::millimeter;
pub(crate) use uom::si::molar_concentration::mole_per_liter;
pub(crate) use uom::si::temperature_interval::{
    degree_celsius, kelvin as kelvin_interval,
};
pub(crate) use uom::si::thermodynamic_temperature::kelvin;
pub(crate) use uom::si::time::day;
pub(crate) use uom::si::volume::liter;

/// Water depth at a Python/API boundary, raw millimeters.
pub(crate) type Depth = f64;
/// Water flux at a Python/API boundary, raw millimeters per day.
pub(crate) type Flux = f64;
/// Time step in raw days.
pub(crate) type Days = f64;
/// Temperature interval (1 °C == 1 K), raw.
pub(crate) type TempDiff = f64;
/// Aqueous concentration in raw mol/L.
pub(crate) type Conc = f64;
/// Water volume in raw liters.
pub(crate) type WaterVol = f64;
/// Amount of substance in raw moles.
pub(crate) type MolesQty = f64;
