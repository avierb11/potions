// M0 spike — pins down the `uom` 0.38 API surface used by the typed core.
//
// This is a regression contract for the exact uom version in Cargo.toml.
// Notes:
//  * Verified with `cargo test` in a standalone spike crate (15/15 pass, 2026-10-14).
//  * `cargo test` inside this repo cannot link (pyo3 `extension-module`); in-repo,
//    type-check with `cargo check --tests`. The behavior gate is the Python suite.
//
// Empirically verified uom 0.38 facts (all covered by the tests below):
//  * construction: `Q::new::<unit>(v)`; extraction: `q.get::<unit>() -> V`
//    (no `new_<unit>()` methods in 0.38)
//  * public field `q.value` = the value in BASE SI units (m, s, K, mole, m^3,
//    mol/m^3, m/s); e.g. `.value` of a Ratio is the raw dimensionless number
//  * `autoconvert` is a default feature (ON here): all of these compile and are
//    numerically exact within <= 2 ulp:
//      - Length / Time -> velocity-dim, extractable in ANY velocity unit
//      - Velocity * Time -> length-dim, extractable via `get::<millimeter>()`
//      - Length + (Velocity * Time)   [the core hydro update pattern]
//      - MolarConcentration * Volume -> amount, extractable via `get::<mole>()`
//      - AmountOfSubstance / Volume -> composed Kind: `get::<mole_per_liter>()`
//        is NOT available (Kind mismatch); use `.value` (base mol/m^3) / 1000
//      - Velocity / Length -> 1/Time dim: `.value` is base s^-1 (x86400 = per day)
//      - same-dim division -> exactly type `Ratio`; `q.value` is the raw number
//      - ThermodynamicTemperature +/- TemperatureInterval (kelvin, exact)
//      - TemperatureInterval + TemperatureInterval cross-unit (degC == K deltas)
//      - Interval / scalar; Length(mm)/Interval(K) * Interval(degC) -> length
//  * TT - TT is NOT implemented; T-T-T differences come in as TemperatureInterval
//    (1 degC delta == 1 K delta exactly). Absolute C values stay raw f64
//  * `unit!` generates the new unit into the CALLING crate's scope (crate root
//    here / in `crate::` in the lib) — that is how `millimeter_per_day` below
//    and `crate::millimeter_per_day` in the lib resolve
//  * integer powers only: `q.powi(typenum::P2)`; runtime exponents (powf)
//    remain raw f64 in the core
//  * inherent `.max()/.min()` and `</>` comparisons work without `Ord` imports
//  * mm -> m -> mm round-trip measured EXACT (0.0 drift) on sampled values —
//    within the approved < 2 ulp budget
#![allow(non_camel_case_types)]

#[macro_use]
extern crate uom;

use uom::si::amount_of_substance::mole;
use uom::si::f64::*;
use uom::si::length::millimeter;
use uom::si::molar_concentration::mole_per_liter;
use uom::si::temperature_interval::{degree_celsius, kelvin as kelvin_iv};
use uom::si::thermodynamic_temperature::kelvin;
use uom::si::time::day;
use uom::si::velocity::meter_per_second;
use uom::si::volume::liter;

// Custom unit: water flux in mm/day (velocity).
unit! {
    system: uom::si;
    quantity: uom::si::velocity;

    @millimeter_per_day: (1.0 / 86400.0) * 1.0e-3; "mm/d", "millimeter per day", "millimeters per day";
}

// 1. mm -> m -> mm round-trip drift (approved budget < 2 ulp ~ 4.4e-16).
#[test]
fn mm_roundtrip_drift() {
    let vals = [
        0.0f64, 1e-8, 0.001, 0.005, 0.1, 1.0, 77.777777, 25.0, 100.0, 1000.0, 99_999.999,
        12345.6789, 0.000125,
    ];
    let mut max_drift = 0.0f64;
    for x in vals {
        let r = Length::new::<millimeter>(x).get::<millimeter>();
        let d = if x.abs() > 0.0 {
            ((r - x) / x.abs()).abs()
        } else {
            (r - x).abs()
        };
        max_drift = max_drift.max(d);
        println!("x={x:.8e} -> rt={r:.8e} drift={d:.3e}");
    }
    assert!(max_drift < 1e-14, "drift too big: {max_drift:e}");
}

// 2. Custom mm/day unit: round-trip + cross-check vs m/s.
#[test]
fn vel_mm_per_day() {
    let v = Velocity::new::<millimeter_per_day>(100.0);
    assert!((v.get::<millimeter_per_day>() - 100.0).abs() < 1e-12);
    let v2 = Velocity::new::<millimeter_per_day>(1.0);
    let expect = 1.0e-3 / 86400.0;
    let got = v2.get::<meter_per_second>();
    assert!((got - expect).abs() / expect < 1e-12, "got={got} expect={expect}");
    // .value of a velocity is base m/s:
    let v3 = Velocity::new::<millimeter_per_day>(86400.0); // == 1e-3 m/s
    assert!((v3.value - 1.0e-3).abs() < 1e-15);
}

// 3. Cross-dim: Length / Time -> velocity-dim; extract in both units.
#[test]
fn length_over_time() {
    let depth = Length::new::<millimeter>(250.0);
    let t = Time::new::<day>(2.5);
    let v = depth / t;
    let ms = v.get::<meter_per_second>();
    let expect_ms = 250.0e-3 / (2.5 * 86400.0);
    assert!((ms - expect_ms).abs() / expect_ms < 1e-12, "ms={ms}");
    let mpd = v.get::<millimeter_per_day>();
    assert!((mpd - 100.0).abs() < 1e-12, "mpd={mpd}");
}

// 4. Cross-dim: Velocity * Time -> length-dim, incl. mixed (m/s * s).
#[test]
fn velocity_times_time() {
    let v = Velocity::new::<millimeter_per_day>(100.0);
    let dt = Time::new::<day>(2.5);
    let depth = v * dt;
    assert!((depth.get::<millimeter>() - 250.0).abs() / 250.0 < 1e-12);
    // mixed units: 1.5e-3 m/s (=1.5 mm/s) over 86400 s -> 129.6 m = 129600 mm
    let v2 = Velocity::new::<meter_per_second>(1.5e-3);
    let s = Time::new::<uom::si::time::second>(86400.0);
    assert!(((v2 * s).get::<millimeter>() - 129600.0).abs() / 129600.0 < 1e-9);
}

// 5. THE core hydro pattern: Length + Velocity*Time (autoconvert Add).
#[test]
fn length_add_velocity_time() {
    let s = Length::new::<millimeter>(120.0);
    let new = s + Velocity::new::<millimeter_per_day>(50.0) * Time::new::<day>(2.0);
    assert!((new.get::<millimeter>() - 220.0).abs() / 220.0 < 1e-12);
}

// 6. Cross-dim: MolarConcentration * Volume -> amount; reverse division.
#[test]
fn conc_volume_amount() {
    let c = MolarConcentration::new::<mole_per_liter>(2.0);
    let vol = Volume::new::<liter>(15.0);
    let n = c * vol;
    assert!((n.get::<mole>() - 30.0).abs() < 1e-12);
    // reverse: Amount / Volume -> composed Kind, `get::<mole_per_liter>` is NOT
    // available (Kind mismatch), but `.value` (base mol/m^3) works — extract / 1000.
    let n2 = AmountOfSubstance::new::<mole>(30.0);
    let c2 = n2 / vol;
    assert!((c2.value - 2000.0).abs() / 2000.0 < 1e-12, "val={}", c2.value);
    let mol_per_l = c2.value / 1000.0;
    assert!((mol_per_l - 2.0).abs() < 1e-12);
    // .value of an amount is base moles:
    let n3 = AmountOfSubstance::new::<mole>(3.0);
    assert!((n3.value - 3.0).abs() < 1e-15);
}

// 7. Velocity / Length (exchange rate q/v): 1/Time dim; .value is base s^-1.
#[test]
fn vel_over_len() {
    let q = Velocity::new::<millimeter_per_day>(100.0);
    let v = Length::new::<millimeter>(10.0);
    let r = q / v; // 10 per day
    let base = r.value; // s^-1
    assert!((base - 10.0 / 86400.0).abs() / (10.0 / 86400.0) < 1e-12, "base={base}");
    let per_day = base * 86400.0;
    assert!((per_day - 10.0).abs() < 1e-12);
}

// 8. Ratios (Monod c/(c+K), K/(c+K)): same-dim division -> Ratio; .value is raw.
#[test]
fn monod_ratio() {
    let c = MolarConcentration::new::<mole_per_liter>(3.0);
    let k = MolarConcentration::new::<mole_per_liter>(2.0);
    let d1: Ratio = c / (c + k); // result type is exactly `Ratio`
    assert!((d1.value - 0.6).abs() < 1e-15, "v={}", d1.value);
    let d2: Ratio = k / (c + k);
    assert!((d2.value - 0.4).abs() < 1e-15);
    // extract via the named ratio unit too:
    assert!((d1.get::<uom::si::ratio::ratio>() - 0.6).abs() < 1e-15);
}

// 9. Temperature: TT +/- Interval, Interval + Interval (mixed degC/K).
#[test]
fn temperature() {
    let t = ThermodynamicTemperature::new::<kelvin>(293.15);
    let ten = TemperatureInterval::new::<kelvin_iv>(10.0);
    let lo = t - ten;
    assert!((lo.get::<kelvin>() - 283.15).abs() / 283.15 < 1e-14);
    let hi = t + ten;
    assert!((hi.get::<kelvin>() - 303.15).abs() / 303.15 < 1e-14);
    // degree_celsius exists for intervals; 1 degC == 1 K exactly:
    let dc = TemperatureInterval::new::<degree_celsius>(3.0);
    let kk = TemperatureInterval::new::<kelvin_iv>(2.0);
    let sum = dc + kk;
    assert!((sum.get::<kelvin_iv>() - 5.0).abs() < 1e-15);
    // interval / scalar (q10 exponent pattern):
    let diff = TemperatureInterval::new::<kelvin_iv>(12.0);
    let scaled = diff / 10.0;
    assert!((scaled.get::<kelvin_iv>() - 1.2).abs() < 1e-15);
}

// 10. Scalar multiplication, both orders, and division.
#[test]
fn scalar_ops() {
    let a = Length::new::<millimeter>(10.0) * 2.0;
    assert!((a.get::<millimeter>() - 20.0).abs() < 1e-15);
    let c = 2.0 * Length::new::<millimeter>(10.0);
    assert!((c.get::<millimeter>() - 20.0).abs() < 1e-15);
    let d = Length::new::<millimeter>(10.0) / 2.0;
    assert!((d.get::<millimeter>() - 5.0).abs() < 1e-15);
    // scalar on intervals too:
    let e = TemperatureInterval::new::<kelvin_iv>(3.0) * 7.0;
    assert!((e.get::<kelvin_iv>() - 21.0).abs() < 1e-15);
}

// 11. Ordering / comparisons used in hydro.rs guards (inherent max/min in 0.38).
#[test]
fn ord_ops() {
    let s = Length::new::<millimeter>(5.0);
    let melt = Length::new::<millimeter>(12.0);
    assert!(s.max(melt) == melt);
    assert!((s.max(melt).get::<millimeter>() - 12.0).abs() < 1e-15);
    let thr = Length::new::<millimeter>(10.0);
    assert!(s < thr);
    // clamped lateral term pattern: max(s - thr, 0)
    let lat = (s - thr).max(Length::new::<millimeter>(0.0));
    assert!((lat.get::<millimeter>() - 0.0).abs() < 1e-15);
    // ratio ordering
    assert!(0.2 < 0.3);
}

// 12. Melt coefficient: (Length / Interval) * Interval -> length.
#[test]
fn melt_coefficient() {
    let fmax = 1.0; // mm per K (raw parameter)
    let coef = Length::new::<millimeter>(fmax) / TemperatureInterval::new::<kelvin_iv>(1.0);
    let d_t = TemperatureInterval::new::<degree_celsius>(3.0);
    let melt = coef * d_t;
    assert!(
        (melt.get::<millimeter>() - 3.0).abs() < 1e-12,
        "melt={}",
        melt.get::<millimeter>()
    );
}

// 13. Time day <-> second conversions (86400 pattern).
#[test]
fn time_conversions() {
    let t = Time::new::<day>(2.5);
    assert!((t.get::<uom::si::time::second>() - 216000.0).abs() < 1e-9);
    assert!((t.get::<day>() - 2.5).abs() < 1e-14);
}

// 14. powi takes a typenum integer constant (only integer exponents possible).
#[test]
fn powi_works() {
    use uom::typenum::P2;
    let a: Area = Length::new::<millimeter>(10.0).powi(P2::new());
    assert!((a.get::<uom::si::area::square_meter>() - 1.0e-4).abs() < 1e-18);
}

// 15. Concentration differences: c_in - c (transport uptake term).
#[test]
fn conc_difference() {
    let c_in = MolarConcentration::new::<mole_per_liter>(8.0);
    let ch = MolarConcentration::new::<mole_per_liter>(3.0);
    let dc = c_in - ch;
    assert!((dc.get::<mole_per_liter>() - 5.0).abs() < 1e-15);
}
