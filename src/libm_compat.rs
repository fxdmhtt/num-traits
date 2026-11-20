//! The `libm` crate's public API, backed by `core`'s float intrinsics and the
//! C library's mathlib, so the without-`std` impls in `float.rs` need no
//! `libm` dependency.
//!
//! # Coverage
//!
//! Every f32/f64 function `libm 0.2.16` exports is here, with the same name and
//! signature, so any `libm::` call site resolves. The f16 and f128 variants are
//! gated on the same `f16_enabled` / `f128_enabled` cfgs `libm` itself uses; on
//! a target without those types neither crate compiles them.
//!
//! Two entries are deliberately absent: `cbrt_round` and `roundeven_impl`.
//! Their parameters are `libm`'s own `support::Float`, `Round` and `FpResult`,
//! which that crate keeps `pub(crate)` unless `unstable-public-internals` is on
//! — so no caller outside `libm` can reach them even in `libm`. Reproducing
//! them would mean reproducing `libm`'s internal type system, which is not what
//! a compatibility layer is for.
//!
//! # Three routes
//!
//! - **`core::f32::math` / `core::f64::math`** for the operations the target
//!   may have a single instruction for. Where it lacks one, LLVM emits a
//!   libcall to the same bare C name the second route names anyway.
//! - **An inherent method** for `fabs`, `copysign`, `fmax`, `fmin`. None of
//!   these emits a symbol.
//! - **A bare `extern "C"` name** for everything else. These are deliberately
//!   unprefixed. Under Arm's mathlib the bare name is the *softfp* entry: it
//!   reads its arguments from the wrong registers and silently computes
//!   garbage. Arm's compiler rewrites the name at compile time, LLVM does not,
//!   so the rewriting has to happen on the archive — every bare name referenced
//!   from it becomes `__hardfp_*`. Prefixing here would cover only this crate's
//!   own calls and miss the ones a consumer inherits.
//!
//! # Contract
//!
//! A `no_std` target with neither the instruction nor a C mathlib providing
//! these names will fail to link. That is what the `libm` feature means in this
//! fork: it selects a float backend, it no longer pulls one in.
//!
//! Not every function here has a caller — `float.rs` uses 66 of them, and the
//! rest exist so that a `libm::` call added upstream resolves instead of
//! failing the build. The `allow(dead_code)` below says that, the same way
//! `libm`'s own `src/macros.rs` opens with `#![allow(unused)]`.

#![allow(dead_code)]

// ---- routed to core::f32::math / core::f64::math ---------------------------

macro_rules! from_core {
    (f32 { $( $f:ident => $g:ident );* $(;)? } f64 { $( $d:ident => $e:ident );* $(;)? }) => {
        $(
            /// Routed to `core::f32::math`, which lowers to a hardware
            /// instruction where the target has one.
            #[inline]
            pub(crate) fn $f(x: f32) -> f32 {
                core::f32::math::$g(x)
            }
        )*
        $(
            /// Routed to `core::f64::math`, which lowers to a hardware
            /// instruction where the target has one.
            #[inline]
            pub(crate) fn $d(x: f64) -> f64 {
                core::f64::math::$e(x)
            }
        )*
    };
}

from_core! {
    f32 {
        floorf => floor; ceilf => ceil; roundf => round; truncf => trunc;
        sqrtf => sqrt; cbrtf => cbrt; roundevenf => round_ties_even
    }
    f64 {
        floor => floor; ceil => ceil; round => round; trunc => trunc;
        sqrt => sqrt; cbrt => cbrt; roundeven => round_ties_even
    }
}

/// Fused multiply-add; `mul_add` takes three arguments, so the two-name macro
/// above would need a second arm for one pair.
#[inline]
pub(crate) fn fmaf(x: f32, y: f32, z: f32) -> f32 {
    core::f32::math::mul_add(x, y, z)
}

/// Fused multiply-add; see [`fmaf`].
#[inline]
pub(crate) fn fma(x: f64, y: f64, z: f64) -> f64 {
    core::f64::math::mul_add(x, y, z)
}

// ---- inherent methods, no symbol -------------------------------------------

/// Absolute value via the `f32` inherent method.
#[inline]
pub(crate) fn fabsf(x: f32) -> f32 {
    x.abs()
}

/// Absolute value via the `f64` inherent method.
#[inline]
pub(crate) fn fabs(x: f64) -> f64 {
    x.abs()
}

/// Sign transfer via the `f32` inherent method.
#[inline]
pub(crate) fn copysignf(x: f32, y: f32) -> f32 {
    x.copysign(y)
}

/// Sign transfer via the `f64` inherent method.
#[inline]
pub(crate) fn copysign(x: f64, y: f64) -> f64 {
    x.copysign(y)
}

/// The larger of two values, ignoring a NaN operand — the `fmax` semantics, and
/// what the `f32` inherent `max` already does.
#[inline]
pub(crate) fn fmaxf(x: f32, y: f32) -> f32 {
    x.max(y)
}

/// The larger of two values, ignoring a NaN operand.
#[inline]
pub(crate) fn fmax(x: f64, y: f64) -> f64 {
    x.max(y)
}

/// The smaller of two values, ignoring a NaN operand.
#[inline]
pub(crate) fn fminf(x: f32, y: f32) -> f32 {
    x.min(y)
}

/// The smaller of two values, ignoring a NaN operand.
#[inline]
pub(crate) fn fmin(x: f64, y: f64) -> f64 {
    x.min(y)
}

// ---- the C library's mathlib ------------------------------------------------

macro_rules! from_c {
    ($( fn $name:ident ( $( $arg:ident : $ty:ty ),* ) -> $ret:ty ; )*) => {
        // A child module, so the declarations can share the names below
        // without colliding with them.
        mod sys {
            #![allow(dead_code)]

            extern "C" {
                $( pub fn $name( $( $arg : $ty ),* ) -> $ret; )*
            }
        }

        $(
            /// Forwarded to the C library's mathlib under its bare name; see
            /// the module docs for why the name is unprefixed.
            #[inline]
            pub(crate) fn $name( $( $arg : $ty ),* ) -> $ret {
                // SAFETY: a plain mathlib function taking and returning floats
                // by value, with no pointers and no state.
                unsafe { sys::$name( $( $arg ),* ) }
            }
        )*
    };
}

from_c! {
    fn acosf(x: f32) -> f32;              fn acos(x: f64) -> f64;
    fn acoshf(x: f32) -> f32;             fn acosh(x: f64) -> f64;
    fn asinf(x: f32) -> f32;              fn asin(x: f64) -> f64;
    fn asinhf(x: f32) -> f32;             fn asinh(x: f64) -> f64;
    fn atanf(x: f32) -> f32;              fn atan(x: f64) -> f64;
    fn atan2f(y: f32, x: f32) -> f32;     fn atan2(y: f64, x: f64) -> f64;
    fn atanhf(x: f32) -> f32;             fn atanh(x: f64) -> f64;
    fn cosf(x: f32) -> f32;               fn cos(x: f64) -> f64;
    fn coshf(x: f32) -> f32;              fn cosh(x: f64) -> f64;
    fn erff(x: f32) -> f32;               fn erf(x: f64) -> f64;
    fn erfcf(x: f32) -> f32;              fn erfc(x: f64) -> f64;
    fn expf(x: f32) -> f32;               fn exp(x: f64) -> f64;
    fn exp2f(x: f32) -> f32;              fn exp2(x: f64) -> f64;
    fn expm1f(x: f32) -> f32;             fn expm1(x: f64) -> f64;
    fn fdimf(x: f32, y: f32) -> f32;      fn fdim(x: f64, y: f64) -> f64;
    fn fmodf(x: f32, y: f32) -> f32;      fn fmod(x: f64, y: f64) -> f64;
    fn hypotf(x: f32, y: f32) -> f32;     fn hypot(x: f64, y: f64) -> f64;
    fn ilogbf(x: f32) -> i32;             fn ilogb(x: f64) -> i32;
    fn ldexpf(x: f32, n: i32) -> f32;     fn ldexp(x: f64, n: i32) -> f64;
    fn lgammaf(x: f32) -> f32;            fn lgamma(x: f64) -> f64;
    fn logf(x: f32) -> f32;               fn log(x: f64) -> f64;
    fn log10f(x: f32) -> f32;             fn log10(x: f64) -> f64;
    fn log1pf(x: f32) -> f32;             fn log1p(x: f64) -> f64;
    fn log2f(x: f32) -> f32;              fn log2(x: f64) -> f64;
    fn nextafterf(x: f32, y: f32) -> f32; fn nextafter(x: f64, y: f64) -> f64;
    fn powf(x: f32, y: f32) -> f32;       fn pow(x: f64, y: f64) -> f64;
    fn remainderf(x: f32, y: f32) -> f32; fn remainder(x: f64, y: f64) -> f64;
    fn rintf(x: f32) -> f32;              fn rint(x: f64) -> f64;
    fn scalbnf(x: f32, n: i32) -> f32;    fn scalbn(x: f64, n: i32) -> f64;
    fn sinf(x: f32) -> f32;               fn sin(x: f64) -> f64;
    fn sinhf(x: f32) -> f32;              fn sinh(x: f64) -> f64;
    fn tanf(x: f32) -> f32;               fn tan(x: f64) -> f64;
    fn tanhf(x: f32) -> f32;              fn tanh(x: f64) -> f64;
    fn tgammaf(x: f32) -> f32;            fn tgamma(x: f64) -> f64;
    // The f64 Bessel functions are the only ones Arm's mathlib exports; the
    // f32 entry points below are built on top of them.
    fn j0(x: f64) -> f64;   fn j1(x: f64) -> f64;   fn jn(n: i32, x: f64) -> f64;
    fn y0(x: f64) -> f64;   fn y1(x: f64) -> f64;   fn yn(n: i32, x: f64) -> f64;
}

// ---- the C entries that report through a pointer ---------------------------

mod sys_ptr {
    #![allow(dead_code)]

    extern "C" {
        pub fn frexpf(x: f32, exp: *mut i32) -> f32;
        pub fn frexp(x: f64, exp: *mut i32) -> f64;
        pub fn modff(x: f32, iptr: *mut f32) -> f32;
        pub fn modf(x: f64, iptr: *mut f64) -> f64;
        pub fn remquof(x: f32, y: f32, quo: *mut i32) -> f32;
        pub fn remquo(x: f64, y: f64, quo: *mut i32) -> f64;
        pub fn lgammaf_r(x: f32, sign: *mut i32) -> f32;
        pub fn lgamma_r(x: f64, sign: *mut i32) -> f64;
    }
}

/// Splits into mantissa and exponent; C reports the exponent through a pointer,
/// this crate's signature returns it in a tuple, as `libm` does.
#[inline]
pub(crate) fn frexpf(x: f32) -> (f32, i32) {
    let mut exp = 0i32;
    // SAFETY: `exp` is a live local for the duration of the call, which is the
    // only thing the C entry does with the pointer.
    let m = unsafe { sys_ptr::frexpf(x, &mut exp) };
    (m, exp)
}

/// Splits into mantissa and exponent; see [`frexpf`].
#[inline]
pub(crate) fn frexp(x: f64) -> (f64, i32) {
    let mut exp = 0i32;
    // SAFETY: as in `frexpf`.
    let m = unsafe { sys_ptr::frexp(x, &mut exp) };
    (m, exp)
}

/// Splits into fractional and integral parts.
#[inline]
pub(crate) fn modff(x: f32) -> (f32, f32) {
    let mut iptr = 0f32;
    // SAFETY: `iptr` is a live local for the duration of the call.
    let frac = unsafe { sys_ptr::modff(x, &mut iptr) };
    (frac, iptr)
}

/// Splits into fractional and integral parts; see [`modff`].
#[inline]
pub(crate) fn modf(x: f64) -> (f64, f64) {
    let mut iptr = 0f64;
    // SAFETY: as in `modff`.
    let frac = unsafe { sys_ptr::modf(x, &mut iptr) };
    (frac, iptr)
}

/// Remainder together with the low bits of the quotient.
#[inline]
pub(crate) fn remquof(x: f32, y: f32) -> (f32, i32) {
    let mut quo = 0i32;
    // SAFETY: `quo` is a live local for the duration of the call.
    let r = unsafe { sys_ptr::remquof(x, y, &mut quo) };
    (r, quo)
}

/// Remainder together with the low bits of the quotient; see [`remquof`].
#[inline]
pub(crate) fn remquo(x: f64, y: f64) -> (f64, i32) {
    let mut quo = 0i32;
    // SAFETY: as in `remquof`.
    let r = unsafe { sys_ptr::remquo(x, y, &mut quo) };
    (r, quo)
}

/// Log-gamma together with the sign of the gamma function.
#[inline]
pub(crate) fn lgammaf_r(x: f32) -> (f32, i32) {
    let mut sign = 0i32;
    // SAFETY: `sign` is a live local for the duration of the call.
    let v = unsafe { sys_ptr::lgammaf_r(x, &mut sign) };
    (v, sign)
}

/// Log-gamma together with the sign of the gamma function; see [`lgammaf_r`].
#[inline]
pub(crate) fn lgamma_r(x: f64) -> (f64, i32) {
    let mut sign = 0i32;
    // SAFETY: as in `lgammaf_r`.
    let v = unsafe { sys_ptr::lgamma_r(x, &mut sign) };
    (v, sign)
}

// ---- entries the C library has no counterpart for --------------------------

/// `10^x`. Arm's mathlib has no `exp10`; `pow` is the C library's own
/// correctly-rounded answer for a constant base.
#[inline]
pub(crate) fn exp10f(x: f32) -> f32 {
    powf(10.0, x)
}

/// `10^x`; see [`exp10f`].
#[inline]
pub(crate) fn exp10(x: f64) -> f64 {
    pow(10.0, x)
}

/// C23 `fmaximum`: NaN wins, and two zeros return `+0`.
#[inline]
pub(crate) fn fmaximumf(x: f32, y: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        f32::NAN
    } else if x == 0.0 && y == 0.0 {
        0.0
    } else if x > y {
        x
    } else {
        y
    }
}

/// C23 `fmaximum`; see [`fmaximumf`].
#[inline]
pub(crate) fn fmaximum(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if x == 0.0 && y == 0.0 {
        0.0
    } else if x > y {
        x
    } else {
        y
    }
}

/// C23 `fminimum`: NaN wins, and two zeros return `-0`.
#[inline]
pub(crate) fn fminimumf(x: f32, y: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        f32::NAN
    } else if x == 0.0 && y == 0.0 {
        -0.0
    } else if x < y {
        x
    } else {
        y
    }
}

/// C23 `fminimum`; see [`fminimumf`].
#[inline]
pub(crate) fn fminimum(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if x == 0.0 && y == 0.0 {
        -0.0
    } else if x < y {
        x
    } else {
        y
    }
}

/// C23 `fmaximum_num`: a NaN operand is ignored. Same as `fmax`, and the same
/// as the `f32` inherent `max`.
#[inline]
pub(crate) fn fmaximum_numf(x: f32, y: f32) -> f32 {
    fmaxf(x, y)
}

/// C23 `fmaximum_num`; see [`fmaximum_numf`].
#[inline]
pub(crate) fn fmaximum_num(x: f64, y: f64) -> f64 {
    fmax(x, y)
}

/// C23 `fminimum_num`: a NaN operand is ignored.
#[inline]
pub(crate) fn fminimum_numf(x: f32, y: f32) -> f32 {
    fminf(x, y)
}

/// C23 `fminimum_num`; see [`fminimum_numf`].
#[inline]
pub(crate) fn fminimum_num(x: f64, y: f64) -> f64 {
    fmin(x, y)
}

/// Bessel J0 at f32 precision, through the f64 entry Arm's mathlib does export.
#[inline]
pub(crate) fn j0f(x: f32) -> f32 {
    j0(x as f64) as f32
}

/// Bessel J1 at f32 precision; see [`j0f`].
#[inline]
pub(crate) fn j1f(x: f32) -> f32 {
    j1(x as f64) as f32
}

/// Bessel Jn at f32 precision; see [`j0f`].
#[inline]
pub(crate) fn jnf(n: i32, x: f32) -> f32 {
    jn(n, x as f64) as f32
}

/// Bessel Y0 at f32 precision; see [`j0f`].
#[inline]
pub(crate) fn y0f(x: f32) -> f32 {
    y0(x as f64) as f32
}

/// Bessel Y1 at f32 precision; see [`j0f`].
#[inline]
pub(crate) fn y1f(x: f32) -> f32 {
    y1(x as f64) as f32
}

/// Bessel Yn at f32 precision; see [`j0f`].
#[inline]
pub(crate) fn ynf(n: i32, x: f32) -> f32 {
    yn(n, x as f64) as f32
}

/// Sine and cosine of one angle. Arm's mathlib has no `sincosf`, and neither
/// does its 262-symbol rename table, so a bare `sincosf` would be left
/// unrewritten. Two calls under names that do get rewritten. `libm` returns
/// `(sin, cos)`; so does this.
#[inline]
pub(crate) fn sincosf(x: f32) -> (f32, f32) {
    (sinf(x), cosf(x))
}

/// Sine and cosine of one angle; see [`sincosf`].
#[inline]
pub(crate) fn sincos(x: f64) -> (f64, f64) {
    (sin(x), cos(x))
}

// ---- f16 / f128 -------------------------------------------------------------

// `libm`'s 39 f16/f128 entry points are not here.
//
// They cannot be routed the way the rest of this file is. There is no C
// counterpart for either type, and every one of them is `libm`'s own software
// implementation — `generic::sqrt`, `generic::ceil`, `generic::fmaximum`,
// `roundeven_impl`, and so on. The only hardware path in the whole set is
// `sqrtf16` under `aarch64 + target_feature = "fp16"`, which is not a target
// this crate builds for.
//
// So there is nothing to hand over to the hardware or to mathlib here.
// Reproducing them would mean reproducing the crate this layer exists to
// retire, which is the opposite of the point.
//
// Nothing in `num-traits` calls them — `float.rs` is f32/f64 only — and no
// target this crate ships for establishes `f16_enabled` or `f128_enabled`.
// A caller that wants them gets a name that does not resolve, which is a
// clearer answer than a lower-precision substitute would be.
