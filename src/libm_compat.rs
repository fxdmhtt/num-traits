//! The `libm` crate's entry points, backed by `core`'s float intrinsics and the
//! C library's mathlib, so the without-`std` impls in `float.rs` need no `libm`
//! dependency.
//!
//! # Three routes
//!
//! - **`core::f32::math` / `core::f64::math`** for the operations the target
//!   may have a single instruction for: `floor`, `ceil`, `round`, `trunc`,
//!   `sqrt`, `cbrt`, `mul_add`. Where the target lacks the instruction, LLVM
//!   emits a libcall to the same bare C name this file would have named anyway,
//!   so the second route below still catches it.
//! - **An inherent method** for `fabs` and `copysign`. Neither emits a symbol.
//! - **A bare `extern "C"` name** for the transcendentals. These are
//!   deliberately unprefixed. Under Arm's mathlib the bare name is the *softfp*
//!   entry: it reads its arguments from the wrong registers and silently
//!   computes garbage. Arm's compiler rewrites the name at compile time, LLVM
//!   does not, so the rewriting has to happen on the archive — every bare name
//!   referenced from it is turned into `__hardfp_*`. Prefixing here would cover
//!   only this crate's own calls and miss the ones a consumer inherits.
//!
//! # Contract
//!
//! A `no_std` target with neither the instruction nor a C mathlib providing
//! these names will fail to link. That is what the `libm` feature means in this
//! fork: it selects a float backend, it no longer pulls one in.

#![allow(dead_code)]

// ---- core::f32::math / core::f64::math --------------------------------------

macro_rules! from_core {
    (f32 { $( $f:ident => $g:ident );* $(;)? } f64 { $( $d:ident => $e:ident );* $(;)? }) => {
        $(
            #[inline]
            pub(crate) fn $f(x: f32) -> f32 {
                core::f32::math::$g(x)
            }
        )*
        $(
            #[inline]
            pub(crate) fn $d(x: f64) -> f64 {
                core::f64::math::$e(x)
            }
        )*
    };
}

from_core! {
    f32 {
        floorf => floor; ceilf => ceil; roundf => round;
        truncf => trunc; sqrtf => sqrt; cbrtf => cbrt
    }
    f64 {
        floor => floor; ceil => ceil; round => round;
        trunc => trunc; sqrt => sqrt; cbrt => cbrt
    }
}

// `mul_add` takes three arguments, so the two-name macro above would need a
// second arm for one function each.
#[inline]
pub(crate) fn fmaf(x: f32, y: f32, z: f32) -> f32 {
    core::f32::math::mul_add(x, y, z)
}

#[inline]
pub(crate) fn fma(x: f64, y: f64, z: f64) -> f64 {
    core::f64::math::mul_add(x, y, z)
}

// ---- inherent methods, no symbol --------------------------------------------

#[inline]
pub(crate) fn fabsf(x: f32) -> f32 {
    x.abs()
}

#[inline]
pub(crate) fn fabs(x: f64) -> f64 {
    x.abs()
}

#[inline]
pub(crate) fn copysignf(x: f32, y: f32) -> f32 {
    x.copysign(y)
}

#[inline]
pub(crate) fn copysign(x: f64, y: f64) -> f64 {
    x.copysign(y)
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
            #[inline]
            pub(crate) fn $name( $( $arg : $ty ),* ) -> $ret {
                // SAFETY: every entry below is a plain mathlib function taking
                // and returning floats by value, with no pointers and no state.
                unsafe { sys::$name( $( $arg ),* ) }
            }
        )*
    };
}

from_c! {
    fn acosf(x: f32) -> f32;            fn acos(x: f64) -> f64;
    fn acoshf(x: f32) -> f32;           fn acosh(x: f64) -> f64;
    fn asinf(x: f32) -> f32;            fn asin(x: f64) -> f64;
    fn asinhf(x: f32) -> f32;           fn asinh(x: f64) -> f64;
    fn atanf(x: f32) -> f32;            fn atan(x: f64) -> f64;
    fn atan2f(y: f32, x: f32) -> f32;   fn atan2(y: f64, x: f64) -> f64;
    fn atanhf(x: f32) -> f32;           fn atanh(x: f64) -> f64;
    fn cosf(x: f32) -> f32;             fn cos(x: f64) -> f64;
    fn coshf(x: f32) -> f32;            fn cosh(x: f64) -> f64;
    fn expf(x: f32) -> f32;             fn exp(x: f64) -> f64;
    fn exp2f(x: f32) -> f32;            fn exp2(x: f64) -> f64;
    fn expm1f(x: f32) -> f32;           fn expm1(x: f64) -> f64;
    fn fdimf(x: f32, other: f32) -> f32; fn fdim(x: f64, other: f64) -> f64;
    fn hypotf(x: f32, other: f32) -> f32; fn hypot(x: f64, other: f64) -> f64;
    fn logf(x: f32) -> f32;             fn log(x: f64) -> f64;
    fn log10f(x: f32) -> f32;           fn log10(x: f64) -> f64;
    fn log1pf(x: f32) -> f32;           fn log1p(x: f64) -> f64;
    fn log2f(x: f32) -> f32;            fn log2(x: f64) -> f64;
    fn powf(x: f32, n: f32) -> f32;     fn pow(x: f64, n: f64) -> f64;
    fn sinf(x: f32) -> f32;             fn sin(x: f64) -> f64;
    fn sinhf(x: f32) -> f32;            fn sinh(x: f64) -> f64;
    fn tanf(x: f32) -> f32;             fn tan(x: f64) -> f64;
    fn tanhf(x: f32) -> f32;            fn tanh(x: f64) -> f64;
}

// ---- the one entry with no C counterpart ------------------------------------

// Arm's mathlib has no `sincosf`, and neither does its 262-symbol hard-float
// rename table, so a bare `sincosf` would be left unrewritten. Two calls under
// the names that do get rewritten. `libm` returns `(sin, cos)`; so does this.

#[inline]
pub(crate) fn sincosf(x: f32) -> (f32, f32) {
    (sinf(x), cosf(x))
}

#[inline]
pub(crate) fn sincos(x: f64) -> (f64, f64) {
    (sin(x), cos(x))
}
