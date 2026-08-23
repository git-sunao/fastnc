"""Specialized Gauss hypergeometric evaluator for Slepian Weber kernels.

This module is intentionally *not* a general replacement for ``hyp2f1``.
It targets the parameter family used by the Weber--Schafheitlin transform,

    F(A, B; C; z),  z = r**2 in [0, 1),

with ``A-B`` fixed by an integer Bessel order and
``C-A-B = lambda`` inherited from the FFTLog exponent.  The small/intermediate
``r`` region is evaluated by the ordinary power series.  Close to ``r=1`` we
use the standard connection formula and two power series in ``u=1-r**2``.

The implementation is vectorized in NumPy ``complex128``.  Degenerate
connection-formula cases and the exact endpoint ``z=1`` are rare in the fastnc
production path and are evaluated with a scalar mpmath fallback so that this
optimization does not change the existing analytic-continuation semantics.
"""
from __future__ import annotations

import numpy as np
from scipy.special import loggamma


DEFAULT_R_SWITCH = 0.95
DEFAULT_RTOL = 3.0e-13
DEFAULT_MAX_TERMS = 10000
_DEGENERATE_TOL = 5.0e-13


def _series(a, b, c, z, *, rtol=DEFAULT_RTOL, max_terms=DEFAULT_MAX_TERMS):
    """Vectorized Taylor series for ``2F1(a,b;c;z)`` with ``|z|<1``."""
    a, b, c, z = np.broadcast_arrays(
        np.asarray(a, dtype=np.complex128),
        np.asarray(b, dtype=np.complex128),
        np.asarray(c, dtype=np.complex128),
        np.asarray(z, dtype=np.complex128),
    )
    term = np.ones(a.shape, dtype=np.complex128)
    total = term.copy()

    # z=0 is exact and common at the first Weber grid point.
    active = np.abs(z) != 0.0
    if not np.any(active):
        return total

    for n in range(int(max_terms)):
        term[active] *= (
            ((a[active] + n) * (b[active] + n))
            / ((c[active] + n) * (n + 1.0))
            * z[active]
        )
        total[active] += term[active]

        scale = np.maximum(1.0, np.abs(total))
        converged = np.abs(term) <= float(rtol) * scale
        active &= ~converged
        if not np.any(active):
            return total

    raise RuntimeError(
        f"specialized hyp2f1 series did not converge within {max_terms} terms"
    )


def _gamma_ratio(num_args, den_args):
    """Return a product of gamma functions via complex log-gamma."""
    out = 0.0j
    for arg in num_args:
        out = out + loggamma(np.asarray(arg, dtype=np.complex128))
    for arg in den_args:
        out = out - loggamma(np.asarray(arg, dtype=np.complex128))
    return np.exp(out)


def _near_one(a, b, c, lam, z, *, rtol=DEFAULT_RTOL, max_terms=DEFAULT_MAX_TERMS):
    """Connection formula evaluated as two series in ``u=1-z``."""
    a, b, c, lam, z = np.broadcast_arrays(
        np.asarray(a, dtype=np.complex128),
        np.asarray(b, dtype=np.complex128),
        np.asarray(c, dtype=np.complex128),
        np.asarray(lam, dtype=np.complex128),
        np.asarray(z, dtype=np.complex128),
    )
    u = 1.0 - z
    if np.any(np.real(u) <= 0.0) or np.any(np.abs(np.imag(u)) > 1.0e-14):
        raise ValueError("near-one Weber hyp2f1 evaluator requires 0 < 1-z < 1")

    g1 = _gamma_ratio([c, lam], [c - a, c - b])
    g2 = _gamma_ratio([c, -lam], [a, b])

    f1 = _series(a, b, 1.0 - lam, u, rtol=rtol, max_terms=max_terms)
    f2 = _series(c - a, c - b, 1.0 + lam, u, rtol=rtol, max_terms=max_terms)
    return g1 * f1 + np.exp(lam * np.log(u)) * g2 * f2


def _needs_reference(lam):
    """Flag logarithmically degenerate connection-formula parameters."""
    lam = np.asarray(lam, dtype=np.complex128)
    nearest = np.rint(lam.real)
    return (np.abs(lam.imag) < _DEGENERATE_TOL) & (
        np.abs(lam.real - nearest) < _DEGENERATE_TOL
    )


def _mpmath_scalar(a, b, c, z):
    """Rare reference fallback, imported lazily to keep the fast path light."""
    import mpmath as mp

    return complex(mp.hyp2f1(complex(a), complex(b), complex(c), mp.mpf(float(z))))


def hyp2f1_weber(
    a,
    b,
    c,
    lam,
    z,
    *,
    r_switch=DEFAULT_R_SWITCH,
    rtol=DEFAULT_RTOL,
    max_terms=DEFAULT_MAX_TERMS,
):
    """Evaluate the Gauss function for the fastnc Weber parameter family.

    Parameters are broadcast in the NumPy sense.  ``z`` must be real and lie in
    ``[0,1]``.  For ``sqrt(z) < r_switch`` the direct ``z`` series is used; near
    one, the connection formula in ``u=1-z`` is used.  The exact endpoint and
    logarithmically degenerate connection-formula cases use a scalar mpmath
    fallback to preserve the previous analytic-continuation behavior.
    """
    a, b, c, lam, z = np.broadcast_arrays(
        np.asarray(a, dtype=np.complex128),
        np.asarray(b, dtype=np.complex128),
        np.asarray(c, dtype=np.complex128),
        np.asarray(lam, dtype=np.complex128),
        np.asarray(z, dtype=np.complex128),
    )
    if np.any(np.abs(z.imag) > 1.0e-14):
        raise ValueError("Weber hyp2f1 requires real z=r**2")
    zr = z.real
    if np.any((zr < 0.0) | (zr > 1.0)):
        raise ValueError("Weber hyp2f1 requires 0 <= z=r**2 <= 1")

    out = np.empty(a.shape, dtype=np.complex128)
    endpoint = zr == 1.0
    near = (~endpoint) & (zr >= float(r_switch) ** 2)
    degenerate = near & _needs_reference(lam)
    direct = (~endpoint) & (~near)
    fast_near = near & (~degenerate)

    if np.any(direct):
        out[direct] = _series(
            a[direct], b[direct], c[direct], z[direct],
            rtol=rtol, max_terms=max_terms,
        )
    if np.any(fast_near):
        out[fast_near] = _near_one(
            a[fast_near], b[fast_near], c[fast_near], lam[fast_near], z[fast_near],
            rtol=rtol, max_terms=max_terms,
        )

    fallback = endpoint | degenerate
    if np.any(fallback):
        flat_out = out.ravel()
        flat_a = a.ravel()
        flat_b = b.ravel()
        flat_c = c.ravel()
        flat_z = zr.ravel()
        for idx in np.flatnonzero(fallback.ravel()):
            flat_out[idx] = _mpmath_scalar(
                flat_a[idx], flat_b[idx], flat_c[idx], flat_z[idx]
            )

    return out
