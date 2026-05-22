"""Projection-convention utilities for resummed 3PCF grids.

The conversion factor is written in terms of the effective spin triple

    sigma_i = epsilon_i * spin_i

at the three vertices.  The old shear natural-component interface
``component=0,1,2,3`` is kept as a compatibility layer and corresponds to

    component 0: sigma = (+2, +2, +2)
    component 1: sigma = (-2, +2, +2)
    component 2: sigma = (+2, -2, +2)
    component 3: sigma = (+2, +2, -2)

For a spin field component with effective spin ``sigma_i``, changing the
reference direction at vertex i by ``Delta alpha_i`` multiplies the component
by ``exp(1j * sigma_i * Delta alpha_i)`` in the convention used here.  The
legacy shear formulas are recovered exactly for ``sigma_i = +/-2``.
"""
from __future__ import annotations

from typing import Literal, Sequence
import numpy as np

# Conversion is only meaningful for an already resummed zeta(theta1, theta2, phi).

Projection = Literal["x", "cross", "ortho", "orthocenter", "cent", "centroid"]


_SHEAR_COMPONENT_SIGMA = {
    0: (2, 2, 2),
    1: (-2, 2, 2),
    2: (2, -2, 2),
    3: (2, 2, -2),
}


def _projection_name(name: str) -> str:
    n = str(name).lower()
    if n in {"x", "cross", "times"}:
        return "x"
    if n in {"ortho", "orthocenter"}:
        return "ortho"
    if n in {"cent", "centroid"}:
        return "cent"
    raise ValueError("projection must be one of 'x'/'cross', 'ortho', or 'centroid'.")


def _as_triple(x, *, name: str, dtype=int) -> tuple:
    """Return ``x`` as a length-3 tuple.

    A scalar is interpreted as ``(x, x, x)``.  This is convenient for the
    common equal-spin case ``spin=(s, s, s)``.
    """
    arr = np.asarray(x)
    if arr.ndim == 0:
        val = dtype(arr.item())
        return (val, val, val)
    vals = tuple(dtype(v) for v in arr.tolist())
    if len(vals) != 3:
        raise ValueError(f"{name} must be a scalar or a length-3 sequence.")
    return vals


def sigma_from_spin_epsilons(
    spin: int | Sequence[int],
    epsilons: Sequence[int] | None = None,
) -> tuple[int, int, int]:
    """Build the effective spin triple ``sigma_i = epsilon_i * spin_i``.

    Parameters
    ----------
    spin : int or sequence of three ints
        Spin at each vertex.  A scalar means equal spins ``(s, s, s)``.
    epsilons : sequence of three ints, optional
        Signs of the components.  If omitted, ``(+1, +1, +1)`` is used.
    """
    s = _as_triple(spin, name="spin", dtype=int)
    eps = (1, 1, 1) if epsilons is None else _as_triple(epsilons, name="epsilons", dtype=int)
    if any(e not in (-1, 1) for e in eps):
        raise ValueError("epsilons must contain only +1 or -1.")
    return tuple(e * si for e, si in zip(eps, s))


def _sigma_triple(
    *,
    component: int | None = None,
    sigma=None,
    spin: int | Sequence[int] | None = None,
    epsilons: Sequence[int] | None = None,
) -> tuple[int, int, int]:
    """Resolve component/sigma/spin+epsilons into an effective spin triple.

    ``sigma`` is preferred over ``component``.  This is intentional because
    generic ``ZetaGrid`` objects still carry a component-axis label, but that
    label is not a shear natural-component index unless ``spin=(2,2,2)``.
    """
    if sigma is not None and spin is not None:
        raise ValueError("Provide either sigma or spin, not both.")

    if sigma is not None:
        return _as_triple(sigma, name="sigma", dtype=int)

    if spin is not None:
        return sigma_from_spin_epsilons(spin, epsilons)

    if component is None:
        raise ValueError("Provide one of component, sigma, or spin.")

    mu = int(component)
    if mu not in _SHEAR_COMPONENT_SIGMA:
        raise ValueError("component must be 0, 1, 2, or 3 for the legacy shear interface.")
    return _SHEAR_COMPONENT_SIGMA[mu]


def natural_component_index_from_sigma(sigma: tuple[int, int, int] | list[int] | np.ndarray) -> int:
    """Infer the shear natural-component index from a spin triple.

    This only applies to shear natural components with entries +/-2.
    For general spins, use ``sigma`` or ``spin``/``epsilons`` directly.
    """
    sig = tuple(int(s) for s in sigma)
    mapping = {v: k for k, v in _SHEAR_COMPONENT_SIGMA.items()}
    if sig not in mapping:
        raise ValueError(
            "Cannot infer shear natural-component index from sigma="
            f"{sig}. Use sigma directly, or pass spin and epsilons."
        )
    return mapping[sig]


def _component_index(component: int | None = None, sigma=None) -> int:
    """Backward-compatible shear-only helper."""
    if component is not None:
        mu = int(component)
        if mu not in (0, 1, 2, 3):
            raise ValueError("component must be 0, 1, 2, or 3.")
        return mu
    if sigma is None:
        raise ValueError("Either component or sigma must be provided.")
    return natural_component_index_from_sigma(sigma)


def _sincos2angbar(psi, delta):
    """Return sin(2 beta_bar), cos(2 beta_bar) for triangle geometry."""
    cos2b = np.cos(delta) + np.sin(2.0 * psi)
    sin2b = np.cos(2.0 * psi) * np.sin(delta)
    norm = np.sqrt(cos2b**2 + sin2b**2)
    return sin2b / norm, cos2b / norm


def _broadcast_triangle(theta1, theta2, phi):
    t1 = np.asarray(theta1, dtype=float)
    t2 = np.asarray(theta2, dtype=float)
    p = np.asarray(phi, dtype=float)

    if t1.ndim == 1 and t2.ndim == 1 and p.ndim == 1:
        return np.meshgrid(t1, t2, p, indexing="ij")
    return np.broadcast_arrays(t1, t2, p)


def _angle(z):
    return np.angle(z)


def _centroid_angles(theta1, theta2, phi):
    """Angles from each vertex to the centroid.

    Vertices are placed at ``0``, ``theta1``, and ``theta2 * exp(-i phi)``.
    The factor of 1/3 in the centroid vectors is irrelevant for angles.
    """
    z3 = theta2 * np.exp(-1j * phi)
    alpha1 = _angle(theta1 + z3)
    alpha2 = _angle(-2.0 * theta1 + z3)
    alpha3 = _angle(theta1 - 2.0 * z3)
    return alpha1, alpha2, alpha3


def _x_reference_angles(phi):
    """Angles of the legacy x/cross reference directions at the three vertices.

    These values are fixed by matching the legacy shear natural-component
    formulas.  With ``sigma=(+2,+2,+2)``, the centroid conversion gives the
    old prefactor ``q1*q2*q3*exp(3j*phi)``.
    """
    return -0.5 * phi, np.zeros_like(phi), -phi


def _legacy_ortho2cent_exp2psi(theta1, theta2, phi):
    """Return exp(2j psi_i) used by the legacy orthocenter -> centroid code."""
    theta3 = np.sqrt(
        np.maximum(theta1**2 + theta2**2 - 2.0 * theta1 * theta2 * np.cos(phi), 0.0)
    )

    def temp(a, b, c):
        cos_phi = (a**2 + b**2 - c**2) / (2.0 * a * b)
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        phi3 = np.arccos(cos_phi)
        cos2psi = ((b**2 - a**2) ** 2 - 4.0 * a**2 * b**2 * np.sin(phi3) ** 2) / 4.0
        sin2psi = (b**2 - a**2) * a * b * np.sin(phi3)
        norm = np.sqrt(cos2psi**2 + sin2psi**2)
        return cos2psi / norm + 1j * sin2psi / norm

    exp2psi3 = temp(theta1, theta2, theta3)
    exp2psi1 = temp(theta2, theta3, theta1)
    exp2psi2 = temp(theta3, theta1, theta2)
    return exp2psi1, exp2psi2, exp2psi3


def x2ortho_factor(
    component: int | None,
    theta1,
    theta2,
    phi,
    *,
    sigma=None,
    spin: int | Sequence[int] | None = None,
    epsilons: Sequence[int] | None = None,
):
    """Phase factor converting x-projection to orthocenter projection.

    For a general component, pass one of

    - ``sigma=(sigma1, sigma2, sigma3)``;
    - ``spin=s`` or ``spin=(s1,s2,s3)`` with ``epsilons=(e1,e2,e3)``.

    The old call ``x2ortho_factor(component, theta1, theta2, phi)`` remains
    valid for shear natural components.
    """
    sig = _sigma_triple(component=component, sigma=sigma, spin=spin, epsilons=epsilons)
    t1, t2, p = _broadcast_triangle(theta1, theta2, phi)
    sin2pb, cos2pb = _sincos2angbar(np.arctan2(t2, t1), np.pi - p)
    beta_bar = 0.5 * np.arctan2(sin2pb, cos2pb)
    return np.exp(-1j * sig[2] * beta_bar)


def ortho2cent_factor(
    component: int | None,
    theta1,
    theta2,
    phi,
    *,
    sigma=None,
    spin: int | Sequence[int] | None = None,
    epsilons: Sequence[int] | None = None,
):
    """Phase factor converting orthocenter projection to centroid projection.

    This generalizes the legacy implementation.  The old code marked this
    conversion as not validated, so the same caveat applies here.
    """
    sig = _sigma_triple(component=component, sigma=sigma, spin=spin, epsilons=epsilons)
    t1, t2, p = _broadcast_triangle(theta1, theta2, phi)
    exp2psi1, exp2psi2, exp2psi3 = _legacy_ortho2cent_exp2psi(t1, t2, p)

    # The legacy shear rule is phase_i**(-sigma_i/2).  Use angles rather than
    # complex fractional powers so integer spins are handled consistently.
    psi1 = 0.5 * np.angle(exp2psi1)
    psi2 = 0.5 * np.angle(exp2psi2)
    psi3 = 0.5 * np.angle(exp2psi3)
    return np.exp(-1j * (sig[0] * psi1 + sig[1] * psi2 + sig[2] * psi3))


def x2cent_factor(
    component: int | None,
    theta1,
    theta2,
    phi,
    *,
    sigma=None,
    spin: int | Sequence[int] | None = None,
    epsilons: Sequence[int] | None = None,
):
    """Phase factor converting x-projection to centroid projection.

    For the old shear natural components this reproduces the direct legacy
    expressions exactly, e.g. ``component=0`` gives
    ``q1*q2*q3*exp(3j*phi)``.
    """
    sig = _sigma_triple(component=component, sigma=sigma, spin=spin, epsilons=epsilons)
    t1, t2, p = _broadcast_triangle(theta1, theta2, phi)

    alpha1, alpha2, alpha3 = _centroid_angles(t1, t2, p)
    alpha_x1, alpha_x2, alpha_x3 = _x_reference_angles(p)

    phase = (
        sig[0] * (alpha1 - alpha_x1)
        + sig[1] * (alpha2 - alpha_x2)
        + sig[2] * (alpha3 - alpha_x3)
    )
    return np.exp(1j * phase)


def projection_factor(
    from_projection: Projection,
    to_projection: Projection,
    theta1,
    theta2,
    phi,
    *,
    component: int | None = None,
    sigma=None,
    spin: int | Sequence[int] | None = None,
    epsilons: Sequence[int] | None = None,
):
    """Return the phase factor converting between projection conventions.

    Use one of the following equivalent interfaces:

    - ``component=0,1,2,3`` for the old shear natural components only;
    - ``sigma=(sigma1, sigma2, sigma3)`` for a general effective spin triple;
    - ``spin=s`` or ``spin=(s1,s2,s3)`` with ``epsilons=(e1,e2,e3)``.

    ``sigma_i`` is the effective spin carried by the chosen component at
    vertex i, usually ``sigma_i = epsilon_i * spin_i``.
    """
    src = _projection_name(from_projection)
    dst = _projection_name(to_projection)
    sig = _sigma_triple(component=component, sigma=sigma, spin=spin, epsilons=epsilons)

    if src == dst:
        t1, t2, p = _broadcast_triangle(theta1, theta2, phi)
        return np.ones_like(t1 + t2 + p, dtype=complex)

    if src == "x" and dst == "ortho":
        return x2ortho_factor(None, theta1, theta2, phi, sigma=sig)
    if src == "ortho" and dst == "x":
        return 1.0 / x2ortho_factor(None, theta1, theta2, phi, sigma=sig)

    if src == "x" and dst == "cent":
        return x2cent_factor(None, theta1, theta2, phi, sigma=sig)
    if src == "cent" and dst == "x":
        return 1.0 / x2cent_factor(None, theta1, theta2, phi, sigma=sig)

    if src == "ortho" and dst == "cent":
        return ortho2cent_factor(None, theta1, theta2, phi, sigma=sig)
    if src == "cent" and dst == "ortho":
        return 1.0 / ortho2cent_factor(None, theta1, theta2, phi, sigma=sig)

    raise RuntimeError("unreachable projection conversion branch")


def convert_projection(
    zeta,
    theta1,
    theta2,
    phi,
    *,
    from_projection: Projection = "x",
    to_projection: Projection = "centroid",
    component: int | None = None,
    sigma=None,
    spin: int | Sequence[int] | None = None,
    epsilons: Sequence[int] | None = None,
):
    """Convert a 3PCF component between projection conventions.

    Parameters
    ----------
    zeta : array_like
        3PCF values with shape ``(ntheta1, ntheta2, nphi)``.
    theta1, theta2, phi : array_like
        Coordinate grids.  The common use case is 1D ``theta1``, 1D
        ``theta2`` and 1D ``phi``.
    from_projection, to_projection : str
        ``'x'``/``'cross'``, ``'ortho'`` or ``'centroid'``.
    component : int, optional
        Old shear natural-component index.  Only valid for spin-2 shear.
    sigma : sequence of three ints, optional
        Effective spin triple ``sigma_i = epsilon_i * spin_i``.
    spin, epsilons : optional
        Alternative way to specify ``sigma``.  ``spin`` may be a scalar for
        equal-spin fields, and ``epsilons`` defaults to ``(+1,+1,+1)``.
    """
    z = np.asarray(zeta)
    fac = projection_factor(
        from_projection,
        to_projection,
        theta1,
        theta2,
        phi,
        component=component,
        sigma=sigma,
        spin=spin,
        epsilons=epsilons,
    )
    return z * fac
