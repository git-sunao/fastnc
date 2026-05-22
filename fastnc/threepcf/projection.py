"""Projection-convention utilities for resummed 3PCF grids."""
from __future__ import annotations

from typing import Literal
import numpy as np

# conversion is only meaningful for an already resummed zeta(theta1,theta2,phi).

Projection = Literal["x", "cross", "ortho", "orthocenter", "cent", "centroid"]


def _projection_name(name: str) -> str:
    n = str(name).lower()
    if n in {"x", "cross", "times"}:
        return "x"
    if n in {"ortho", "orthocenter"}:
        return "ortho"
    if n in {"cent", "centroid"}:
        return "cent"
    raise ValueError("projection must be one of 'x'/'cross', 'ortho', or 'centroid'.")


def natural_component_index_from_sigma(sigma: tuple[int, int, int] | list[int] | np.ndarray) -> int:
    """Infer the shear natural-component index from a spin triple.

    This only applies to shear natural components with entries +/-2.
    """
    sig = tuple(int(s) for s in sigma)
    mapping = {
        (2, 2, 2): 0,
        (-2, 2, 2): 1,
        (2, -2, 2): 2,
        (2, 2, -2): 3,
    }
    if sig not in mapping:
        raise ValueError(
            "Cannot infer shear natural-component index from sigma="
            f"{sig}. Pass component=0,1,2,3 explicitly, or use a shear "
            "natural-component spin triple."
        )
    return mapping[sig]


def _component_index(component: int | None = None, sigma=None) -> int:
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


def x2ortho_factor(component: int, theta1, theta2, phi):
    """Phase factor converting x-projection to orthocenter projection."""
    mu = _component_index(component)
    t1, t2, p = _broadcast_triangle(theta1, theta2, phi)
    sin2pb, cos2pb = _sincos2angbar(np.arctan2(t2, t1), np.pi - p)
    if mu in (0, 1, 2):
        return cos2pb - 1j * sin2pb
    if mu == 3:
        return cos2pb + 1j * sin2pb
    raise ValueError("component must be 0, 1, 2, or 3.")


def ortho2cent_factor(component: int, theta1, theta2, phi):
    """Phase factor converting orthocenter projection to centroid projection.

    This follows the legacy implementation.  The old code marked this as not
    validated, so use this factor with the same caution for precision work.
    """
    mu = _component_index(component)
    t1, t2, p = _broadcast_triangle(theta1, theta2, phi)
    t3 = np.sqrt(np.maximum(t1**2 + t2**2 - 2.0 * t1 * t2 * np.cos(p), 0.0))

    def temp(a, b, c):
        cos_phi = (a**2 + b**2 - c**2) / (2.0 * a * b)
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        phi3 = np.arccos(cos_phi)
        cos2psi = ((b**2 - a**2) ** 2 - 4.0 * a**2 * b**2 * np.sin(phi3) ** 2) / 4.0
        sin2psi = (b**2 - a**2) * a * b * np.sin(phi3)
        norm = np.sqrt(cos2psi**2 + sin2psi**2)
        return cos2psi / norm + 1j * sin2psi / norm

    exp2psi3 = temp(t1, t2, t3)
    exp2psi1 = temp(t2, t3, t1)
    exp2psi2 = temp(t3, t1, t2)

    out = np.ones_like(exp2psi3, dtype=complex)
    for j, phase in enumerate([1.0, exp2psi1, exp2psi2, exp2psi3]):
        if j == mu:
            out *= phase
        else:
            out *= np.conj(phase)
    return out


def x2cent_factor(component: int, theta1, theta2, phi):
    """Phase factor converting x-projection to centroid projection.

    This is the direct legacy expression used for the shear natural
    components.  ``phi`` is the real-space opening angle between ``theta1``
    and ``theta2``.
    """
    mu = _component_index(component)
    t1, t2, p = _broadcast_triangle(theta1, theta2, phi)

    v = t1 + t2 * np.exp(-1j * p)
    q1 = v / np.conj(v)

    v = -2.0 * t1 + t2 * np.exp(-1j * p)
    q2 = v / np.conj(v)

    v = t1 - 2.0 * t2 * np.exp(-1j * p)
    q3 = v / np.conj(v)

    if mu == 0:
        return q1 * q2 * q3 * np.exp(3j * p)
    if mu == 1:
        return np.conj(q1) * q2 * q3 * np.exp(1j * p)
    if mu == 2:
        return q1 * np.conj(q2) * q3 * np.exp(3j * p)
    if mu == 3:
        return q1 * q2 * np.conj(q3) * np.exp(-1j * p)
    raise ValueError("component must be 0, 1, 2, or 3.")


def projection_factor(
    from_projection: Projection,
    to_projection: Projection,
    theta1,
    theta2,
    phi,
    *,
    component: int | None = None,
    sigma=None,
):
    """Return the phase factor converting between shear projections."""
    src = _projection_name(from_projection)
    dst = _projection_name(to_projection)
    mu = _component_index(component, sigma)

    if src == dst:
        t1, t2, p = _broadcast_triangle(theta1, theta2, phi)
        return np.ones_like(t1 + t2 + p, dtype=complex)

    if src == "x" and dst == "ortho":
        return x2ortho_factor(mu, theta1, theta2, phi)
    if src == "ortho" and dst == "x":
        return 1.0 / x2ortho_factor(mu, theta1, theta2, phi)

    if src == "x" and dst == "cent":
        return x2cent_factor(mu, theta1, theta2, phi)
    if src == "cent" and dst == "x":
        return 1.0 / x2cent_factor(mu, theta1, theta2, phi)

    if src == "ortho" and dst == "cent":
        return ortho2cent_factor(mu, theta1, theta2, phi)
    if src == "cent" and dst == "ortho":
        return 1.0 / ortho2cent_factor(mu, theta1, theta2, phi)

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
):
    """Convert a shear natural component between projection conventions.

    Parameters
    ----------
    zeta : array_like
        3PCF values with shape ``(ntheta1, ntheta2, nphi)``.
    theta1, theta2, phi : array_like
        Coordinate grids.  The common use case is 1D ``theta1``, 1D
        ``theta2`` and 1D ``phi``.
    from_projection, to_projection : str
        ``'x'``/``'cross'``, ``'ortho'`` or ``'centroid'``.
    component, sigma : optional
        Natural-component index or spin triple.  If ``component`` is omitted,
        it is inferred from ``sigma``.
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
    )
    return z * fac

