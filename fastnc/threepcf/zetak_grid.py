"""Double-Hankel transformed zeta_k modes on the 3PCF real-space grid."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable
import logging
import time
import numpy as np

from ..hankel.wrapper import DoubleHankelConfig, double_hankel_transform
from .grid import FFTGrid
from .hkernel_grid import HKernelGrid, HKernel, HKernelKey
from .spin import EffectiveSpinTriple, SpinSpec, as_effective_spin_triple
from .zeta_grid import ZetaGrid


@dataclass(frozen=True)
class ZetaKKey:
    """Deduplication key for one ``zeta_k`` double-Hankel transform."""

    hkey: HKernelKey
    m: int
    n: int
    Sigma: int


@dataclass
class ZetaKMode:
    """One opening-angle coefficient ``zeta_k(theta1, theta2)``.

    ``value`` stores the full FFTLog real-space grid.  Use :meth:`get_value`
    for the user-facing/downsampled grid and :meth:`get_value_fft` for the
    full FFT grid explicitly.
    """

    grid: FFTGrid
    key: ZetaKKey
    value: np.ndarray
    source_k: float
    source_sigma: tuple[int, int, int]

    def __post_init__(self):
        self.value = np.asarray(self.value)
        if self.value.shape != self.grid.shape_theta_fft:
            raise ValueError(
                f"ZetaKMode.value must have shape {self.grid.shape_theta_fft}; "
                f"got {self.value.shape}."
            )

    @property
    def theta1(self) -> np.ndarray:
        """Length ``|X_2-X_1|`` in the X1-reference convention."""
        return self.grid.theta

    @property
    def theta2(self) -> np.ndarray:
        """Length ``|X_3-X_1|`` in the X1-reference convention."""
        return self.grid.theta

    @property
    def theta1_fft(self) -> np.ndarray:
        return self.grid.theta_fft

    @property
    def theta2_fft(self) -> np.ndarray:
        return self.grid.theta_fft

    @property
    def value_fft(self) -> np.ndarray:
        return self.value

    def get_value(self) -> np.ndarray:
        """Return ``zeta_k`` on the user-facing theta grid."""
        return self.grid.downsample_theta_array(self.value, axis1=0, axis2=1)

    def get_value_fft(self) -> np.ndarray:
        """Return ``zeta_k`` on the full FFTLog theta grid."""
        return self.value

    def get_zeta_k(self) -> np.ndarray:
        """Return ``zeta_k`` on the user-facing theta grid."""
        return self.get_value()

    def get_zeta_k_fft(self) -> np.ndarray:
        """Return ``zeta_k`` on the full FFTLog theta grid."""
        return self.get_value_fft()


@dataclass
class ZetaKGrid:
    """In-memory storage of deduplicated ``zeta_k`` modes."""

    spin: tuple[int, int, int]
    kmax: float
    grid: FFTGrid
    modes: dict[ZetaKKey, ZetaKMode] = field(default_factory=dict)
    aliases: dict[tuple[tuple[int, int, int], int], ZetaKKey] = field(default_factory=dict)
    active_epsilons: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)
    logger: logging.Logger | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.spin = tuple(int(x) for x in self.spin)
        self.kmax = float(self.kmax)
        self.spin_spec = SpinSpec(self.spin)

    @staticmethod
    def two_k(k: float) -> int:
        return int(round(2.0 * float(k)))

    @staticmethod
    def key_from_sigma_k(sigma: tuple[int, int, int], k: float) -> ZetaKKey:
        eff = as_effective_spin_triple(sigma)
        hkey = HKernelGrid.key_from_sigma_k(sigma, k)
        m, n = eff.bessel_orders(float(k))
        return ZetaKKey(hkey=hkey, m=int(m), n=int(n), Sigma=int(eff.Sigma))

    def _alias_key(self, epsilon: tuple[int, int, int], k: float) -> tuple[tuple[int, int, int], int]:
        return tuple(int(e) for e in epsilon), self.two_k(k)

    def sigma_from_epsilon(self, epsilon: tuple[int, int, int]) -> tuple[int, int, int]:
        return self.spin_spec.sigma_from_epsilon(epsilon)

    def k_values(self, epsilon: tuple[int, int, int]) -> np.ndarray:
        return as_effective_spin_triple(self.sigma_from_epsilon(epsilon)).k_values(self.kmax)

    def add(self, mode: ZetaKMode) -> None:
        mode.grid.validate_same(self.grid)
        self.modes[mode.key] = mode

    def get(self, key: ZetaKKey) -> ZetaKMode:
        return self.modes[key]

    def get_for_epsilon(self, epsilon: tuple[int, int, int], k: float) -> ZetaKMode:
        """Return the stored ``ZetaKMode`` for ``(epsilon, k)``.

        The returned mode stores the full FFTLog theta grid internally.  Use
        ``mode.get_value()`` for the user-facing grid and
        ``mode.get_value_fft()`` for the full FFT grid.
        """
        key = self.aliases.get(self._alias_key(epsilon, k))
        if key is None:
            key = self.key_from_sigma_k(self.sigma_from_epsilon(epsilon), k)
        return self.modes[key]

    def get_for_epsilon_fft(self, epsilon: tuple[int, int, int], k: float) -> np.ndarray:
        """Return ``zeta_k`` for ``(epsilon, k)`` on the full FFTLog theta grid."""
        return self.get_for_epsilon(epsilon, k).get_value_fft()

    def get_value_for_epsilon(self, epsilon: tuple[int, int, int], k: float) -> np.ndarray:
        """Return ``zeta_k`` for ``(epsilon, k)`` on the user-facing theta grid."""
        return self.get_for_epsilon(epsilon, k).get_value()

    def get_value_for_epsilon_fft(self, epsilon: tuple[int, int, int], k: float) -> np.ndarray:
        """Return ``zeta_k`` for ``(epsilon, k)`` on the full FFTLog theta grid."""
        return self.get_for_epsilon(epsilon, k).get_value_fft()

    def __contains__(self, key: ZetaKKey) -> bool:
        return key in self.modes

    def _compute_mode(
        self,
        hkernel: HKernel,
        *,
        sigma: tuple[int, int, int],
        k: float,
        hankel_config: DoubleHankelConfig,
        bin_width_logtheta: float | None = None,
    ) -> ZetaKMode:
        hkernel.grid.validate_same(self.grid)
        eff = as_effective_spin_triple(sigma)
        m, n = eff.bessel_orders(float(k))
        prefactor = ((-1j) ** eff.Sigma) / (2.0 * np.pi) ** 3
        integrand = hkernel.value * self.grid.ELL1**2 * self.grid.ELL2**2
        _, _, zeta = double_hankel_transform(
            self.grid.ell,
            self.grid.ell,
            prefactor * integrand,
            m,
            n,
            config=hankel_config,
            bin_width_logtheta=bin_width_logtheta,
        )
        return ZetaKMode(
            grid=self.grid,
            key=self.key_from_sigma_k(sigma, float(k)),
            value=zeta,
            source_k=float(k),
            source_sigma=tuple(int(x) for x in sigma),
        )

    def compute_epsilon(
        self,
        Hgrid: HKernelGrid,
        epsilon: tuple[int, int, int],
        *,
        hankel_config: DoubleHankelConfig,
        bin_width_logtheta: float | None = None,
        force: bool = False,
    ) -> list[ZetaKMode]:
        Hgrid.grid.validate_same(self.grid)
        eps = tuple(int(e) for e in epsilon)
        sigma = self.sigma_from_epsilon(eps)
        out: list[ZetaKMode] = []
        for k in self.k_values(eps):
            key = self.key_from_sigma_k(sigma, float(k))
            alias = self._alias_key(eps, float(k))
            if key not in self.modes or force:
                hkernel = Hgrid.get_for_epsilon(eps, float(k))
                t_k = time.perf_counter()
                self.modes[key] = self._compute_mode(
                    hkernel,
                    sigma=sigma,
                    k=float(k),
                    hankel_config=hankel_config,
                    bin_width_logtheta=bin_width_logtheta,
                )
                if self.logger is not None:
                    self.logger.debug("3PCF zetak epsilon=%s k=%+.1f FFTLog finished in %.3f s", eps, float(k), time.perf_counter() - t_k)
            self.aliases[alias] = key
            out.append(self.modes[key])
        return out

    def compute_all_epsilons(
        self,
        Hgrid: HKernelGrid,
        epsilons: Iterable[tuple[int, int, int]] | None = None,
        *,
        hankel_config: DoubleHankelConfig,
        bin_width_logtheta: float | None = None,
        force: bool = False,
    ) -> "ZetaKGrid":
        if epsilons is None:
            epsilons = self.spin_spec.representative_epsilons()
        epsilons = tuple(tuple(int(e) for e in eps) for eps in epsilons)
        self.active_epsilons = epsilons
        for eps in epsilons:
            self.compute_epsilon(
                Hgrid,
                tuple(eps),
                hankel_config=hankel_config,
                bin_width_logtheta=bin_width_logtheta,
                force=force,
            )
        return self

    def _resolve_component(self, *, epsilon=None, component=None):
        if epsilon is not None and component is not None:
            raise ValueError("Specify at most one of epsilon or component.")
        if epsilon is None and component is None and len(self.active_epsilons) == 1:
            epsilon = self.active_epsilons[0]

        if epsilon is not None:
            idx, conjugated = self.spin_spec.component_index_from_epsilon(epsilon)
            comp = self.spin_spec.component(idx)
            requested_epsilon = tuple(int(e) for e in epsilon)
            requested_sigma = self.spin_spec.sigma_from_epsilon(requested_epsilon)
            stored_epsilon = comp.epsilon
            return idx, requested_epsilon, requested_sigma, stored_epsilon, conjugated
        if component is not None:
            comp = self.spin_spec.component(component)
            return comp.index, comp.epsilon, comp.sigma, comp.epsilon, False
        comp = self.spin_spec.component(0)
        return comp.index, comp.epsilon, comp.sigma, comp.epsilon, False

    def k_values_for(self, *, epsilon=None, component=None) -> np.ndarray:
        """Return the user-facing k values for an epsilon/component."""
        _, _, sigma, _, _ = self._resolve_component(epsilon=epsilon, component=component)
        return as_effective_spin_triple(sigma).k_values(self.kmax)

    def zeta_k_array(self, *, epsilon=None, component=None) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], tuple[int, int, int], int]:
        """Return ``(k_values, zeta_k, sigma, epsilon, component)``.

        The returned arrays are user-facing.  If ``epsilon`` is the conjugate of
        a stored representative, the stored modes are conjugated and the k labels
        are mapped consistently with the existing natural-component convention.
        """
        idx, requested_epsilon, requested_sigma, stored_epsilon, conjugated = self._resolve_component(
            epsilon=epsilon, component=component
        )
        stored_sigma = self.spin_spec.sigma_from_epsilon(stored_epsilon)
        stored_k = as_effective_spin_triple(stored_sigma).k_values(self.kmax)
        zeta = [self.get_for_epsilon(stored_epsilon, float(k)).get_value() for k in stored_k]
        zeta_arr = np.asarray(zeta)
        if conjugated:
            return -stored_k, np.conjugate(zeta_arr), requested_sigma, requested_epsilon, int(idx)
        return stored_k, zeta_arr, requested_sigma, requested_epsilon, int(idx)

    def zeta_k_array_fft(self, *, epsilon=None, component=None) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], tuple[int, int, int], int]:
        """Return ``zeta_k`` modes on the full FFTLog theta grid.

        The return signature matches :meth:`zeta_k_array`, but the second
        element has shape ``(nk, ntheta_fft, ntheta_fft)``.
        """
        idx, requested_epsilon, requested_sigma, stored_epsilon, conjugated = self._resolve_component(
            epsilon=epsilon, component=component
        )
        stored_sigma = self.spin_spec.sigma_from_epsilon(stored_epsilon)
        stored_k = as_effective_spin_triple(stored_sigma).k_values(self.kmax)
        zeta = [self.get_for_epsilon(stored_epsilon, float(k)).get_value_fft() for k in stored_k]
        zeta_arr = np.asarray(zeta)
        if conjugated:
            return -stored_k, np.conjugate(zeta_arr), requested_sigma, requested_epsilon, int(idx)
        return stored_k, zeta_arr, requested_sigma, requested_epsilon, int(idx)

    def get_zeta_k_array(self, *, epsilon=None, component=None):
        """Alias of :meth:`zeta_k_array` using explicit getter naming."""
        return self.zeta_k_array(epsilon=epsilon, component=component)

    def get_zeta_k_array_fft(self, *, epsilon=None, component=None):
        """Alias of :meth:`zeta_k_array_fft` using explicit getter naming."""
        return self.zeta_k_array_fft(epsilon=epsilon, component=component)

    def component_specs_for_resum(self):
        """Return component specs available for all-component resummation."""
        if self.active_epsilons:
            comps = []
            seen = set()
            for eps in self.active_epsilons:
                idx, _ = self.spin_spec.component_index_from_epsilon(eps)
                if idx not in seen:
                    comps.append(self.spin_spec.component(idx))
                    seen.add(idx)
            return tuple(comps)
        return self.spin_spec.components()

    def resum(
        self,
        delta_phi,
        *,
        phase: str = "nu",
        normalization: float = 1.0,
        bin_width: float | None = None,
        config=None,
    ) -> ZetaGrid:
        """Resum all stored components into an x-projection real-space grid.

        The returned :class:`ZetaGrid` stores every available independent
        component with shape ``(ncomponent, ntheta1, ntheta2, nphi)``.  Projection
        conversion is intentionally not done here; call
        ``ZetaGrid.to_projection(...)`` on the returned object instead.
        """
        delta_phi_arr = np.asarray(delta_phi, dtype=float)
        values = []
        sigmas = []
        epsilons = []
        components = []

        for comp in self.component_specs_for_resum():
            k_values, zeta_k_fft, sigma, eps, idx = self.zeta_k_array_fft(component=comp.index)
            values.append(
                resum_multipoles(
                    zeta_k_fft,
                    k_values,
                    delta_phi_arr,
                    sigma,
                    phase=phase,
                    normalization=normalization,
                    bin_width=bin_width,
                )
            )
            sigmas.append(tuple(int(x) for x in sigma))
            epsilons.append(tuple(int(x) for x in eps))
            components.append(int(idx))

        return ZetaGrid(
            grid=self.grid,
            delta_phi=delta_phi_arr,
            values_fft=np.asarray(values),
            sigmas=tuple(sigmas),
            epsilons=tuple(epsilons),
            components=tuple(components),
            projection="x",
            phase=phase,
            normalization=normalization,
            bin_width=bin_width,
            config=config,
            spin=self.spin,
        )

# ---------------------------------------------------------------------------
# Opening-angle resummation.  These helpers are local to ZetaKGrid because
# resummation is only defined for zeta_k modes.

def opening_angle_phase_values(k_values: np.ndarray, sigma: tuple[int, int, int] | EffectiveSpinTriple, *, phase: str = "nu") -> np.ndarray:
    """Return Fourier phase labels for opening-angle resummation."""
    spin = as_effective_spin_triple(sigma)
    k_values = np.asarray(k_values, dtype=float)
    if phase == "nu":
        return spin.nu(k_values)
    if phase == "k":
        return k_values
    raise ValueError("phase must be either 'nu' or 'k'.")


def resummation_matrix(k_values: np.ndarray, delta_phi: np.ndarray, sigma: tuple[int, int, int] | EffectiveSpinTriple, *, phase: str = "nu", bin_width: float | None = None) -> np.ndarray:
    labels = opening_angle_phase_values(k_values, sigma, phase=phase)
    delta_phi = np.asarray(delta_phi, dtype=float)
    mat = np.exp(1j * labels[:, None] * delta_phi[None, :])
    if bin_width is not None:
        width = float(bin_width)
        fac = np.ones_like(labels, dtype=complex)
        nz = np.abs(labels) > 0.0
        fac[nz] = (np.exp(1j * labels[nz] * width) - 1.0) / (1j * labels[nz] * width)
        mat = fac[:, None] * mat
    return mat


def resum_multipoles(zeta_k: np.ndarray, k_values: np.ndarray, delta_phi: np.ndarray, sigma: tuple[int, int, int] | EffectiveSpinTriple, *, phase: str = "nu", normalization: float = 1.0, bin_width: float | None = None) -> np.ndarray:
    """Resum zeta_k(theta1,theta2) into zeta(theta1,theta2,DeltaPhi)."""
    zeta_k = np.asarray(zeta_k)
    if zeta_k.ndim != 3:
        raise ValueError("zeta_k must have shape (nk, ntheta1, ntheta2).")
    mat = resummation_matrix(k_values, delta_phi, sigma, phase=phase, bin_width=bin_width)
    out = np.tensordot(zeta_k, mat, axes=(0, 0))
    return normalization * out

