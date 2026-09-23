"""Resummed real-space 3PCF grid object."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .config import ThreePCFConfig
from .grid import FFTGrid
from .projection import Projection, _projection_name, convert_projection


@dataclass
class ZetaGrid:
    """Resummed real-space three-point correlation functions on a grid.

    This is the final user-facing grid object.  It stores all independent
    natural components.  ``values`` has shape

        (ncomponent, ntheta1, ntheta2, nphi)

    and is defined on ``grid.theta`` for both side-length axes and
    ``delta_phi`` for the opening-angle axis.  Projection conversion acts on
    this object through :meth:`to_projection` and returns a new ``ZetaGrid``.
    """

    grid: FFTGrid
    delta_phi: np.ndarray
    values: np.ndarray | None = None
    sigmas: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)
    epsilons: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)
    components: tuple[int, ...] = field(default_factory=tuple)
    projection: str = "x"
    phase: str = "nu"
    normalization: float = 1.0
    bin_width: float | None = None
    config: ThreePCFConfig | None = None
    spin: tuple[int, int, int] | None = None
    values_fft: np.ndarray | None = None

    def __post_init__(self):
        self.delta_phi = np.asarray(self.delta_phi, dtype=float)
        if self.values_fft is not None:
            self.values_fft = np.asarray(self.values_fft)
        if self.values is None:
            if self.values_fft is None:
                raise ValueError("Either values or values_fft must be provided.")
            self.values = self._downsample_values_fft(self.values_fft)
        else:
            self.values = np.asarray(self.values)
        self.sigmas = tuple(tuple(int(x) for x in sig) for sig in self.sigmas)
        self.epsilons = tuple(tuple(int(x) for x in eps) for eps in self.epsilons)
        self.components = tuple(int(c) for c in self.components)
        self.projection = _projection_name(self.projection)
        if self.spin is not None:
            self.spin = tuple(int(x) for x in self.spin)

        ncomp = len(self.components)
        expected = (ncomp, self.grid.n_theta, self.grid.n_theta, self.delta_phi.size)
        if self.values.shape != expected:
            raise ValueError(f"values must have shape {expected}; got {self.values.shape}.")
        if self.values_fft is not None:
            expected_fft = (ncomp, self.grid.n_theta_fft, self.grid.n_theta_fft, self.delta_phi.size)
            if self.values_fft.shape != expected_fft:
                raise ValueError(f"values_fft must have shape {expected_fft}; got {self.values_fft.shape}.")
        if len(self.sigmas) != ncomp:
            raise ValueError("sigmas must have one entry per component.")
        if len(self.epsilons) != ncomp:
            raise ValueError("epsilons must have one entry per component.")
        if len(set(self.components)) != ncomp:
            raise ValueError("components must be unique.")

    def _downsample_values_fft(self, values_fft: np.ndarray) -> np.ndarray:
        """Select user-facing theta axes from full FFT-grid zeta values."""
        return self.grid.downsample_theta_array(values_fft, axis1=1, axis2=2)

    @property
    def theta1(self) -> np.ndarray:
        return self.grid.theta

    @property
    def theta2(self) -> np.ndarray:
        return self.grid.theta

    @property
    def theta(self) -> np.ndarray:
        return self.grid.theta

    @property
    def theta_user(self) -> np.ndarray:
        return self.grid.theta_user

    @property
    def theta_fft(self) -> np.ndarray:
        return self.grid.theta_fft

    @property
    def down_sampler(self) -> np.ndarray:
        return self.grid.down_sampler

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.values.shape

    @property
    def ncomponent(self) -> int:
        return len(self.components)

    @property
    def ntheta1(self) -> int:
        return self.grid.n_theta

    @property
    def ntheta2(self) -> int:
        return self.grid.n_theta

    @property
    def nphi(self) -> int:
        return self.delta_phi.size

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)

    def get_values(self) -> np.ndarray:
        """Return zeta on the user-facing theta grid."""
        return self.values

    def get_values_fft(self) -> np.ndarray:
        """Return zeta on the full FFTLog theta grid."""
        if self.values_fft is None:
            raise RuntimeError("This ZetaGrid does not store full FFT-grid values.")
        return self.values_fft

    def get_zeta(self) -> np.ndarray:
        """Return zeta on the user-facing theta grid."""
        return self.get_values()

    def get_zeta_fft(self) -> np.ndarray:
        """Return zeta on the full FFTLog theta grid."""
        return self.get_values_fft()

    def component_axis_index(self, component: int) -> int:
        component = int(component)
        try:
            return self.components.index(component)
        except ValueError as exc:
            raise KeyError(f"component={component} is not stored in this ZetaGrid.") from exc

    def epsilon_axis_index(self, epsilon: tuple[int, int, int]) -> int:
        """Return the stored component-axis index for an epsilon label.

        If ``spin`` is available, this method accepts both representative and
        conjugate epsilon labels and maps them to the stored natural component.
        Otherwise it falls back to exact matching against ``self.epsilons``.
        """
        eps = tuple(int(e) for e in epsilon)
        if self.spin is not None:
            from .spin import SpinSpec

            idx, _ = SpinSpec(self.spin).component_index_from_epsilon(eps)
            component = SpinSpec(self.spin).component(idx).index
            return self.component_axis_index(component)
        try:
            return self.epsilons.index(eps)
        except ValueError as exc:
            raise KeyError(f"epsilon={eps} is not stored in this ZetaGrid.") from exc

    def get_for_epsilon(self, epsilon: tuple[int, int, int]) -> np.ndarray:
        """Return one natural component on the user-facing theta grid."""
        return self.values[self.epsilon_axis_index(epsilon)]

    def get_for_epsilon_fft(self, epsilon: tuple[int, int, int]) -> np.ndarray:
        """Return one natural component on the full FFTLog theta grid."""
        return self.get_values_fft()[self.epsilon_axis_index(epsilon)]

    def component_values(self, component: int) -> np.ndarray:
        """Return values for one component with shape ``(ntheta1, ntheta2, nphi)``."""
        return self.values[self.component_axis_index(component)]

    def component_values_fft(self, component: int) -> np.ndarray:
        """Return one component on the full FFTLog theta grid."""
        return self.get_values_fft()[self.component_axis_index(component)]

    def component_zeta(self, component: int) -> np.ndarray:
        """Return one component on the user-facing theta grid."""
        return self.component_values(component)

    def component_zeta_fft(self, component: int) -> np.ndarray:
        """Return one component on the full FFTLog theta grid."""
        return self.component_values_fft(component)

    def component_sigma(self, component: int) -> tuple[int, int, int]:
        return self.sigmas[self.component_axis_index(component)]

    def component_epsilon(self, component: int) -> tuple[int, int, int]:
        return self.epsilons[self.component_axis_index(component)]

    def at_phi_index(self, i: int, *, component: int | None = None) -> np.ndarray:
        if component is None:
            return self.values[:, :, :, int(i)]
        return self.component_values(component)[:, :, int(i)]

    def at_theta_index(self, i: int, j: int, *, component: int | None = None) -> np.ndarray:
        if component is None:
            return self.values[:, int(i), int(j), :]
        return self.component_values(component)[int(i), int(j), :]

    def nearest_phi_index(self, phi: float) -> int:
        return int(np.argmin(np.abs(self.delta_phi - float(phi))))

    def nearest_theta_indices(self, theta1: float, theta2: float) -> tuple[int, int]:
        i = int(np.argmin(np.abs(self.grid.theta - float(theta1))))
        j = int(np.argmin(np.abs(self.grid.theta - float(theta2))))
        return i, j

    def at_phi(self, phi: float, *, component: int | None = None) -> np.ndarray:
        return self.at_phi_index(self.nearest_phi_index(phi), component=component)

    def at_theta(self, theta1: float, theta2: float, *, component: int | None = None) -> np.ndarray:
        i, j = self.nearest_theta_indices(theta1, theta2)
        return self.at_theta_index(i, j, component=component)

    def to_projection(self, projection: Projection) -> "ZetaGrid":
        """Return a copy converted to another projection convention.

        ``ZetaGrid`` stores all components, so projection conversion is applied
        component-by-component.  The source projection is the current
        ``self.projection``; the common creation path from ``ZetaKGrid`` always
        produces ``projection='x'``.
        """
        dst = _projection_name(projection)
        src = _projection_name(self.projection)
        if src == dst:
            return self

        converted = []
        for axis, component in enumerate(self.components):
            converted.append(
                convert_projection(
                    self.values[axis],
                    self.grid.theta,
                    self.grid.theta,
                    self.delta_phi,
                    from_projection=src,
                    to_projection=dst,
                    sigma=self.sigmas[axis],
                )
            )

        converted_fft = None
        if self.values_fft is not None:
            converted_fft = []
            for axis, component in enumerate(self.components):
                converted_fft.append(
                    convert_projection(
                        self.values_fft[axis],
                        self.grid.theta_fft,
                        self.grid.theta_fft,
                        self.delta_phi,
                        from_projection=src,
                        to_projection=dst,
                        sigma=self.sigmas[axis],
                    )
                )
            converted_fft = np.asarray(converted_fft)

        return ZetaGrid(
            grid=self.grid,
            delta_phi=self.delta_phi,
            values=np.asarray(converted),
            values_fft=converted_fft,
            sigmas=self.sigmas,
            epsilons=self.epsilons,
            components=self.components,
            projection=dst,
            phase=self.phase,
            normalization=self.normalization,
            bin_width=self.bin_width,
            config=self.config,
            spin=self.spin,
        )
