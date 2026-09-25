# Slepian route contract

Status: Normative numerical and implementation contract.

Detailed derivations, experiments, and rejected alternatives are retained in
`docs/notes/slepian-performance.md`. Human-oriented notebooks and the TeX/PDF
report are local under `dev/slepian/`.

## Scope

Slepian, semi-analytic, and numeric 3PCF algorithms consume angular Fourier
coordinates `ell` at fixed redshift or after projection. A native
`Bispectrum2D` and a projected `Bispectrum3D` therefore share the same route
interfaces after an angular representation has been constructed.

The first production Slepian implementation is a native-2D reference path.
LOS integration remains a separate projection problem. Optimized
coefficient-level LOS integration may be added only after it reproduces a
transparent node-by-node reference calculation.

## Radial transform conventions

The FFTLog/Mellin expansion uses complex exponents with a fixed real bias and
Fourier-spaced imaginary indices. A double Bessel transform is reduced by
homogeneity to an overall scale power and a unit Weber-Schafheitlin function of
one ratio

```text
scale = max(x, theta)
ratio = min(x, theta) / scale,  0 < ratio <= 1.
```

`WeberGeometry` stores the scale/ratio reconstruction and unique ratio mapping.
Primitive Weber values are evaluated once per unique ratio and cached by
geometry, Mellin exponent, Bessel-order pair, tolerance, diagonal policy, and
evaluation method. Factor-dependent finite-band diagonal corrections do not
belong in the universal primitive cache.

## Method separation

Optimization choices are independent method-valued configuration fields, not
interacting boolean flags. The currently implemented Weber choice is:

```python
SlepianConfig(
    weber_method="direct",  # direct | interpolated
)
```

The agreed interface for the later unequal-order regular contribution will add
an independent field:

```python
SlepianConfig(
    regular_method="quadrature",  # quadrature | full_matrix | low_rank
)
```

`regular_method` is a design contract, not an available configuration option
in the current implementation.

Each accepted slower method remains permanently selectable as the reference
for the next optimization. Caches for different methods are distinct.

The required development ladders are

```text
direct Weber -> interpolated Weber

direct regular x quadrature
  -> full uncompressed F_ab contraction
  -> low-rank F_ab contraction
```

Do not implement or validate multiple ladder steps in one change.

## Weber interpolation

The interpolated method tabulates the complex unit Weber function in
`t = log(-log(ratio))` and interpolates real and imaginary parts. Ratios near
one may be evaluated directly, and a request with no more interpolation-region
ratios than table nodes falls back to direct evaluation because constructing a
table would cost more.

Interpolation must be validated for every relevant Bessel-order pair and
complex Mellin exponent. Pointwise Weber agreement is necessary but not
sufficient: final validation must include actual FFTLog coefficients and the
resulting ZetaK contribution.

The current direct evaluator uses a power series for
`hyp2f1(A, B; C, ratio**2)`. It is numerically unreliable near `ratio -> 1` at
large imaginary Mellin index, and interpolation tables inherit that error.
This is a known limitation, not evidence that interpolation alone failed. The
planned repair and measured residuals are in `docs/todo.md`.

## Constant and nonconstant legs

Equal-order constant-leg kernels contain a contact contribution. Future
unequal-order kernels may also contain regular and Heaviside-supported terms.
The kernel primitive must expose these pieces explicitly rather than relying
on upper-level equality checks.

The current equal-order contact result is the regression benchmark. Introduce
the unequal-order decomposition before constructing the general regular
matrix.

## Full Mellin matrix and low rank

For `regular_method="quadrature"`, reconstruct the required radial functions
from their Mellin sums and perform the remaining regular radial integral
directly. This deliberately slower calculation is the independent reference.

For `regular_method="full_matrix"`, exchange the Mellin sums and radial
integral, construct the exact uncompressed `F_ab(rho)`, cache it by geometry,
orders, exponents, support, and numerical controls, and perform the full
coefficient contraction. Validate final ZetaK against direct quadrature before
proceeding.

Only then may `regular_method="low_rank"` compress the established matrix.
Keep the full matrix selectable. Rank selection must test both matrix
reconstruction and final coefficient-weighted ZetaK error; a Frobenius norm
alone is insufficient because small singular directions may receive large
Mellin weights.

## Result boundary

The Slepian calculator owns FFTLog power sums, Weber geometries, primitive
kernels, interpolation tables, and future `F_ab` resources. It constructs a
route-independent `ZetaKTable` directly on the requested theta bins. Final
opening-angle and shear-projection assembly is shared with other routes.

The calculator must preserve a direct benchmark path, report unsupported
representations explicitly, and never hide a route fallback that changes the
meaning of the requested term or epsilon tuple.
