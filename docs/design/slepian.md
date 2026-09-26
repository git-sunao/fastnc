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

The implemented unequal-order regular reference uses an independent field:

```python
SlepianConfig(
    regular_method="quadrature",
    regular_n_x=256,
    regular_x_padding=20.0,
)
```

`quadrature` reconstructs the radial functions and integrates directly over a
logarithmic `x` grid. The grid contains the requested theta values in addition
to its regular logarithmic samples. `full_matrix` constructs and caches the
uncompressed `F_ab(theta1, theta2)` before contracting the source-dependent
Mellin coefficients. `low_rank` constructs the same matrix transiently, applies
an independent SVD at each `(theta1, theta2)`, discards the full matrix, and
caches only the truncated factors. `regular_low_rank_rank` fixes a common rank;
when it is `None`, `regular_low_rank_rtol` selects the smallest common rank that
satisfies the relative Frobenius-tail tolerance at every theta pair.

The current quadrature accepts only a positive even difference between the
canonical Bessel orders. Such kernels have one-sided Heaviside support. Odd
canonical-order differences have regular pieces on both sides and a singular
diagonal limit; they remain unsupported until the required distributional or
principal-value prescription is established.

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

For a constant radial factor, the remaining closure kernel is the distribution

```text
C_pq(x, theta) = integral_0^infinity d ell ell
                 J_p(ell x) J_q(ell theta).
```

Use `x` for the internal radial integration variable and `theta` for the target
separation throughout this route. Do not introduce `y` or `rho` as aliases for
these coordinates.

For integer signed Bessel orders, represent the kernel as

```text
C_pq(x, theta)
  = c_delta(p, q) delta(x - theta) / x
  + Theta(theta - x) C_pq^<(x, theta)
  + Theta(x - theta) C_pq^>(x, theta),

c_delta(p, q) = cos[pi (p - q) / 2].
```

The cosine is exactly `0`, `+1`, or `-1` for integer orders and must be
evaluated by integer parity, not floating-point trigonometry. The two regular
pieces are orientation-specific Weber-Schafheitlin continuations; they are not
inferred by swapping array axes after evaluation. Either regular piece may
vanish for a particular order pair, but the primitive still reports its
support explicitly.

The low-level constant-leg primitive returns a decomposition with three
logical fields: the exact contact coefficient, the regular evaluator or
values, and the regular support (`x < theta`, `x > theta`, or both). It does
not apply FFTLog coefficients, perform the outer `x` integral, or modify a
`ZetaKTable`. Assembly is responsible for contracting each piece.

For `p = q`, the regular pieces vanish and the usual Bessel closure relation
is recovered. Signed equal-canonical-order cases obtain the same result with
the appropriate contact sign. This current contact-only calculation is the
regression benchmark and must remain numerically unchanged when the
decomposition is introduced.

The off-diagonal values already produced by the general power-law Weber
primitive are not, by themselves, an implementation of this constant-leg
distribution. The diagonal contact term and the regular continuation have
different mathematical and numerical treatment. Introduce and test this
decomposition before constructing the general regular matrix.

## Full Mellin matrix and low rank

For `regular_method="quadrature"`, reconstruct the required radial functions
from their Mellin sums and perform

```text
integral dx x R_single(x) R_double(x, theta1) C_reg(x, theta2)
```

directly. This deliberately slower calculation is the implemented independent
reference. The contact contribution is evaluated separately and added to the
regular result; it is never approximated on the `x` grid.

For `regular_method="full_matrix"`, exchange the Mellin sums and radial
integral, construct the exact uncompressed
`F_ab(theta1, theta2)`, cache it by geometry, orders, exponents, support, and
numerical controls, and perform the full coefficient contraction. This path is
implemented and must reproduce direct quadrature before any later compression.
Finite-band diagonal corrections are evaluated per Mellin basis, preserving
linearity and keeping `F_ab` independent of the source coefficients.

`regular_method="low_rank"` compresses the established matrix while keeping
the full matrix selectable. The compressed object records its retained rank and
global relative reconstruction error. Rank selection must test both matrix
reconstruction and final coefficient-weighted ZetaK error; a Frobenius norm
alone is insufficient because small singular directions may receive large
Mellin weights. The automatic tolerance controls matrix reconstruction only and
is therefore not an end-to-end accuracy guarantee.

## Result boundary

The Slepian calculator owns FFTLog power sums, Weber geometries, primitive
kernels, interpolation tables, and future `F_ab` resources. It constructs a
route-independent `ZetaKTable` directly on the requested theta bins. Final
opening-angle and shear-projection assembly is shared with other routes.

The calculator must preserve a direct benchmark path, report unsupported
representations explicitly, and never hide a route fallback that changes the
meaning of the requested term or epsilon tuple.
