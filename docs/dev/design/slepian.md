# Slepian route contract

Status: Normative numerical and implementation contract.

Detailed derivations, experiments, and rejected alternatives are retained in
`docs/dev/notes/slepian-performance.md`. Human-oriented notebooks and the TeX/PDF
report are local under `dev/slepian/`.

## Scope

Slepian, semi-analytic, and numeric 3PCF algorithms consume angular Fourier
coordinates `ell` at fixed redshift or after projection. A native
`Bispectrum2D` and a projected `Bispectrum3D` therefore share the same route
interfaces after an angular representation has been constructed.

The native-2D path remains the fixed-redshift reference calculation. A
projected 3D Slepian expression is evaluated on the fixed angular `ell` grid
with `k = (ell + shift) / chi(z)` at every projector node. Its Mellin
coefficients are therefore functions of the LOS node, while the Mellin
exponents and the regular matrix `F_ab` remain fixed by the calculation grid.
The resulting contact and regular contributions are integrated with the LOS
grid and physical weights owned by the projector.

The projected implementation must reproduce the transparent node-by-node
native-2D calculation. For `full_matrix`, each node contracts its factorized
coefficients with the shared full `F_ab`. For `low_rank`, the LOS integral is
performed after direct contraction with the low-rank factors. It must not form
the generally full-rank LOS-averaged matrix `bar_C_ab`.

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
    regular_quadrature="ratio_gauss",  # ratio_gauss | legacy_log
    regular_matrix_implementation="vectorized",  # vectorized | scalar
    regular_n_ratio=64,
    regular_ratio_min=0.05,
)
```

`quadrature` treats the open branches separately. For each constant-leg target
`theta_*`, Gauss-Legendre nodes `r` lie strictly inside
`[regular_ratio_min, 1]`; the two coordinates are `x_< = theta_* r` and
`x_> = theta_*/r`. Their radial measures are respectively
`theta_*^2 r dr` and `theta_*^2 r^(-3) dr`. Thus the distributional point
`x=theta_*` is never sampled and no artificial zero is inserted at the branch
boundary. `legacy_log` preserves the former global logarithmic trapezoidal
integral as a selectable benchmark; its controls remain `regular_n_x` and
`regular_x_padding`. `full_matrix` constructs and caches the uncompressed
`F_ab(theta1, theta2)` with the same open, branch-separated ratio quadrature
before contracting the source-dependent Mellin coefficients. The matrix axes
are `(double Mellin index a, single Mellin index b, theta1, theta_constant)`.
Consequently its construction depends on the angular grid and numerical
hyperparameters but not on source coefficients, cosmology, or the legacy
global `x` grid. `low_rank` constructs the same matrix transiently, applies
an independent SVD at each `(theta1, theta2)`, discards the full matrix, and
caches only the truncated factors. `regular_low_rank_rank` fixes a common rank;
when it is `None`, `regular_low_rank_rtol` selects the smallest common rank that
satisfies the relative Frobenius-tail tolerance at every theta pair.

For `full_matrix` and `low_rank` with ratio-Gauss quadrature, `vectorized`
batches every `(theta_constant, branch, ratio node)` coordinate and evaluates
the double Weber kernel once per Mellin exponent. It changes only the order of
the finite quadrature sums. `scalar` preserves the original nested-loop
implementation as a direct regression and performance reference. With direct
Weber evaluation the two implementations must agree to floating-point
roundoff. Interpolated Weber values may differ at the interpolation-error level
because the scalar and batched requests span different spline intervals.

The current quadrature accepts only a positive even difference between the
canonical Bessel orders. Such kernels have one-sided Heaviside support. Odd
canonical-order differences have regular pieces on both sides and a singular
diagonal limit; they remain unsupported until the required distributional or
principal-value prescription is established.

Each accepted slower method remains permanently selectable as the reference
for the next optimization. Caches for different methods are distinct.

With `diagonal_correction="brute"`, the power-law Weber evaluator is replaced
by the finite-band direct Bessel integral when the radial ratio is greater than
or equal to `weber_brute_min_ratio`. This extends the former exact-diagonal
correction to the neighborhood where the hypergeometric series is numerically
ill-conditioned.

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

For the central Mellin mode `exponent = 0` and an even canonical Bessel-order
difference, the direct evaluator does not call the general hypergeometric
series. If the large-argument order exceeds the small-argument order,
`B = 1 - d` is a nonpositive integer and `hyp2f1` is evaluated as its finite
degree-`d - 1` polynomial. The opposite orientation and equal-order regular
part are returned as exact zero; the equal-order contact contribution remains
separate. This branch preserves the exact signed-order parity factor.

For other exponents, the direct evaluator is piecewise. Below an adaptive
ratio boundary it uses the ordinary power series for
`hyp2f1(A, B; C, ratio**2)`. At and above the boundary it uses the `z = 1`
connection formula, evaluates both hypergeometric series in
`1 - ratio**2`, and constructs the Gamma-function coefficients in log space.
The boundary is
`max(0.25, min(1 / sqrt(2), 8 / abs(Im(exponent))))`: low imaginary indices
retain the balanced-series boundary, while high indices switch earlier to
avoid catastrophic cancellation in the ordinary series. Real integer values
of the connection exponent require a separate limiting formula and therefore
remain on the ordinary-series path unless handled by the eta-zero finite
polynomial branch above.

## Constant and nonconstant legs

The three bispectrum legs are not interchangeable in the Slepian-to-`ZetaK`
calculation. `ZetaK` retains multipoles of the opening angle opposite leg 1,
so leg 1 is the distinguished leg defining the requested angular multipole.
Eliminating leg 1 as the constant Slepian leg would remove the angular
information that `ZetaK` is meant to retain. The Slepian route must therefore
not support a constant leg 1.

Using one-based physical leg labels, the supported and unsupported cases are

```text
constant leg 2 -> supported by the Slepian route
constant leg 3 -> supported by the Slepian route
constant leg 1 -> unsupported by the Slepian route
```

In Python tuple indices these are respectively `constant_legs == (1,)`,
`constant_legs == (2,)`, and the unsupported `constant_legs == (0,)`. Legs 2
and 3 must be treated symmetrically. Their exchange must consistently map the
radial factors, angular orders, Bessel orders, Fourier mode, and target-theta
axes while keeping leg 1 fixed. This is a restricted leg-2/leg-3 symmetry, not
a full three-leg canonicalization.

SPT is maintained as a three-dimensional physical model, not as a native-2D
model parameterized by an angular power spectrum. At each LOS redshift,
`k_i = ell_i / chi(z)`, so the ratios in the SPT kernel obey
`k_i / k_j = ell_i / ell_j`. The finite Slepian decomposition can therefore
be attached to the 3D terms and carried into their projected 2D
representations without defining a separate 2D SPT model.

`SPTMatterBispectrum3D` exposes all seven tree-level harmonic terms for each
of pairs `12` and `31`. The `m=+/-1` coefficient is stored as two separable
radial products and must not be collapsed into a nonseparable term.
`SPTGalaxyBispectrum3D` additionally exposes the
quadratic-bias `m=0` term and the tidal-bias `m=0,+/-2` terms for those pairs.
Every such additive term carries both `NumericExpression3D` and
`SlepianExpression3D`. The matter pair-23 tree contribution is split into
`p=0,2,4` terms carrying numeric and semi-analytic representations. Galaxy
pair-23 terms remain numeric-only. Splitting the former pair aggregate is
required because one additive term may expose only one representation of each
concrete type, and because hybrid route selection operates term by term.

Odd angular harmonics on nonconstant SPT legs do not by themselves imply an
odd-order constant-leg kernel. Odd-order constant-leg regular support is a
separate, more general case. When it occurs, its contact coefficient is zero
and both open regions `x < theta` and `x > theta` must be integrated; neither
branch may be dropped.

## Spin scope

The Slepian radial transforms use the effective spin
`sigma_i = epsilon_i * spin_i`. For an allowed opening-angle mode `k`, define
`m = Sigma/2 + k`, `n = Sigma/2 - k`, and `Sigma = sigma1 + sigma2 + sigma3`.
The single radial order is `n1 + sigma1`. For constant leg 3 the double and
constant order pairs are `(n2 + sigma2 - m, m)` and
`(n3 + sigma3 - n, n)`; constant leg 2 exchanges the latter two physical
legs and transposes the result back to the public theta-axis order. The final
radial result carries the phase `(-i)**Sigma`.

This construction applies equally when `sigma1` is nonzero. It is the direct
Slepian route obtained after inserting the Fourier-triangle closure delta and
performing the three angular integrals separately; it does not use the
numeric route's intermediate `G_Lk` coupling. All even integer spin triples
and every representative epsilon are supported. More generally, integer spin
triples produce integer or half-integer `k` according to the parity of
`Sigma`, provided `m` and `n` are integers.

For a constant radial leg, comparisons with finite-band numeric or brute-force
routes require care. Slepian uses the exact distributional Bessel closure,
whereas a finite ell interval replaces that delta distribution by a broadened
kernel. Reference-vertex spin can amplify this truncation residual. Validate
the Slepian expression itself against direct radial quadrature before using a
finite-band route as an accuracy reference.

A term with constant leg 1, or any other structure unsupported by the Slepian
formalism, must not be forced into a Slepian representation. Under a future
term-wise hybrid policy it falls back first to a semi-analytic representation
and then to the numeric representation when no semi-analytic representation
is available. An explicitly requested pure Slepian route must report the term
as unsupported rather than silently applying that fallback.

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

`F_ab` and its low-rank factors depend on the angular calculation grid,
Bessel orders, Mellin exponents, theta geometry, and numerical controls. They
do not depend on the bispectrum model, cosmology, projector weights, or LOS
nodes. Replacing the source model or changing its physical parameters must
recompute the node-dependent Mellin coefficients but preserve compatible
`F_ab` resources. A different LOS grid changes the coefficient samples and
quadrature, not the universal regular matrix.

`SlepianCalculator` therefore separates source-dependent FFTLog power sums
from structural geometries, Weber resources, constant-leg kernels, and regular
matrices. A bispectrum state-token change or `ThreePCF.set_bispectrum()` clears
the power sums and result tables while retaining the calculator and compatible
structural caches. A theta-grid change discards the calculator because it
changes the transform geometry.

## Result boundary

The Slepian calculator owns FFTLog power sums, Weber geometries, primitive
kernels, interpolation tables, and future `F_ab` resources. It constructs a
route-independent `ZetaKTable` directly on the requested theta bins. Final
opening-angle and shear-projection assembly is shared with other routes.

The calculator must preserve a direct benchmark path, report unsupported
representations explicitly, and never hide a route fallback that changes the
meaning of the requested term or epsilon tuple.

## Analytic validation hierarchy

An analytic native-2D toy is the primary normalization and convergence
reference; brute-force integration is a later independent integration test,
not the definition of the exact result. The first maintained toy is

```text
B = A ell1^2 exp(-a ell1^2) ell2^2 exp(-b ell2^2),
```

with constant leg 3, zero angular orders, and scalar spin. In the fastnc
theta-axis convention its exact result is the product of a single Gaussian
Hankel transform on `theta2` and a Gaussian double-Bessel transform on
`(theta2, theta1)`, including the route's `(2 pi)^-2` normalization. This test
must cover every retained `ZetaK` mode and their finite Fourier sum in `Zeta`.
Although the bispectrum has no explicit opening-angle dependence, its
double-Bessel transform generally has nonzero modes: the scalar assembly is
`sum_k ZetaK_k exp(i k phi)`. The test must therefore verify both the mode
amplitudes and the resulting phi dependence, and scan the Mellin grid so
increasing resolution cannot silently reduce accuracy.

Subsequent analytic references should add angular harmonics and then a regular
constant-leg kernel. Spinful brute comparisons remain useful only after these
component-level analytic contracts pass.
