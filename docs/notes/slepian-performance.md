# Slepian route: radial transforms and performance decisions

Status: Development note; non-normative.

Authoritative design: `docs/design/slepian.md`.

Purpose: Preserve derivations, experiments, benchmark observations, and the
reasoning behind the staged implementation. If this note conflicts with the
authoritative design, the design document wins.

This note records which parts of the Slepian/Weber investigation are current
design decisions and which parts remain experiments. It is the stable entry
point for `dev/slepian`; notebook output alone is not the current
implementation contract.

## Mathematical source

The derivation is in `cross3pcf (22).pdf` and `cross3pcf (23).pdf`, especially
"Alternative approach; Slepian and Eisenstein" and Appendix B, "Efficient
evaluation of Slepian integrals for SPT bispectrum and bihalofit". These PDFs
are research notes outside this repository. `2104.10169v2.pdf` was examined as
an alternative integral-to-sum method, but was not adopted.

For one FFTLog power, the expensive double radial transform has the form

```text
I_nu(x, theta) = integral d ell ell^(nu+1) J_p(ell x) J_q(ell theta).
```

The Weber-Schafheitlin expression can be organized as

```text
I_nu(x, theta) = s^(-nu-2) W_nu,p,q(r),
s = max(x, theta),  r = min(x, theta) / max(x, theta).
```

The special-function part is therefore one-dimensional in `r`. The orientation
`x <= theta` or `x > theta` determines which Bessel order is the small-argument
order and must be retained when the two orders differ.

## Current production decisions

These decisions are implemented in `fastnc/threepcf/slepian.py`.

1. Evaluate directly on the requested `theta` bins. The tuned numeric FFTLog
   grid is not required by the Slepian route.
2. Construct one `WeberGeometry` per coordinate pair. It stores `scale`,
   `ratio`, and the diagonal mask; orientation follows from the coordinates.
3. Evaluate the unit Weber function only for unique log-ratios. Reconstruct the
   matrix by scattering those values and multiplying by the scale power.
4. Cache primitive Weber matrices, geometries, and FFTLog power sums in
   `SlepianCalculator`. A primitive key includes geometry, Mellin exponent,
   Bessel orders, tolerance, and diagonal policy.
5. Keep the factor-dependent finite-band diagonal correction outside the
   primitive cache.
6. Clear calculator state when target geometry or relevant `ThreePCF` state
   changes.

On the development machine, target-grid and unique-ratio evaluation reduced
the native-2D toy calculation from order 10 seconds on the full tuned grid to
order 0.1 seconds for ten target bins. This is diagnostic, not an API timing
guarantee. A pointwise Weber test and a numeric-route comparison protect the
result numerically.

## Constant-leg distribution and unequal orders

Suppose the third separable radial factor is constant. After the angular
algebra, its two Bessel orders will be denoted by the signed integers `p` and
`q`. The kernel entering the remaining radial calculation is

```text
C_pq(x, theta) = integral_0^infinity d ell ell
                 J_p(ell x) J_q(ell theta).
```

Here and below, `x` is the variable integrated by the later radial assembly and
`theta` is the requested output separation. Earlier experiments sometimes
called these variables `y` and `rho`; those aliases are retired because they
obscure which coordinate is integrated and which is external.

This integral is a distribution, not an ordinary function at `x = theta`.
Its useful decomposition is

```text
C_pq(x, theta)
  = c_delta(p, q) delta(x - theta) / x
  + Theta(theta - x) C_pq^<(x, theta)
  + Theta(x - theta) C_pq^>(x, theta).
```

The contact coefficient follows from the nonoscillatory part of the common
large-argument Bessel asymptotics. Since

```text
J_p(z) approximately sqrt(2 / (pi z))
                 cos(z - pi p / 2 - pi / 4),
```

the phase difference of the product is `pi(p-q)/2`, giving

```text
c_delta(p, q) = cos[pi (p - q) / 2].
```

For integer orders this coefficient must be computed exactly from
`(p-q) mod 4`: it is `+1` for a difference divisible by four, `-1` for a
difference congruent to two modulo four, and zero for an odd difference. This
also incorporates `J_-n(z) = (-1)^n J_n(z)` without separately canonicalizing
the two orders. Thus a contact term can survive for unequal orders of the same
parity; equality of canonical orders is sufficient but not necessary.

For `p = q`, Bessel closure gives only the contact distribution and both
regular pieces vanish. The same is true, up to the exact sign, for the signed
equal-canonical-order cases already accepted by the current implementation.
For general unequal orders, the Weber-Schafheitlin continuation gives ordinary
functions away from the diagonal. The formula differs between `x < theta` and
`x > theta` because the small-argument Bessel order changes with orientation.
These are the two Heaviside-supported regular pieces above. Their exact
normalization and the order pairs for which either side vanishes must be
validated against the source derivation and high-precision integration before
being promoted to the normative formula.

For a constant third leg, `SlepianCalculator` uses
`p = n3 + sigma3 - n` and `q = n`; for a constant second leg it uses the
symmetric pair `p = n2 + sigma2 - m` and `q = m` and restores the public
theta-axis ordering after the internal contraction. The single transform has
order `n1 + sigma1`, while the overall phase is `(-i)**Sigma`. This directly
supports nonzero reference-vertex spin and has been checked against direct
radial quadrature. The `G_Lk` angular coupling belongs to the generic numeric
route; it is not an additional factor in this direct separable construction.
The earlier `_contact_sign(p, q)` implementation returned a sign only when the
canonical orders were equal and was limited to the contact-only subset. The
current implementation replaces that check with an explicit constant-leg
decomposition and direct regular quadrature for one-sided kernels. A constant
first leg remains intentionally unsupported because eliminating it would lose
the opening-angle multipole retained by `ZetaK`.

The implemented low-level decomposition object is independent of
`SlepianCalculator`. Its mathematical payload is:

```text
contact_coefficient : exact integer in {-1, 0, +1}
regular_less        : C_pq^<(x, theta), supported only on x < theta
regular_greater     : C_pq^>(x, theta), supported only on x > theta
```

No finite-band diagonal replacement belongs in this object. It neither
contracts Mellin coefficients nor performs the outer `x` integration. The
calculator consumes the contact piece separately and contracts the regular
pieces with `regular_method="quadrature"` using

```text
integral dx x R_single(x) R_double(x, theta1) C_reg(x, theta2).
```

The original benchmark uses a logarithmic integration grid extending beyond
the requested theta range by `regular_x_padding` and explicitly inserts every
requested theta point. Because the regular kernel is represented as zero at
the distributional boundary, trapezoidal panels adjacent to that inserted
point introduce a grid-dependent notch. It is retained as
`regular_quadrature="legacy_log"`, not used as the default.

The replacement uses open Gauss-Legendre nodes in the ratio coordinate. For
each constant-leg target `theta_*`, the lower branch sets `x=theta_* r` and
the upper branch sets `x=theta_*/r`, with `r` in
`[regular_ratio_min,1]`. Including the radial measure gives

```text
I_< = integral dr theta_*^2 r
      R_1(theta_* r) R_2(theta_* r, theta_1) C^<(theta_* r, theta_*),

I_> = integral dr theta_*^2 r^(-3)
      R_1(theta_*/r) R_2(theta_*/r, theta_1) C^>(theta_*/r, theta_*).
```

No quadrature node lies at `r=1`. A spin `(2,4,6)` toy scan with 32 Mellin
samples and three theta bins gave total norms `1.20069e5`, `1.18923e5`, and
`1.19043e5` for 64, 128, and 256 ratio nodes. The last change is about 0.1%;
this is a numerical observation for that toy, not a universal error bound.

The same branch-separated rule now constructs the reusable Mellin matrix
directly. For double Mellin index `a`, single Mellin index `b`, and target pair
`(theta_1, theta_*)`, each branch accumulates

```text
F_ab(theta_1, theta_*) = sum_i w_i J_branch(r_i)
    S_b(x_branch(r_i)) D_a(x_branch(r_i), theta_1)
    C_branch(x_branch(r_i), theta_*),
```

where `J_< = theta_*^2 r` and `J_> = theta_*^2 r^(-3)`. Contracting this
matrix with the two FFTLog coefficient vectors reproduces direct ratio
quadrature. `full_matrix` and `low_rank` therefore no longer inherit the
artificial boundary notch of the legacy global-`x` trapezoid. The legacy
matrix remains available by selecting `regular_quadrature="legacy_log"`.

Increasing the open quadrature order also drives nodes close to diagonals of
the nonconstant double radial transform. The hypergeometric Weber series is
ill-conditioned there. The direct finite-band Bessel correction therefore
applies throughout `ratio >= weber_brute_min_ratio`, rather than only at exact
equality. A zero reciprocal-gamma prefactor is detected before evaluating the
hypergeometric function; this avoids evaluating a divergent continuation on a
regular branch whose amplitude is exactly zero.

Tests protect the unchanged
equal-order result, signed contact parity, one-sided and two-sided support, the
analytic `(p,q)=(0,2)` regular kernel, and the axis ordering and measure of the
quadrature contraction.

The kernel primitive can expose odd canonical-order differences, for which
both orientations are nonzero, but the first calculator quadrature deliberately
rejects them. A toy convergence scan grew rather than converged when samples
approached `x=theta`; an ordinary trapezoidal rule does not define this
singular diagonal limit. Production quadrature is therefore restricted to a
positive even canonical-order difference and one-sided support. Odd differences
require a separate distributional or principal-value derivation before they
can be treated as an ordinary `x` integral.

## Low-rank experiment

The proposed low-rank approximation acts on a reusable matrix over pairs of
Mellin indices. For each radial geometry index `t`, schematically,

```text
F_t[a, b] approximately equals
    sum_(r=1)^R U_t[a, r] S_t[r] Vh_t[r, b].
```

Cosmology-dependent Mellin coefficients can then be contracted with rank-`R`
factors rather than the full `a,b` matrix. The constant-leg SPT experiments
found rapid singular-value decay. Rank 6 typically gave pair-level errors near
`1e-5` to `1e-4`; recorded end-to-end term errors varied by term and reached a
few `1e-3`. One-time matrix/SVD preparation took about 14 seconds, while later
contractions were millisecond-scale.

Low-rank compression is now implemented for the supported constant-leg regular
kernel. The earlier experiment predates `SlepianRepresentation2D` and
`SlepianCalculator`; its numerical observations remain historical context.
The production implementation provides explicit-rank and matrix-tolerance
selection, compressed-factor caching without retaining the full matrix, and
comparison against both the uncompressed matrix and direct `x` quadrature.

The matrix tolerance is not a final-`ZetaK` error contract. Current toy terms
show that coefficient weighting can amplify discarded singular directions, so
rank scans must continue to report both errors. Non-contact and projected terms
remain outside the current implementation scope.

The implementation order is fixed as follows. Unequal Bessel orders on a
constant leg produce a regular Heaviside-supported contribution in addition to
any contact delta contribution. Implement that regular contribution first with
the exact, uncompressed matrix

```text
F_ab(theta) = integral dx K_a^double(x, theta)
                          x^(-mu_b-1) C_reg(x).
```

Cache `F_ab` as geometry/order/exponent state and validate its full contraction
against direct radial quadrature. Low-rank compression is a later optimization
of this established matrix contract; it must not be introduced while the
regular kernel or its normalization is still being validated.

## Interpolation experiment

`double_radial_interpolation_usage.ipynb` tested sparse interpolation of the
assembled double-radial result. Errors were strongly mode- and signal-dependent
and cubic interpolation was not uniformly better. It is inconclusive rather
than an accepted optimization. Interpolating the one-dimensional unit Weber
function remains possible, but current code evaluates unique ratios directly
and caches them.

## Staged implementation and benchmark policy

Introduce the remaining optimizations one at a time. Each accepted method must
remain available as the reference for the next method; an optimization must
not replace or delete its benchmark path. Use explicit method-valued options,
not interacting boolean flags:

```python
SlepianConfig(
    weber_method="direct",        # "direct" | "interpolated"
    regular_method="quadrature",  # "quadrature" | "full_matrix" | "low_rank"
)
```

The two choices are independent. This allows, for example, a full `F_ab`
calculation with direct Weber evaluation, so an interpolation error cannot be
mistaken for a matrix-assembly error. Adding another algorithm later extends a
method set instead of creating contradictory combinations such as multiple
`use_*` flags.

The required development order is:

1. **One-dimensional Weber evaluator.** Preserve `weber_method="direct"` as
   the primitive reference. Add `"interpolated"` for a table of
   `W_nu,p,q(r)`, tabulated in `t=log(-log(r))` to resolve the rapid variation
   near `r -> 1`. Ratios at or above `weber_interpolation_max_ratio` are
   evaluated directly; the default boundary is `0.8`. If the request contains
   no more off-diagonal ratios than table nodes, direct evaluation is cheaper
   and is used automatically. Validate complex values for each Mellin exponent
   and Bessel-order pair before using the interpolator in a radial transform.
   The current direct `hyp2f1` power series is not a reliable reference near
   `r -> 1` at large imaginary Mellin index; its observed artificial jaggedness
   and planned connection-formula repair are recorded in
   `docs/todo.md`. Until that repair is complete, neither direct
   fallback above `0.8` nor an interpolation table sampled from it should be
   interpreted as a high-accuracy result over the full FFTLog index range.
2. **Constant-leg kernel decomposition.** Implemented. Replace upper-level equality checks
   on Bessel orders with a kernel primitive that exposes contact coefficient,
   regular contribution, and Heaviside support. The existing equal-order
   contact result remains unchanged and is the regression benchmark.
3. **Direct regular quadrature.** Implemented. For `regular_method="quadrature"`, first
   reconstruct the single and double radial transforms from their Mellin sums,
   then perform the remaining `x` integral. This deliberately slower path is
   the independent reference for `F_ab` assembly.
4. **Full Mellin matrix.** Implemented. For `regular_method="full_matrix"`, exchange the
   Mellin sums and the regular radial integral, construct the exact uncompressed
   `F_ab(theta1, theta2)`, cache it by geometry/order/exponent state, and perform
   the full contraction. Final `ZetaK` is tested directly against `"quadrature"`.
5. **Low-rank matrix.** Implemented. `regular_method="low_rank"` caches the
   truncated SVD factors and leaves `"full_matrix"` permanently selectable.
   Validation reports both matrix reconstruction and final `ZetaK`; rank is not
   accepted solely from a Frobenius-norm error.

The permanent reference ladders are therefore

```text
direct Weber -> interpolated Weber

direct x quadrature -> full F_ab contraction -> low-rank F_ab contraction
```

Every stage needs focused unit tests and a Japanese development notebook that
shows the new option together with its immediate reference. Benchmark caches
for different methods independently so a result produced by one evaluator is
never returned under another evaluator's cache key.

## Spin brute-force validation after ratio quadrature

The spin `(2,4,6)` validation was rerun after introducing branch-separated
ratio quadrature, using `regular_n_ratio=128`, `regular_ratio_min=0.05`,
`weber_brute_min_ratio=0.95`, `n_ell=64`, and `kmax=8`. Relative maximum
residuals against the fine brute-force result for the four representative
epsilon components changed as follows:

```text
epsilon          legacy regular integral    ratio Gauss integral
(+,+,+)                  0.903                      0.540
(-,+,+)                  0.972                      0.289
(+,-,+)                  0.751                      0.348
(+,+,-)                  0.147                      0.037
```

The corresponding relative L2 residuals for ratio Gauss were `0.226`,
`0.141`, `0.156`, and `0.035`. This is a substantial improvement but not a
validation of components zero through two. The brute coarse/fine relative
maximum differences themselves were `0.129`, `0.066`, `0.139`, and `0.078`.
Before changing the spin algebra or regular formula again, hold the fine brute
result fixed and scan one Slepian control at a time: `kmax`, Mellin `n_ell`,
`regular_n_ratio`, and direct versus interpolated Weber evaluation. The local
archive is `dev/slepian/slepian_spin_brute_validation_ratio_gauss.npz`.

Holding that brute result and all other Slepian controls fixed, the worst
component `epsilon=(+,+,+)` gave relative maximum residuals `0.540`, `0.380`,
`0.135`, `0.103`, and `0.101` at `kmax=8`, `12`, `16`, `20`, and `24`.
The Slepian result changed by `0.0189` in relative maximum norm and `0.0081`
in relative L2 norm from `kmax=20` to `24`. The mode sum is therefore close
to convergence for this toy by `kmax=24`; the remaining relative L2 residual
against brute fine is `0.0594`. Subsequent convergence tests should use
`kmax=24` and vary Mellin `n_ell` next rather than attributing the remaining
difference to the regular kernel formula.

With `kmax=24` fixed, increasing the Mellin grid gave the following residuals
for the same component:

```text
n_ell       brute max residual       brute L2 residual
  64              0.1007                   0.0594
  96              0.0454                   0.0518
 128              0.0447                   0.0515
 160              0.0453                   0.0505
```

Successive Slepian L2 changes were `0.0410`, `0.0192`, and `0.0182`, while
the maximum changes were `0.131`, `0.0623`, and `0.0730`. The nonmonotonic
maximum at 160 occurs at the largest theta pair and the phi bin nearest pi;
the brute value lies between the 128- and 160-point predictions there. Thus
the global result is approaching a stable roughly five-percent L2 difference,
but local angular extrema are not cleanly Mellin-converged. Increasing `n_ell`
also introduces larger imaginary Mellin indices, precisely where the current
Weber evaluator is known to become unreliable. The next isolation test should
compare direct and interpolated Weber evaluation at a fixed affordable Mellin
grid before increasing `n_ell` further.

## Analytic Gaussian contact reference

A separate validation family avoids treating finite-band brute integration as
the exact answer. For scalar spin and constant leg 3, define

```text
B(ell1, ell2, ell3)
  = A ell1^2 exp(-a ell1^2) ell2^2 exp(-b ell2^2).
```

In the fastnc coordinate convention, `theta2` is the radius of the single
leg-1 transform and `(theta2, theta1)` are the two radii of the leg-2 double
transform. Define

```text
S_a(t) = exp[-t^2/(4a)] [1 - t^2/(4a)] / (2a^2),

D_b(r,s) = -d/db {
    exp[-(r^2+s^2)/(4b)] I_0(rs/(2b)) / (2b)
}.
```

Writing `u=rs/(2b)` gives

```text
D_b(r,s) = exp[-(r^2+s^2)/(4b)] I_0(u) / (2b)
  * [1/b - (r^2+s^2)/(4b^2)
     + rs I_1(u)/(2b^2 I_0(u))].
```

The exact 3PCF is therefore

```text
zeta(theta1, theta2)
  = A S_a(theta2) D_b(theta2, theta1) / (2 pi)^2,
```

independent of phi. With `ell` in `[1e-3,500]`, taper fraction `0.1`, and the
theta range `[0.003,0.02]`, the maximum error was `3.9e-3`, `5.4e-6`, and
`2.3e-2` for `n_ell=64`, `128`, and `256`. The excellent 128-point agreement
fixes the normalization and axis convention without brute force. The loss of
accuracy at 256 is direct end-to-end evidence that adding larger imaginary
Mellin indices can expose Weber instability. The local script and notebook are
`dev/slepian/slepian_analytic_gaussian_contact.py` and
`dev/slepian/slepian_analytic_gaussian_contact.ipynb`.

## Canonical validation path

Use `dev/slepian-old/threepcf_slepian_toy.ipynb` for the earlier end-to-end implementation. It
defines one native-2D term with numeric and Slepian representations, compares
the `ZetaK` modes, and reports target-grid and cache behavior. Files under
`dev/slepian-old` contains historical development evidence, not the public
route API.
