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

Low-rank compression is **not in the current production route**. The experiment
predates `SlepianRepresentation2D` and `SlepianCalculator` and covers only the
constant-leg regular SPT case. Promotion requires:

- an error contract on final `ZetaK`, not only matrix norms;
- rank selection across terms, modes, and geometries;
- cache invalidation and memory accounting in the current calculator;
- benchmarks against the already-fast unique-ratio implementation;
- support or an explicit rejection policy for non-contact and projected terms.

The implementation order is fixed as follows. Unequal Bessel orders on a
constant leg produce a regular Heaviside-supported contribution in addition to
any contact delta contribution. Implement that regular contribution first with
the exact, uncompressed matrix

```text
F_ab(rho) = integral dy K_a^double(y, rho)
                        y^(-mu_b-1) C_reg(y).
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
2. **Constant-leg kernel decomposition.** Replace upper-level equality checks
   on Bessel orders with a kernel primitive that exposes contact coefficient,
   regular contribution, and Heaviside support. The existing equal-order
   contact result remains unchanged and is the regression benchmark.
3. **Direct regular quadrature.** For `regular_method="quadrature"`, first
   reconstruct the single and double radial transforms from their Mellin sums,
   then perform the remaining `x` integral. This deliberately slower path is
   the independent reference for `F_ab` assembly.
4. **Full Mellin matrix.** For `regular_method="full_matrix"`, exchange the
   Mellin sums and the regular radial integral, construct the exact uncompressed
   `F_ab(rho)`, cache it by geometry/order/exponent state, and perform the full
   contraction. Validate final `ZetaK` against `"quadrature"` before proceeding.
5. **Low-rank matrix.** Only after the full matrix contract is established,
   add `regular_method="low_rank"`. Keep `"full_matrix"` permanently
   selectable. Validate both matrix reconstruction and final `ZetaK`; do not
   select rank solely from a Frobenius-norm error.

The permanent reference ladders are therefore

```text
direct Weber -> interpolated Weber

direct x quadrature -> full F_ab contraction -> low-rank F_ab contraction
```

Every stage needs focused unit tests and a Japanese development notebook that
shows the new option together with its immediate reference. Benchmark caches
for different methods independently so a result produced by one evaluator is
never returned under another evaluator's cache key.

## Canonical validation path

Use `dev/slepian/threepcf_slepian_toy.ipynb` for current code. It
defines one native-2D term with numeric and Slepian representations, compares
the `ZetaK` modes, and reports target-grid and cache behavior. Files under
`dev/slepian` contains historical development evidence, not the public
route API.
