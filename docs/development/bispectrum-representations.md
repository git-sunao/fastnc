# Bispectrum representation refactor

## Status and goal

This document records the design agreed before implementing the Slepian and
Weber-Schafheitlin route. The implementation branch is
`codex/bispectrum-representations`, based on the `v2` branch.

The immediate goal is to refactor the bispectrum package so that numerical,
Slepian, and semi-analytic calculations can consume explicit mathematical
representations of additive bispectrum terms. The Slepian and Weber kernels
themselves are not part of this first refactor.

The longer-term public workflow is expected to resemble

```python
b3d = BiHalofitBispectrum3D(config_b3d)
los = LineOfSightProjector(config_los)
threepcf = ThreePCF.from_b3d_los(b3d, los, config_3pcf)
zeta = threepcf.compute()
```

The same 3PCF algorithms must also be usable for a 3D bispectrum at one fixed
redshift and for a directly supplied 2D angular bispectrum.

## Fixed decisions

1. A 3D bispectrum is a function `B3D(k1, k2, k3, z)`.
2. A 2D angular bispectrum is a function `B2D(ell1, ell2, ell3)`.
3. No generic `q` coordinate and no public redshift-slice object are introduced.
4. All 3PCF route calculators operate in angular Fourier coordinates `ell` and
   return results on angular real-space coordinates `theta`.
5. At fixed redshift, a 3D expression is presented to an angular calculator by
   the substitution

   ```text
   k_i = ell_i / chi(z),  equivalently ell_i = chi(z) k_i.
   ```

6. A bispectrum is an additive collection of numerically stable terms.
7. One physical term may provide several alternative mathematical
   representations. A planner selects exactly one representation for a given
   requested calculation.
8. Terms do not own route caches. A route calculator owns or is given the
   caches it needs during evaluation.
9. Grid classes store coordinates, values, labels, and provenance. They do not
   implement expensive numerical calculations.
10. Low-level numerical kernels consume arrays, scalars, and callables. They do
    not depend on Grid, bispectrum-model, or ThreePCF APIs.

## Mathematical representations

The word representation means all information required to execute a later
calculation unambiguously. Representations describe mathematics; they do not
store calculated grids or route caches.

### Numeric expression

A 3D numeric expression evaluates

```text
B_a(k1, k2, k3, z; state).
```

A 2D numeric expression evaluates

```text
B_a(ell1, ell2, ell3; state).
```

The numeric representation is the universal fallback and the reference used
to test optimized representations. Models should provide it for every term
when a direct stable evaluation exists.

### Slepian expression

A 3D Slepian expression describes a term of the form

```text
C_a(z) product_i [f_i^a(k_i, z) exp(i n_i^a phi_i)],
sum_i n_i^a = 0.
```

It contains the coefficient, the three radial functions, their powers or
Mellin shifts, and their angular phases. At fixed `z` it is angularized as

```text
f_i^a(k_i, z) -> f_i^a(ell_i / chi(z), z).
```

A native 2D Slepian expression contains the corresponding functions of
`ell_i` directly. The Slepian calculator uses this representation to obtain
Mellin coefficients, Weber-Schafheitlin kernels, and `ZetaK` without passing
through angular bispectrum multipoles or `HKernel`.

### Semi-analytic expression

A 3D semi-analytic expression describes

```text
B_a(k1, k2, k3, z)
  = U(k2/k, k3/k) V(k2, k3, z) (k1/k)^p W(k1, z),
k = sqrt(k2^2 + k3^2).
```

At fixed `z`, it is angularized by

```text
U(k2/k, k3/k) -> U(ell2/ell, ell3/ell)
V(k2, k3, z)  -> V(ell2/chi, ell3/chi, z)
W(k1, z)      -> W(ell1/chi, z),
ell = sqrt(ell2^2 + ell3^2).
```

The scale ratios in `U` and `(k1/k)^p` are unchanged because the factors of
`chi` cancel. The semi-analytic calculator produces angular bispectrum
multipoles. The later `B_L -> H_k -> ZetaK` stages are shared with the numeric
route.

### Direct multipole expression

Some terms have a finite or otherwise explicitly known set of angular Fourier
multipoles. Such a term may provide a direct multipole representation and skip
numeric angular decomposition. This is an optimized producer of the same
bispectrum-multipole product used by the numeric and semi-analytic paths.

## Terms and composition

`BispectrumTerm3D` and `BispectrumTerm2D` are additive physical contributions.
A term is the smallest contribution that can be evaluated independently and
stably, not necessarily the smallest symbolic monomial.

This distinction matters when separately divergent or ill-conditioned pieces
cancel. For example, a regularized combination of the SPT `31` and `12`
`p=-2` pieces should be one term if evaluating or routing the pieces
independently would lose precision.

A term has a stable name and one or more representations:

```python
BispectrumTerm3D(
    name="tree:F2:31:p=0",
    representations=(
        NumericExpression3D(...),
        SlepianExpression3D(...),
        SemiAnalyticExpression3D(...),
    ),
)
```

Terms may be scaled and added. Scaling should be immutable:

```python
scaled = term.scaled_by(weight)
combined = term_a + term_b
```

The coefficient may be a constant or a state/redshift-dependent callable.
Composition is flattened to weighted leaf terms before planning:

```python
for weighted_term in bispectrum.iter_terms():
    ...
```

A complete bispectrum and an individual term may implement a common component
protocol, but a composite bispectrum and a leaf term do not need to be the same
concrete class.

## State and invalidation

The 3D bispectrum owns physical state shared by its terms, including cosmology,
power spectra, biases, and model parameters. Shared state should not be copied
into every term.

The preferred update is immutable:

```python
updated = b3d.with_state(cosmology=new_cosmology, b1=1.8)
```

If a mutable compatibility method is retained, it must replace the internal
state and increment a revision token. Model-dependent cache keys include that
token so stale values are not reused after an update.

Route-independent universal kernels and model-dependent values have different
invalidation rules:

```text
state independent:
    multipole coupling matrices
    Weber-Schafheitlin kernels
    semi-analytic angular kernels

state dependent:
    power-spectrum values
    Mellin coefficients
    LOS-integrated coefficients
    term amplitudes
```

Terms do not store these caches. Initially, each calculator may own an in-memory
cache. Cache objects should be injectable so that Slepian and semi-analytic
calculators can later share Mellin coefficients if profiling shows that this is
valuable.

## Angularization of 3D expressions

Angularization is an adapter operation, not a 3PCF calculator operation. For a
numeric expression at fixed redshift,

```python
def evaluate_angularized(ell1, ell2, ell3):
    return expression.evaluate(
        ell1 / chi,
        ell2 / chi,
        ell3 / chi,
        z,
        state=state,
    )
```

No persistent fixed-redshift bispectrum object is required. The adapter may
return a lightweight resolved representation or pass bound callables and
arrays directly to the calculator.

For a Mellin expansion

```text
f(k, z) = sum_m c_m(z) k^(nu_m),
```

the angularized form is

```text
f(ell/chi, z)
  = sum_m [c_m(z) chi^(-nu_m)] ell^(nu_m).
```

Thus `chi^(-nu_m)` belongs in the angularized or LOS-integrated coefficient,
not in the universal Weber kernel.

## Route algorithms

### Numeric route

```text
B3D numeric terms
    -> numeric LOS projection or fixed-z angularization
    -> B2D numeric terms
    -> numeric B2D multipoles
    -> multipole coupling H_k
    -> double Hankel transform ZetaK
    -> opening-angle assembly Zeta
```

For a native `B2D`, the projection/Angularization step is absent. This route is
the universal fallback.

### Slepian route

```text
B3D Slepian term
    -> fixed-z angularization or coefficient-level LOS integration
    -> angular Slepian expression
    -> Mellin coefficients
    -> Weber-Schafheitlin radial kernels
    -> direct ZetaK contribution
    -> opening-angle assembly Zeta
```

A native `B2D` may provide the same angular Slepian expression directly. The
calculator is identical after angularization.

The first LOS implementation should favor a transparent reference algorithm,
such as evaluating angularized contributions at LOS nodes and integrating
them. Coefficient-level LOS integration is an optimization to add after the
reference result is established.

### Semi-analytic route

```text
B3D semi-analytic term
    -> fixed-z angularization or coefficient-level LOS integration
    -> angular U, V, W, p expression
    -> Mellin expansion of W
    -> precomputed angular kernel K_L^(nu+p)
    -> B2D multipole contribution
    -> H_k
    -> ZetaK
    -> Zeta
```

A native `B2D` may provide an angular semi-analytic expression directly.

## Projection and route are independent choices

The calculation route and LOS strategy are separate axes:

```text
route:
    numeric
    Slepian
    semi-analytic

LOS strategy:
    project the numeric bispectrum first
    evaluate a route at each LOS node and then integrate
    integrate Mellin or other coefficients first
```

The optimized LOS strategies must reproduce the transparent reference
strategy within configured numerical tolerance.

## Planned public API

Direct 2D angular bispectrum:

```python
threepcf = ThreePCF.from_b2d(b2d, config_3pcf)
zeta = threepcf.compute()
```

Fixed-redshift 3D bispectrum:

```python
threepcf = ThreePCF.from_b3d(
    b3d,
    z=z,
    chi=chi,
    config=config_3pcf,
)
zeta_at_z = threepcf.compute()
```

The fixed-redshift path uses the raw angularized bispectrum and does not
silently apply an LOS weight. If `chi` is omitted in a future convenience API,
the cosmology and distance convention used to obtain it must be explicit in
the model state or configuration.

LOS-projected 3PCF:

```python
threepcf = ThreePCF.from_b3d_los(b3d, los, config_3pcf)
zeta = threepcf.compute()
```

Intermediate products should be available on request:

```python
b2d = threepcf.compute_b2d()
b2dm = threepcf.compute_b2d_multipoles()
plan = threepcf.plan()
```

Requesting an intermediate product is independent of the route selected for
the final 3PCF. For example, a Slepian final calculation may still compute
numeric bispectrum multipoles for validation.

## Calculator and storage boundaries

Route calculators operate on angular representations and `ell` arrays. They do
not receive a `Bispectrum3D`, `Bispectrum2D`, Grid, or ThreePCF instance.

Pure numerical kernels have signatures conceptually similar to

```python
decompose_angular_multipoles(values, delta_beta, modes, basis) -> arrays
calculate_hkernel(multipoles, coupling_values, weights) -> arrays
transform_hkernel(ell, hkernel, bessel_orders, config) -> arrays
evaluate_weber_kernel(exponent, orders, ratios, config) -> arrays
contract_slepian(coefficients, kernels, geometry) -> arrays
```

Workflow or assembly code extracts representations, performs angularization,
queries caches, calls numerical kernels, and stores results in Grid objects.

## Package ownership and the legacy object model

The representation refactor replaces, rather than extends, the object model in
the current `bispectrum/base.py`. The old model makes a generic `evaluate()`
method the center of the package and connects projection, interpolation, and
multipole calculations through methods and inheritance. That dependency
direction does not fit the term/representation/calculator design.

The target domain structure is

```text
fastnc/bispectrum/
    bispectrum.py       # Bispectrum3D and Bispectrum2D term aggregates
    terms.py            # indivisible terms and typed weighted terms
    representations.py  # alternative mathematical descriptions of a term
    support.py           # domain/support value types
    models/              # SPT, BiHalofit, one-halo, and later models
```

`Bispectrum3D` and `Bispectrum2D` are typed aggregates of weighted terms, not
abstract evaluator wrappers. A bispectrum is not itself a term. Addition of
bispectra concatenates and flattens their term collections; a term remains the
smallest unit assigned to one route. The 3D and 2D aggregate and weighted-term
types remain distinct instead of using runtime dimensionality checks.

The target model does not retain the following behavior from `base.py`:

- automatic wrapping of subclass `evaluate()` methods through
  `__init_subclass__`;
- mutable persistent `default_kwargs` hidden inside domain objects;
- route entry points such as `Bispectrum2D.multipole()` on a bispectrum;
- inheritance whose purpose is only to make unrelated derived objects expose
  the same `evaluate()` method.

Physical model classes construct and own term aggregates and shared physical
state. Their numeric evaluation is the sum of the selected numeric
representations. SPT, one-halo, and BiHalofit models must be migrated to this
structure before new Slepian model classes are added.

## Disposition of existing modules

Existing files are not kept or deleted as indivisible units. Pure mathematical
and numerical components are retained, while wrappers that depend on the old
bispectrum object graph are replaced.

- `support.py` remains a domain/value module and can be reused directly.
- `halofit.py` is a standalone physical/numerical implementation and remains
  usable. Moving it under a later `models` or `physics` namespace is optional
  and is not required for the representation refactor.
- The basis definitions and pure transforms in `decompose.py` are reusable.
  Callers coupled to old bispectrum or multipole objects are not part of the
  retained interface.
- Coordinate transforms, grid preparation, packing, and interpolation kernels
  in `interpolate.py` should be extracted as array-based functions. The old
  interpolated bispectrum wrappers should not define the new architecture.
- Angular sampling, quadrature, basis normalization, and array-shape logic in
  `multipole.py` should be extracted into calculators or pure kernels. Lazy
  multipole wrappers, Grid/calculator mixing, and direct dependencies on
  `Bispectrum2D` should be retired after equivalence tests pass.
- The mathematical regulator in `regulator.py` is reusable, but regulator
  selection belongs to the numeric route rather than to a bispectrum model.
- LOS geometry, windows, and quadrature are projection concerns and move out
  of `bispectrum/`. Route-independent projection kernels belong in a projection
  package. Numeric node evaluation, Mellin-coefficient integration, and other
  route-specific assembly belong to their respective route packages.
- Grid classes are passive storage for coordinates, values, labels, and
  provenance. Calculator methods are removed from them rather than migrated.

The intended high-level ownership is therefore

```text
bispectrum/   physical terms, representations, state, and support
projection/   route-independent LOS geometry and integration primitives
routes/       numeric, Slepian, and semi-analytic calculators and assembly
grids/        passive result storage
```

## Migration from the current package

The refactor remains incremental, but preservation of an old wrapper is not a
design objective. Existing implementations remain temporarily available only
until their reusable kernels and numerical behavior have replacement tests.

1. Introduce representation value types and weighted term composition without
   changing existing model behavior.
2. Express a small SPT matter-bispectrum subset as terms and verify that the
   sum of numeric expressions reproduces the existing `evaluate` method.
3. Replace the generic `CompositeBispectrum` prototype with typed 3D and 2D
   weighted terms and aggregates in a new `bispectrum.py`.
4. Move shared state revision, support, numeric evaluation, and immutable
   scaling/composition into the new aggregates; do not build them on the old
   `base.py` inheritance mechanism.
5. Migrate SPT matter and galaxy terms, then one-halo and BiHalofit terms, and
   verify each aggregate against the previous direct evaluation.
6. Introduce corresponding native 2D terms and verify direct 2D numeric
   evaluation.
7. Extract the angular multipole decomposition kernel from
   `BispectrumMultipole2DCalculator` so it consumes arrays/callables rather than
   a concrete `Bispectrum2D` object.
8. Verify that one fixed-z 3D expression and one native 2D expression use the
   same angular multipole kernel.
9. Adapt existing semi-analytic term classes into `SemiAnalyticExpression`
   producers while retaining their proven FFTLog and angular-kernel code.
10. Move LOS code out of `bispectrum/`, split route-independent projection
   primitives from route-specific assembly, and remove `isinstance` dispatch.
11. Refactor the 3PCF workflow and passive Grid storage only after the
   bispectrum representation boundary is stable.
12. Remove `base.py` and obsolete interpolation/multipole wrappers after all
   production imports have migrated and numerical-equivalence tests pass.
13. Implement the Slepian and Weber-Schafheitlin route on top of the new
   expression API.

Interpolation and persistent disk-cache redesign are not part of the first
step unless required to preserve existing behavior.

## Required tests

The refactor is not complete without tests for the following contracts:

1. Numeric term sums reproduce existing SPT and BiHalofit evaluations.
2. Scaling and addition do not mutate source terms.
3. A term can expose multiple representations without double counting.
4. Missing representations produce informative errors.
5. State updates change dependent evaluations and do not reuse stale caches.
6. Universal kernels remain reusable across physical-state updates.
7. Fixed-z angularization uses `k_i = ell_i / chi(z)` on every leg.
8. Native 2D and angularized fixed-z expressions pass through the same numeric
   angular multipole kernel.
9. Numeric and semi-analytic multipoles agree for terms that support both.
10. Numeric and future Slepian `ZetaK` agree for terms that support both.
11. A stable grouped term is not split across routes.
12. LOS optimized strategies agree with the node-by-node reference strategy.

## Open questions

These decisions should be made from concrete term implementations rather than
premature abstraction:

- Exact names and fields of the 3D and 2D representation dataclasses.
- The first public form of state updates: immutable only or immutable plus a
  mutable compatibility wrapper.
- Whether projected 2D terms preserve per-3D-term provenance by default.
- How route selection is configured for individual terms.
- The initial LOS implementation for Slepian terms and the threshold for adding
  coefficient-level integration.
- Which caches should eventually be persistent on disk.

These open questions do not change the fixed boundaries above: 3D expressions
are functions of `(k1, k2, k3, z)`, 2D expressions are functions of
`(ell1, ell2, ell3)`, angular calculators operate only on `ell`, and the
`k_i = ell_i / chi(z)` substitution belongs to the adapter/projection layer.
