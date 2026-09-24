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

Angularization is a projection/assembly operation, not a 3PCF calculator
operation. The public API maps a typed 3D bispectrum to a typed angular
bispectrum:

```python
los = LOSProjector.delta_like(z=z, chi=chi)
b2d = los.project(b3d)
value = b2d.evaluate_numeric(ell1, ell2, ell3)
```

`LOSProjector.project` is the assembly boundary that knows both the 3D and 2D
bispectrum APIs. It preserves additive term names and returns a
`Bispectrum2D`; it does not expose a bound closure as the projected result.
The numeric LOS sampling and quadrature functions remain independent internal
array/callable primitives beneath this boundary.

Native and projected angular bispectra use the same `Bispectrum2D` aggregate.
Their construction history belongs to their term representations, not to
separate bispectrum subclasses. A native numeric term contains a
`NumericExpression2D`; a projected numeric term contains a
`ProjectedNumericRepresentation2D`. Both implement the
`NumericRepresentation2D` capability consumed by downstream calculators.
The projected representation retains its weighted source term, exact source
representation, projector, kernel combination, and an immutable executable
projection-rule object. It therefore remains a deferred projection recipe
rather than an anonymous closure or a sampled result. The current
`NumericLOSProjectionRule` is stateless; future Slepian and semi-analytic rules
can carry their own representation-specific projection behavior.

Representation projection is type-dispatched in `projection/rules.py`:

```text
NumericRepresentation3D
    -> ProjectedNumericRepresentation2D
SlepianRepresentation3D
    -> ProjectedSlepianRepresentation2D       (future)
SemiAnalyticRepresentation3D
    -> ProjectedSemiAnalyticRepresentation2D  (future)
```

`LOSProjector.project` applies the registered rule to every representation of
every source term and then assembles the resulting `BispectrumTerm2D` objects.
An unregistered representation raises `NotImplementedError`; projection must
never silently discard a source representation. This permits native and
projected terms to coexist in one `Bispectrum2D`, while route calculators ask
only for the representation capability they consume.

An exact fixed-redshift evaluation is constructed with
`LOSProjector.delta_like(z=z, chi=chi)`, not with a narrow sampled
`Kernel1D`. It is an ordinary `LOSProjector` from the caller's point of
view, but performs no LOS quadrature: it evaluates the 3D expression directly
at `k_i = ell_i / chi` and the specified `z`. Extended kernels use the regular
constructor, whose default geometrical prefactor is `chi**-4`. The delta-like
projector applies no such prefactor.

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

## Numeric 2D interpolation

Interpolation is another implementation of the 2D numeric capability, not a
new bispectrum type. `InterpolatedNumericRepresentation2D` retains its source
`NumericRepresentation2D`, interpolation configuration, source-state token,
and a self-owned in-memory cache. Its `evaluate(ell1, ell2, ell3)` method
therefore has the same contract as native and projected numeric
representations.

The initial interpolation coordinates are
`(log ell2, log ell3, mu23)`, with

```text
mu23 = (ell1^2 - ell2^2 - ell3^2) / (2 ell2 ell3).
```

This rectangularizes the closed-triangle domain used by the numeric multipole
route without sampling invalid triples of side lengths. Values are
interpolated linearly by default; logarithmic interpolation of the bispectrum
value is not assumed because terms may change sign.

`b2d.interpolate(config)` returns a new `Bispectrum2D` in which each numeric
term representation is replaced by an interpolated representation. It does
not mutate the source bispectrum or append a second numeric capability to the
same term. The exact source remains available through
`InterpolatedNumericRepresentation2D.source_representation`; term names,
coefficients, non-numeric representations, and the `Bispectrum2D` interface
are preserved. Term-level caches are built by default.

Multiple terms can share one interpolation table by composing them before
interpolation:

```python
b_grouped = b2d.combine_numeric_terms(
    name="combined",
    terms=("term1", "term2"),
)
b_interpolated = b_grouped.interpolate(config)
```

The combined term contains a `NumericSumRepresentation2D` whose components
are the original coefficient-bearing terms. It intentionally exposes only a
numeric representation: combining route-specific single-form expressions is
not assumed to preserve their Slepian or semi-analytic structure. Unselected
terms remain separate unless `keep_unselected=False` is requested. This keeps
grouping independent from interpolation and makes any loss of route-specific
representations explicit.

Each representation owns a mutable private cache slot containing an immutable
`TriangleInterpolationCache`. The cache stores the axes, sampled values,
SciPy interpolator, and source-state token. Evaluation lazily builds a missing
cache and rebuilds a stale cache when the source token changes. Cache mutation
is lock-protected; the physical source, interpolation configuration, and
representation identity remain immutable. Persistent disk caching is a
separate future layer.

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

## 3PCF tables and tuned FFT grids

`HKernelTable`, `ZetaKTable`, and `ZetaTable` are passive calculated results.
They do not know which route produced them and do not run coupling, FFTLog,
Slepian, or angular-assembly algorithms. Concrete route calculators construct
these tables.

The numeric route uses `fastnc.hankel.TunedFFTGrid`. Given the final evenly
log-spaced target theta bins, this object creates a higher-resolution FFTLog
ell grid and matching full theta grid such that the target bins occur at the
integer `down_sampler` indices. Numeric assembly creates this grid before the
HKernel calculation, passes its full ell axis to the HKernel calculator, uses
the same `xy` and axes for the double FFTLog, and downsamples both theta axes
without interpolation.

`HKernelTable` retains the `TunedFFTGrid` because its values live on the full
FFTLog ell grid. `ZetaKTable` retains only the final target theta coordinates
and downsampled values. This is the common route boundary:

```text
numeric:
    BispectrumMultipole -> HKernelTable -> ZetaKTable -> ZetaTable

Slepian:
    Bispectrum3D + LOSProjector -> ZetaKTable -> ZetaTable
```

The full numeric FFTLog theta array is an intermediate calculator result, not
part of the route-independent `ZetaKTable` contract. A Slepian calculator can
therefore construct the same table directly on the requested theta bins.
Table keys remain hashable route-supplied labels so spin/coupling key design
can be fixed when the corresponding calculators are implemented.

As of version `2.0.38`, route execution is managed by one `ThreePCF` object
rather than separate calculator classes for every stage. It retains coupling
matrices and their shared cache sessions, and later will retain the Weber and
Mellin resources used by the Slepian route. The object is initialized with
config, a 2D bispectrum, theta, phi, and a route, and owns the source
coordinates and tuned FFT grid. The route is fixed at construction and may be
changed explicitly with `set_route()`. Its `multipoles()`, `hkernel()`, and
`zetak()`, and `zeta()` methods expose intermediate stages for validation
while sharing the same route state. Low-level operations such as the
Fourier-mode contraction remain pure array functions.

`set_theta()` rebuilds the tuned grid and invalidates HKernel, ZetaK, and
Zeta results. `set_phi()` preserves all radial results and invalidates only
final Zeta assembly. `set_bispectrum()` preserves the grid and coupling
resources while invalidating source-dependent results. Input coordinates are
copied and made read-only. `set_route()` preserves the source, coordinates,
grid, and reusable coupling resources, while invalidating the multipole facade
and all route-dependent result tables.

Route is execution policy, not result identity. Consequently neither the
ZetaK cache key nor the Zeta cache key contains the route name. Both caches are
keyed only by their requested epsilon set.
`set_route()` clears these caches before another policy can run, so omitting
route introduces no stale-result ambiguity. In the future mixed policy, each
calculator contributes to the same physical `ComponentModeKey(epsilon,
two_k)` and the numeric, Slepian, and semi-analytic contributions are summed
before the common `ZetaKTable` is exposed. Route provenance may be recorded as
diagnostic metadata, but it must not split physically identical modes into
different result keys.

Explicit epsilon requests are strict and must belong exactly to
`SpinSpec.representative_epsilons()`. At a vertex with `spin_i == 0`, epsilon
does not represent a conjugation degree of freedom, so the representative has
`epsilon_i=+1`; explicitly supplying `-1` raises `ValueError`. A conjugate but
non-representative component also raises `ValueError` identifying its
representative rather than silently returning or relabeling that result.
Repeated valid representatives are removed while preserving request order.
Conjugate reconstruction can be added later as an explicit API after its
k-reversal and projection conventions are independently tested.

`ThreePCF.zeta()` performs the route-independent opening-angle assembly. For
each stored representative epsilon triple it obtains the physical k modes
through the `ZetaKTable` alias mapping and evaluates

```text
Zeta_epsilon(theta1, theta2, phi)
    = sum_k ZetaK_epsilon,k(theta1, theta2) exp(i nu_k phi),
nu_k = k + (sigma3 - sigma2) / 2.
```

The canonical assembled `ZetaTable` is in the x/cross projection and stores
the effective spin triple for every component. `ThreePCF.zeta()` defaults to
`projection="x"`; passing `projection="ortho"` or `projection="centroid"`
converts the cached x-projection table when the result is returned. Converted
projections are deliberately not cached because the phase conversion is cheap.
Equivalently,
`ZetaTable.to_projection(projection)` applies the existing spin-aware phase
conversion component by component and returns a new passive table. Neither API
reruns the bispectrum, coupling, or Hankel calculations. The current assembly
evaluates the Fourier series at the supplied phi coordinates; finite phi-bin
averaging is not yet part of this API.

The numeric implementation first asks `BispectrumMultipole` for canonical
full-Fourier coefficients. Cosine and sine storage conventions are converted
inside the multipole package; the 3PCF manager therefore contracts only
coefficients satisfying the common complex Fourier convention. For each
effective spin it obtains a retained `CouplingMatrix`, computes

```text
H_k(ell2, ell3) = sum_L B_L(ell2, ell3) G_Lk(psi_ell),
```

and deduplicates the result by `HKernelKey(sigma1, two_nu)`. This canonical
storage key is not the physical component label. Both `HKernelTable` and
`ZetaKTable` therefore retain an immutable alias mapping from
`ComponentModeKey(epsilon=(epsilon1, epsilon2, epsilon3), two_k=2*k)` to the
corresponding deduplicated storage key. Callers use `get_for_mode(epsilon, k)`
for physical lookup and `get(key)` only when inspecting canonical storage.
Only epsilon modes explicitly included when constructing the table receive an
alias; conjugate components are not inferred until their k-sign and complex
conjugation convention is implemented and tested. The subsequent
double Hankel transform always replaces its `xy` option with
`TunedFFTGrid.xy`, checks the returned full theta coordinates, and selects the
target theta bins by integer indices. `ZetaKKey` contains the H-kernel key,
the two Bessel orders, and the total effective spin, so direct Slepian output
can later use the same route-independent labels.

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

## Bispectrum multipoles

Angular bispectrum multipoles are route-independent mathematical objects. They
live in the top-level `fastnc/multipole` package rather than in either
`fastnc/bispectrum` or a 3PCF route. `BispectrumMultipole` stores the source
bispectrum, basis convention, and calculator object. It does not require an
`ell` grid or mode range and does no numerical evaluation at construction.

The normal user entry point is a thin construction facade:

```python
bm = BispectrumMultipole.from_numeric(
    config,
    b2d,
    basis="fourier",
)

value = bm.evaluate(mode, ell2, ell3)
values = bm.evaluate(modes, ell2, ell3)
```

`from_numeric` constructs and retains a
`NumericBispectrumMultipoleCalculator`; it does not call the numerical kernel.
`evaluate` delegates to that concrete calculator only when values are
requested. A scalar mode returns the broadcast `ell` shape; a one-dimensional
mode array returns `(n_modes, *ell_shape)` and evaluates all requested modes in
one sampling pass. Future `from_slepian` and `from_semi_analytic` constructors
construct their own concrete calculators. All route-dependent work remains in
those calculator objects.

No generic calculator protocol, calculator factory, or intermediate evaluated
data class is introduced. A calculator returns an `ndarray` directly because
the caller already supplied the modes and coordinates. If an explicit sampled
table is later needed, it will be a separate passive object carrying modes,
coordinates, values, and conventions.

A sampled multipole table is a separate future passive object. Its explicit
mode and `ell` grids must not be folded into `BispectrumMultipole`.

The basis name is a calculation-wide convention rather than a quadrature
setting.  A standalone `BispectrumMultipole` receives it explicitly, while a
`ThreePCF` receives it once through `ThreePCFConfig.basis` and passes the same
value to both its multipole calculator and coupling object.  The current
choices are strings (`cosine`, `sine`, `fourier`, and `legendre`); a dedicated
basis class is not justified until basis objects own behavior beyond their
finite Fourier expansion.

`CouplingKernel` is the real-valued Fourier primitive and is the only layer
that owns coupling cache and exact-support zeros.  `CouplingMatrix` expands a
requested cosine, sine, Fourier, or Legendre mode into a finite set of Fourier
modes, evaluates those primitives, and resums them.  No new-basis zero test or
small-value threshold is applied after resummation.  Cosine and Legendre
couplings remain real; sine coupling is generally purely imaginary.  The
Fourier cache remains real-valued in every case.  `BispectrumMultipole`
exposes only `evaluate`; it has no basis-conversion convenience interface.

The sine coupling convention is the odd sector of the full-angle real Fourier
basis on `[0, 2 pi)` and its finite Fourier resummation is generally purely
imaginary.  A length-only `Bispectrum2D(ell1, ell2, ell3)` is even in the signed
relative angle and therefore has no physical full-angle sine component.  The
current numeric sine decomposer is still a half-range `[0, pi]` diagnostic;
until an orientation-aware 2D source representation exists, an end-to-end
sine `ThreePCF` must not be interpreted as the full-angle odd sector.

The basis resummations are tested independently against direct quadrature of
the original angular coupling integral below, at, and above `psi=pi/4`.
End-to-end Fourier, cosine, and Legendre calculations agree through HKernel,
ZetaK, and Zeta for a finite Legendre-mode source.  A Gaussian radial toy
bispectrum with only `L=0,+/-2` also gives agreement between the numeric and
brute-force 3PCF routes at better than one percent on the validation grid.

The dependency direction is

```text
Bispectrum2D or another supported source
    -> route-specific multipole calculator
    -> BispectrumMultipole
    -> HKernel calculator
```

`Bispectrum2D` has no multipole convenience method and does not import the
multipole package.

## Calculator and storage boundaries

Low-level numerical kernels operate on angular values and `ell` arrays. They do
not receive a `Bispectrum3D`, `Bispectrum2D`, Grid, or ThreePCF instance. A
calculator facade may receive a typed domain object at the assembly boundary,
extract its callable capability, and then invoke those kernels.

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
    halofit.py           # standalone Halofit/BiHalofit numerical model
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

The old object graph is stored under `legacy/bispectrum_object_api/`. This is
a non-importable reference archive, not a compatibility layer. It contains the
former evaluator bases and wrappers, the unfinished and unvalidated
`bispectrum/analytic` implementation, and the old high-level `ThreePCF` API.
Active modules must not import from the archive. A useful numerical component
returns to active code only after it is extracted behind the new dependency
boundary and independently tested.

## Disposition of existing modules

Existing files are not kept or deleted as indivisible units. Pure mathematical
and numerical components are retained, while wrappers that depend on the old
bispectrum object graph are replaced.

- `support.py` remains a domain/value module and can be reused directly.
- `halofit.py` is a standalone physical/numerical implementation and remains
  usable. Moving it under a later `models` or `physics` namespace is optional
  and is not required for the representation refactor.
- The basis definitions and pure transforms formerly in `bispectrum/decompose.py`
  are reusable and now live in `multipole/decompose.py`. Callers coupled to old
  bispectrum or multipole objects are not part of the retained interface.
- Coordinate transforms, grid preparation, packing, and interpolation kernels
  in the archived `interpolate.py` may later be extracted as array-based
  functions. The old interpolated bispectrum wrappers do not define the new
  architecture.
- Angular sampling, quadrature, basis normalization, and array-shape logic in
  the archived `multipole.py` may later be extracted into calculators or pure
  kernels. Lazy multipole wrappers, Grid/calculator mixing, and direct
  dependencies on `Bispectrum2D` are retired.
- The mathematical regulator in the archived `regulator.py` is potentially
  reusable, but regulator selection belongs to the numeric route rather than
  to a bispectrum model.
- LOS geometry, windows, and quadrature are projection concerns and move out
  of `bispectrum/`. Route-independent projection kernels belong in a projection
  package. Numeric node evaluation, Mellin-coefficient integration, and other
  route-specific assembly belong to their respective route modules.
- Grid classes are passive storage for coordinates, values, labels, and
  provenance. Calculator methods are removed from them rather than migrated.

The intended high-level ownership is therefore

```text
fastnc/
    bispectrum/          physical terms, representations, state, and support
    multipole/           route-independent multipole results and producers
    projection/          LOS kernels, geometry, and projection primitives
    threepcf/
        conventions/     shared spin and projection conventions
        numeric.py       numeric 3PCF route assembly
        slepian.py       angular Slepian expression -> ZetaK -> Zeta
        semi_analytic.py angular U/V/W expression -> multipoles -> ...
```

The route modules are specifically bispectrum-to-3PCF algorithms and therefore
live directly under `threepcf`, not in an ambiguous top-level `routes` package.
Multipole production is top-level because it produces a reusable mathematical
object rather than a 3PCF result.
Projection is top-level because converting a 3D expression into an angular or
LOS-integrated expression is useful independently of a final 3PCF route.

The active projection layer separates pure numerical primitives from assembly.
`numeric_los.py` consumes only arrays and callables and contains LOS sampling
and quadrature. These primitives are internal implementation tools rather than
top-level user API. `LOSProjector` is the assembly boundary: its public
`project(b3d)` method accepts a `Bispectrum3D` and returns a `Bispectrum2D`,
preserving additive term names. It stores only LOS coordinates, `Kernel1D`
objects, an explicit prefactor, and the angular-to-comoving shift. It does not
select a 3PCF route or own route caches. Coefficient integration remains a
separate public operation for future Slepian and Mellin representations.
Following the version 2 projection convention, the prefactor defaults to
`chi**-4`. A fixed-redshift benchmark is created by
`LOSProjector.delta_like(...)`; it evaluates exactly at the requested point
without a sampled kernel, quadrature, or geometrical prefactor. The former
object-based projectors remain archived.

## Migration from the current package

The refactor remains incremental, but preservation of an old wrapper is not a
design objective. Existing implementations remain temporarily available only
until their reusable kernels and numerical behavior have replacement tests.

As of version `2.0.24`, steps 1--8 below are implemented. The old bispectrum
object graph, unfinished semi-analytic package, and old high-level `ThreePCF`
entry point are in `legacy/bispectrum_object_api/`. The computing Grid
pipeline formerly under `fastnc/threepcf` is in
`legacy/threepcf_grid_pipeline/`. Neither archive is an active import.
Route-independent projection primitives have been extracted into
`fastnc/projection`, and the new route ownership exists under
`fastnc/threepcf`. Shared spin and output-projection conventions live under
`fastnc/threepcf/conventions`. Route-independent multipole products and their
numeric producer live under `fastnc/multipole`. No Slepian or semi-analytic
route is implemented yet.

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
9. Define and validate `SemiAnalyticExpression` from the mathematics and
   reference calculations. Do not migrate the archived, unvalidated analytic
   package wholesale; recover individual kernels only after independent tests.
10. Extract required LOS primitives from the archive into a projection package,
   separating route-independent geometry from route-specific assembly.
11. Refactor the 3PCF workflow and passive Grid storage only after the
   bispectrum representation boundary is stable.
12. Keep the retired `base.py` and obsolete interpolation/multipole wrappers
   only in the non-importable reference archive.
13. Implement the Slepian and Weber-Schafheitlin route on top of the new
   expression API.

Term-wise in-memory numeric 2D interpolation and explicit numeric term
composition are implemented. Automatic grouping policies and persistent
disk-cache design remain later optimizations.

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

## Independent brute-force validation

`fastnc.threepcf.bruteforce` is the active reference calculation for checking
the numeric route.  `BruteForceX3PCF` accepts the same current `Bispectrum2D`
object used by `ThreePCF`, but deliberately bypasses bispectrum multipoles,
the multipole coupling matrix, HKernel, and the two-dimensional FFTLog.  It
instead performs a one-dimensional FFTLog in the common Fourier scale and
direct numerical quadrature over the two triangle-angle variables.  Agreement
therefore tests the assembled numeric route against an algorithmically
independent calculation.

The brute-force solver returns the X-projection for one representative spin
component selected by the active `SpinSpec` convention.  Its ell interval,
radial FFTLog resolution, angular quadrature resolution, and optional adaptive
refinement are explicit in `BruteForce3PCFConfig`.  Numerical comparisons must
converge both calculations separately before attributing a discrepancy to the
formalism.  This module was recovered from
`legacy/threepcf_grid_pipeline/fastnc/threepcf/bruteforce.py`, but the active
implementation imports no legacy code and the archive remains reference-only.
