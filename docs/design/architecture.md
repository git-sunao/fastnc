# fastnc v2 architecture

Status: Normative design contract.

This document records the current package boundaries and public calculation
model. Detailed reasoning, migration history, and rejected alternatives are in
`docs/notes/bispectrum-refactor.md`.

## Coordinates and domain objects

A three-dimensional bispectrum is a function
`B3D(k1, k2, k3, z)`. A native angular bispectrum is a function
`B2D(ell1, ell2, ell3)`. The public model introduces neither a generic `q`
coordinate nor a mandatory redshift-slice object. At fixed redshift,
angularization uses `k_i = ell_i / chi(z)`.

`Bispectrum3D` and `Bispectrum2D` are typed aggregates of weighted additive
terms. A bispectrum is not itself a term. Addition concatenates and flattens
term collections; a term remains the smallest unit assigned to one calculation
route.

## Terms and representations

Each term may provide alternative mathematical representations:

- a numeric representation evaluates the term at supplied coordinates;
- a Slepian representation contains the powers, phases, radial factors, and
  other information required for Mellin/Weber evaluation;
- a semi-analytic representation contains the corresponding `U`, `V`, `W`,
  power, and angular structure;
- a direct multipole representation may provide an angular multipole without
  first sampling the full angular bispectrum.

A representation describes mathematics. It does not own sampled grids, route
caches, LOS state, or downstream calculators. Every term must have a numeric
representation so the numeric route remains the universal fallback.

A semi-analytic representation stores the factorization `U`, `V`, `W`, `p`
and the Mellin expansion of `W`, but owns no route cache. Projection retains
that rule and the `LOSProjector`. `SemiAnalyticCalculator` projects the
redshift-dependent Mellin coefficients before constructing the bispectrum
multipoles, without sampling a projected angular bispectrum over the triangle
angle. The result then uses the shared `BispectrumMultipole -> HKernel ->
ZetaK` stages.

Term addition produces a new term whose guaranteed representation is numeric.
It must not claim that separability, Slepian structure, or semi-analytic
structure survives an arbitrary sum. A caller can therefore combine selected
terms before constructing a shared numeric interpolation.

An interpolated numeric representation is itself a numeric representation. It
owns its interpolation table and cache and evaluates numeric values through the
same interface as a direct expression. Interpolation may be attached per term;
combined interpolation is obtained by first constructing the desired numeric
term sum.

## State and cache ownership

Physical model state includes cosmology, power spectra, redshift-dependent
parameters, and model parameters. Updating state invalidates derived physical
quantities, which are recomputed lazily when requested.

Terms and representations do not own route caches. Route calculators own
Mellin coefficients, Weber kernels, coupling resources, FFTLog state, and
sampled intermediate results. Cache keys must include every mathematical or
numerical choice that changes a result. Changing a source, route, or target
coordinate invalidates only dependent state.

## Projection

Projection is independent of the 3PCF route. Projection code belongs in
`fastnc/projection`, not in bispectrum models or route modules.

`LOSProjector` is a configured projection operator. It consistently owns both
the physical projection rule and its immutable numerical LOS grid: kernels,
the geometrical prefactor, the angular-to-comoving mapping, `z` and `chi`
nodes, and the quadrature rule. These concerns should remain distinguishable
inside the object for signatures and cache invalidation, but they do not have
different public owners.

Projecting a source constructs representation-specific deferred recipes.
Consequently, standalone numeric projection remains convenient:

```python
b2d = los.project(b3d, sample_combination=samples)
b2d.evaluate_numeric(ell1, ell2, ell3)
```

The returned projected `Bispectrum2D` owns a deferred recipe containing its
source and projector. A projected numeric representation performs LOS
integration at bispectrum evaluation. Projected Slepian and semi-analytic
representations retain redshift-dependent source information and expose the
same projector to their route calculators, which apply LOS integration at the
result or coefficient stage. The projector itself does not know the route.

Coefficient-level projection preserves the factorized LOS structure. At each
node it contracts the redshift-dependent one-dimensional Mellin coefficients
with the low-rank factors of `F_ab`, then performs LOS quadrature only on the
reduced rank-space integrand. It must not materialize or cache a dense
LOS-integrated coefficient matrix `bar_C_ab`. That matrix remains a useful
mathematical identity, not a runtime data object.

Native and projected representations may coexist in one `Bispectrum2D`.
`ThreePCF` must not branch on a bispectrum-wide `is_projected` flag. It
dispatches on explicit representation capabilities or types: native
representations require no LOS operation, while projected representations
provide their source and projector. This keeps route selection local to each
term and permits mixed construction histories.

`Kernel1D` stores an unnormalized radial kernel. `KernelSet` groups the kernels
and uses the ordinary LOS prefactor, conventionally `chi**-4`, unless exact
fixed-redshift evaluation is requested. The fixed-redshift constructor binds a
single exact `(z, chi)` point and does not approximate a delta function with a
narrow sampled window.

Projecting a `Bispectrum3D` constructs a `Bispectrum2D`. Each projected term
retains the representations supported by a mathematically valid projection
rule. Downstream route code sees only the resulting `Bispectrum2D` contract and
does not depend on whether it was defined natively or obtained by projection.

## Multipoles and coupling

Route-independent angular multipole objects belong in `fastnc/multipole`.
`BispectrumMultipole` is a lazy facade: construction stores the source, basis,
configuration, and concrete calculator; numerical work begins at
`evaluate(mode, ell2, ell3)`.

The basis is selected once in `ThreePCFConfig` and is used consistently by the
bispectrum multipole calculator and coupling calculation. Fourier coupling is
the cached primitive with exact-support zeros. Cosine, sine, and Legendre
couplings are finite Fourier resummations. They do not introduce independent
small-value thresholds or zero tests. Sine coupling is generally imaginary;
its physical interpretation requires an orientation-aware source.

`Bispectrum2D` does not import the multipole package and has no multipole
convenience method.

## ThreePCF and routes

The basic interface is

```python
threepcf = ThreePCF(config, bispectrum2d, theta, phi, route="numeric")
```

`theta` and `phi` are part of the calculation state. `set_theta()` rebuilds the
tuned radial grid and invalidates radial and angular results. `set_phi()` keeps
radial results and invalidates final angular assembly. `set_bispectrum()` keeps
route-independent grid resources and invalidates source-dependent results.
`set_route()` clears route-dependent results and resources.

The numeric route is

```text
Bispectrum2D numeric representation
  -> BispectrumMultipole
  -> HKernelTable
  -> ZetaKTable
  -> ZetaTable
```

The Slepian route may proceed directly from a supported angular representation
to `ZetaKTable`; the semi-analytic route may contribute through its appropriate
multipole or radial kernel. The agreed hybrid-route contract is term-wise: use
the requested specialized representation where available and fall back to
interpolated or direct numeric evaluation otherwise. Contributions from
different routes are summed into the same route-independent table keys.

`route="hybrid"` implements this contract term by term. Terms carrying a
Slepian representation are evaluated by `_zetak_slepian`; remaining terms use
a coefficient-factorized semi-analytic representation when available and
otherwise fall back to numeric angular decomposition. The latter two produce
bispectrum multipoles through separate calculators and then share coupling,
HKernel, and HKernel-to-ZetaK processing. Public `zetak()` alone combines and
caches the contributions.
The planner assigns each term exactly once, so a term carrying both numeric
and Slepian representations is not double counted.

`threepcf.zeta(projection="x")` returns the cached x-projection by default and
may convert it cheaply to another shear projection such as `"centroid"` at
return time. Projection is not part of the Zeta cache key.

## Passive tables and grids

`HKernelTable`, `ZetaKTable`, and `ZetaTable` are passive result containers.
They store coordinates, values, mode/spin labels, and provenance; they do not
run coupling, FFTLog, Slepian, or assembly calculations.

The numeric route uses `fastnc.hankel.TunedFFTGrid`. It builds a high-resolution
logarithmic theory grid whose integer down-sampling points coincide with the
requested logarithmic `theta` bins, avoiding interpolation of the final theory
prediction. HKernel uses the full matched ell grid; ZetaK stores only requested
theta bins after the double FFTLog and down-sampling.

Table keys encode physical mode identity, including the requested representative
epsilon tuple where applicable. They do not contain the route because a
`ThreePCF` instance has one active route and route changes clear the cache. For
spin zero, only epsilon `+1` is valid; unsupported epsilon requests raise an
error rather than silently returning a representative with different meaning.

## Numerical kernel boundary

Low-level kernels accept arrays, scalars, and callables. They do not receive
Grid, bispectrum-model, or `ThreePCF` objects. Assembly code is the only layer
that extracts capabilities from domain objects, selects representations,
queries caches, and stores results.

Active package ownership is:

```text
fastnc/bispectrum/   terms, representations, aggregates, physical models
fastnc/projection/   LOS kernels, geometry, and projectors
fastnc/multipole/    route-independent multipole objects and coupling
fastnc/hankel/       tuned FFTLog grids and transforms
fastnc/threepcf/     ThreePCF facade, passive tables, and route algorithms
```

Code under `dev/legacy/` is local reference material only. Active code must
never import it. Recover a useful component by extracting it into the active
architecture and adding an independent test for its contract.
