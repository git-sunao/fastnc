# Semi-analytic route development notes

Status: Non-normative implementation note. The equations follow Appendix C,
Eqs. (C1)-(C11), of the current cross3pcf calculation note and must be checked
against independent numeric multipoles before becoming a normative contract.

## Representation and projected coefficients

The generic 3D contribution is

```text
B(k1,k2,k3;z)
  = U(k2/k,k3/k) V(k2,k3;z) (k1/k)^p W(k1;z),
k = sqrt(k2^2+k3^2).
```

Only `W(k1;z)` is expanded in complex powers,
`W(k1;z) = sum_n w_n(z) k1^nu_n`. The angular kernel

```text
K_L^(nu)(r) = (1/2pi) int_0^(2pi) dphi s(r,phi)^nu exp(-i L phi)
```

depends on neither redshift nor projection kernels and is a reusable route
cache. With the present definition of `s`,
`r = 2 ell2 ell3/(ell2^2+ell3^2)` and is unchanged by
`k_i = ell_i/chi`.

For a projected term, do not first construct a fully projected angular
bispectrum and then numerically decompose it. Replace the redshift-dependent
FFTLog coefficient by

```text
d_n^ABC(ell2,ell3)
  = int dchi W^ABC(chi)/chi^4
      V(ell2/chi,ell3/chi;z(chi)) w_n(z(chi))/chi^nu_n.
```

The projected multipole is then

```text
B_L^ABC(ell2,ell3)
  = U(ell2/ell,ell3/ell)
      sum_n d_n^ABC(ell2,ell3) ell^nu_n K_L^(nu_n+p)(r),
ell = sqrt(ell2^2+ell3^2).
```

Thus a projected semi-analytic representation retains the 3D expression and
`LOSProjector`. `SemiAnalyticCalculator` owns the universal `K_L` tables and
performs the coefficient-level LOS projection and final contraction.
`Bispectrum2D.evaluate_numeric()` is not an intermediate of this route.

## Current implementation boundary

The generic representation and calculator are now implemented without adding
any physical semi-analytic model. A model must provide `U`, `V`, `p`, Mellin
exponents `nu_n`, and the redshift-dependent coefficients `w_n(z)`. The
projected representation remains passive; all evaluation is calculator-owned,
matching the Slepian route's code structure.

The generic calculator is covered by two independent toy checks. A finite
Fourier polynomial compares the semi-analytic result with numeric angular
decomposition. A complex-conjugate Mellin pair compares coefficient-first LOS
projection with a transparent benchmark that constructs each fixed-redshift
multipole before LOS integration. The latter also verifies that changing only
the redshift-dependent coefficients reuses the universal angular-kernel cache.

The SPT matter pair-23 contribution is the first physical implementation. It
is expanded into additive `p=0,2,4` terms, each carrying the same contribution
in numeric and semi-analytic form. Their sum is tested against the unsplit SPT
formula, and the projected multipoles are compared in the development notebook.

The note's Eq. (C15) prints the pair-23 zero mode as `23/7 P2 P3`. Directly
expanding its Eq. (C13) instead gives

```text
2 F2 = 12/7 + (k2/k3 + k3/k2) cos(phi) + 2/7 cos(2 phi),
```

and hence `B_0 = 12/7 P2 P3`. This apparent typo must be resolved before Eq.
(C15) is used as a regression reference.

## Required implementation order

1. Inspect the pair-23 multipole residual and warm-cache timing over broad
   scale ratios.
2. Extend the physical semi-analytic model coverage only after that comparison
   is accepted.
3. Add low-rank `V` and other optimizations only when a model requires them.
