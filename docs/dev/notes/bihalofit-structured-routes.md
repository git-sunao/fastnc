# BiHalofit structured-route formulation

Status: design note preceding implementation. The exact numeric formula is the
reference. No fiducial-shape approximation may be exposed as an alternative
representation of the exact physical term without an explicit approximation
identity.

## Definitions

At redshift `z`, define

```text
q_i = k_i r_sigma(z),
D(q,z) = 1 / (1 + e_n(z) q),

E(k,z) = [(1 + f_n q^2)/(1 + g_n q + h_n q^2)] P_L(k,z)
       + [1/(m_n q^mu_n + n_n q^nu_n)] [1/(1 + (p_n q)^-3)].
```

For ordered side lengths `k_min <= k_mid <= k_max`, the shape variables are

```text
r1 = k_min / k_max,
r2 = (k_mid + k_min - k_max) / k_max.
```

Both are scale-free but depend on the complete triangle, including the angle
when parameterized by two sides and their included angle.

## Exact 3-halo decomposition

The implemented 3-halo contribution is

```text
B_h3 = D1 D2 D3 sum_cyc [2 F2(i,j) E_i E_j + 2 d_n q_k E_i E_j],
```

where `k` is opposite pair `(i,j)`. With `R_i = D_i E_i`, one pair is

```text
B_h3^(ij|k) = R_i R_j D_k [2 F2(i,j) + 2 d_n q_k].
```

For pair angle `phi_ij`,

```text
2 F2(i,j)
  = 12/7
  + (k_i/k_j + k_j/k_i) cos(phi_ij)
  + (2/7) cos(2 phi_ij)

  = 12/7
  + 1/2 (k_i/k_j + k_j/k_i)
      [exp(+i phi_ij) + exp(-i phi_ij)]
  + 1/7 [exp(+2i phi_ij) + exp(-2i phi_ij)].
```

Each cyclic pair therefore has an exact finite separable expansion:

- one `m=0` F2 term with radial factors `(R_i, R_j, D_k)`;
- four `m=+/-1` terms, since each phase occurs with both `k_i/k_j` and
  `k_j/k_i` radial powers;
- two `m=+/-2` terms with radial factors `(R_i, R_j, D_k)`;
- one additional `m=0` term with radial factors `(R_i, R_j, q_k D_k)` and
  coefficient `2 d_n(z)`.

Thus `B_h3` has 24 exact primitive terms across three cyclic pairs. This is
the same finite angular family as tree-level SPT, multiplied by a non-constant
radial damping factor on the opposite leg.

Mathematically this is Slepian-separable. It is not yet accepted by the current
calculator because `_slepian_leg_layout` requires exactly leg 2 or leg 3 to be
constant. BiHalofit is therefore the first physical reason to implement the
general three-nonconstant-leg Slepian transform. Until then, these primitives
may be represented semi-analytically by Mellin-expanding one selected leg, but
this is a calculator limitation rather than a failure of separability.

## Exact and fixed-shape 1-halo terms

Define

```text
H(q,z;r1,r2)
  = 1 / [a_n(r1,z) q^alpha_n(r2,z)
         + b_n(z) q^beta_n(r2,z)]
    / [1 + 1/(c_n(z) q)],
```

with

```text
a_n(r1,z)     = 10^[A_n(z) - 0.310 r1^gamma_n(z)],
alpha_n(r2,z) = min(10^[A_alpha(z) + C_alpha(z) r2^2],
                    1 - 2 n_s/3),
beta_n(r2,z)  = 10^[A_beta(z) + 0.007 r2].
```

The exact term is

```text
B_h1(k1,k2,k3;z)
  = product_i H(q_i,z; r1(k1,k2,k3), r2(k1,k2,k3)).
```

Although written as a product over legs, it is not separable: every factor
contains the same full-triangle shape variables. The exact term must remain
numeric unless shape dependence itself is expanded in a controlled basis.

For fixed fiducial values `(r1*,r2*)`,

```text
B_h1^fid(k1,k2,k3;z | r1*,r2*)
  = product_i H(q_i,z;r1*,r2*)
```

is exactly separable into three non-constant radial factors. It can use the
generalized Slepian calculator or a semi-analytic Mellin expansion. However,
`B_h1^fid` is not another representation of exact `B_h1`; it is a different,
approximate model. The API must expose `(r1*,r2*)` and approximation status.
Its numeric representation must evaluate the same frozen-shape formula, so
numeric-versus-structured validation compares identical models.

## Proposed term architecture

The exact `BiHalofitBispectrum3D` should contain:

```text
bihalofit:Bh1:exact
bihalofit:Bh3:<pair>:<harmonic/radial label>
```

`Bh1:exact` initially has only a numeric representation. Every primitive Bh3
term has a numeric representation and, after calculator support is ready, an
exact structured representation. Summing primitive numeric terms must recover
legacy `get_bihalofit(..., which="Bh3")` before any route test.

The fixed-shape model should be explicit, for example
`BiHalofitFiducialShapeBispectrum3D`, or be returned by an unmistakably named
factory as a separate bispectrum. It contains `bihalofit:Bh1:fixed-shape` with
matching numeric and structured representations. It must not silently replace
exact Bh1 in the default model.

## Implementation and validation order

1. Extract coefficient and radial helper functions from `Halofit` without
   changing its legacy numeric result.
2. Split exact Bh3 into 24 named primitive numeric terms and compare their sum
   with monolithic Bh3 over random, equilateral, flattened, and squeezed
   triangles across redshift.
3. Add passive three-nonconstant-leg Slepian expressions to those terms.
4. Extend the Slepian calculator for three non-constant radial legs, validating
   analytic/native-2D toys before fixed-redshift Bh3.
5. Add exact Bh3 semi-analytic representations only where they provide a
   distinct computational advantage; do not duplicate a route only for API
   symmetry.
6. Implement explicit fixed-shape Bh1 with identical numeric and structured
   formulas. Measure its error against exact Bh1 over triangle shape.
7. Investigate a controlled basis or low-rank expansion in `(r1,r2)` for exact
   Bh1. One fiducial shape is a benchmark and approximation, not an exact route.

## Open numerical questions

- The generalized three-leg expression introduces three Mellin coefficient
  sets. A direct post-Mellin contraction is rank three and may be too costly.
  Contraction order and reusable geometry tensors require a derivation first.
- Dropping the opposite-leg `D(k,z)` to reuse the constant-leg calculator would
  be an uncontrolled model approximation and is not proposed.
- Legacy squeezed-safe evaluation combines cyclic F2 terms before taking the
  squeezed limit. Primitive terms expose large cancellations. Their numeric
  reference needs stable grouped summation or a grouped squeezed-limit path.
- The fitted `alpha_n` includes clipping. Fixed-shape and future shape-basis
  implementations must reproduce that branch exactly.
