# RNS modulus for the Ring-LWE schemes: security review

Question: can the NTT-friendly prime `q` of the Ring-LWE schemes be replaced
by a product `q = q1 q2` of distinct NTT-friendly primes (`q_i = 1 mod 2n`),
so that the native code works in two u64 residues instead of u128 words,
without invalidating BMMS21's proofs or PyFE4AI's own derivations?

References: BMMS21 = Bermudo Mera, Karmakar, Marc, Soleimanian, PKC 2022 /
ePrint 2021/046 (`references/BMMS21.pdf`); ACFGU18 = Abdalla, Catalano,
Fiore, Gay, Ursu, CRYPTO 2018 (`references/ACFGU18.pdf`).

## Summary

| scheme | verdict | rests on |
| --- | --- | --- |
| `sife/ring_lwe` (selective, Sec. 4) | safe | Sec. 4.1 games; one step (G6) needs `gcd(Delta, q) = 1` or the shift argument below |
| `mife/ring_lwe` (ACFGU18 compiler) | safe | ACFGU18 Sec. 3.3 / Fig. 4 works over any `Z_L` |
| `mcfe/ring_lwe` (identity-based MCFE) | safe | BMMS21 Sec. 6, Thm. 5, App. D.2: additive group `Z_q` only |
| `mcfe/ring_lwe_decentralized` (GDSum) | safe | BMMS21 App. D.4: masks cancel in `Z_q`, any `q` |
| `mcfe/ring_lwe_threshold` (own design) | safe, opt-in (2026-10, section 10) | Shamir over `Z_q` is Shamir over each `F_{q_i}` (CRT) when every `q_i > s` (checked); replicated sharing, flooding, label hash and decoding are independent of the factorisation |
| `sife/fullysec_ring_lwe` (adaptive, Sec. 5) | **not safe without a new argument** | Thm. 2 (Ring-LHL) and App. B are stated and proved for prime `q` |

The prototype (`ring_modulus="rns"`) was first offered for MIFE, MCFE and
dMCFE Ring-LWE only; the threshold MCFE takes it since the re-review of
section 10 (four factors at its 212 to 218 bits). The SIFE key generators
(`sife/ring_lwe`, `sife/fullysec_ring_lwe`) refuse it.

## 1. Hardness of RLWE / mhe-RLWE with a composite `q`

* Concrete security, as PyFE4AI uses it. `RING_LWE_128_LOG_Q` was produced
  with the lattice estimator for LWE with `q = 2^k` (`tools/estimate_ring_lwe.py`),
  i.e. already for a composite modulus: the attacks it models (primal uSVP,
  BDD, hybrids, dual) see `q` only as a number. The 128-bit bound is a bound
  on `log2 q` whatever the factorisation, which is also how the Homomorphic
  Encryption Standard tables are read for the RNS moduli of SEAL / OpenFHE.
  BMMS21's own implementation uses a product of three primes (Sec. 7.1, "CRT
  representation", and the CRT moduli of Table 1) and estimates it the same way.
  Re-checked for the prototype's composite moduli by running the estimator on
  the exact `q` (section 9).
* Asymptotic reductions. BMMS21 cites LPR10 (App. A.3, Theorem 6), stated
  for a prime `q = 1 mod 2n` bounded by `poly(n)`. PyFE4AI's moduli are not
  `poly(n)`-bounded in that theorem's sense either, and its derivation never
  used Theorem 6. Decision RLWE is hard for every modulus by Peikert, Regev,
  Stephens-Davidowitz (STOC 2017, "Pseudorandomness of Ring-LWE for any ring
  and modulus").
* Attacks through the factors. Reducing a sample `(a, a s + e)` mod `q1`
  gives RLWE mod `q1` with the same error, i.e. a smaller modulus-to-noise
  ratio: harder, not easier. The ring `x^n + 1` splits completely mod each
  `q_i` exactly as it does mod the prime `q` used today, so the known
  "weak PLWE instance" conditions (roots of small order) are not met either.
* Theorem 1 (mhe-RLWE from RLWE, Sec. 3.1). The reduction adds `a r' + f'`
  to the RLWE sample and argues with Gaussians over `Z^(n + n l)`
  (Lemma 1); a uniform `b` stays uniform in `R_q` for any `q`. Nothing
  depends on `q` being prime.

## 2. The selective IPFE (Sec. 4, Theorem 3)

Games of Fig. 2:

* G0 -> G1, G2 -> G3, G4 -> G5: algebraic identities in the commutative ring
  `R_q`; any `q`.
* G1 -> G2: mhe-RLWE (section 1); G5 -> G6: RLWE with secret `s*` in two
  samples `(a, a s* + e*)`, `(u, u s* + f*)`: any `q`. No step needs `a` or
  `u` to be a unit.
* G3 -> G4: Lemma 1 on `Z^n` (the columns of `S, E, F`); `q` does not appear.
* G6, the last step: the paper writes `u'' alpha_i + Delta x^b_i =
  Delta((Delta^-1 u'' + b) alpha_i + x^0_i)` with "`Delta^-1` the inverse of
  `Delta = floor(q / K)` in `Z_q`", which needs `gcd(Delta, q) = 1`
  (automatic for a prime `q`). The inverse is not needed:
  `u'' alpha_i + Delta x^b_i = (u'' + b Delta 1_R) alpha_i + Delta x^0_i`, and
  `u'' -> u'' + b Delta 1_R` is a bijection of `R_q` that keeps `u''` uniform;
  `u''` appears nowhere else, so the view does not depend on `b`. With the
  batching of Remark 1 the same shift is applied per batch position `k`
  (`u''_k + b Delta 1^k_R`). The prototype also enforces `gcd(Delta, q) = 1`
  (a failure has probability about `2 / 2^39`; the next prime is taken), so
  the paper's text applies verbatim.
* Sec. 2.1's isomorphism `R_q = Z_q^n` holds for `q = q1 q2` by CRT, since
  `x^n + 1` splits into distinct linear factors mod each `q_i`.
* Correctness (Sec. 4.2, Lemma 2) is a bound on integers: `||noise||_inf <
  floor(q / 2K)`, so `Dec2` (centred lift, rounding to a multiple of
  `Delta`) decodes for any `q` of that size. The Gaussian widths of Sec. 4.2
  do not depend on `q`.

## 3. MIFE compiler (ACFGU18 Sec. 3.3, Fig. 4, Theorem 3; BMMS21 Sec. 6)

ACFGU18 builds the compiler for an IPFE with a linear encoding into a group
of order `L` (Property 1) and linear encryption (Property 2), for any integer
`L`; they instantiate it with an RSA modulus `N`. Here `L = q`:

* the pads `u_i` are uniform in `Z_q^m` (`rand_uniform_vector`, `randrange(q)`):
  a one-time pad in the additive group, perfect for any `q`;
* `E(gamma, noise) = Delta gamma + noise mod q` is linear mod `q`
  (`Delta (gamma mod q) = Delta gamma mod q`), and `Add(Enc(x), x') =
  Enc(x + x' mod q)` holds with identical distributions;
* the `3X` condition of footnote 12 concerns the Gaussian widths, not `q`.

## 4. MCFE compiler (BMMS21 Sec. 6, Theorem 5, App. D.2)

The random oracle is programmed as `H(u'_i, l') = r_{i,l'} - u_i` in
`Z_q^k`: uniform for any `q`. `label_pad` reduces blocks of `bits(q) + 128`
bits mod `q` (`hash_to_zq_vector`): statistically `2^-128`-close to uniform
per coefficient for any `q`, as for the prime. The public `a` is expanded
the same way (`seeded_polynomial`).

## 5. Decentralised MCFE (BMMS21 App. D.4, Theorems 7 and 8)

GDSum masks `M_i = sum_{j > i} PRF(K_ij) - sum_{j < i} PRF(K_ji)` cancel in
`Z_q` and each is uniform when one pairwise key is unknown: additive group
only, any `q`.

## 6. Threshold MCFE (`mcfe/ring_lwe_threshold`, PyFE4AI's design)

* Shamir over `Z_q` with `q = q1 q2`: by CRT `Z_q = F_q1 x F_q2`, and
  uniform polynomial coefficients mod `q` are independent uniform
  coefficients mod each `q_i`, so the sharing is two independent field
  sharings. Reconstruction (`lagrange_coefficient_at_zero`) inverts
  differences `x_m - x_k` of the points `1..s`; privacy of `t - 1` shares
  needs the Vandermonde matrix on `{0} union S` to be invertible: both hold
  when every `q_i > s` (all points and differences are units). The Lagrange
  coefficients `lambda_k` are units, so each `w_k = lambda_k (...) + e_k` is
  distributed per prime as in the field case.
* Flooding: `e_k <- D_{Z, 2^kappa B}` smudges an integer noise of norm `B`;
  the argument and the decoding budget are over the integers.
* So RNS would be sound under `s < min q_i`, but the flooded `q` has 212 bits
  at `ring_n` 8192 (at least four primes below 2^62), and three quarters of
  a partial decryption is the flooding sampler. Not adopted; the key
  generator refuses `ring_modulus="rns"`.

## 7. Adaptive IPFE (`sife/fullysec_ring_lwe`, Sec. 5, App. C, App. B)

* Theorem 2 (Ring-LHL) is stated for "`x^n + 1` splits into `n` linear
  factors modulo a prime `q`"; its proof (App. B, Lemmas 6-9) counts ideals of
  `R_q` and the fraction `2^n q^(-eps n) (q^mk / ((q^m - 1)...(q^m - q^(k-1))))^n`
  of bad matrices over the field. With `q = q1 q2` the ideal lattice and the
  count change (ideals are pairs of ideals mod `q1` and `q2`).
* App. C uses that some element of `a` / `u` is invertible and that `f_{a,u}`
  is not surjective only if `u = s a` for a scalar `s in Z_q`; over `Z_q1 x
  Z_q2`, `u` can be proportional to `a` modulo one prime only, a larger bad
  set (the implementation's `_invertible` / `independent_at_every_slot`
  checks are written for the field case).
* The parameters (`sigma' ~ q^(2/m)`) also depend on `q`.
* A composite-modulus Ring-LHL would be a new result (or a reduction to a
  regularity lemma for composite moduli whose hypotheses would have to be
  checked against App. C). Until then: prime `q` only.

## 8. PyFE4AI's derivations and checks

* `bmms21_selective_parameters` derives the lower bound
  `L = 2 p (noise + 1) + 1` as before; the RNS modulus is
  `q = q1 q2 >= L` with `q1` the largest NTT prime below `sqrt(L)`, `q2` the
  smallest NTT prime with `q1 q2 >= L` (so `q1 < q2`, both `= 1 mod 2n`, both
  below 2^62), then `q2` advanced until `gcd(floor(q / p), q) = 1`. `q`
  exceeds the prime of the default path by a factor below `1 + 2^-16`
  (`1 + 2^-25` at 80 bits) and has the same bit length in every
  configuration tested (`ring_n` 2048 to 8192, 61 to 88 bits). The 128-bit
  check (`check_ring_lwe`, `ring_lwe_auto_n`) reads the same table with this
  `q`; `standard` picks the same ring dimension.
* NTT root. `_find_primitive_root` searched `g^((q - 1) / 2n)` and accepted
  it if `root^(2n) = 1` and `root^n != 1`. For a prime this means
  `root^n = -1`; for a composite `q` it does not (a root `= 1` mod one prime
  passes), and `g^((q - 1) / 2n)` rarely has order dividing `2n`. The check is
  now `root^n = -1 mod q` (the same roots for every prime `q`), and for an
  RNS modulus the root is the CRT of the per-prime roots (registered from
  `q_factors`, checked: product `= q`, each factor prime and `= 1 mod 2n`).
  `root^n = -1` mod each `q_i` makes `root^2` of order `n` mod each prime, so
  the negacyclic NTT is a ring isomorphism mod `q`.
* Parameter cache: `_param_verification` recomputes `q` and its factors from
  the configuration and compares them; the record carries
  `"ring_modulus": "rns"` and `q_factors`, absent for the default (whose
  records are unchanged), and is kept in `param-rns.json` beside the
  prime's `param.json`. A prime record never verifies for an RNS
  configuration and vice versa; a planted cache cannot choose the factors.
  The factors are public (there is no trapdoor in them; 40-bit factors of an
  80-bit `q` are found instantly anyway).
* Formats: ciphertexts stay in the standard representation (coefficients in
  `[0, q)`): binary format 1 (`wb = ceil(bits(q) / 8)`) and the JSON
  decimal strings are unchanged; the native code converts residues at the
  boundary (Garner). The public parameters gain `q_factors` (RNS only).

## 9. 128-bit check of the composite moduli

Lattice estimator (commit 53da598, `third_party/lattice-estimator`), the
attacks and settings of `tools/estimate_ring_lwe.py` (MATZOV cost model,
primal uSVP / BDD / BDD hybrid, dual / dual hybrid; secret and errors
`D_{Z, sigma}`, unlimited samples, `jobs=1`), run on the exact moduli of the
FL configurations (3 or 10 clients, `eta = 1`, `bound_y = 1`; `sigma` the
RLWE width of Sec. 4.2 that the table is read at):

| `ring_n` | clients, `bound_x` | `sigma` | `q` bits (factors) | table bound | log2 cost, RNS `q` | log2 cost, prime `q` |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 4096 | 3, 100 | 8.22 | 80 (40 + 40) | 111 | 189.4 | 189.4 |
| 4096 | 3, 2 | 8.22 | 63 (32 + 32) | 111 | 251.8 | 251.8 |
| 4096 | 10, 100 | 8.22 | 86 (43 + 43) | 111 | 174.2 | 174.2 |
| 8192 | 3, 100 | 8.27 | 82 (41 + 41) | 218 | 407.4 | 407.4 |

The estimator gives the composite modulus and the prime of the same size
the same cost (it reads `q` only through its size), all far above 128 bits.

## 10. Threshold MCFE re-review (2026-10)

Since section 6 the default flooding is uniform (integer, native) and the
default sharing for `t < s` is replicated (multipliers 0 and 1); the flooded
`q` (212 to 218 bits under `standard`) is now the product of
`k = max(2, ceil(bits / 60))` NTT primes, but `k = 4` (factors up to 62 bits)
up to 248 bits, the most the native ring holds, with a fifth only where the
last would exceed 62 bits (`rns_ntt_modulus_for`, then
`rns_ntt_modulus(..., factors=k)`:
`k - 1` largest NTT primes below `L^(1/k)`, the last the smallest with a
product `>= L` that is none of them, advanced until `gcd(floor(q / p), q) = 1`;
for `k = 2` the modulus of the earlier sections). Above 248 bits the modulus
has five or more factors and runs on the Python path (the key generator
warns). Item by item:

* **RLWE hardness, MCFE layer**: sections 1, 3 and 4 (any modulus; the
  estimator reads `q` by size, and `q` has the bit length of the prime).
* **Shamir sharing** (`t = s`, the default there): CRT, one field sharing per
  factor; the points `1..s` and their differences are units when every
  `q_i > s`, which the key generator and every server check
  (`check_rns_servers`; the factors are about `2^54`). The Lagrange
  coefficients are units.
* **The Lagrange leak** (`docs/security.rst`, threshold layers) concerns the
  rational multiplier `M` of the decryption noise on the integers: 0 or 1 for
  the replicated sharing and 1 for Shamir with `t = s`, whatever the
  factorisation of `q`.
* **Replicated and seeded sharing**: additive in `Z_q` (pieces uniform mod
  `q`; seeded pieces `hash_to_zq_vector` mod `q`, within `2^-130` of uniform
  per value for any `q`).
* **Flooding**: the smudging bound (uniform: `|nu| / 2^(tau + 1)` per value;
  Gaussian: the statistical distance of `D_{Z, sigma_f}` and its shift by `nu`) is a statement
  about integers; reducing mod `q` is a function of the sum and cannot
  increase the statistical distance. The decoding budget
  `s 2^tau + B < q / (2p)` uses the size of `q` only.
* **Label hash**: `"shake128-rejection"` samples exactly uniformly on
  `[0, q)` for any `q`; `"shake256-wide"` within `2^-128` per value.
* **Binding and noise derivation**: `q` is in `pp_binding` (and with it the
  noise context); `q_factors` is added where present, so key shares of a
  prime `q` and of an RNS `q` never serve each other's public parameters,
  and the digests of prime public parameters are unchanged. Construction ids
  are unchanged (the sharing and flooding constructions are the same); the
  parameter records carry `ring_modulus` and `q_factors` and are recomputed
  and compared on load; RNS records live in `param-rns.json`, beside the
  prime's `param.json` (alternating the configurations in one folder loads
  both).
* **FACCT and the Gaussian flooding sampler**: integer samples, independent
  of `q`; the native kernels reduce them mod each factor.
* **NTT root**: the CRT of the factors' roots with `root^n = -1` mod every
  factor (section 8); the native ring runs one 64-bit NTT per factor and
  returns standard values, so every result equals the Python code on the
  composite `q` (tests/test_threshold_rns.py compares NTTs, products,
  encryption, key shares and partial decryptions).

Measured (standard profile, 3 clients, 10^4 values, `s = t = 3`, 1 thread,
host): NTT of 8192 coefficients 1.33 ms (256-bit words) -> 0.59 ms (four
64-bit primes, conversions included), product 4.19 -> 1.47 ms; partial
decryption with binary ciphertexts 19 -> 11 ms, JSON 57 -> 45 ms; client 30
-> 26.5 ms.
