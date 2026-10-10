# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor bumps may break
compatibility).

## [Unreleased]

Privacy fixes: every scheme that was insecure in v0.2.0 now implements a
published construction (four threshold variants add an *experimental*
threshold layer of our own on top of one), and every scheme carries a
review status. **Keys, ciphertexts and partial decryptions of the affected
schemes are not compatible with v0.2.0.** Read the README *Scheme Status*
table and *Known Issues* before relying on any privacy property.

### Security

- **MCFE-Paillier** implements ABMPSW20 (Abdalla et al., SCN 2020) Fig. 3:
  `c = (1 + N)^x H(l)^s mod N^2` with a full-domain hash into `Z_{N^2}^*`,
  `s ~ D_{Z,sigma}` with `sigma > sqrt(kappa + 2n log2(2X)) N^{5/2}`, and the
  single aggregated key `sum y s`. Previously a single ciphertext revealed a
  client's `<x_i, y_i>` to the key holder, and keys were brute-forceable.
- **MCFE-LWE / MCFE-FullySec-LWE** implement ABMPSW20 Fig. 4
  (`c = round_q0(Z . H(l) + floor(q/K) x)`, aggregated key `sum y Z`) with
  parameters derived from the Appendix C.3 constraints (selective and
  adaptive choice of `m0` respectively). Previously the key holder learned
  `x_i mod l` and `x_i` up to a few candidates.
- **MCFE-Damgard** implements ABMPSW20 Fig. 2 (`D_k`-MDDH, `k = mddh_k`,
  default 1, where it coincides with CDGPP18). Previously two ciphertexts of
  one client under two labels revealed its plaintexts.
- **tMCFE-DDH** is CDGPP18 with the aggregated key Shamir-shared over `Z_q`;
  any `t` servers interpolate `[H(l)]^d` in the exponent. Previously the
  combiner could strip each client's MD5-scalar label mask.
- **tMCFE-LWE** is ABMPSW20 Fig. 4 with a Shamir-shared key and
  noise-flooded partial decryptions (each is an LWE sample under the
  server's share instead of an exact linear function of it). The noise is
  derived from a per-server secret and `(label, enrolled set)`, so repeated
  requests cannot be averaged. The threshold layer is our own, so the
  scheme's status is *experimental*.
- **Paillier parameter caches** (SIFE, MIFE, MCFE, dMCFE) no longer store
  the primes `p, q`: the factorization decrypts SIFE ciphertexts directly
  and MCFE ciphertexts from two labels. Caches written by earlier versions
  are regenerated with a fresh `N`.
- **Labels**: ndarray element labels were plain JSON strings, so a chosen
  user label could reproduce an element's label (two ciphertexts under one
  label leak `x - x'`); user and element labels are now hashed in disjoint
  domains (`pyfe4ai.schemes.mcfe.labels`).
- Threshold key shares record a digest of their fusion weights and servers
  refuse ciphertexts presented under other weights.
- **SIFE-Paillier** samples keys from `D_{Z,sigma}`, `sigma > sqrt(kappa)
  N^{5/2}` (ALS16); keys used to be drawn from a range small enough to
  brute-force from the public key.
- **MIFE** Damgard, Paillier, LWE, FullySec-LWE and Ring-LWE follow
  ACFGU18 (CRYPTO 2018) Fig. 4: each slot encrypts `x_i + u_i mod L` under
  its single-input IPFE with a one-time pad `u_i`, the functional key adds
  `z = sum <u_i, y_i>`, and decryption decodes only the sum (ALS16 Figs. 7, 9
  and 8, ABDP15, and BMMS21 Sec. 6 as the per-slot IPFEs). Previously the
  key holder learned every client's `<x_i, y_i>`. MIFE-Paillier keys are
  Gaussian (they used to be brute-forceable).
- **tMIFE-LWE** Shamir-shares the padded MIFE-LWE key and floods each
  partial decryption with `D_{Z, 2^kappa B}` (`B` bounds the decryption
  noise; `q` enlarged), derandomised per request (*experimental*).
- **dMCFE-DDH, dMCFE-LWE, dMCFE-Paillier** follow ABMPSW20 App. D.1, Fig. 14
  (DMCFE from MCFE): encryption and decryption are those of the reviewed
  MCFE, and each client publishes an MCFE2 (DCR, Fig. 3) encryption of its
  key contribution under the label `H(y)`; zero-sum MCFE2 keys make the
  shares combine to exactly the MCFE1 functional key. Previously the
  combiner could strip each client's label mask, and dMCFE-Paillier keys
  were brute-forceable.
- **MCFE-Ring-LWE** implements BMMS21 (PKC 2022) Sec. 6, the
  identity-based MCFE-compiler: clients encrypt `x_i + u_i + H(u'_i, l)` and
  the key for `(y, l)` carries the label's pad term, so keys are bound to a
  label. **dMCFE-Ring-LWE** follows BMMS21's identity-based DMCFE (ABKW19 +
  GDSum, realised with pairwise PRF zero-sum masks). **tMCFE-Ring-LWE**
  Shamir-shares the identity-based key and floods partial decryptions
  (*experimental*). Previously all three masked inputs with a small public
  label scalar.
- **FH-MCFE** implements AGT21b (TCC 2021) Sec. 6.2 as recapped in NPS24
  (IACR CiC 2024) Sec. 2: one function-hiding IPFE per client
  (`pyfe4ai.schemes.mcfe.fh_ipe`, ACFGU18 Fig. 12 with `n = 1` and LV16
  padding), ciphertexts carry `H1(tag)` and every functional key embeds a
  fresh zero sharing. **dMCFE-FH-Multi-IPE** implements NPS24 Fig. 1
  (selective FH-DMCFE): the zero sharing lives in the encryption keys and
  each client derives its own key with `H2(tag-f)`; `tag-f` is bound to the
  whole weight vector and keys are derived deterministically, so a client
  issues one key per `tag-f`. **tMCFE-FH-Multi-IPE** Shamir-shares the
  FH-MCFE key exponents; partial decryptions are combined in `G_T`
  (*experimental*). Previously all three masked inputs with a public label
  scalar and ran over the symmetric group `SS512`, where SXDH fails; the
  family now defaults to `MNT224` and rejects symmetric groups.
- **All pairing-based schemes** (SIFE FH-IPE and partial FH-IPE, MIFE
  FH-IPE and FH-Multi-IPE, quadratic Quad, SGP and MI-SGP, the FH-MCFE
  family) default to the asymmetric group `MNT224` instead of the symmetric
  `SS512` and reject symmetric groups: their constructions assume SXDH or a
  generic asymmetric bilinear group, and DDH is easy in a symmetric one.
- **Review of the remaining schemes** against their papers:
  - SIFE-DDH, SIFE-DDH-Dynamic and MIFE-DDH published ciphertext components
    as the integer product of two residues mod `p`; trial division (or
    factoring) recovered the plaintext from the ciphertext alone. Every
    element is now reduced mod `p`, and keys mod `q` (a key with a large
    weight used to reveal the master secret / the MIFE pads).
  - SIFE partial-FH-IPE (Gay20 Fig. 8) handed the master secret key
    `(b, U, V)` to every encryptor, so any client decrypted everything and
    recovered every key's weights. `get_private_keys` now returns `None` and
    `Enc'` is run by the key generator (`sec_encrypt`). The default subspace
    dimension equalled `eta` (a square `M` reveals `V`); it is now
    `min(3, eta - 1)` and `M` is resampled until it has full rank.
  - SIFE FH-IPE (KLMMRW18) had no privacy for `eta = 1` (`B* = 1`) and let
    anyone recognise zero vectors; vectors are now padded to `(x, 1, 0)`,
    `(y, 0, 1)`.
  - MIFE FH-IPE ran independent per-slot instances and decoded every slot,
    revealing each `<x_i, y_i>` and allowing slot keys of different
    functional keys to be recombined; it now implements ACFGU18 Fig. 12.
  - MIFE in federated learning: MIFE has no labels, so one functional key
    combines any ciphertexts of a client and reveals differences between
    its values. `encrypt_gradient` / `aggregate_gradients` refuse MIFE
    schemes (use MCFE).
  - tMIFE-DDH is rebuilt on MIFE-DDH with the functional key Shamir-shared
    mod `q` (as tMCFE-DDH); key shares are bound to their weights.
  - Quadratic Quad now implements Gay20 Fig. 4 (it encrypted `x (x) y` with
    an IPFE before): ciphertexts of `4n + 2m + 4` G1 and `m` G2 elements.
  - Gaussian samples wider than `2^40` (FullySec-LWE keys and noise) were
    rounded doubles whose low bits were all zero; they now use the
    arbitrary-precision sampler.
  - charm serialises the identity of G1 into a point outside the group;
    `pyfe4ai.utils.pairing_backend.PairingGroup` writes it explicitly.
- **Independent verification** of the rebuilt schemes found:
  - Function-hiding keys (FH-MCFE, FH-DMCFE, tMCFE-FH, MIFE FH-IPE)
    carried `weights_id`, an unsalted SHA-256 of `y`: any key holder could
    brute-force `y` and link keys for the same function. Keys now carry a
    random `key_id` (FH-DMCFE shares only `tag_f`); decryption no longer
    needs or checks `y`, and threshold servers do not need it either.
    Clients receive their slot key without the outer `A'` (only key
    generation uses it).
  - SIFE partial-FH-IPE keys reveal `M^T y`, which for small
    `eta - subspace_dim` usually determines a bounded integer `y`; the
    default `subspace_dim` is now 1 and the leakage is documented.
  - tMIFE-DDH applied key shares to unchecked ciphertext elements: an
    element outside the order-q subgroup leaked one bit of a server's share
    through the Legendre symbol. Servers (and MIFE-DDH decryption) now
    require subgroup elements. All threshold servers refuse a key share
    issued to another server.
- **DOT18 and DSGPP18 reviews:**
  - Quadratic SGP handed the master secret key `(s, t)` to every encryptor;
    as in DSGPP18 Fig. 2 the public key is now `(g1^s, g2^t)`, encryption
    uses it only and `get_private_keys` returns `None`. Floats were
    truncated silently; inputs are validated.
  - MIFE FH-Multi-IPE (DOT18 Sec. 3.1) produced deterministic keys for
    `n = 1` with `k = 1`; it now requires `n >= 2`, `k >= 1` and
    `len(lst_nid) == n`, checks that every slot is present, and its
    decryption helpers no longer ask for `y`.
  - Quadratic MI-SGP had no published basis (per-client copies of SGP): the
    key holder learned every client's `x_i^T F_i y_i` and could mix sub-keys.
    It is replaced by `quadratic/miqfe`, the MIFE for quadratic functions of
    Agrawal, Goyal, Tomida (TCC 2022, Sec. 5): per-user encryption keys made
    of master ciphertexts (a corrupted user learns its own key, not the master
    secret), function
    class `sum_{i<k} c[(i,j,k,l)] x_i[j] x_k[l]` (cross-user terms), built from
    the TAO20 FH-IPFE and the paper's SK-MCFE schemes (Figs. 1 and 9).

### Added

- `security_profile` config key for every key generator
  (`pyfe4ai.schemes.profiles`). `"toy"` (default) leaves configs unchanged,
  so all defaults stay as they were. `"standard"` presets 128-bit-oriented
  sizes: a 3072-bit DL group (RFC 3526 MODP group 15), a Paillier `N` of
  about 3072 bits (`bit_length = 1536`), Ring-LWE `kappa = 128` with `ring_n = "auto"`
  (smallest dimension whose derived `q` meets the Homomorphic Encryption
  Standard 128-bit bound, re-checked after derivation, flooding included)
  and the BN254 pairing group (about 100-bit, with a `FESecurityWarning`).
  The ABMPSW20 MCFE-LWE family (`mcfe/lwe`, `fullysec_lwe`, `lwe_threshold`,
  `lwe_decentralized`) gets `lwe_n = "auto"`: the smallest dimension whose
  derived `q` meets `lwe_utils.DENSE_LWE_128_LOG_Q`, a lattice-estimator
  table for LWE(n0, q, ceil(2 sqrt(n0))) with a uniform secret (the
  assumption of the scheme's proof), regenerated by
  `tools/estimate_dense_lwe.py`; under `standard` its decentralised variant
  combines key shares under a DCR modulus of at least 3072 bits.
  Explicit values below the profile, and the schemes without a usable
  secure parameter set (LWE with a public matrix: SIFE / MIFE `lwe`,
  `fullysec_lwe`, MIFE `lwe_threshold`; dMCFE-Paillier), raise
  `FEValidationError`.
- `lwe_n = "auto"` for the MCFE-LWE family; `lwe_utils.DENSE_LWE_128_LOG_Q`,
  `check_dense_lwe`, `dense_lwe_n`, `resolve_lwe_n`;
  `profiles.STANDARD_DCR_PRIME_BITS`; `modular_utils.pow_signed_many`
  (fixed-base exponentiation of one base to many exponents).
- `ring_n = "auto"`; `ring_lwe_utils.HE_STANDARD_128_LOG_Q`,
  `check_he_standard`, `he_standard_ring_n`, `resolve_ring_n`;
  `crypto_utils.RFC3526_PRIMES`.
- `pyfe4ai.utils.validation.int_vector`: scheme inputs must be integers
  (integral floats accepted); non-integral values raise
  `FEValidationError` instead of being truncated.
- `pyfe4ai.utils.ring_lwe_utils.bmms21_parameters`, `check_ring_n` and
  `DEFAULT_RING_N`; `dlog_bound` crypto option for SIFE-DDH and MIFE-DDH;
  `FHIPE.keygen(..., r=...)`.
- `pyfe4ai.schemes.status` (`SchemeStatus`, `get_scheme_status`, exported
  from `pyfe4ai`): status (`reviewed` / `experimental` / `unreviewed` /
  `known-insecure`), fidelity, reference and issue for all registered
  schemes; key generators and crypto objects of known-insecure schemes emit
  `pyfe4ai.FESecurityWarning`; `FESchemeWrapper.status`;
  `markdown_table()` renders the README table, which a test keeps in sync.
- `dlog_bound` crypto option for MCFE-DDH and tMCFE-DDH (results beyond the
  default `10^(precision + 2)` decrypted to `None`).
- `XofRandom` (SHAKE-256 bit source for derandomised sampling) and an `rng`
  argument for `discrete_gaussian_bigint`.
- `discrete_gaussian_bigint` (arbitrary-width rounded Gaussian),
  `hash_to_zn2_star`, `hash_to_zq_vector`, `abmpsw20_lwe_parameters`,
  `round_to_q0`, `decode_rounded_inner_product`.
- Config keys `stat_param` (ABMPSW20 schemes, SIFE-Paillier; default 64) and
  `mddh_k` (MCFE-Damgard).
- `pyfe4ai.utils.pairing_backend.check_asymmetric` (detects symmetric
  groups by pairing two `G1` elements) and `DEFAULT_PAIRING_GROUP`.
- `pyfe4ai.schemes.mcfe.fh_ipe` (`FHIPE`),
  `pyfe4ai.schemes.mcfe.decentralize` (MCFE2 and zero-sum keys for
  Fig. 14), `labels.chunk_label`, `MCFERingLWE.key_chunks` (the ML adapter
  passes the resulting `n_chunks` when requesting Ring-LWE keys).
- `tag_f` argument of the dMCFE-FH-Multi-IPE key derivation.

### Changed (breaking)

- MCFE-Paillier: `sk = {s}`, `dk = {k}`, `ct = {c}`. MCFE-LWE /
  FullySec-LWE: `sk = {z}`, `dk = {sk_y}`, `ct = {c}`; the config keys
  `bound_u` and `label_modulus` are ignored and parameters are no longer
  cached in `./config`. MCFE-Damgard: `sk = {s}`, `dk = {d}`, `ct = {c}`, no
  `h` in the public parameters.
- tMCFE-DDH: `sk = {s}`, `dk = {d, sid, lst_sid}` (label-independent),
  `ct = {c, label}`, partial decryption `{c_y, u_d, label, sid,
  lst_sid_enrolled}`; key shares also carry `weights_id`. tMCFE-LWE:
  `dk = {sk_y, sid, lst_sid, weights_id, noise_key}`, partial decryption
  `{w, c_y, label, sid, lst_sid_enrolled}`, validated like the DDH threshold
  scheme. The `label` fields hold the encoded label.
- The ndarray helpers of all rebuilt MCFE schemes encrypt each element
  under its own `ElementLabel` (one ciphertext per slot and label); labels
  are hashed through `encode_label`, which also changes MCFE-DDH
  ciphertexts relative to v0.2.0.
- Decrypting under a wrong label returns `None` for MCFE-Paillier and
  MCFE-Damgard and an arbitrary in-range value for the LWE schemes.
- MIFE Damgard / Paillier / LWE / FullySec-LWE / Ring-LWE: client keys
  carry the pad `u`; functional keys carry `z` (`{k1, k2, z}`, `{k, z}`,
  `{sk_y, z}`); decryption needs a ciphertext for every slot of the key.
  tMIFE-LWE key shares carry `noise_key`.
- dMCFE-DDH / -LWE / -Paillier: key shares are MCFE2 ciphertexts
  `{c, nid, weights_id}`, every client must contribute (with zero weights
  if absent), and the combined key is the MCFE1 key.
- MCFE / dMCFE / tMCFE-Ring-LWE: functional keys are bound to a label
  (`credentials` need `label`, optionally `n_chunks`;
  `derive_function_decryption_key_share(fusion_weight, label, n_chunks)`);
  ndarray ciphertexts pack values into chunks of `ring_n` coefficients.
- FH-MCFE family: client keys `{imsk}` (dMCFE adds `t_tilde`, `seed`),
  functional keys `{dk: {nid: {k, z}}, weights_id}`; tMCFE key shares
  `{k, z, sid, lst_sid, weights_id}` and partial decryptions
  `{w, ct_digest, sid, lst_sid_enrolled}`. The config keys `sec_level`,
  `u_bound` and `label_modulus` are ignored and parameters are no longer
  cached.
- Decryption keys of the SIFE schemes (DDH, LWE, Damgard, FullySec-LWE,
  Ring-LWE) carry their weights `y`, and decryption raises
  `FESchemeError` for other weights; MIFE/FH keys carry `weights_id`.
  Ciphertext lengths are checked.
- SIFE-FullySec-LWE: `K = 2 eta X Y + 1`, encoding `floor(q / K) x`
  (caches carry a construction tag and are regenerated).
- SIFE-Ring-LWE / MIFE / MCFE: `ring_n` defaults to 32 (it was claimed to be
  derived but always was 32) and must be a power of two.
- MIFE-DDH: client keys `{h, u}` (no secret slot key), `ct = {t, c}`, keys
  must cover every slot. tMIFE-DDH: shares `{d, z, sid, lst_sid,
  weights_id}`, partial decryptions `{c_y, w, ct_digest, sid,
  lst_sid_enrolled}`. MIFE FH-IPE: `sk = {imsk}`, `ct = {ct}`,
  `dk = {dk, z, weights_id}`, no parameter cache. Quad: `ct = {ct_x, ct_y,
  ct}`, `dk = {f, k}`, no parameter cache.
- SIFE FH-IPE ciphertexts and keys have `eta + 2` components.
- Pairing-based schemes: `pairing_group_param` defaults to `MNT224`;
  `SS512` and `SS1024` raise `FEValidationError`. Parameters, keys and
  ciphertexts created over `SS512` cannot be reused (cached parameters are
  regenerated).

### Changed

- `hash_to_zq_vector` cuts one SHAKE-256 stream instead of calling SHAKE-256
  once per coordinate (hashing a label of MCFE-LWE at n0 = 8192 drops from
  about 12 ms to 3.6 ms). Its outputs change, so ciphertexts of `mcfe/lwe`,
  `fullysec_lwe`, `lwe_threshold`, `lwe_decentralized`, `mcfe/ring_lwe` and
  `ring_lwe_decentralized` are not compatible with earlier versions.
- MCFE2 key shares of the decentralised MCFE schemes (`ddh_decentralized`,
  `lwe_decentralized`, `paillier_decentralized`) raise one label hash to all
  key coordinates with a fixed-base method (about 4 times faster at 3072
  bits; same outputs).
- Safe primes are generated by sieving candidates `q` and `2q + 1` against
  the primes below 2^16 before the primality tests: a 1536-bit safe prime
  takes seconds instead of tens of seconds (all Paillier setups and the
  dMCFE modulus benefit). The tests themselves are unchanged.
- `group_generator_fe` (DL schemes) returns the RFC 3526 group for 2048- and
  3072-bit requests instead of generating a fresh safe prime; smaller
  sizes, including the default, are generated as before.

### Removed

- `quadratic/multi_input_sgp` (replaced by `quadratic/miqfe`, see above).
- `mife/ddh_hybrid_alpha` (HybridAlpha): it duplicated MIFE-DDH without the
  paper's inference-prevention filter, and the published design leaks
  individual updates (MIFE without labels; a stateless filter that allows
  keys for `S` and `S` minus one client).
- Unused modules `schemes/lwe_base.py`, `schemes/ring_lwe_base.py`,
  `schemes/paillier_base.py` and the helpers `label_scalar_from_hash`,
  `decode_fullysec_lwe_inner_product`.

### Fixed

- SIFE-FullySec-LWE decoded `+-eta X Y` with a random sign (`K` was
  `2 eta X Y`). SIFE-Damgard could not decrypt in an instance holding the
  encryption key. SIFE-LWE's ndarray helper skipped the `bound_x` check.
  The MIFE Ring-LWE key generator overwrote the SIFE parameter cache.
- Ring-LWE plaintext modulus (SIFE, MIFE, MCFE): `p` was exactly
  `2 l X Y`, so inner products of `+-l X Y` were ambiguous; it is now
  `2 l X Y + 2` (parameter caches are regenerated).

### Documentation

- README: *Scheme Status* table, rewritten *Known Issues*, the scheme table
  cites the implemented constructions (ACFGU18 Fig. 4, ABMPSW20, BMMS21
  Sec. 6, AGT21b, NPS24, ACFGU18 Fig. 12, Gay20 Fig. 4); the ALS16 link
  pointed to a different paper (ePrint 2016/011, now 2015/608); landing
  page matrix updated; the docs index no longer claims a uniform
  `setup / keygen / encrypt / decrypt` API.

## [0.2.0] — 2026-10-07

Correctness and security fixes. **Not compatible with v0.1.0 keys or
ciphertexts for MCFE-DDH and the DDH threshold schemes; ciphertexts of the
label-based LWE / Ring-LWE / FH-Multi-IPE MCFE schemes produced by v0.1.0 do
not decrypt correctly with v0.2.0 (label scalar mapping changed).** Several multi-client
schemes still have known privacy issues — read
[Known Issues / Security Notice](README.md#known-issues--security-notice)
before using them.

### Security

- **MCFE-DDH** (`pyfe4ai.MCFE`) re-implemented following CDGPP18: the label is
  hashed to a vector of group elements (SHAKE-256 into the quadratic-residue
  subgroup), every client slot has an independent key, and the functional key
  is the aggregate `d = Σ y_i · s_i`. Previously the label entered as a public
  MD5 scalar and keys carried per-client components, so a functional-key holder
  could recover an individual client's plaintexts. Because CDGPP18 ciphertexts
  are deterministic per label, the ndarray helpers derive a distinct label per
  element.
- **MIFE-DDH**: every input slot now has its own IPFE master key; previously all
  clients shared `w` / `g_a` and one client could unmask another's ciphertext.
- **DDH threshold** (tMIFE-DDH, tMCFE-DDH): Shamir share points start at 1 and
  sharing is done over the group order (point 0 used to be the secret itself,
  so server `sid_0` alone could decrypt); share and combine require at least
  `t` shares from the enrolled set, reject duplicate or unknown partial
  decryptions, and validate `1 ≤ t ≤ s`.
- Discrete-log tables for integer groups are built in memory and no longer
  loaded from disk (the pickle-based cache in the working directory could run
  arbitrary code).
- LWE-family label scalars never map to 0 (about 1 in `label_modulus` labels,
  12.5% with the default of 8, used to remove the mask entirely). The structural leakage of LWE label masking remains; see
  Known Issues.

### Fixed

- MCFE-DDH decryption for `eta ≥ 2` (returned `None`).
- DDH threshold Lagrange coefficients are computed modulo the group order
  instead of with float division and truncation (enrolled subsets with
  non-integer coefficients, e.g. `{sid_1, sid_3}`, failed); `setup` no longer raises `IndexError` and combining no longer
  hard-codes divisors for `eta ≥ 2`.
- `get_decryption_keys` of the five decentralized MCFE schemes raises
  `NotImplementedError` instead of returning the exception class; the ML
  adapter reports unsupported threshold/decentralized variants clearly.
- Pairing and quadratic test modules skip cleanly when `charm-crypto` is not
  installed.

### Changed (breaking)

- MCFE-DDH formats: `sk = {"s"}`, `dk = {"d": [d0, d1]}`, `ct = {"c"}`.
- MCFE-DDH ndarray helpers use per-element derived labels
  (`json.dumps([label, layer, index])`).
- MCFE-DDH `get_decryption_keys` raises `FEValidationError` on an unknown
  client id instead of silently skipping it; `decrypt` validates the length of
  `d`, the ciphertext range, and that ciphertext and weight lengths match.
- Threshold `L()` takes a `modulus` argument; partial decryptions carry `sid`
  and `lst_sid_enrolled`, which `combine_decrypt` validates.
- LWE label scalar mapping changed from `h mod M` to `(h mod (M − 1)) + 1`
  (centered, never 0), so v0.1.0 ciphertexts of the label-based LWE /
  Ring-LWE / FH-Multi-IPE MCFE schemes no longer decrypt; keys are unaffected.
  `label_modulus < 2` raises `FEValidationError`.
- `dlog_solver.load_or_build_dlog_table` ignores its `filepath` argument:
  integer-group tables are no longer read from or written to `./config/crypto`,
  and existing cache files there are ignored.

### Added

- `benchmarks/fl_path_coverage.py`: enumerates federated-learning path support
  over all 39 registry entries (`--output json` available).
- Regression tests for `eta ≥ 2`, `t < s` subsets that exclude `sid_0`,
  single-server rejection, duplicate shares, independent client keys, label
  binding, and the dlog cache.
- README *Known Issues / Security Notice*.

### Documentation

- `sec_param` is documented as a modulus bit length, not a security level.
- README bibliography: TAPFed DOI, venues for Gay20, DOT18 and the Ring-LWE
  construction, complete list of pairing-dependent modules.
- Landing page aligned with the Known Issues.

## [0.1.0] — 2026-08-05

Initial public release.

[Unreleased]: https://github.com/spire-studio/pyfe4ai/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/spire-studio/pyfe4ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/spire-studio/pyfe4ai/releases/tag/v0.1.0
