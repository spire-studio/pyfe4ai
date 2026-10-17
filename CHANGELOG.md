# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor bumps may break
compatibility).

## [Unreleased]

### Security

- **MIFE-Ring-LWE** and the **Ring-LWE MCFE** schemes (`mcfe/ring_lwe` and
  its threshold and decentralized variants) now take the Gaussian widths of
  BMMS21 Sec. 4.2 for messages of norm `3X`: the MIFE compiler of ACFGU18
  (Fig. 4, footnote 12) needs an IPFE that is secure for that bound, since
  the hybrids of its Lemma 3 encrypt `x^j - x^1 + x'^1` in one slot. Up to
  v0.3.0 the widths were sized for `X` and dimension `n * eta`, which meets
  the condition only for nine or more clients (with three clients `sigma1`
  was 0.58 of the required width). No attack is known. The plaintext modulus
  `p` is still sized for `X`; `q` grows by 3 to 4 bits, and the `standard`
  profile keeps the same ring dimension for the benchmark configurations.
  Cached parameters of these schemes are re-derived (construction
  `acfgu18-bmms21/v5`).

### Performance

- **Native backend** in Rust (`pyfe4ai._rust_native`, PyO3; sources in
  `rust/`, `rust/core` has no dependencies), built by `pip install` when
  cargo is available or by `tools/build_rust.py`.
  `pyfe4ai.utils.native_backend` selects `auto` / `python` / `native`
  (`PYFE4AI_BACKEND`) and threads (`PYFE4AI_THREADS`): Ring-LWE MIFE / MCFE
  encryption, decryption and key derivation of a whole tensor per call, with
  the GIL released and optional threads; dense-LWE MCFE label hashing and
  dot products; ristretto255; BLS12-381 (below). Same outputs as the Python code
  (checked bit for bit by the tests); Python fallback for unsupported
  parameters.
- **Sampling randomness of the native backend**: each tensor call expands one
  32-byte OS seed with ChaCha20, one stream per chunk, and wipes seed and
  states at its end (no state across calls); before, every sample byte was
  read from the OS. Same distribution. ChaCha20 is not an SP 800-90A DRBG
  (relevant only to FIPS validation). The Python code still reads the OS
  for every byte. Faster on macOS (0.4-0.6 ms per 10^4
  elements, `getentropy`), unchanged on Linux (`getrandom` is cheap).
- Ring-LWE in Python: `NTT(a)`, `NTT(pk)` and `NTT(sk_y)` cached, `r`
  transformed once, decryption products summed in the NTT domain (same
  outputs). Dense-LWE MCFE: keys parsed once per object.
- MCFE Ring-LWE 10^4-element FL round (standard profile, 3 clients): client
  230 -> about 3.5 ms, aggregator 368 -> about 9.5 ms in a single thread
  (Linux, binary ciphertexts); see `benchmarks/PERFORMANCE.md`.
- **Threshold Ring-LWE MCFE natively under `standard`**: 256-bit words in
  `rust/core` for moduli up to 2^253 (the flooded `q` has 212 bits at
  `ring_n` 8192, beyond the 125-bit words), so its encryption and key
  derivation run natively; so do the Shamir sharing of the key, the parsing,
  digests and NTT-domain products of a partial decryption, and the
  combination, with the values of the Python code (Shamir coefficients drawn
  as `SystemRandom.randrange(0, q)` draws them, from the per-call DRBG). The
  flooding noise is still `discrete_gaussian_bigint`, drawn in one call per
  polynomial (same draws); ciphertext tags are serialised once for all
  servers (same tags). 10^4 elements, 3 clients, `s = 3`, `t = 2` (host):
  client 213.5 -> 19.6 ms, key shares 276 -> 41 ms, partial decryption 528 ->
  211 ms per server (three quarters of it the flooding sampler), combination
  18.0 -> 4.3 ms.
- **Pairings on BLS12-381 in Rust** (arkworks; `pyfe4ai.utils.bls12_381`
  behind charm's `PairingGroup` interface): a pairing takes 0.9 ms (charm:
  MNT224 2.4 ms, BN254 13.7 ms), G2 exponentiation 0.4 ms (3.2 / 0.9 ms).
  Products of pairings take one final exponentiation (`multi_pair`; a
  product of 8: 2.6 ms, charm MNT224 19.4 ms). The bounded discrete log in
  GT keeps its baby-step table per base and bound. SGP decryption pairs
  `a_i` with `prod_j b_j^{F_ij}` (2 n + 1 pairings instead of 2 n^2 pairings
  and n^2 GT exponentiations). End to end (`report` profile) SGP 320 -> 36
  ms, Quad 506 -> 67 ms, MIFE / MCFE / dMCFE / tMCFE FH-Multi-IPE 425 /
  889 / 945 / 1565 -> 25 / 82 / 79 / 136 ms (charm MNT224 in Docker before,
  same Mac).
- **Products of powers on BLS12-381 in one native call**: every product of
  powers in the pairing schemes is one multi-scalar multiplication (Straus
  over the GLV decomposition in G1 / G2, Pippenger from 512 terms, a
  Frobenius decomposition of the exponent in GT), powers of one base share
  one table, and vectors of group elements are decoded in one call (equal
  encodings once). Single exponentiations: G2 0.37 -> 0.16 ms (arkworks
  used plain double-and-add in G2), GT 0.49 -> 0.22 ms; GT deserialisation
  0.81 -> 0.08 ms (membership by `f^p = f^u` in the cyclotomic subgroup
  instead of `f^r = 1`). Keys and ciphertexts are unchanged; charm groups
  keep the loops. End to end (`report`, host): MIFE / MCFE / dMCFE / tMCFE
  FH-Multi-IPE 24 / 79 / 79 / 138 -> 18 / 47 / 48 / 70 ms, SGP 35 -> 30 ms,
  Quad 65 -> 48 ms (setup 16 -> 6 ms); see `benchmarks/PERFORMANCE.md`.
- **u64 words for moduli of 62 and 63 bits** (the NTT keeps its values in
  [0, q) there instead of Harvey's lazy [0, 4q)): the `bound_x = 2`
  configuration, at 63 bits since the 3X fix, is back in 64-bit words --
  client 2.77 -> 2.04 ms, aggregator 7.25 -> 5.14 ms (10^4 elements).
- **The exact FACCT sampler in Rust**: `sampler="facct"` keeps the Ring-LWE
  encryption native, with the getrandbits calls and decisions of
  `discrete_gaussian_facct` (its known answers hold natively; the rare
  acceptance tests the fixed-point bound cannot decide use the Python
  formula). 10^4 elements: 355 ms (Python) -> 18.3 ms; 0.45 us per sample.
  The default stays the rounded sampler.
- `ring_lwe_utils` NTTs and `poly_mul` run natively when the extension has a
  context for (q, n) and the caller's root (the same values): SIFE Ring-LWE
  encrypts natively (200 -> 16 ms, eta 10), SIFE FullySec Ring-LWE decrypts
  in 67 instead of 290 ms; dMCFE Ring-LWE key shares and their combination
  are native (25.3 -> 4.7 ms per client, 12.6 -> 2.0 ms); single-vector MCFE
  label pads are hashed natively.
- **Offline phases for Z_p^* and DCR**: `precompute(label, shapes)` /
  `encrypt_lst_ndarray(..., precomputed=...)` and `prepare_key(dk, label,
  shapes)` for MCFE-DDH, MCFE-Damgard and MCFE-Paillier (and their
  decentralized variants; used by `precompute_gradient` and
  `prepare_aggregation`), `precompute(shapes)` for SIFE- / MIFE-Paillier
  and MIFE-DDH; the same ciphertexts. Online, 20 elements: client 212 /
  212 / 717 -> 0.67 / 0.67 / 0.39 ms, aggregator 220 / 225 / 717 -> 4.6 /
  6.9 / 0.9 ms. Discrete-log tables are keyed by value (the old string
  keys are still read).

- **Wide Gaussians in Rust**: `discrete_gaussian_bigint` (the noise of SIFE
  FullySec Ring-LWE and ALS16, the flooding noise of tMIFE-LWE, tMCFE-LWE and
  tMCFE-Ring-LWE) runs natively for widths below 2^448 when `rng` is an
  `XofRandom` or `None`, with the integers of the MPFR reference for the same
  bits: a fixed-point evaluation whose error is far below its rounding
  margin, and the reference formula for the samples it cannot decide (about
  2^-45). Per sample 8-12 -> 0.55 us at 2^49, about 10 -> 2.0 us at 2^200;
  SIFE FullySec Ring-LWE encryption 964 -> 120 ms, ALS16 483 -> 189 ms per
  element, tMCFE-Ring-LWE partial decryption 225 -> 107 ms per server. With
  `rng=None` the bytes come from the per-call DRBG instead of `os.urandom`
  per call (same distribution).

### Added

- Opt-in exact discrete Gaussian sampler `"facct"`
  (`sampling_utils.discrete_gaussian_facct`): FACCT-style double sampling
  (Zhao, Steinfeld and Sakzad 2020) with a CDT base computed for
  `sigma / k`, so every width is sampled exactly as requested, to about
  2^-185 statistical distance per sample. Select it with `sampler="facct"`
  on `discrete_gaussian_matrix` / `discrete_gaussian_bigint`, the
  `gaussian_sampler()` context manager, `set_gaussian_sampler()` or
  `PYFE4AI_GAUSSIAN_SAMPLER=facct`. The default stays the rounded
  continuous Gaussian. Its output for a given bit source is platform
  independent (known-answer tests included); it reads `os.urandom` in
  64 KiB blocks (fork- and thread-safe). Not constant-time in CPython.
- `benchmarks/benchmark_sampling.py`: rounded vs exact sampler, per width
  and (`--e2e`) per MCFE Ring-LWE round.

- `ct_format="binary"` for Ring-LWE MIFE / MCFE ciphertexts (crypto config;
  `FESchemeWrapper(crypto_options=...)`): a versioned compact encoding, 37%
  of the JSON size. JSON stays the default; decryption reads both.
- Offline / online MCFE Ring-LWE: `MCFERingLWE.precompute(label, shapes)` and
  `encrypt_lst_ndarray(..., precomputed=...)`; ML adapter
  `precompute_gradient`, `encrypt_gradient(precomputed=...)`,
  `prepare_aggregation` and `aggregate_gradients(prepared=...)`.
- `benchmarks/bench_fl_compare.py` (per-family FL timings as JSON) and
  `benchmarks/bench_offline_potential.py`.
- `pyfe4ai.utils.pairing_backend.multiexp(bases, exps, group=None)`,
  `pow_many(base, exps, group=None)` and `deserialize_many(group,
  encodings)`, native on BLS12-381 and loops on charm groups;
  `PairingGroup.deserialize_many` on BLS12-381.
- `sife/`, `mife/`, `mcfe/ddh_ristretto255` (status *unreviewed*): the DDH
  constructions in the prime-order group ristretto255 (RFC 9496; labels
  hashed with RFC 9380 `hash_to_ristretto255`), selected with
  `group="ristretto255"` on the `ddh` variants. About 126-bit security;
  0.07 ms per element and client for MCFE on the Rust backend (3072-bit
  group: 10.8 ms), 75 instead of 936 bytes per element.

### Changed

- **Pairing groups (breaking for pairing keys and ciphertexts):** every
  pairing-based scheme (SIFE FH-IPE and partial FH-IPE, MIFE FH-IPE and
  FH-Multi-IPE, MCFE / dMCFE / tMCFE FH-Multi-IPE, Quad, SGP, MIQFE) now
  defaults to **BLS12-381** from the Rust extension, under both profiles.
  The `toy` default was MNT224 (charm): toy pairing keys, public parameters
  and ciphertexts of earlier versions are not readable any more (pass
  `pairing_group_param="MNT224"` to keep using them with charm installed).
  The `standard` profile uses BLS12-381 instead of BN254 (about 100-bit) and
  still warns, since BLS12-381 gives about 120-126 bits after the exTNFS
  attacks on its 381-bit field (Barbulescu and Duquesne, J. Cryptol. 2019):
  `FESecurityWarning: <family>/<variant>: BLS12-381 gives about 120 to
  126-bit security after the exTNFS attacks, below the 128-bit target of the
  'standard' profile`. charm-crypto stays the backend of every other curve
  (MNT224, BN254, ...) and is no longer needed for the defaults.
- `group.hash(msg, G1 | G2)` on BLS12-381 is the RFC 9380 hash_to_curve
  (BLS12381G1/G2_XMD:SHA-256_SSWU_RO_, domain tag
  `PYFE4AI-V01-CS01-with-...`), used by the FH-MCFE label hashes.
- The pairing-scheme tests run on `PYFE4AI_TEST_PAIRING_GROUP` (default
  BLS12-381) and skip when no backend implements it.
- The `ddh` key generators refuse a `group` other than `"modp"` instead of
  ignoring it.

### Fixed

- `EncryptedGradient.to_dict()` failed with `KeyError: ()` on the packed
  ciphertext layers of the Ring-LWE MCFE schemes (`mcfe/ring_lwe` and its
  threshold and decentralized variants), and `from_dict()` could not rebuild
  them. Packed layers are now written as
  `{"packed": {"shape", "first_chunk", "chunks"}}`; the per-element format
  is unchanged. `from_dict()` checks the packed layout: shapes as for
  per-element layers, chunks numbered consecutively across layers, each chunk
  with a batch length `k >= 1`, the `k` of a layer adding up to its number of
  elements, and only a layer's last chunk shorter than the others.
  `to_dict()` raises `TypeError` for chunks holding `bytes` (a binary
  ciphertext format has no JSON form).


## [0.3.0] — 2026-10-13

Privacy fixes: every scheme that was insecure in v0.2.0 now implements a
published construction (four threshold variants add an *experimental*
threshold layer of our own on top of one, and dMCFE-DDH an *experimental*
carry mask), and every scheme carries a
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
  derived from a per-server secret, the label and the server's Lagrange
  coefficient for the enrolled set, so repeated requests cannot be
  averaged. The threshold layer is our own, so the
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
  each client derives its own key from an explicit `tag-f` and its own
  weights (`H2(tag-f)`), deterministically, one key per client and `tag-f`.
  **tMCFE-FH-Multi-IPE** Shamir-shares the FH-MCFE key exponents (servers
  hold their shares as group elements); partial decryptions are combined in
  `G_T` (*experimental*). Previously all three masked inputs with a public label
  scalar and ran over the symmetric group `SS512`, where SXDH fails; the
  family now defaults to `MNT224` and rejects symmetric groups.
- **All pairing-based schemes** (SIFE FH-IPE and partial FH-IPE, MIFE
  FH-IPE and FH-Multi-IPE, quadratic Quad, SGP and MIQFE, the FH-MCFE
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
    dimension equalled `eta` (a square `M` reveals `V`); it is now 1 (see
    below) and `M` is resampled until it has full rank.
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
- **SIFE-Paillier and MIFE-Paillier** decryption returned whatever the
  decryption equation produced. The ciphertext `ct0 = 1 + N`, `ct1 = 1`
  turns `ct0^{-k}` into `-k mod N`, so anyone shown the output of a
  decryption (e.g. an inference client) learnt the functional key.
  Decryption now checks ciphertext lengths and ranges and returns `None`
  unless the result is a valid encoding within `+-eta X Y`
  (`modular_utils.check_zn2_elements`, `decode_paillier`).
- **dMCFE-DDH** reduced every client's key contribution mod `q` before the
  MCFE2 layer, whose decryption is the *integer* sum: anyone holding the
  published shares read the carry (`0..n-1`) of that sum, a comparison
  oracle on the clients' secrets. ABMPSW20 states Fig. 14 for MCFEs without
  a discrete log in decryption only. Contributions are now masked with
  `q rho_i`, `rho_i < 2^(stat_param + bits(n))` derived from a new per-client
  key and `H(y)`; the scheme becomes *experimental* (our own argument).
- **dMCFE-LWE** reduced every contribution mod `q` as well, so the integer
  sum also told how many `y_i t_i` were negative in each coordinate. The
  contributions are now sent over the integers, as the functional key
  `sum y_i Z_i in Z^(n0 + m0)` of ABMPSW20 Fig. 4 / Fig. 14.
- **SIFE-LWE and SIFE-Ring-LWE** encoded a plaintext as `trunc(x q / p)`
  instead of `floor(q / p) x`. The two differ by a small term that is not
  linear in `x`, while ABDP15's proof relies on the linear ciphertext
  homomorphism of its PKE and BMMS21's last game on `floor(q/K)` times the
  message; the residue was hidden only statistically by the ciphertext
  noise. Both now use `floor(q / p) x mod q` (`center_lwe_vector`,
  `center_matrix`); found while an independent review of the new
  `sife/fullysec_ring_lwe` flagged the same encoding there. Ciphertexts
  change; decryption is unaffected.
- **tMIFE-LWE** (experimental) enlarged `q` for its flooding noise but kept
  the `m` of the base modulus, so the binary `r` of a ciphertext had about
  half the length the leftover hash lemma needs for `A^T r` to be close to
  uniform (toy defaults: `m = 753` where `2139` is needed); ABDP15's
  semantic-security argument then fails. `m` and `q` are now iterated to a
  fixed point; cached parameters are regenerated.
- **tMIFE-LWE** (experimental) servers decrypted any ciphertexts they were
  handed. A partial decryption is linear in `ct0`, so a combiner submitting
  `ct0 = c e_j` for chosen `c` read the shared functional key off the
  partials coordinate by coordinate (the flooding hides a fixed noise term,
  not a chosen multiple of the key). Clients now tag every ciphertext with
  one HMAC-SHA256 per server, under per-(client, server) keys sampled by
  `setup`, and a server refuses ciphertexts whose tag for it does not verify.
  A corrupted client can still choose its own slot's ciphertext, which
  reveals only that slot's key part.
- **tMCFE-Ring-LWE** (experimental) servers decrypted any ciphertexts they
  were handed; a partial decryption is linear in `ct1`, so a combiner
  recovered the functional-key parts `S_i y_i` exactly in a few queries
  with chosen `ct1`. As for tMIFE-LWE, clients now tag every ciphertext (and
  every ndarray chunk) for every server, over the (chunk) label and the
  ciphertext; servers refuse untagged, altered or other-label ciphertexts.
- **tMCFE-FH-Multi-IPE** (experimental) servers decrypted whatever round the
  combiner submitted (FH ciphertexts hide their label). Ciphertexts now
  carry the encoded label, tagged together with the ciphertext for every
  server, and `share_decrypt` takes the label the server agrees to decrypt.
- **tMIFE-LWE and tMCFE-Ring-LWE** seeded the flooding noise with the JSON of
  the request, so the same ciphertexts with an extra field, another server's
  tag or (Ring-LWE) coefficients written as `v + q` drew fresh noise, and
  averaging such partials removes it. The noise is now seeded with a digest
  of the values the key part depends on (`ct0` for tMIFE-LWE; `ct1` and the
  label or chunk for tMCFE-Ring-LWE), so neither the encoding, nor extra
  fields, nor the other half of a ciphertext re-tagged by a corrupted client
  draws fresh noise (`threshold_utils.ciphertexts_digest`); the enrolled set
  enters only through the server's Lagrange coefficient (two sets with the
  same coefficient drew different noise), and tMCFE-Ring-LWE's noise keys
  no longer change when a key is re-shared for another `n_chunks`. A
  corrupted client varying its own slot still draws fresh noise (averaging
  needs about `2^(2 stat_param)` partials; documented); tMCFE-Ring-LWE
  also refuses ciphertexts of the wrong shape
  (`ring_lwe_utils.canonical_ring_ciphertext`). Tags use the shared
  `threshold_utils.client_server_mac_keys`, `tag_ciphertext` and
  `check_ciphertext_tags`.
- **tMCFE-DDH** servers decrypted whatever label the submitted ciphertexts
  carried (the key is label-independent), so they could not restrict
  decryption to the rounds they mean to serve: a combiner could have any
  round decrypted at any time. `share_decrypt` now takes the label the
  server agrees to decrypt and refuses ciphertexts under any other.
- **tMCFE-FH-Multi-IPE** (experimental) handed every server Shamir shares of
  the FH-MCFE key exponents `e_i` in `Z_p`. The key hides `y` only as
  `[e_i]_2`: `t` colluding servers (one if `t = 1`) interpolated `e_i` and
  solved for the weights. Servers now hold their shares in the exponent
  (`[e^(k)]_2`, `[z^(k)]_T`) and raise their pairing product to their
  Lagrange coefficient; `t` servers interpolate the FH-MCFE key itself.
- **dMCFE-FH-Multi-IPE** used `tag-f = (tag_f, H(y))` over all clients'
  weights, so a client's key share depended on the others' weights: a
  corrupted client derived its share for a guess of the honest clients'
  weights and learnt whether the guess was right from whether decryption
  succeeded. As in NPS24 Fig. 1, a client's share now depends on an explicit
  `tag_f` and its own `y_i` only; a client object refuses a second weight
  vector under a `tag_f` it already used.
- **ML adapter**: `FESchemeWrapper` holds the master key, and
  `aggregate_gradients` issued a key for any weights, so weights `(1, 0)` or
  a subset of the clients decrypted one client's gradient; labels could be
  reused. It now needs a ciphertext and a non-zero weight for every enrolled
  client (`allow_partial=True` relaxes this), refuses a second encryption by
  a client under one label and a second weight vector for one label, and is
  documented as a single-trust-domain simulation.
- **`standard` profile**: the statistical parameter `stat_param` kept its
  default 64 and accepted any explicit value (`stat_param = 1` gave
  flooding at statistical distance about 1/2), although the profile claims
  128 bits and refuses weaker explicit values. It is now 128 (the `kappa` of
  the Ring-LWE schemes) and lower values are refused: a distance of
  `2^-128` per sample instead of `2^-64`. ABMPSW20 moduli grow by about 64
  bits (`mcfe/lwe` picks `n0` of about 7200 to 8200 instead of 4600 to
  5600), tMCFE-Ring-LWE moves to `ring_n = 16384`, and tMIFE-LWE to
  `lwe_n` of about 6000, for which `WIDE_LWE_128_LOG_RATIO` now extends to
  `n = 6144` (same estimator and method).
- **`standard` profile**: a config that was a `Mapping` but not a `dict`
  (e.g. `MappingProxyType`) skipped the profile while the key generator
  recorded `security_profile = "standard"` (`sife/ddh` ran on a 128-bit
  group). Every `Mapping` is now resolved.
- **Parameter cache**: cached discrete-log groups and DCR moduli were used
  without any check (a planted 5-bit group or a prime "RSA modulus" was
  accepted, also under `standard`). DL groups are now checked for primality,
  size, a safe-prime structure (a cached cofactor `r` with a small `q`
  passed a first version of the check), the order of their generators and,
  at RFC 3526 sizes, for being the RFC prime; DCR moduli for size,
  compositeness and prime factors below 2^16, Paillier bases for not being
  `1 mod N` (`crypto_utils.is_dl_group`, `is_dcr_modulus`). Everything else
  a cache holds is now either re-derived on load or expanded from a stored
  seed: the Ring-LWE parameters and public polynomial `a`
  (`ring_lwe_utils.seeded_polynomial`; `a = 0` used to be accepted), the
  ABDP15 / tMIFE-LWE parameters including `sigma_f` (`lwe_n = 1`,
  `sigma_q = 0`, `m = 8` or `sigma_f = 1` used to be accepted), the ALS16
  parameters and matrix `A`, the Paillier base `g = H(seed)^(2N)`
  (`crypto_utils.paillier_base`; `g = N^2 - 1` used to reveal the
  plaintexts) and the Damgard element `h`, hashed into the group (a planted
  `h = g` had a known discrete logarithm).
  A file that fails, or does not parse (e.g. after an interrupted write), is
  regenerated, and files are written atomically. Since no check can tell
  whether someone knows the factors of a DCR modulus, the `standard` profile
  never loads or stores the Paillier moduli and the dMCFE `N2` (they are
  generated for every key generator); `toy` keeps the cache.
- **Ring-LWE parameters (SIFE, MIFE, MCFE, tMCFE, dMCFE)**: `bmms21_parameters`
  cites BMMS21 Sec. 4.2 but used `sigma = 1` and `sigma1 = 2 sqrt(l) X`,
  which miss its smoothing conditions (the mhe-RLWE condition of G1 -> G2
  needs `sigma sqrt(1 - ...) > eta(Z^(n + n l))`, about 5.6, and had 0.71;
  G3 -> G4 needs `sigma' >= sqrt(2) eta(Z^n)` and had `sigma' = 1`). The
  widths now follow Sec. 4.2 (`ring_lwe_utils.bmms21_selective_parameters`),
  which adds about 10 bits to `q`. The `standard` profile checked the
  moduli against the Homomorphic Encryption Standard's bound, which assumes
  errors of width 3.2 (the standard SIFE-Ring-LWE had `sigma1 = 2`) and which
  the lattice estimator puts at about 125 bits; it now uses
  `RING_LWE_128_LOG_Q`, estimated with the same tool and rule as the LWE
  tables for widths 2 to 16 and `n = 1024` to `32768`, read at the smallest
  width among the RLWE problems of the proof (about 8). Cached parameters
  are regenerated.
- **Threshold schemes** assume honest-but-curious servers; this is now stated
  in the README, `SECURITY.md` and `threshold_utils`. Nothing proves a
  partial decryption correct, so `combine_decrypt` and
  `decrypt_lst_ndarray_ct` of every threshold scheme accept partial
  decryptions for several enrolled sets and raise `FESchemeError` if their
  results disagree (`threshold_utils.combine_consistently`).

### Added

- `security_profile` config key for every key generator
  (`pyfe4ai.schemes.profiles`). `"toy"` (default) leaves configs unchanged,
  so all defaults stay as they were. `"standard"` presets 128-bit-oriented
  sizes: a 3072-bit DL group (RFC 3526 MODP group 15), a Paillier `N` of
  about 3072 bits (`bit_length = 1536`), Ring-LWE `kappa = 128` with `ring_n = "auto"`
  (smallest dimension whose derived `q` meets `RING_LWE_128_LOG_Q`, a
  lattice-estimator table read at the smallest RLWE width of the scheme's
  proof, re-checked after derivation, flooding included)
  and the BN254 pairing group (about 100-bit, with a `FESecurityWarning`).
  The ABMPSW20 MCFE-LWE family (`mcfe/lwe`, `fullysec_lwe`, `lwe_threshold`,
  `lwe_decentralized`) gets `lwe_n = "auto"`: the smallest dimension whose
  derived `q` meets `lwe_utils.DENSE_LWE_128_LOG_Q`, a lattice-estimator
  table for LWE(n0, q, ceil(2 sqrt(n0))) with a uniform secret (the
  assumption of the scheme's proof), regenerated by
  `tools/estimate_dense_lwe.py`; under `standard` its decentralised variant
  combines key shares under a DCR modulus of at least 3072 bits.
  Explicit values below the profile, and the schemes without a usable
  secure parameter set (SIFE / MIFE `fullysec_lwe` and dMCFE-Paillier),
  raise `FEValidationError`.
- ABDP15 LWE IPFE (`sife/lwe`, `mife/lwe`) under `standard`: `lwe_n = "auto"`
  picks the smallest dimension whose `(q, sigma)` meets
  `lwe_utils.WIDE_LWE_128_LOG_RATIO`, a lattice-estimator table of the
  largest `log2(q / sigma)` per dimension and error width for the LWE problem
  of the public key `A S + E` (regenerated by `tools/estimate_wide_lwe.py`);
  explicit dimensions are checked against it. New config key `matrix_mode`
  (key generators and crypto objects; forwarded by the ML adapter):
  `"memory"` keeps the expanded public matrix (n = 1088: 0.1 s per
  encryption, 2.7 GB), `"stream"` expands it for every batch (3.4 s, 0.5 GB).
  `pyfe4ai.utils.lwe_matrix` (`SeededMatrix`, `LimbMatrix`),
  `pyfe4ai.schemes.abdp15_lwe`, `sampling_utils.rand_bit_matrix`.
  tMIFE-LWE gets the same `lwe_n = "auto"` at its flooding-enlarged `q`
  (about 6000 for `X = Y = 50` with `stat_param = 128`; the table covers
  dimensions up to 6144), and `standard` fixes its `matrix_mode` to
  `"stream"` (its `A` would take more than 400 GB in memory): key generation
  38 minutes, one encryption 572 s, 10.6 s each in batches of 64. `"memory"` mode refuses a matrix larger than the
  machine's physical memory.
- `sife/fullysec_ring_lwe`: the adaptively secure Ring-LWE IPFE of BMMS21
  (Bermudo Mera, Karmakar, Marc, Soleimanian, PKC 2022, Sec. 5), the ring
  counterpart of ALS16. The paper takes `m = 3` key polynomials, but under
  its own constraints (Appendix C.1: the Ring-LHL width `sigma'` grows like
  `q^(2/m)`, the decryption noise like `q^(4/m)`) no modulus exists for
  `m <= 4`; `m = "auto"` takes the smallest `m >= 5` that admits one (within
  the 128-bit Ring-LWE bound under `standard`). Widths follow Theorems 1 and 2
  and Appendix C.1 for signed inputs, with complexity leveraging over
  `(2X + 1)^(2 eta)`; one vector per ciphertext. `standard`: eta = 10,
  X = Y = 50 gives n = 8192, m = 8 (encrypt 2.4 s, decrypt 0.5 s).
  `ring_lwe_utils.bmms21_adaptive_parameters`, `bmms21_adaptive_log2_q`,
  `bmms21_adaptive_choice`.
- `lwe_n = "auto"` for the MCFE-LWE family; `lwe_utils.DENSE_LWE_128_LOG_Q`,
  `check_dense_lwe`, `dense_lwe_n`, `resolve_lwe_n`;
  `profiles.STANDARD_DCR_PRIME_BITS`; `modular_utils.pow_signed_many`
  (fixed-base exponentiation of one base to many exponents).
- `ring_n = "auto"`; `ring_lwe_utils.RING_LWE_128_LOG_Q` (regenerated by
  `tools/estimate_ring_lwe.py`), `ring_lwe_bound`, `check_ring_lwe`,
  `ring_lwe_auto_n`, `resolve_ring_n`, `bmms21_selective_parameters`;
  `crypto_utils.RFC3526_PRIMES`.
- `pyfe4ai.utils.validation.int_vector`: scheme inputs must be integers
  (integral floats accepted); non-integral values raise
  `FEValidationError` instead of being truncated.
- `pyfe4ai.utils.ring_lwe_utils.bmms21_parameters`, `check_ring_n` and
  `DEFAULT_RING_N`; `FHIPE.keygen(..., r=...)`.
- `dlog_bound` crypto option for SIFE-DDH, MIFE-DDH, MCFE-DDH and
  tMCFE-DDH (results beyond the default `10^(precision + 2)` decrypt to
  `None`).
- `pyfe4ai.schemes.status` (`SchemeStatus`, `get_scheme_status`, exported
  from `pyfe4ai`): status (`reviewed` / `experimental` / `unreviewed` /
  `known-insecure`), fidelity, reference and issue for all registered
  schemes; key generators and crypto objects of known-insecure schemes emit
  `pyfe4ai.FESecurityWarning`; `FESchemeWrapper.status`;
  `markdown_table()` renders the README table, which a test keeps in sync.
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

- tMCFE-DDH: `share_decrypt(dict_ct, credentials, dk, lst_sid_enrolled,
  label)`, `compute_lst_ndarray_ct(..., lst_sid_enrolled, label)` and
  `decrypt_lst_ndarray_ct(partials, label)` take the label to decrypt;
  `combine_decrypt(partials, label=None)` checks it when given.
- Exports and registry: `HybridAlphaMIFE` and `HybridAlphaMIFEKeyGenerator`
  are removed (see *Removed*); `MultiInputQuadraticSGP` and
  `MultiInputQuadraticSGPKeyGenerator` are replaced by `MultiInputQuadratic`
  and `MultiInputQuadraticKeyGenerator`, the ML adapter's registry key
  `("quadratic", "multi_input_sgp")` by `("quadratic", "miqfe")` and
  `config_types.MultiInputSGPConfig` by `MultiInputQuadraticConfig`. The
  static method `ThresholdMIFE.L` is removed (use
  `threshold_utils.lagrange_coefficient_at_zero`).
- tMCFE-FH-Multi-IPE: key shares hold `k` as `G2` elements and `z` as a `GT`
  element (serialised) instead of integers, plus `mac_keys`; client keys
  carry `mac_keys`; ciphertexts are `{ct, label, tags}`;
  `share_decrypt(dict_ct, credentials, dk, lst_sid_enrolled, label)`,
  `compute_lst_ndarray_ct(..., lst_sid_enrolled, label)`,
  `decrypt_lst_ndarray_ct(partials, label)`, `combine_decrypt(partials,
  label=None)`; partial decryptions carry `label`.
- tMCFE-Ring-LWE: client keys and key shares carry `mac_keys`, ciphertexts
  and ndarray chunks `tags`.
- dMCFE-FH-Multi-IPE: `derive_function_decryption_key_share(weights, tag_f)`
  takes this client's weights (or a dict, of which only its entry is used)
  and a required non-empty `tag_f`.
- tMIFE-LWE: client keys carry `mac_keys` (`{sid: key}`), key shares
  `mac_keys` (`{nid: key}`) and ciphertexts `tags` (`{sid: tag}`);
  ciphertexts without valid tags are refused.
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

- SIFE-LWE, MIFE-LWE and tMIFE-LWE publish the matrix `A` as a 32-byte seed
  (`pp["A_seed"]` replaces the `m x lwe_n` digit matrix `pp["A"]`), expanded
  with SHAKE-128, and encrypt with exact numpy products over 31-bit limbs
  (whole batches in the ndarray helpers); `q` is the largest prime with the
  derived bit length. Public parameters, keys and cached parameters are not
  compatible with earlier versions.
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
- `dlog_solver.PairingDLogCacheMixin`, which had no users (it trusted a
  dlog table read from `./config/crypto`), and the unused helpers
  `crypto_utils.md5_hash` and `group_generator_paillier`.
- The `jsonpickle` dependency (never imported).
- The unused modules `schemes/lwe_base.py`, `schemes/ring_lwe_base.py` and
  `schemes/paillier_base.py` and the helpers `label_scalar_from_hash` and
  `decode_fullysec_lwe_inner_product`.
- The repository-root `config.yaml`, an unused copy of the bundled
  `pyfe4ai/config.yaml`.

### Build

- `Dockerfile.pairing`: the deprecated `continuumio/miniconda3` base is
  replaced by `condaforge/miniforge3:26.5.3-0`, pinned by digest; the PBC
  1.0.0 archive is checked against its SHA-256 before it is built; the image
  runs as an unprivileged user. `environment.yml` uses the `conda-forge`
  channel.

### Fixed

- `benchmarks/benchmark_ndarray_helpers.py` ran every group before applying
  `--group`, so `--group sife` failed without `charm-crypto`, and its warm-up
  rounds ran after the timed ones. It now runs the warm-up and then only the
  selected groups.
- The ndarray helpers truncated `x 10^precision` toward zero (`0.57` became
  `0.56` at precision 2) and turned NaN and infinities into `-2^63`. They now
  round to nearest and refuse non-finite or too large values
  (`validation.scale_to_int`).
- Ring-LWE MIFE / MCFE / dMCFE and FH-MCFE keys truncated fractional weights
  (`0.5` became `0`), and `threshold_utils.fusion_weight_id` gave `1.5` and
  `1` the same digest; weights now go through `validation.int_vector`.
  Ring-LWE MIFE / MCFE plaintext rows of different lengths were cut or
  padded silently and are now refused; so are ciphertexts of the wrong shape
  (short polynomials were truncated by `zip`) and ndarray layers whose
  `first_chunk` is negative or beyond the key (a negative index picked chunk
  keys from the end of the list).
- MIFE-LWE sized its noise for `eta`-dimensional vectors, but decryption adds
  the noise of all `n` slots, so the decryption margin shrank with
  `sqrt(n)`. As in ACFGU18 (footnote 14), the ALS16 / ABDP15 parameters are
  now chosen as if the vectors had dimension `n * eta`; cached parameters are
  regenerated.
- The ABDP15 plaintext modulus `p` was random in its bit range, so
  `lwe_n = "auto"` could pick different dimensions on different runs (the
  README's `n = 832` for `eta = 1` was one such draw; the deterministic
  choice is 768).
  `p` is now the smallest prime of that size.
- SIFE / MIFE FullySec-LWE (ALS16) used `m = 1.01 n log2 q`; ALS16 Theorem 2
  needs `m >= 4 n log2 q`. The recipe now uses `4 n log2 q` (also inside
  the width derivation), sizes the row norm `B_tau` of `tau` with `sqrt(m)`
  instead of `sqrt(n)` (with the larger `m`, the old bound let about 0.1% of
  extreme decryptions fail at toy sizes), uses at least the 12-sigma tail
  cut for its `omega(sqrt(log n))` (`log2 n` alone left dimensions below
  4096 a few sigma of margin) and takes the largest prime of the
  derived size (a random draw could exceed it by one bit); cached parameters
  are regenerated. The `standard` refusal now quotes the resulting sizes
  (`n = 3328` to `4352`, 80 to 145 MB per ciphertext).
- `sife/fullysec_ring_lwe` with an explicit `ring_n` and `m = "auto"` under
  `standard` took `m = 5`, whose `q` exceeds the bound, and refused; it now
  takes the smallest `m` that meets the bound.
- SIFE-Paillier truncated fractional plaintexts and weights (`[1.7, 2.9]`
  encrypted as `[1, 2]`, a weight `0.5` became `0`); the same `abs(int(v))`
  checks were in MIFE/MCFE LWE, Paillier and Damgard, dMCFE-LWE /
  -Paillier and partial FH-IPE. They now refuse non-integral values
  (`validation.strict_int`), and so do MCFE-DDH key derivation, encryption
  and decryption (`gp.mpz(2.5)` was 2); rationals are checked exactly
  (`Fraction(2^60 + 1, 2)` passed through a float). `scale_to_int` refuses a
  negative precision;
  a key generator config that is not a `Mapping` is refused (it could skip
  the security profile).
- MCFE-DDH and tMCFE-DDH put the plaintext into the message of the
  exception for a malformed plaintext, and `int_vector` the offending value;
  messages now name only the problem (and the type).
- SIFE FH-IPE decryption required the weights `y` although its keys hide
  them; `fusion_weight` is now optional (checked if given).
- The logging configuration is read with `yaml.safe_load` semantics
  (`SafeLoader`); dependencies have lower bounds (`gmpy2>=2.2` for
  `prev_prime`, `numpy>=1.22`, `PyYAML>=5.4`).
- The ndarray combiners of the threshold schemes crashed with a `TypeError`
  (or returned NaN, tMCFE-LWE) when an element's partial decryptions
  disagreed, and took the output shape from the first server; they now
  check that all servers' outputs have one structure (lists of arrays) and
  raise `FESchemeError` for malformed outputs and for an element that does
  not decrypt (`threshold_utils.combine_elementwise`).
- The ndarray helpers of the other schemes did the same for an element that
  does not decrypt (result outside the decryption bound, or key and
  ciphertext do not match): a `TypeError` in most schemes, a silent NaN in
  MCFE-Ring-LWE, MIFE FH-Multi-IPE, `quad` and `sgp`. They now raise
  `FESchemeError` naming the element (`ipfe.decoded_layer`), as
  `ml_adapter.compute_linear` does for a result that does not decrypt.
- SIFE-Paillier `decrypt_lst_ndarray_ct` decoded inline without the checks
  of `decrypt` (ciphertexts in `Z_{N^2}^*`, result in range), so a wrong key
  gave wrapped values instead of an error; it now calls `decrypt` per
  element.
- ML adapter: `aggregate_gradients` on a SIFE wrapper returned the first
  client's result when given several; it now refuses more than one. Fractional
  weights were quantised but the result was not scaled back (weights `0.5`
  gave `10^precision` times the sum); the result is now rescaled, and
  fractional weights without a precision, or not representable at it, are
  refused. The policy does not stop very unequal weights, which separate
  two clients' gradients in one aggregate; this is now documented.
  `EncryptedGradient.from_dict` allocated whatever shape it was given; it now
  bounds the shape and requires one entry per element.
- SIFE-FullySec-LWE decoded `+-eta X Y` with a random sign (`K` was
  `2 eta X Y`). SIFE-Damgard could not decrypt in an instance holding the
  encryption key. SIFE-LWE's ndarray helper skipped the `bound_x` check.
  The MIFE Ring-LWE key generator overwrote the SIFE parameter cache.
- Ring-LWE plaintext modulus (SIFE, MIFE, MCFE): `p` was exactly
  `2 l X Y`, so inner products of `+-l X Y` were ambiguous; it is now
  `2 l X Y + 2` (parameter caches are regenerated).
- Safe primes below 64 bits could have one bit too many (`next_prime`
  stepped past the size, about 1% of 16-bit draws), which made a test fail
  intermittently.
- Key generators accepted `n` and a `lst_nid` of another length; bounds,
  noise widths and moduli were then sized for `n` slots (MCFE-LWE with
  `n = 2` and 10 clients decrypted 160 instead of 810). They now raise
  `FEValidationError`, as threshold schemes do for repeated server ids.

### Documentation

- README: *Scheme Status* table, rewritten *Known Issues*, the scheme table
  cites the implemented constructions (ACFGU18 Fig. 4, ABMPSW20, BMMS21
  Sec. 6, AGT21b, NPS24, ACFGU18 Fig. 12, Gay20 Fig. 4); the ALS16 link
  pointed to a different paper (ePrint 2016/011, now 2015/608); landing
  page matrix updated; the docs index no longer claims a uniform
  `setup / keygen / encrypt / decrypt` API.
- README and docs: the usage example's advice (adding `standard` next to an
  explicit `sec_param = 128` is refused), `sife/fullysec_ring_lwe` in the
  scheme table and features, the landing page's scheme count (39), the size
  of the dMCFE-DDH modulus (about 6410 bits with the carry mask), the docs index's group size
  (3072 bits); a Sphinx page for `pyfe4ai.utils.validation`; `config_types`
  lists `stat_param`, `matrix_mode`, `m`, `lwe_n = "auto"` and the crypto
  config's `dlog_bound`; the benchmarks README lists `benchmark_ntt.py`.
- Tests: a bare `pytest` from the repository root works without an install
  (`pythonpath`); the core lane no longer collects pairing tests
  (`test_fh_ipe.py`, `test_pairing_groups.py`); the ACFGU18 pad test no
  longer fails when a padded slot hits the plain value by chance (about 0.3%
  of runs for MIFE FullySec-LWE); the README status table is compared in full
  with `status.markdown_table()`.
- Docs site: a *Security & Status* page whose scheme table and counts are
  rendered from `pyfe4ai.schemes.status` at build time; `GLOSSARY.md`, which
  was never rendered and named config keys that no longer exist
  (`poly_degree`, `group`), is now `glossary.rst`, corrected and extended
  (`security_profile`, `stat_param`, `matrix_mode`, `label`, `tag_f`,
  `mac_keys`); every class is documented once instead of twice; the
  ML-adapter and quantization docstrings render (they were NumPy style),
  as do 16 over-indented `Args:` sections and 57 bullet items; placeholder
  docstrings are replaced; nitpicky build warnings went from 171 to 0.
- Landing page: experimental schemes are marked in the matrix, MIQFE is
  listed, the ML-adapter card no longer claims to wrap any scheme, a
  security-profile card, the toy-parameter warning in the snippet, how to
  install `charm-crypto`, social-preview tags, a navigation bar that fits
  phones, and WCAG AA contrast.
- README, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue and
  PR templates: vulnerabilities are reported through GitHub private
  vulnerability reporting; the prerequisites of `charm-crypto`; a checklist
  for adding a scheme; LWE dimensions default to 32–64 (not 16–64); the
  examples and benchmarks READMEs list every script and say which need
  `charm-crypto`.

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

[Unreleased]: https://github.com/spire-studio/pyfe4ai/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/spire-studio/pyfe4ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/spire-studio/pyfe4ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/spire-studio/pyfe4ai/releases/tag/v0.1.0
