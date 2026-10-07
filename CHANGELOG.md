# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor bumps may break
compatibility).

## [Unreleased]

Privacy fixes: five multi-client schemes and SIFE-Paillier now implement
published constructions, and every scheme carries a review status.
**Keys, ciphertexts and partial decryptions of the affected schemes are not
compatible with v0.2.0.** Read the README *Scheme Status* table and *Known
Issues* before relying on any privacy property.

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
  server's share instead of an exact linear function of it).
- **SIFE-Paillier** samples keys from `D_{Z,sigma}`, `sigma > sqrt(kappa)
  N^{5/2}` (ALS16); keys used to be drawn from a range small enough to
  brute-force from the public key.
- Newly documented as known-insecure: MIFE `lwe`, `fullysec_lwe`,
  `ring_lwe`, `damgard`, `paillier` and tMIFE-LWE omit ACFGU18's one-time
  pad and reveal every client's `<x_i, y_i>` to the key holder; MIFE-Paillier
  and dMCFE-Paillier keys are brute-forceable.

### Added

- `pyfe4ai.schemes.status` (`SchemeStatus`, `get_scheme_status`, exported
  from `pyfe4ai`): status (`reviewed` / `unreviewed` / `known-insecure`),
  fidelity, reference and issue for all 39 registered schemes; key
  generators of known-insecure schemes emit `pyfe4ai.FESecurityWarning`;
  `FESchemeWrapper.status`. A test keeps the README table in sync.
- `discrete_gaussian_bigint` (arbitrary-width rounded Gaussian),
  `hash_to_zn2_star`, `hash_to_zq_vector`, `abmpsw20_lwe_parameters`,
  `round_to_q0`, `decode_rounded_inner_product`.
- Config keys `stat_param` (ABMPSW20 schemes, SIFE-Paillier; default 64) and
  `mddh_k` (MCFE-Damgard).

### Changed (breaking)

- MCFE-Paillier: `sk = {s}`, `dk = {k}`, `ct = {c}`. MCFE-LWE /
  FullySec-LWE: `sk = {z}`, `dk = {sk_y}`, `ct = {c}`; the config keys
  `bound_u` and `label_modulus` are ignored and parameters are no longer
  cached in `./config`. MCFE-Damgard: `sk = {s}`, `dk = {d}`, `ct = {c}`, no
  `h` in the public parameters.
- tMCFE-DDH: `sk = {s}`, `dk = {d, sid, lst_sid}` (label-independent),
  `ct = {c, label}`, partial decryption `{c_y, u_d, label, sid,
  lst_sid_enrolled}`. tMCFE-LWE: `dk = {sk_y, sid, lst_sid}`, partial
  decryption `{w, c_y, label, sid, lst_sid_enrolled}`, validated like the DDH
  threshold scheme.
- The ndarray helpers of all reviewed MCFE schemes encrypt each element
  under its own derived label (one ciphertext per slot and label).
- Decrypting under a wrong label returns `None` for MCFE-Paillier and
  MCFE-Damgard and an arbitrary in-range value for the LWE schemes.

### Documentation

- README: *Scheme Status* table, rewritten *Known Issues*, scheme table cites
  ABMPSW20; landing page matrix updated; the docs index no longer claims a
  uniform `setup / keygen / encrypt / decrypt` API; known-insecure modules
  carry a warning in their docstring.

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
