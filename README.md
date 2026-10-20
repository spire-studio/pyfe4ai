<p align="center">
  <img src="assets/branding/logo-simple.png" alt="PyFE4AI header" width="40%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/github/v/release/spire-studio/pyfe4ai?color=green" alt="Latest release">
  <img src="https://img.shields.io/badge/license-Apache%202.0-orange" alt="License Apache 2.0">
  <img src="https://github.com/spire-studio/pyfe4ai/actions/workflows/tests.yml/badge.svg" alt="Tests">
  <img src="https://img.shields.io/badge/docs-Sphinx-blueviolet?logo=readthedocs&logoColor=white" alt="Docs Sphinx">
  <img src="https://img.shields.io/badge/crypto-functional%20encryption-critical" alt="Functional Encryption">
</p>

<p align="center">
  🌐 <a href="https://spire-studio.github.io/pyfe4ai/"><strong>Project Page</strong></a> ·
  📖 <a href="https://spire-studio.github.io/pyfe4ai/docs/"><strong>Documentation</strong></a> ·
  🐛 <a href="https://github.com/spire-studio/pyfe4ai/issues"><strong>Issues</strong></a>
</p>

# PyFE4AI

PyFE4AI is a research-oriented Python library for functional encryption in
trustworthy AI systems.
It currently focuses on inner-product and quadratic
functional encryption, with implementations spanning single-input, multi-input,
multi-client, threshold, decentralized, and function-hiding settings.

> [!WARNING]
> **Experimental, under active development — not recommended for production
> use.** PyFE4AI is a research prototype that has not been audited.
> Interfaces, configuration options, defaults and serialized formats (keys,
> threshold key shares, ciphertexts, partial decryptions, parameter records,
> pickles) can change substantially between versions, without a deprecation
> period. When upgrading, regenerate keys and key shares rather than reusing
> them: this version refuses the threshold key shares of v0.3.0 and
> regenerates some cached parameter records (see the upgrade notes in
> [`CHANGELOG.md`](CHANGELOG.md)). Read the
> [Scheme Status](#scheme-status) and
> [Known Issues](#known-issues--security-notice) before relying on any
> privacy property.

## Quick Start

```bash
conda env create -f environment.yml
conda activate pyfe4ai
pip install -e ".[test]"        # editable with test extras
python -m pytest -m core -q     # run the core test suite
```

Every scheme is built from its id with `pyfe4ai.create()`, e.g.
`kg, SIFEDDH = pyfe4ai.create("sife/ddh", eta=3)` (see [Usage](#usage));
`pyfe4ai.list_schemes()` lists the ids.

The pairing-based FE families run on BLS12-381 from the Rust extension
(`python tools/build_rust.py`, needs a Rust toolchain). Other pairing curves
(MNT224, BN254, ...) and the full test suite under them
(`PYFE4AI_TEST_PAIRING_GROUP=MNT224 python -m pytest -q`) need charm-crypto;
for them,
additionally install:

```bash
pip install -e ".[test,pairing]"
```

`charm-crypto-framework` is built from source and needs GMP, OpenSSL and
[PBC 1.0.0](https://crypto.stanford.edu/pbc/) installed first (e.g.
`brew install gmp openssl@3 pbc` on macOS; on Linux, build PBC as in
[`docker/Dockerfile.pairing`](docker/Dockerfile.pairing)). The Docker image of
[Pairing-Based Schemes](#pairing-based-schemes) has all of them.

`pip install` also builds an optional Rust extension when a Rust toolchain
(`cargo`) is on the `PATH` (`python tools/build_rust.py` in a source
checkout; `PYFE4AI_NO_NATIVE=1` skips it). It runs the Ring-LWE MIFE / MCFE,
the threshold MCFE and the dense-LWE MCFE hot paths natively, with the same
outputs as the Python code, which remains the fallback: one MCFE Ring-LWE
round of 10^4 gradient elements under `standard` takes about 18 ms per client
and 6 ms for the aggregator, against 198 ms and 280 ms with v0.3.0 (0.25 ms
per client online after an offline phase). It also provides ristretto255 for
the `ddh_ristretto255` schemes and the BLS12-381 pairing group, the default
of the pairing-based schemes. `PYFE4AI_BACKEND=python` (or
`pyfe4ai.backend.set_backend("python")`) forces the Python implementation. See
`pyfe4ai.backend` and
[`benchmarks/PERFORMANCE.md`](benchmarks/PERFORMANCE.md).

## Usage

```python
import pyfe4ai

x = [2, 1, 3]
y = [4, 5, 6]

# Key generation. sec_param is the bit length of the DDH modulus, not a
# security level: 128 keeps the demo fast but is NOT secure. For a 3072-bit
# group, replace "sec_param": 128 with "security_profile": "standard" (see
# Security Profiles; an explicit sec_param below 3072 is then refused).
kg, SIFEDDH = pyfe4ai.create("sife/ddh", {"sec_param": 128, "eta": len(x)})
kg.setup()
pp = kg.get_public_parameters()
sk = kg.get_private_keys()
dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})

# Encrypt (needs pp + sk)
encryptor = SIFEDDH({"precision": 3, "keys": {"pp": pp, "sk": sk}})
ct = encryptor.encrypt(x)

# Decrypt (needs pp + dk)
decryptor = SIFEDDH({"precision": 3, "keys": {"pp": pp}})
result = decryptor.decrypt(ct, dk, y)
print("<x, y> =", result)   # → 31
```

See `examples/sife_minimal.py` for the full runnable version.

`pyfe4ai.create(scheme, config)` takes a scheme id `"family/variant"` and
returns the scheme's key generator, built from the config but not yet set
up, and its crypto class, whose objects encrypt (with `pp` and `sk`) and
decrypt (with `pp` and a functional key). The classes are also importable
by name (`from pyfe4ai.schemes.sife import SIFEDDH, SIFEDDHKeyGenerator`;
see [Naming Convention](#naming-convention)). The scheme registry answers
what is available: `pyfe4ai.list_schemes()` lists the ids
(`pyfe4ai.list_schemes("mcfe", capabilities="threshold")` the threshold
MCFEs), `pyfe4ai.get_scheme("mcfe/ring_lwe")` gives a scheme's classes,
kind, capabilities and review status.

## Key Features

- **Multiple FE families** — DDH, LWE, Ring-LWE, FullySec-LWE, FullySec-Ring-LWE (adaptively secure, BMMS21 Sec. 5), Paillier, and Damgård instantiations across SIFE, MIFE, and MCFE
- **Threshold & decentralized** — `tMIFE`, `tMCFE`, `dMCFE` for distributed trust settings
- **Function-hiding** — pairing-based FH-IPE and FH-Multi-IPE with full inner-product privacy
- **Quadratic FE** — SGP and Quad (single-input, public-key) and multi-input quadratic FE with per-user keys (AGT22)
- **Security profiles** — `security_profile="standard"` selects 3072-bit DL / DCR groups, lattice-estimator-sized LWE and Ring-LWE dimensions and BLS12-381 pairings; the default `"toy"` keeps the fast demo parameters
- **ML adapter** — `pyfe4ai.ml` for encrypted linear inference and federated gradient aggregation, including threshold MCFE (Ring-LWE) with its decryption servers run in-process
- **Native backend** — optional Rust extension (`pyfe4ai._rust_native`) for the Ring-LWE, dense-LWE and threshold hot paths, ristretto255 and the BLS12-381 pairing group, with the Python code as reference and fallback (`pyfe4ai.backend`)
- **Precision toolkit** — `pyfe4ai.ml.quantization` for float ↔ integer quantization with configurable bit-width
- **Discrete-log solvers** — in-memory dlog table (default) with on-the-fly BSGS as optional backup in `primitives/groups/dlog_solver.py`
- **NTT acceleration** — number-theoretic transform for Ring-LWE polynomial multiplication in `primitives/lattice/ring_lwe_utils.py`
- **Structured exceptions** — `FEError` hierarchy in `core/exceptions.py`

## Implemented Schemes

| Family | Instantiations | Paper(s) |
|--------|---------------|----------|
| **SIFE** (single-input) | DDH, DDH-Dynamic, LWE | [ABDP15](https://eprint.iacr.org/2015/017.pdf) (PKC '15) |
| | Damgård-DDH, FullySec-LWE, Paillier | [ALS16](https://eprint.iacr.org/2015/608.pdf) (CRYPTO '16) |
| | Ring-LWE | [BMMS21](https://eprint.iacr.org/2021/046.pdf) (PKC '22), Sec. 4 |
| | FullySec-Ring-LWE (adaptive) | BMMS21, Sec. 5 |
| | FH-IPE *(pairing)* | [KLMMRW](https://eprint.iacr.org/2016/440.pdf) (SCN '18) |
| | Partial-FH-IPE *(pairing)* | [Gay20](https://eprint.iacr.org/2020/093.pdf) (PKC '20) |
| **MIFE** (multi-input) | DDH, Damgård-DDH, LWE, FullySec-LWE, Paillier | [ACFGU18](https://eprint.iacr.org/2017/972.pdf) (CRYPTO '18), Fig. 4 + ALS16 / ABDP15 |
| | Ring-LWE | ACFGU18 Fig. 4 as in [BMMS21](https://eprint.iacr.org/2021/046.pdf), Sec. 6 |
| | FH-IPE *(pairing)* | ACFGU18, Fig. 12 |
| | FH-Multi-IPE *(pairing)* | [DOT18](https://eprint.iacr.org/2018/061.pdf) (PKC '18), Sec. 3.1 |
| **MCFE** (multi-client) | DDH | [CDGPP18](https://eprint.iacr.org/2017/989.pdf) (ASIACRYPT '18) |
| | Damgård-DDH (MDDH), Paillier (DCR), LWE, FullySec-LWE | [ABMPSW20](https://eprint.iacr.org/2020/788.pdf) (SCN '20), Figs. 2–4 |
| | Ring-LWE | BMMS21, Sec. 6 (identity-based MCFE) |
| | FH-Multi-IPE *(pairing)* | [AGT21b](https://doi.org/10.1007/978-3-030-90453-1_8) (TCC '21), Sec. 6.2 |
| **Decentralized** | dMCFE-DDH, dMCFE-LWE, dMCFE-Paillier | ABMPSW20, App. D, Fig. 14 (DMCFE from MCFE) |
| | dMCFE-Ring-LWE | BMMS21, Sec. 6 ([ABKW19](https://eprint.iacr.org/2019/020.pdf) + GDSum) |
| | dMCFE-FH-Multi-IPE *(pairing)* | [NPS24](https://doi.org/10.62056/andkp2fgx) (IACR CiC '24), Fig. 1 |
| **Threshold** | tMIFE (DDH, LWE), tMCFE (DDH, LWE, Ring-LWE, FH-Multi-IPE) | [Xu+24](https://doi.org/10.1109/TDSC.2024.3350206) (IEEE TDSC '24) ¹ |
| **Quadratic** | SGP | [DSGPP18](https://eprint.iacr.org/2018/206.pdf) (ePrint), Fig. 2 |
| | Multi-input quadratic FE (cross-user terms) | [AGT22](https://eprint.iacr.org/2022/1168.pdf) (TCC '22), Sec. 5 |
| | Quad (public-key) | [Gay20](https://eprint.iacr.org/2020/093.pdf) (PKC '20), Fig. 4 |

> ¹ Threshold variants add a Shamir secret-sharing layer on top of the
> corresponding MIFE or MCFE scheme; there is no separate standalone paper
> for each combination. tMCFE-DDH shares the single aggregated functional
> key of CDGPP18 and tMIFE-DDH the MIFE-DDH key, both interpolated in the
> exponent. tMIFE-LWE, tMCFE-LWE and tMCFE-Ring-LWE additionally flood partial decryptions with noise, and
> tMCFE-FH-Multi-IPE shares the FH-MCFE key exponents (servers hold their
> shares only as group elements) and combines partial decryptions in `G_T`; these four threshold layers are our own (status
> *experimental*, see the module docstrings). In tMIFE-LWE, tMCFE-Ring-LWE
> and tMCFE-FH-Multi-IPE clients tag every ciphertext for every server
> (HMAC, over the label too), and servers only decrypt ciphertexts tagged for
> them under the label they serve. For `t < s`, tMIFE-LWE, tMCFE-LWE and
> tMCFE-Ring-LWE share the key with a replicated (0/1) sharing rather than
> Shamir's, whose partial decryptions reveal the decryption noise modulo
> small integers there (v0.3.0 is affected, see
> [Known Issues](#known-issues--security-notice)). The opt-in
> `sharing="replicated-seeded"` expands all key pieces but one from secret
> 256-bit seeds, so a server stores at most one full piece; its key shares hide
> the key only computationally (SHAKE-256 keyed by the seeds as a PRF), and full
> replicated sharing stays the default for `t < s`. All threshold variants assume honest-but-curious servers (see
> [Known Issues](#known-issues--security-notice)).

> **⚠️ Research prototype.** Every scheme except the three new DDH variants
> over ristretto255 (*unreviewed*) has been checked against its paper, but
> four threshold layers and the dMCFE-DDH carry mask are our own
> (*experimental*) and the default parameters are toy-sized. Check the
> [Scheme Status](#scheme-status) table and the
> [Known Issues / Security Notice](#known-issues--security-notice) before
> relying on any privacy property.

## Scheme Status

Every registered scheme carries a review status, declared with the scheme
in the registry [`pyfe4ai/core/registry.py`](pyfe4ai/core/registry.py) and
available at runtime as `pyfe4ai.get_scheme_status("family/variant")` (and
`pyfe4ai.get_scheme(...).status`, `FESchemeWrapper(...).status`). Instantiating a key generator or crypto
object of a known-insecure scheme emits a `pyfe4ai.FESecurityWarning`. The
table below is rendered by `pyfe4ai.core.status.markdown_table()` and a
test keeps it in sync.

- **reviewed**: checked against the cited construction; no defect known.
  *faithful* = the published construction as written; *adapted* = the
  published construction with the documented deviation in the last column.
- **experimental**: implemented and checked here, but the construction or
  its security argument is our own (*ad-hoc*) rather than published; no
  defect known.
- **unreviewed**: not yet checked against its paper; no defect is known,
  but none was looked for systematically either.
- **known-insecure**: a concrete privacy defect is known; the scheme computes
  correct results but provides no meaningful privacy. Use it only for
  functionality or performance experiments.

Currently 34 reviewed, 5 experimental and 3 unreviewed out of 42 registered
schemes; none is known to be insecure. The unreviewed ones are the DDH
schemes over ristretto255 (`*/ddh_ristretto255`, selected with
`group="ristretto255"`), new code.

| Scheme | Status | Fidelity | Construction | Notes |
|---|---|---|---|---|
| `sife/ddh` | reviewed | faithful | ABDP15 Construction 1 (DDH) | signed x, y; decryption bound dlog_bound (default 10^(precision + 2)) |
| `sife/ddh_ristretto255` | unreviewed | unknown | ABDP15 Construction 1 (DDH) in ristretto255 | ristretto255 (RFC 9496, about 126-bit security: Pollard rho in a group of order 2^252) instead of a 3072-bit Z_p^* subgroup; group arithmetic in Rust (curve25519-dalek) with the native extension, else in pure Python and not constant-time; new code, not yet reviewed |
| `sife/lwe` | reviewed | adapted | ABDP15 full version (ePrint 2015/017), LWE | the LWE instantiation as revised in the full version (no r-ary decomposition); its selective IND-CPA proof is the full version's Thm. 4.4 with Thms. 6.6 and 6.7 (Bourse 2017 Thm. 4.3.5 with 8.2.2, 8.2.3, 8.3.2), not PKC Thm. 2, which does not cover noisy LWE keys; it reduces to LWE with errors sigma / (1 + 2X sqrt(eta)), the width standard reads the 128-bit table at; signed X/Y-bounded inputs with centred decoding; GoFE parameter recipe; pk noise E <- D_{Z,sigma_q} exactly (FACCT) under standard, the library-wide sampler under toy; A expanded from a public seed (SHAKE-128) |
| `sife/paillier` | reviewed | faithful | ALS16 (DCR) | D_{Z,sigma} realised as a high-precision rounded Gaussian |
| `sife/damgard` | reviewed | faithful | ALS16 DDH (ACFGU18 Fig. 7, k = 1) | h = g^a with a discarded |
| `sife/ring_lwe` | reviewed | adapted | BMMS21 Sec. 4 (selective) + Remark 1 batching | signed messages and weights (p = 2lXY + 2, centred decoding); widths from the Sec. 4.2 constraints (the paper's Table 1 misses its smoothing conditions); one ciphertext packs up to ring_n vectors; single NTT prime q; ring_n defaults to the toy value 32 |
| `sife/fullysec_ring_lwe` | reviewed | adapted | BMMS21 Sec. 5 (adaptive) | m >= 5 key polynomials instead of m = 3, which admits no modulus under the paper's App. C.1 constraints (sigma' >= q^(2/m) from Ring-LHL); signed inputs (p = 2lXY + 2, leveraging over (2X+1)^(2l)); one vector per ciphertext. Experimental: ndarray_layout = "packed" (up to ring_n vectors per ciphertext, leveraging over the span of the differences) rests on this project's own argument, independently checked, not externally reviewed |
| `sife/fullysec_lwe` | reviewed | adapted | ALS16 LWE (ACFGU18 Fig. 8) | signed inputs (K = 2 eta X Y + 1, centred decoding); GoFE parameter recipe with m = 4 n log2 q (ALS16 Theorem 2); D_{Z,sigma} realised as a rounded Gaussian |
| `sife/ddh_dynamic` | reviewed | adapted | ABDP15 Construction 1, variable length | vectors of length <= eta handled as zero padding (prefix keys, truncated ciphertexts) |
| `sife/fh_ipe_pairing` | reviewed | adapted | KLMMRW18 Sec. 3 | vectors padded to (x, 1, 0) / (y, 0, 1), so eta = 1 and zero vectors are safe; keys in G1, ciphertexts in G2; generic bilinear group model; the encryption key is master-secret equivalent |
| `sife/part_fh_ipe_pairing` | reviewed | adapted | Gay20 Fig. 8 (CPA, k = 1) | M is a random full-rank public matrix (subspace_dim < eta, default 1), published with V^T M in the clear; keys leak M^T y, which for small eta - subspace_dim often determines a bounded y; public Enc needs bounded t; Enc' is run by the key generator; no QANIZK |
| `mife/ddh` | reviewed | adapted | ACFGU18 Fig. 4 + ABDP15 (DDH) | ACFGU18 instantiates Fig. 4 with ALS16; here the selectively secure ABDP15 IPFE in every slot |
| `mife/ddh_ristretto255` | unreviewed | unknown | ACFGU18 Fig. 4 + ABDP15 (DDH) in ristretto255 | ristretto255 (RFC 9496, about 126-bit security: Pollard rho in a group of order 2^252) instead of a 3072-bit Z_p^* subgroup; group arithmetic in Rust (curve25519-dalek) with the native extension, else in pure Python and not constant-time; new code, not yet reviewed |
| `mife/lwe` | reviewed | adapted | ACFGU18 Fig. 4 + ABDP15 (LWE) | ACFGU18 instantiates Fig. 4 with ALS16; here the selectively secure ABDP15 IPFE (same two-step decryption; full-version proof, see sife/lwe), with parameters for dimension n * eta (footnote 14); the slot challenges of the ACFGU18 reduction differ by up to 4X, so standard reads the 128-bit table at sigma / (1 + 4X sqrt(eta)); slots share A, expanded from a public seed (SHAKE-128); pk noise exact (FACCT) under standard |
| `mife/paillier` | reviewed | adapted | ACFGU18 Fig. 4 + ALS16 (DCR, Fig. 9) | slots share N and g; D_{Z,sigma} realised as a high-precision rounded Gaussian |
| `mife/damgard` | reviewed | adapted | ACFGU18 Fig. 4 + ALS16 (DDH, Fig. 7) | slots share the group and h (the matrix A of Fig. 7) |
| `mife/ring_lwe` | reviewed | adapted | BMMS21 Sec. 6 MIFE-compiler (ACFGU18) | slots share a; parameters sized for dimension n * eta, with the Sec. 4.2 widths for bound 3X (ACFGU18 Fig. 4, footnote 12) |
| `mife/fullysec_lwe` | reviewed | adapted | ACFGU18 Fig. 4 + ALS16 (LWE, Fig. 8) | slots share A; parameters derived for dimension n * eta and bound 3X (ACFGU18 footnote 14), m = 4 n log2 q |
| `mife/fh_ipe_pairing` | reviewed | adapted | ACFGU18 Fig. 12 (k = 1) | one r per key shared by all slots; per-slot FH-IPFE as in mcfe/fh_ipe (LV16 padding of App. B); asymmetric pairing (default BLS12-381); keys carry a random id, never a digest of y |
| `mife/fh_multi_ipe_pairing` | reviewed | adapted | DOT18 Sec. 3.1 (bounded, k-LIN, k = sec_level) | n >= 2 enforced (zero sharing r_i); no Remark B.1 SKE layer, so hiding assumes every slot contributes a ciphertext; slot i's client holds B^_i and can decode its own slot of a key; keys are group elements only, decryption does not need y |
| `mife/ddh_threshold` | reviewed | adapted | ACFGU18 Fig. 4 + ABDP15 + Shamir (TAPFed setting) | the functional key (d_i, z) is Shamir-shared mod q; any t servers interpolate in the exponent (threshold-ElGamal style), as mcfe/ddh_threshold |
| `mife/lwe_threshold` | experimental | ad-hoc | MIFE-LWE + replicated / Shamir sharing + smudging | MIFE layer as published; the threshold layer (statistical flooding, derandomised noise, enlarged q, clients' HMAC tags on ciphertexts so that servers only decrypt ciphertexts made by the clients, replicated key sharing for t < s, where Shamir shares let partials reveal the decryption noise modulo small integers) is our own; the opt-in sharing='replicated-seeded' expands all key pieces but one from secret 256-bit seeds and hides the key shares only computationally (SHAKE-256 keyed by the seeds as a PRF); full replicated sharing stays the default for t < s; standard refuses it (at the reduction width of mife/lwe it needs the very end of the 128-bit table, 5888 to 6144, for a few small bounds and more for most others) |
| `mcfe/ddh` | reviewed | faithful | CDGPP18 |  |
| `mcfe/ddh_ristretto255` | unreviewed | unknown | CDGPP18 in ristretto255 | ristretto255 (RFC 9496, about 126-bit security: Pollard rho in a group of order 2^252) instead of a 3072-bit Z_p^* subgroup; labels hashed with RFC 9380 hash_to_ristretto255; group arithmetic in Rust (curve25519-dalek) with the native extension, else in pure Python and not constant-time; new code, not yet reviewed |
| `mcfe/lwe` | reviewed | adapted | ABMPSW20 Fig. 4, selective parameters | signed messages and weights (K = 2nXY + 1, centred decoding); t <- D_{Z,sigma_t} exactly (FACCT) by default, rounded selectable under toy |
| `mcfe/paillier` | reviewed | faithful | ABMPSW20 Fig. 3 | D_{Z,sigma} realised as a high-precision rounded Gaussian |
| `mcfe/damgard` | reviewed | faithful | ABMPSW20 Fig. 2 (D_k-MDDH) | k = mddh_k (default 1, where it equals CDGPP18) |
| `mcfe/ring_lwe` | reviewed | adapted | BMMS21 Sec. 6 MCFE-compiler (identity-based) | functional keys are bound to a label; ndarray values are packed into one ciphertext per chunk, each chunk with its own label pad; parameters as mife/ring_lwe (widths for bound 3X) |
| `mcfe/fullysec_lwe` | reviewed | adapted | ABMPSW20 Fig. 4, adaptive parameters | signed messages and weights (K = 2nXY + 1, centred decoding); t <- D_{Z,sigma_t} exactly (FACCT) by default, rounded selectable under toy; m0 grows with n log2(2X + 1) |
| `mcfe/fh_multi_ipe_pairing` | reviewed | adapted | AGT21b Sec. 6.2 (as recapped in NPS24 Sec. 2) | FH-IPFE = ACFGU18 Fig. 12 (n = 1, SXDH) with LV16 padding; asymmetric pairing (default BLS12-381); keys carry a random id, never a digest of y |
| `mcfe/fh_multi_ipe_pairing_threshold` | experimental | ad-hoc | AGT21b FH-MCFE + Shamir | FH-MCFE layer as published; the key exponents Shamir-shared in the exponent ([e^(k)]_2, [z^(k)]_T), partial decryptions in G_T and labelled, client-tagged ciphertexts are our own; FH-IPFE = ACFGU18 Fig. 12 (n = 1, SXDH) with LV16 padding; asymmetric pairing (default BLS12-381); keys carry a random id, never a digest of y |
| `mcfe/fh_multi_ipe_pairing_decentralized` | reviewed | adapted | NPS24 Fig. 1 (selective FH-DMCFE) | explicit tag-f, each client's key depends on (tag-f, y_i) only and is derived deterministically; one key per client and tag-f (enforced per client object); FH-IPFE = ACFGU18 Fig. 12 (n = 1, SXDH) with LV16 padding; asymmetric pairing (default BLS12-381); keys carry a random id, never a digest of y |
| `mcfe/ddh_threshold` | reviewed | adapted | CDGPP18 + Shamir (TAPFed setting) | the aggregated key is Shamir-shared; any t servers interpolate in the exponent (threshold-ElGamal style) |
| `mcfe/ddh_decentralized` | experimental | ad-hoc | ABMPSW20 Fig. 14 (MCFE1 = CDGPP18) + carry mask | Fig. 14 is stated for MCFEs without a discrete log in decryption; contributions are reduced mod q and masked by q rho_i (rho_i from a per-client key and H(y)) so that the carry of their sum stays hidden -- our own argument; MCFE2 = ABMPSW20 Fig. 3 (DCR); the zero-sum MCFE2 keys (pairwise masks) are sampled in-process |
| `mcfe/lwe_threshold` | experimental | ad-hoc | ABMPSW20 Fig. 4 + replicated / Shamir sharing + flooding | MCFE layer as published; the threshold layer (replicated key sharing for t < s, Shamir for t = s) and its LWE-based flooding argument are our own (see the module docstring); the opt-in sharing='replicated-seeded' expands all key pieces but one from secret 256-bit seeds and hides the key shares only computationally (SHAKE-256 keyed by the seeds as a PRF); full replicated sharing stays the default for t < s |
| `mcfe/lwe_decentralized` | reviewed | adapted | ABMPSW20 Fig. 14 (MCFE1 = Fig. 4) | MCFE2 = ABMPSW20 Fig. 3 (DCR); the zero-sum MCFE2 keys (pairwise masks) are sampled in-process |
| `mcfe/ring_lwe_threshold` | experimental | ad-hoc | BMMS21 identity-based MCFE + replicated / Shamir sharing + smudging | MCFE layer as published; the threshold layer (statistical flooding, derandomised noise, enlarged q, clients' HMAC tags on ciphertexts, replicated key sharing for t < s, where Shamir shares let partials reveal the decryption noise modulo small integers) is our own; the opt-in sharing='replicated-seeded' expands all key pieces but one from secret 256-bit seeds and hides the key shares only computationally (SHAKE-256 keyed by the seeds as a PRF); full replicated sharing stays the default for t < s; by default binary ciphertexts (format 2: ct0 cut to the batch length and every value strictly below q, tags over the parameters' digest and the bytes, binary partials and key shares), an RNS modulus (q a product of at least three NTT primes, each above s) and split key shares (sk_y once per weights, the label terms per round); ct_format='json', ring_modulus='prime' and key_shares='full' select the formats of earlier versions |
| `mcfe/ring_lwe_decentralized` | reviewed | adapted | BMMS21 Sec. 6 identity-based DMCFE (ABKW19 + GDSum) | GDSum realised as pairwise PRF zero-sum masks (its full specification is only in the ePrint version); pairwise secrets sampled in-process |
| `mcfe/paillier_decentralized` | reviewed | adapted | ABMPSW20 Fig. 14 (MCFE1 = Fig. 3) | MCFE2 = ABMPSW20 Fig. 3 (DCR); the zero-sum MCFE2 keys (pairwise masks) are sampled in-process |
| `quadratic/quad` | reviewed | adapted | Gay20 Fig. 4 over Fig. 8 (CPA) | signed inputs (discrete log in [-n m B^3, n m B^3]); CPA variant (no QANIZK); asymmetric pairing (default BLS12-381) |
| `quadratic/sgp` | reviewed | adapted | DSGPP18 Fig. 2 | public key pk = (g1^s, g2^t) as in the paper; signed inputs with one bound B for x, y and F (discrete log in [-n^2 B^3, n^2 B^3]); generic bilinear group model |
| `quadratic/miqfe` | reviewed | adapted | AGT22 Sec. 5 (MIFE for quadratic functions) | cross-user terms only (Def. 2.9); per-user encryption keys made of master ciphertexts; building blocks TAO20 FH-IPFE, Fig. 9 and Fig. 1 as in the paper; PRF = HMAC-SHA512; pos-security (every user contributes); k = mddh_k >= 2 |

## Security Profiles

Every key generator accepts a `security_profile` entry in its config
([`pyfe4ai/core/profiles.py`](pyfe4ai/core/profiles.py)). The default,
`"toy"`, keeps the small default parameters -- fast, and **not secure**; a
config without the entry behaves exactly as before. `"standard"` fills in
parameters aiming at 128-bit classical security and rejects explicit values
below them with `FEValidationError`:

```python
kg, SIFEDDH = pyfe4ai.create("sife/ddh", {"eta": 3, "security_profile": "standard"})  # 3072-bit group
```

| Family | `standard` parameters | Measured cost (one core, 2 clients) |
|---|---|---|
| Discrete log: DDH, Damgård / MDDH (SIFE, MIFE, MCFE, tMIFE, tMCFE) | `sec_param = modulus_length = 3072`: RFC 3526 MODP group 15 (NIST SP 800-57 Part 1: 3072 bits for 128-bit security) | setup < 0.1 s; encrypt and aggregate < 1 s |
| `mcfe/ddh_decentralized` | as above; key shares (masked with `stat_param` = 128 bits against carries) are combined under a DCR modulus of about 6410 bits | setup ≈ 2 min (two 3207-bit safe primes); key share 0.7 s |
| DCR: Paillier (SIFE, MIFE, MCFE) | `bit_length = 1536`: two 1536-bit safe primes, N ≈ 3072 bits, generated for every key generator (a cached modulus could have known factors) | setup a few seconds (two 1536-bit safe primes) |
| Ring-LWE (SIFE, MIFE, MCFE, tMCFE, dMCFE) | κ = `sec_param` = 128 and `ring_n = "auto"`: the smallest n in {1024, …, 32768} whose derived q (including the threshold flooding) meets a lattice-estimator bound for Ring-LWE with Gaussian secret and errors (`ring_lwe_utils.RING_LWE_128_LOG_Q`, rows by width 2–16; regenerate with `tools/params/estimate_ring_lwe.py`), read at the smallest RLWE width of the scheme's proof -- about 8 with the BMMS21 Sec. 4.2 widths, giving log2 q ≤ 31 / 57 / 111 / 218 / 435 / 872 (the Homomorphic Encryption Standard's 27 / 54 / 109 / 218 / 438 / 881 is about 125 bits by the same estimator); an explicit `ring_n` is checked against the same bound | Python backend: SIFE / MIFE (eta = 10, X = Y = 50): n = 4096; one ciphertext holds up to 4096 vectors at the same cost, encrypt 0.4–0.5 s (0.12 ms per vector when full), decrypt 0.04 s (SIFE) / 0.08 s (MIFE). MCFE, 10 000 gradient elements (X = 10^4, Y = 10): `ring_lwe` n = 4096, encrypt 0.36 s per client (exact sampler), aggregate 0.16 s; measured with v0.3.0: `ring_lwe_threshold` (3 servers, t = 2) n = 16384, encrypt 0.84 s per client, key share 0.21 s and partial decryption 0.67 s per server, combine 0.02 s; `ring_lwe_decentralized` n = 4096, encrypt 0.21 s per client, key share 0.02 s per client and label, decrypt 0.18 s. With the Rust extension the FL path is far faster (one thread, `standard`, 3 clients; MCFE 10^4 elements: about 18 ms per client with the exact sampler, 0.25 ms online after an offline phase, and 6 ms to aggregate; tMCFE with its defaults -- binary ciphertexts, a 212-bit RNS q of four factors, split key shares -- `ring_n` 8192, s = 3: 27 ms per client, 8 ms per partial decryption and 11 / 14 ms for the round's key shares at t = 2 / t = 3, 29 / 39 ms for the servers and the combiner together; with the JSON format, a prime q and whole key shares of earlier versions 39 ms, 65 / 55 ms and 41 / 43 ms), see `benchmarks/PERFORMANCE.md`. Functional keys of the MCFE variants are bound to a label (round) |
| Adaptively secure Ring-LWE: `sife/fullysec_ring_lwe` (BMMS21 Sec. 5) | as Ring-LWE, plus `m = "auto"`: the smallest number m ≥ 5 of key polynomials whose q meets the same bound (the paper's m = 3 admits no modulus under its own constraints, see the module docstring); widths from Theorems 1–2 and Appendix C.1 with complexity leveraging over (2X+1)^(2·eta); exact FACCT Gaussians by default (`gaussian_sampler`); experimental `ndarray_layout = "packed"`: up to n vectors per ciphertext, leveraging over the span of the differences | eta = 4, X = Y = 4: n = 4096, m = 17, setup 0.25 s, encrypt 0.12 s, decrypt 66 ms; eta = 10, X = Y = 50: n = 8192, m = 8, setup 1.0 s, encrypt 0.31 s, decrypt 0.13 s; packed eta = 1, X = 100, Y = 1: n = 4096, m = 12, 4096 elements per ciphertext (182 bytes each), setup 75 ms, encrypt 79 ms, decrypt 46 ms per ciphertext (Rust extension, exact sampler) |
| Dense LWE: the ABMPSW20 MCFE family (`mcfe/lwe`, `fullysec_lwe`, `lwe_threshold`, `lwe_decentralized`) | `lwe_n = "auto"`: the smallest n0 in {4096, 4608, …, 16384} whose derived q meets a [lattice-estimator](https://github.com/malb/lattice-estimator) bound for LWE(n0, q, ⌈2√n0⌉) with a uniform secret -- the assumption of the scheme's proof (`lwe_utils.DENSE_LWE_128_LOG_Q`, e.g. log2 q ≤ 118 / 173 / 227 for n0 = 4096 / 6144 / 8192; regenerate with `tools/params/estimate_dense_lwe.py`); an explicit `lwe_n` is checked against the same table. No public matrix: a label is hashed to n0 + m0 values. `lwe_decentralized` combines key shares under a DCR modulus of ≥ 3072 bits, so its key derivation is not post-quantum | n0 = 7168–8192 for small bounds (stat_param 128); Python backend: about 7 ms per element and client to encrypt and 6–8 ms per element to aggregate (one label per element); with the Rust extension 93 / 101 ms for 200 elements (client / aggregator, 3 clients); dMCFE key share ≈ 90 s per client |
| LWE with a public matrix: ABDP15 (`sife/lwe`, `mife/lwe`) | `lwe_n = "auto"`: the smallest n in {512, 576, …, 6144} whose (q, σ′) meets a lattice-estimator bound on log2(q/σ′) for wide-error LWE with a uniform secret (`lwe_utils.WIDE_LWE_128_LOG_RATIO`, rows by width; regenerate with `tools/params/estimate_wide_lwe.py`), read at σ′ = σ/(1 + B√eta), the error width of the LWE problem that the security proof reduces to (B = 2X, or 4X per MIFE slot; ABDP15 full version, Thm. 4.4 with the width derived in the proof of its Thm. 6.6 = Bourse 2017, Thm. 8.2.2; its Sec. 6.2 parameter list states the stronger (ℓB + 1)σ′), not at the public key's own σ, which gives 17–67% larger dimensions for 8 ≤ X = Y ≤ 1000; an explicit `lwe_n` is checked the same way. A (m ≈ n·log2 q rows) is published as a 32-byte seed and expanded with SHAKE-128. `matrix_mode` (key generator and crypto configs, forwarded by the ML adapter): `"memory"` keeps A expanded, `"stream"` expands it again for every batch | n = 1280 / 1536 for X = Y = 50 / 256 and eta = 10 (A: 1.3·10^8 / 2.2·10^8 entries; 896 / 1088 when the table was read at σ); MIFE is sized as if its vectors had dimension n·eta (ACFGU18, footnote 14), e.g. n = 1408 for 100 slots of dimension eta = 1 with X = Y = 50 (1664 for eta = 10). Encryptor alone, n = 1536: `"memory"` 0.24 s per encryption and 5.3 GB, `"stream"` 6.9 s and 0.7 GB (0.11 s and 3.4 s at the former n = 1088); the ndarray helpers encrypt in batches of 64 (eta = 1, n = 1088: 6.6 vs 44 ms per element). Key generation (setup, the noise of the keys sampled exactly) 8.3 s / 17 s for X = Y = 50 / 256, 47 s for the 100 MIFE slots. The public key is m × eta per slot (46 MB of JSON for eta = 10), so keep eta small. `mife/lwe_threshold` (experimental) is refused, see below |
| Pairing: FH-IPE, FH-MCFE family, quadratic FE | BLS12-381 (Rust extension; the `toy` default too): about **120–126-bit** security after the exTNFS attacks, below the target; a `FESecurityWarning` says so | `report` benchmark profile: SGP 30 ms, Quad 48 ms, MCFE FH-Multi-IPE 47 ms end to end; a pairing 0.8 ms |
| Refused | `sife/fullysec_lwe`, `mife/fullysec_lwe` (ALS16), `mife/lwe_threshold`, `mcfe/paillier_decentralized` | `FEValidationError`: no parameter set that is both secure and usable. ALS16 at a 128-bit dimension (lattice estimator, n = 3328–4352 for X = Y ≤ 50 and eta ≤ 10, with m = 4n·log2 q as its Theorem 2 requires): A has 1–2·10^10 entries and a ciphertext 2.9–4.5·10^6 coordinates of 219–258 bits (80–145 MB); its LWE-to-mheLWE reduction loses about n^6 in the noise ratio, so the parameters cannot be tightened -- for adaptive security use `sife/fullysec_ring_lwe`, otherwise the Ring-LWE scheme of the same family. tMIFE-LWE: read at σ′, the LWE of its flooding-enlarged q (stat_param 128, two clients) needs a dimension at the very end of the table (5888–6144, A: 6.5–7.8·10^9 entries) for a few small bounds, e.g. X = Y ∈ {1, 2, 3, 8, 10} at eta = 1, and beyond it for most others (X = Y = 4–6, X = Y ≥ 12, most eta ≥ 2; about 6500 for X = Y = 50, q of 217 bits; read at σ it took 5952, with key generation 38 min and one encryption 572 s in `"stream"` mode); use `mife/ddh_threshold`, or `mcfe/ring_lwe_threshold` with labels. The opt-in KL-divergence accounting of the flooding (`flooding_accounting = "kl"`, t = s and an explicit `max_partial_decryptions` budget, a weaker guarantee of about stat_param − 6.85 bits) would bring it to about 4864 at σ′ for s = t = 3 and 2^40 flooded values (A: 4.1·10^9 entries, `"stream"` mode), but `standard` refuses tMIFE-LWE with either accounting; "kl" applies to it under `toy`. dMCFE-Paillier needs a DCR modulus above 15000 bits for combining key shares |

The statistical parameter `stat_param` (rounding and leftover-hash slack,
flooding, masks) is 128 under `standard`, a statistical distance of
`2^-128` per sample; lower explicit values are refused. The profile sizes the
hardness assumptions only; the other caveats under
[Known Issues](#known-issues--security-notice) (research prototype, no
side-channel hardening, trusted parameter cache) still apply.

## Repository Layout

- `pyfe4ai/`: installable Python package
  - `core/`: what every scheme shares
    - `registry.py`: the registry of the schemes (classes, kind, review status, capabilities, refusals) and `create()`
    - `base.py`: key-generator and crypto base classes, the parameter cache
    - `config.py`: the config class of every scheme and the rules that combine options
    - `formats/`, `versions.py`: the serialized objects and their version markers
    - `status.py`: review status of every registered scheme (the table above)
    - `profiles.py`: `security_profile` (`"toy"` / `"standard"`)
    - `one_time.py`: one-time precomputations of the offline / online encryption
    - `exceptions.py`: `FEError` exception hierarchy; `constants.py`: scheme types and defaults
  - `schemes/`: the schemes
    - `sife/`: single-input families
    - `mife/`: multi-input families
    - `mcfe/`: multi-client families
    - `quadratic/`: quadratic functional encryption families
    - `common/`: DDH and EC-DDH base classes, `abdp15_lwe.py` (shared ABDP15 parameters and seeded public matrix of SIFE- and MIFE-LWE)
    - `threshold/`: shared modules of the threshold schemes: `sharing.py` (Shamir, replicated and seeded replicated sharing), `binding.py` (`pp_binding`, construction ids, client MAC tags, sharing ids, state versions), `codec.py` (JSON and binary encodings, strict parsing), `store.py` (key-share stores and caches), `roles.py` (server checks, flooding noise, combination), `kl.py` (KL accounting of the flooding)
  - `primitives/`: cryptographic building blocks
    - `lattice/`: LWE / Ring-LWE parameters and NTT (`lwe_utils.py`, `ring_lwe_utils.py`), seeded public LWE matrices (`lwe_matrix.py`), arithmetic mod q (`matrix_utils.py`)
    - `sampling.py`: discrete Gaussian samplers; `hashing.py`: hashing to groups and to `Z_q` (label hashes)
    - `groups/`: safe-prime and DCR groups (`integer_groups.py`), ristretto255, modular arithmetic, in-memory dlog table + BSGS (`dlog_solver.py`)
    - `pairing/`: pairing layer (BLS12-381 from the Rust extension, other curves from charm-crypto)
  - `backend/`: native (Rust) or Python implementation, threads
  - `ml/`: ML integration (encrypted inference, FL aggregation, threshold servers) and `quantization.py` (float ↔ integer)
  - `_validation.py`: strict integer checks and rounding of scheme inputs
- `rust/`: sources of the native extension (`rust/native`, PyO3) and of its computation core (`rust/core`)
- `docs/`: Sphinx API documentation (see [API Documentation](#api-documentation))
- `tests/`: end-to-end and regression tests (`sife/`, `mife/`, `mcfe/`, `quadratic/`)
- `examples/`: minimal runnable usage examples
- `benchmarks/`: lightweight benchmark scripts
- `tools/`: `build_rust.py` (builds the native extension); `params/` (`estimate_dense_lwe.py`, `estimate_wide_lwe.py`, `estimate_ring_lwe.py`, `estimate_rns_modulus.py` regenerate the LWE and Ring-LWE security tables with SageMath and the lattice estimator)
- `site/`: project landing page (published with the docs to GitHub Pages)
- `pyproject.toml`: package metadata and the dependencies (extras `test`, `pairing`, `doc`)
- `environment.yml`: Conda environment (Python and the conda-forge packages); the dependencies come from `pyproject.toml` (`pip install -e ".[test]"`)
- `docker/Dockerfile.pairing`: Docker-based pairing-enabled test environment

## Naming Convention

Scheme modules are organized by family and then named by instantiation:

- `<family>/<instantiation>.py`
- `<family>/<instantiation>_<variant>.py`

Examples:

- `pyfe4ai/schemes/sife/ddh.py`
- `pyfe4ai/schemes/sife/damgard_ddh.py`
- `pyfe4ai/schemes/mife/ddh_threshold.py`
- `pyfe4ai/schemes/mcfe/ring_lwe_threshold.py`

The `instantiation` segment may describe a security assumption, a concrete
cryptosystem, or a named construction lineage, depending on what is most
accurate for the scheme:

- `ddh`: DDH-based discrete-log style construction
- `lwe`: LWE-based instantiation
- `paillier`: Paillier-based instantiation
- `damgard_ddh`: Damgard-style DDH instantiation

A scheme is identified by `<family>/<variant>` (`sife/ddh`, `mcfe/damgard`,
`mcfe/ring_lwe_threshold`; the variant is the module name, but `damgard`
for `damgard_ddh.py`). Its classes are named after that id, one name per
class: `Threshold` or `Decentralized` for those variants, then the family
(`SIFE`, `MIFE`, `MCFE`, `Quadratic`), then the rest of the variant in
CamelCase with its acronyms in capitals; the key generator adds
`KeyGenerator`, the config class `Config`:

| Scheme | Crypto class | Key generator |
|---|---|---|
| `sife/ddh` | `SIFEDDH` | `SIFEDDHKeyGenerator` |
| `mcfe/ring_lwe` | `MCFERingLWE` | `MCFERingLWEKeyGenerator` |
| `mcfe/ring_lwe_threshold` | `ThresholdMCFERingLWE` | `ThresholdMCFERingLWEKeyGenerator` |
| `mcfe/ddh_decentralized` | `DecentralizedMCFEDDH` | `DecentralizedMCFEDDHKeyGenerator` |
| `mife/fh_multi_ipe_pairing` | `MIFEFHMultiIPEPairing` | `MIFEFHMultiIPEPairingKeyGenerator` |
| `quadratic/miqfe` | `QuadraticMIQFE` | `QuadraticMIQFEKeyGenerator` |

## Running Tests

```bash
python -m pytest -q                # all tests
python -m pytest -m core -q        # core FE/IPFE smoke lane only
python -m pytest -m extended -q    # pairing-heavy & quadratic families only
```

The `core` lane is the main public release lane. The `extended` lane covers
pairing-heavy and quadratic families that remain publicly available but are kept
outside the minimal smoke path.

With the Rust extension built, the tests compare the native code with the
Python reference; run the suite on each backend, and the Rust core's own tests
(`tests/test_rust_core.py` also loads the core's C ABI once it is built):

```bash
PYFE4AI_BACKEND=native python -m pytest -q   # error if the extension is not built
PYFE4AI_BACKEND=python python -m pytest -q   # the Python reference implementation
(cd rust/core && cargo test --release && cargo build --release)
```

CI runs the core lane without the extension and, in the `rust-native` job,
builds the extension and runs `cargo test`, the full suite on the native
backend (Python 3.10 and 3.13) and the full suite on the Python backend
(Python 3.10).

## Examples

Minimal runnable examples are grouped as follows:

- Core FE/IPFE
  - `examples/sife_minimal.py`
  - `examples/mife_minimal.py`
  - `examples/mcfe_threshold_demo.py`
- LWE and threshold/decentralized variants
  - `examples/lwe_minimal.py`
  - `examples/mife_lwe_minimal.py`
  - `examples/mcfe_lwe_minimal.py`
  - `examples/mcfe_lwe_decentralized_demo.py`
  - `examples/mcfe_lwe_threshold_demo.py`
  - `examples/mcfe_ring_lwe_threshold_demo.py` (threshold MCFE over Ring-LWE, scheme classes and ML adapter)
  - `examples/mife_lwe_threshold_demo.py`
- ML / privacy-preserving AI
  - `examples/ml_encrypted_inference.py`
  - `examples/ml_fl_aggregation.py`
- Pairing-based extended families
  - `examples/mife_fh_multi_ipe_minimal.py`
  - `examples/mcfe_fh_multi_ipe_minimal.py`
  - `examples/mcfe_fh_multi_ipe_decentralized_demo.py`
  - `examples/mcfe_fh_multi_ipe_threshold_demo.py`
- Quadratic FE
  - `examples/quadratic_sgp_minimal.py`
  - `examples/quadratic_quad_minimal.py`

For a short walkthrough of the example set, see `examples/README.md`.

Lightweight benchmark scripts are available under `benchmarks/`.
For benchmark entry points, see `benchmarks/README.md`.

## API Documentation

Browse the full API reference online at
**<https://spire-studio.github.io/pyfe4ai/docs/>** (rebuilt automatically on
every merge to `main`).

To build the docs locally instead:

```bash
pip install -e ".[doc]"          # install Sphinx + theme
cd docs && make html             # build HTML docs
open _build/html/index.html      # view locally
```

## Pairing-Based Schemes

The schemes in the following files need a pairing backend: the Rust
extension for BLS12-381 (the default), `charm-crypto-framework` for other
curves:

- `pyfe4ai/schemes/sife/fh_ipe_pairing.py`
- `pyfe4ai/schemes/sife/part_fh_ipe_pairing.py`
- `pyfe4ai/schemes/mife/fh_ipe_pairing.py`
- `pyfe4ai/schemes/mife/fh_multi_ipe_pairing.py`
- `pyfe4ai/schemes/mcfe/fh_ipe.py`
- `pyfe4ai/schemes/mcfe/fh_multi_ipe_pairing.py`
- `pyfe4ai/schemes/mcfe/fh_multi_ipe_pairing_threshold.py`
- `pyfe4ai/schemes/mcfe/fh_multi_ipe_pairing_decentralized.py`
- `pyfe4ai/schemes/quadratic/sgp.py`
- `pyfe4ai/schemes/quadratic/miqfe.py`
- `pyfe4ai/schemes/quadratic/quad.py`

All of them run over an asymmetric (type-3) pairing group, `BLS12-381` by
default (`pairing_group_param`, e.g. `MNT224` or `BN254` with charm; up to
v0.3.0 the default was `MNT224`, whose keys and ciphertexts BLS12-381 cannot
read); their constructions assume
SXDH or a generic asymmetric bilinear group, so symmetric groups (`SS512`,
`SS1024`) are rejected with `FEValidationError`.

`charm-crypto-framework` supports macOS, Linux, and Windows, but pairing-based
setups may still require additional system libraries depending on the platform.
If you plan to work heavily with these schemes, a Linux or Docker environment is
still the most reproducible option.

To run the pairing-enabled environment in Docker:

```bash
docker build -f docker/Dockerfile.pairing -t pyfe4ai-pairing .
docker run --rm pyfe4ai-pairing
```

## Known Issues / Security Notice

No registered scheme is currently known to be insecure (should one be found,
the [Scheme Status](#scheme-status) table marks it *known-insecure* and its
key generator emits a `pyfe4ai.FESecurityWarning`). The following
limitations remain:

| Issue | Affected schemes | Consequence |
|---|---|---|
| No labels in multi-input FE | all MIFE schemes and quadratic `miqfe` | By definition, the holder of a functional key may combine any ciphertext of each slot. Encrypting several inputs per client under one key epoch -- e.g. a gradient element by element -- therefore reveals the differences between that client's values. The ML adapter refuses MIFE for gradient aggregation; use MCFE (labelled ciphertexts) for federated learning. |
| Experimental threshold layers | tMIFE-LWE, tMCFE-LWE, tMCFE-Ring-LWE, tMCFE-FH-Multi-IPE | The threshold layers (noise flooding, Shamir sharing in the exponent) are our own and have no published security proof; see the module docstrings. |
| Honest-but-curious decryption servers | all threshold schemes (tMIFE, tMCFE) | Fewer than `t` servers learn nothing beyond the result, but partial decryptions carry no proof of correctness: one server that deviates changes the result without notice. `combine_decrypt` accepts partial decryptions for several enrolled sets (a list of `{sid: partial}` mappings) and raises `FESchemeError` if they disagree; this detects one faulty server (not which one) if every server is left out of some set, e.g. the `t + 1` subsets of size `t` of `t + 1` servers. tMCFE-DDH and tMCFE-FH-Multi-IPE servers decrypt only ciphertexts under the label they are asked for (`share_decrypt(..., label)`). tMCFE-Ring-LWE key shares and partial decryptions carry a `sharing_id`: store and transmit it with them. A combiner refuses partials of two sharings of one key (`MixedSharingsError`), which would otherwise decode to a wrong value inside the decoding bound; partials without an id combine only with each other, so a storage or transport layer that drops the id everywhere would let two sharings be mixed unnoticed. |
| One ciphertext per client and label | all MCFE / DMCFE schemes | As in the papers, a client must not encrypt two different inputs under one label (or tag); the scheme classes do not enforce it (the ML adapter refuses a second encryption per client and label within one wrapper, and FH-DMCFE client objects a second key per `tag_f`). |
| Rounded Gaussian by default outside the exact-default schemes | ALS16 (`fullysec_lwe`), the DCR schemes, and under `toy` ABDP15 SIFE / MIFE LWE (and the keys of tMIFE-LWE) | The proofs sample `D_{Z,sigma}`; these schemes follow the library-wide sampler, which by default rounds a continuous Gaussian: in double precision up to `2^40` (shared Box-Muller radii, ties to even, nothing beyond 8.57 sigma: 2^-25..2^-4 from i.i.d. `D_{Z,sigma}` per key or ciphertext per the October 2026 crypto review) and with MPFR at `bits(sigma) + 320` bits beyond. `sampler="facct"`, `gaussian_sampler("facct")`, `set_gaussian_sampler("facct")` or `PYFE4AI_GAUSSIAN_SAMPLER=facct` select the exact sampler for them (about 2^-185 per sample). The Ring-LWE MIFE / MCFE family, the dense-LWE MCFE keys `t` and SIFE FullySec Ring-LWE sample exactly by default and record it in their public parameters; for them the process-wide choice and the environment variable do not apply, and `standard` refuses `"rounded"`. ABDP15 (`sife/lwe`, `mife/lwe`; `mife/lwe_threshold` is refused under `standard`) does the same for its public keys' noise under `standard`, and under `toy` (also the keys of `mife/lwe_threshold`) follows its config's `gaussian_sampler`, else the library-wide choice; its public parameters record the sampler either way. The threshold flooding always uses the MPFR rounded sampler. Neither sampler is constant-time. |
| Toy default parameters | all | `sec_param` is a modulus bit length (default 128), not a security level; LWE dimensions default to 32–64 and the Ring-LWE dimension to 32; the statistical parameter `stat_param` defaults to 64. All far below any real security level (a key column of SIFE-LWE at its defaults falls to plain LLL in minutes). Pass `security_profile="standard"` for 128-bit-oriented parameters -- except for pairings (BLS12-381, about 120–126-bit) and the refused ALS16, tMIFE-LWE and dMCFE-Paillier schemes; see [Security Profiles](#security-profiles). |
| Parameter cache | key generators that cache parameters (most; not MCFE `lwe`, `fullysec_lwe`, `lwe_threshold`, the FH-MCFE family, MIFE `fh_ipe_pairing` and quadratic `quad` and `miqfe`) | Public parameters are written to and re-loaded from `<cache>/authority/...`, where `<cache>` is the `cache_dir` entry of the key generator's config, else `PYFE4AI_CACHE_DIR`, else the per-user cache directory (`~/Library/Caches/pyfe4ai` on macOS, `$XDG_CACHE_HOME/pyfe4ai` or `~/.cache/pyfe4ai` on Linux, `%LOCALAPPDATA%\pyfe4ai\Cache` on Windows). A cached file is used only if it matches the configuration; discrete-log groups are checked for primality, size, safe-prime structure, generator order and (at RFC 3526 sizes) for being the RFC prime; DCR moduli for size, compositeness and small factors (their factors are never stored); all other cached values are re-derived on load or expanded from a stored seed (LWE and Ring-LWE parameters and public matrices, the Paillier base, the Damgard `h`); under `standard` the lattice schemes re-check their moduli against the 128-bit tables. An unreadable file is regenerated, and files are written atomically. No check can tell whether someone knows the factors of a DCR modulus, so under `standard` the Paillier moduli and the dMCFE `N2` are generated for every key generator, never loaded or stored; under `toy` a planted modulus with known factors is still used: use a cache directory that nobody else can write to. |
| The ML adapter plays every party | `pyfe4ai.ml` | `FESchemeWrapper` runs the key generator, so it holds the master key and acts as clients, authority and decryptor in one trust domain; use it for experiments, not as a deployment. For `mcfe/ring_lwe_threshold` it also runs the decryption servers (`partial_decrypt_gradients`, `combine_partial_decryptions` show a deployment's messages, and a `PreparedAggregation` holds the servers' label key shares), so there the threshold protects nothing. By default `aggregate_gradients` needs a ciphertext and a non-zero weight for every enrolled client (`allow_partial=True` relaxes this), a client encrypts at most once per label and a label is aggregated with one weight vector; these checks are per wrapper object, and they do not stop very unequal weights (e.g. `(W, 1)` with `W` above twice the largest gradient entry), which separate two clients' gradients in one aggregate -- as with any inner-product FE, the weights are the key holder's choice. A pickled or copied wrapper (`multiprocessing`, Ray) carries the master key and every client's key: ship a worker its client's own crypto object (the scheme class built from `wrapper.pp` and `wrapper.kg.get_private_keys(nid)`), not the wrapper. |

**v0.3.0 and earlier are affected: Lagrange-multiplier leak in the threshold
LWE layers** -- severity High, practical impact Medium; fixed after v0.3.0,
see [`CHANGELOG.md`](CHANGELOG.md).

* **What.** `mife/lwe_threshold`, `mcfe/ring_lwe_threshold` and
  `mcfe/lwe_threshold` Shamir-shared the functional key, for every `t < s`.
  An honest partial decryption `lambda_k^E F(k) + e_k` differs from its
  simulation from the result and the key shares of a coalition `C` by
  `M nu`: `nu` is the decryption noise, which depends on the clients'
  secret LWE errors and encryption randomness, and `M` is a product of
  Lagrange coefficients (`lambda_k^E prod_{j in C} (k - j) / (0 - j)` when
  `|C| = t - 1`). Whenever `M = r / d` is not an integer, `d` times the
  partial minus the computable part reveals `nu mod d`, whatever the
  flooding width.
* **Who can read it.** For every `t < s` with `s >= 3`:
  * **`t - 1` corrupted servers.** For example, with `s = 3, t = 2`, server 2
    corrupted and `E = {1, 3}`, they read `nu mod 4` from every tMIFE-LWE
    partial and every tMCFE-Ring-LWE coefficient. Combining the partials of
    all enrolled sets for one ciphertext raises the modulus: 4 at
    `(s, t) = (3, 2)`, 144 at `(5, 3)`, about `2^25` at `(10, 6)`, and at
    most `2^25.5` for `s <= 10`.
  * **Fewer corrupted servers.** These can suffice: one corrupted server
    gives modulus 3 at `(4, 3)` and 5 at `(5, 4)`.
  * **No corrupted server at all**, for many `(s, t)`: the aggregator
    alone, who sees the partials and the result, reads `nu` modulo
    `(s - 1)!` for `t = 1`, 6 at `(5, 2)`, 2 at `(6, 3)`, 120 at `(7, 2)`,
    2 at `(7, 4)` and 2 at `(9, 5)`.
  * **`(s, t) = (2, 1)`.** No residue leaks, only the integer multiplier 2.
    That multiplier, like the integer multipliers up to 4 at `(3, 2)` and 45
    at `(5, 3)`, multiplies the statistical distance of the flooding.
* **Consequences.** With `nu` known exactly, the tMIFE-LWE functional key
  `(sk_y, z)` would follow from the partials by linear algebra. At the
  `standard` parameters the leak is partial: at most about 25 bits of `nu`
  per ciphertext for `s <= 10`, against a noise bound of about `2^67`
  (tMIFE-LWE) or `2^71` (tMCFE-Ring-LWE) in the benchmark configurations. No key or plaintext recovery has been shown. In tMCFE-LWE,
  whose flooding argument is computational, the same multipliers fix exact
  relations among the flooding terms and narrow the error of its LWE
  argument over the enrolled sets of one label, up to 9-fold at `(3, 2)`
  and 869-fold at `(5, 3)`.
* **Rating.** High, because it is the default for every `t < s`, it does
  not depend on the flooding width, and it sometimes needs no corrupted
  server. Practical impact Medium, because no key or plaintext recovery is
  shown at `standard` parameters.
* **Not affected.** `t = s`, and the DDH and pairing threshold layers
  (`mife/ddh_threshold`, `mcfe/ddh_threshold`,
  `mcfe/fh_multi_ipe_pairing_threshold`), which interpolate exactly in the
  exponent.
* **Fix.**
  * For `t < s`, the three layers now default to `sharing="replicated"`.
    Its multipliers are 0 or 1, so the flooding argument holds. `q`,
    `ring_n` and `lwe_n` are unchanged; a server holds `C(s - 1, t - 1)` key
    pieces.
  * `sharing="shamir"` with `t < s` is refused by the key generator under
    `standard` and draws an `FESecurityWarning` under `toy`.
  * Servers refuse Shamir key shares with `t < s` unless they are created
    with `legacy_shamir="warn"`. Key shares of v0.3.0 are refused in any
    case, because their flooding noise was drawn by another derivation
    (`noise_version`).
  * A server holds `C(s - 1, t - 1)` full pieces (126 at `(10, 5)`). The
    opt-in `sharing="replicated-seeded"` brings that down to at most one full
    piece. It deals pieces of the same kind, one per `(t - 1)`-subset and
    assigned the same way, but expands all of them but one from secret
    256-bit seeds; its partial decryptions are those of a replicated key
    share holding the expanded pieces. Its key shares therefore hide the key
    only **computationally**, assuming SHAKE-256 keyed by those seeds is a
    PRF (a standard keyed-sponge assumption, of the kind the derandomised
    flooding noise makes), where full replicated sharing hides it perfectly.
    Full replicated sharing stays the default for `t < s`.
* **Upgrading from v0.3.0** (LWE threshold layers with `t < s`):
  **re-share the threshold keys with this version.**
  * Run its key generator (`setup()`, then `get_private_keys` and
    `get_decryption_keys`) and hand out the new client keys and key shares.
    Discard the old key shares.
  * The replicated sharing has its own parameter record, so this also draws
    a new public matrix (or polynomial). Ciphertexts made under the old keys
    still need the old key shares.
  * This version refuses v0.3.0 key shares (`FEKeyError`: another flooding
    derivation, `noise_version` 1), so finish rounds encrypted under the old
    keys with v0.3.0 before upgrading; the leak remains for those rounds.
  * `legacy_shamir="warn"`, e.g.
    `ThresholdMIFELWE({"id": sid, "legacy_shamir": "warn", "keys": {...}})`,
    serves Shamir key shares with `t < s` of this version (an explicit
    `sharing="shamir"` under `toy`) with an `FESecurityWarning`; otherwise
    they are refused (`FESchemeError`). There is no environment variable for
    this. Public parameters that record the `standard` profile are refused
    even with `legacy_shamir="warn"`.
  * Public parameters now record `threshold_construction` and
    `security_profile`, and servers refuse constructions they do not know.

Fixed in v0.3.0 (see [`CHANGELOG.md`](CHANGELOG.md)): every
scheme previously marked *known-insecure* now implements a published
construction or, for four threshold variants and dMCFE-DDH, an
*experimental* layer of our own on top of one. MCFE `paillier`, `lwe`, `fullysec_lwe` and `damgard` follow
ABMPSW20 (Figs. 3, 4 and 2); the MIFE variants `damgard`, `paillier`,
`lwe`, `fullysec_lwe` and `ring_lwe` add ACFGU18's one-time pad, so the key
holder learns only the sum; dMCFE-DDH, -LWE and -Paillier follow ABMPSW20
Fig. 14; MCFE-, dMCFE- and tMCFE-Ring-LWE follow BMMS21's identity-based
(D)MCFE; the FH-MCFE family follows AGT21b and NPS24 over an asymmetric
pairing; tMCFE-DDH shares the CDGPP18 key; tMIFE-LWE, tMCFE-LWE and
tMCFE-Ring-LWE flood partial decryptions with derandomised noise; SIFE,
MIFE and dMCFE Paillier keys are Gaussian (no longer brute-forceable);
Paillier parameter caches no longer store the factorization of `N`; user
labels and ndarray element labels can no longer collide; all pairing-based
schemes default to the asymmetric group `MNT224` and reject symmetric ones.
A review of the remaining schemes fixed: SIFE- and MIFE-DDH ciphertexts that
were not reduced modulo `p` (their factors revealed the plaintext) and keys
not reduced modulo `q`; the master secret key handed to every encryptor of
SIFE partial-FH-IPE; KLMMRW FH-IPE without privacy for `eta = 1`; MIFE FH-IPE
decoding every slot separately (now ACFGU18 Fig. 12); Quad now implements
Gay20 Fig. 4; FullySec-LWE decoding `+-eta X Y` with a random sign; Gaussian
samples above `2^40` that lost their low bits; the G1 identity, which charm
serialised into a non-member point. MIFE HybridAlpha was removed. Quadratic
SGP is public-key as in DSGPP18 (encryptors used to receive the master key);
MIFE FH-Multi-IPE requires `n >= 2`; the ad-hoc multi-input SGP, which
revealed every client's quadratic value, is replaced by the AGT22 MIFE for
quadratic functions.

Fixed in v0.2.0: MCFE-DDH re-implemented per CDGPP18; independent per-client
keys in MIFE-DDH; DDH threshold sharing (1-based Shamir points over the group
order, modular Lagrange coefficients, ≥ t share check, `eta ≥ 2`);
`get_decryption_keys` of decentralized schemes raises instead of returning
`NotImplementedError`; no more pickle-based dlog cache; pairing tests skip
cleanly without charm-crypto.

## Disclaimer

> **⚠️ Research Prototype — Not for Production Use**
>
> PyFE4AI is a research-oriented library and has **not** undergone a formal
> security audit. Code organization and APIs may change at any point.
>
> The purpose of this project is to support research and proof-of-concept
> implementations. **It should not be used in production** or in any setting
> where real-world security guarantees are required.
>
> In particular:
> - Implementations have not been hardened against side-channel attacks
>   (timing, cache, power analysis, etc.).
> - Parameter choices in examples and tests are selected for fast execution,
>   not for cryptographic strength.
> - Several schemes write generated public parameters into local `config/`
>   subdirectories during setup.

## Citing

If you use PyFE4AI in your research, please cite it (a technical report is in
preparation and will be added here once published):

```bibtex
@software{pyfe4ai2026,
  author  = {{Spire Studio}},
  title   = {{PyFE4AI}: Python-based Functional Encryption
             for {AI} Security and Privacy},
  year    = {2026},
  version = {0.3.0},
  license = {Apache-2.0},
  url     = {https://github.com/spire-studio/pyfe4ai}
}
```

Citation metadata is also available in [`CITATION.cff`](CITATION.cff) — use
GitHub's **"Cite this repository"** button in the sidebar for APA/BibTeX.

## License

This project is licensed under the Apache License 2.0. See the
[`LICENSE`](LICENSE) file for details.

## Project Policies

- Contribution guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security reporting: [`SECURITY.md`](SECURITY.md)
- Citation metadata: [`CITATION.cff`](CITATION.cff)
- Code of conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
