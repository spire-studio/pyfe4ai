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

## Quick Start

```bash
conda env create -f environment.yml
conda activate pyfe4ai
pip install -e ".[test]"        # editable with test extras
python -m pytest -m core -q     # run the core test suite
```

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
[`Dockerfile.pairing`](Dockerfile.pairing)). The Docker image of
[Pairing-Based Schemes](#pairing-based-schemes) has all of them.

`pip install` also builds an optional Rust extension when a Rust toolchain
(`cargo`) is on the `PATH` (`python tools/build_rust.py` in a source
checkout; `PYFE4AI_NO_NATIVE=1` skips it). It runs the Ring-LWE MIFE / MCFE
and the dense-LWE MCFE hot paths natively -- 50x to 70x faster for the FL
aggregation path -- with the same outputs as the Python code, which remains
the fallback; it also provides ristretto255 for the `ddh_ristretto255`
schemes and the BLS12-381 pairing group, the default of the pairing-based
schemes. See `pyfe4ai.utils.native_backend` and
[`benchmarks/PERFORMANCE.md`](benchmarks/PERFORMANCE.md).

## Usage

```python
from pyfe4ai import SIFE, SIFEKeyGenerator

x = [2, 1, 3]
y = [4, 5, 6]

# Key generation. sec_param is the bit length of the DDH modulus, not a
# security level: 128 keeps the demo fast but is NOT secure. For a 3072-bit
# group, replace "sec_param": 128 with "security_profile": "standard" (see
# Security Profiles; an explicit sec_param below 3072 is then refused).
kg = SIFEKeyGenerator({"sec_param": 128, "eta": len(x)})
kg.setup()
pp = kg.get_public_parameters()
sk = kg.get_private_keys()
dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})

# Encrypt (needs pp + sk)
encryptor = SIFE({"precision": 3, "keys": {"pp": pp, "sk": sk}})
ct = encryptor.encrypt(x)

# Decrypt (needs pp + dk)
decryptor = SIFE({"precision": 3, "keys": {"pp": pp}})
result = decryptor.decrypt(ct, dk, y)
print("<x, y> =", result)   # → 31
```

See `examples/sife_minimal.py` for the full runnable version.

## Key Features

- **Multiple FE families** — DDH, LWE, Ring-LWE, FullySec-LWE, FullySec-Ring-LWE (adaptively secure, BMMS21 Sec. 5), Paillier, and Damgård instantiations across SIFE, MIFE, and MCFE
- **Threshold & decentralized** — `tMIFE`, `tMCFE`, `dMCFE` for distributed trust settings
- **Function-hiding** — pairing-based FH-IPE and FH-Multi-IPE with full inner-product privacy
- **Quadratic FE** — SGP and Quad (single-input, public-key) and multi-input quadratic FE with per-user keys (AGT22)
- **Security profiles** — `security_profile="standard"` selects 3072-bit DL / DCR groups, lattice-estimator-sized LWE and Ring-LWE dimensions and BLS12-381 pairings; the default `"toy"` keeps the fast demo parameters
- **ML adapter** — `utils/ml_adapter.py` for encrypted linear inference and federated gradient aggregation
- **Precision toolkit** — `utils/quantization.py` for float ↔ integer quantization with configurable bit-width
- **Discrete-log solvers** — in-memory dlog table (default) with on-the-fly BSGS as optional backup in `utils/dlog_solver.py`
- **NTT acceleration** — number-theoretic transform for Ring-LWE polynomial multiplication in `utils/ring_lwe_utils.py`
- **Structured exceptions** — `FEError` hierarchy in `utils/exceptions.py`

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
> them under the label they serve. All threshold variants assume honest-but-curious servers (see
> [Known Issues](#known-issues--security-notice)).

> **⚠️ Research prototype.** Every scheme has been checked against its
> paper, but four threshold layers and the dMCFE-DDH carry mask are our own
> (*experimental*) and the default parameters are toy-sized. Check the
> [Scheme Status](#scheme-status) table and the
> [Known Issues / Security Notice](#known-issues--security-notice) before
> relying on any privacy property.

## Scheme Status

Every registered scheme carries a review status, kept in
[`pyfe4ai/schemes/status.py`](pyfe4ai/schemes/status.py) and available at
runtime as `pyfe4ai.get_scheme_status(family, variant)` (and
`FESchemeWrapper(...).status`). Instantiating a key generator or crypto
object of a known-insecure scheme emits a `pyfe4ai.FESecurityWarning`. The
table below is rendered by `pyfe4ai.schemes.status.markdown_table()` and a
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
| `sife/lwe` | reviewed | adapted | ABDP15 Constructions 2 + 3 (LWE) | k = 0 (no r-ary decomposition); signed X/Y-bounded inputs with centred decoding; GoFE parameter recipe; noise is a rounded continuous Gaussian; A expanded from a public seed (SHAKE-128) |
| `sife/paillier` | reviewed | faithful | ALS16 (DCR) | D_{Z,sigma} realised as a high-precision rounded Gaussian |
| `sife/damgard` | reviewed | faithful | ALS16 DDH (ACFGU18 Fig. 7, k = 1) | h = g^a with a discarded |
| `sife/ring_lwe` | reviewed | adapted | BMMS21 Sec. 4 (selective) + Remark 1 batching | signed messages and weights (p = 2lXY + 2, centred decoding); widths from the Sec. 4.2 constraints (the paper's Table 1 misses its smoothing conditions); one ciphertext packs up to ring_n vectors; single NTT prime q; ring_n defaults to the toy value 32 |
| `sife/fullysec_ring_lwe` | reviewed | adapted | BMMS21 Sec. 5 (adaptive) | m >= 5 key polynomials instead of m = 3, which admits no modulus under the paper's App. C.1 constraints (sigma' >= q^(2/m) from Ring-LHL); signed inputs (p = 2lXY + 2, leveraging over (2X+1)^(2l)); one vector per ciphertext, or with ndarray_layout = "packed" up to ring_n (leveraging over the span of the differences) |
| `sife/fullysec_lwe` | reviewed | adapted | ALS16 LWE (ACFGU18 Fig. 8) | signed inputs (K = 2 eta X Y + 1, centred decoding); GoFE parameter recipe with m = 4 n log2 q (ALS16 Theorem 2); D_{Z,sigma} realised as a rounded Gaussian |
| `sife/ddh_dynamic` | reviewed | adapted | ABDP15 Construction 1, variable length | vectors of length <= eta handled as zero padding (prefix keys, truncated ciphertexts) |
| `sife/fh_ipe_pairing` | reviewed | adapted | KLMMRW18 Sec. 3 | vectors padded to (x, 1, 0) / (y, 0, 1), so eta = 1 and zero vectors are safe; keys in G1, ciphertexts in G2; generic bilinear group model; the encryption key is master-secret equivalent |
| `sife/part_fh_ipe_pairing` | reviewed | adapted | Gay20 Fig. 8 (CPA, k = 1) | M is a random full-rank public matrix (subspace_dim < eta, default 1), published with V^T M in the clear; keys leak M^T y, which for small eta - subspace_dim often determines a bounded y; public Enc needs bounded t; Enc' is run by the key generator; no QANIZK |
| `mife/ddh` | reviewed | adapted | ACFGU18 Fig. 4 + ABDP15 (DDH) | ACFGU18 instantiates Fig. 4 with ALS16; here the selectively secure ABDP15 IPFE in every slot |
| `mife/ddh_ristretto255` | unreviewed | unknown | ACFGU18 Fig. 4 + ABDP15 (DDH) in ristretto255 | ristretto255 (RFC 9496, about 126-bit security: Pollard rho in a group of order 2^252) instead of a 3072-bit Z_p^* subgroup; group arithmetic in Rust (curve25519-dalek) with the native extension, else in pure Python and not constant-time; new code, not yet reviewed |
| `mife/lwe` | reviewed | adapted | ACFGU18 Fig. 4 + ABDP15 (LWE) | ACFGU18 instantiates Fig. 4 with ALS16; here the selectively secure ABDP15 IPFE (same two-step decryption), with parameters for dimension n * eta (footnote 14); slots share A, expanded from a public seed (SHAKE-128) |
| `mife/paillier` | reviewed | adapted | ACFGU18 Fig. 4 + ALS16 (DCR, Fig. 9) | slots share N and g; D_{Z,sigma} realised as a high-precision rounded Gaussian |
| `mife/damgard` | reviewed | adapted | ACFGU18 Fig. 4 + ALS16 (DDH, Fig. 7) | slots share the group and h (the matrix A of Fig. 7) |
| `mife/ring_lwe` | reviewed | adapted | BMMS21 Sec. 6 MIFE-compiler (ACFGU18) | slots share a; parameters sized for dimension n * eta, with the Sec. 4.2 widths for bound 3X (ACFGU18 Fig. 4, footnote 12) |
| `mife/fullysec_lwe` | reviewed | adapted | ACFGU18 Fig. 4 + ALS16 (LWE, Fig. 8) | slots share A; parameters derived for dimension n * eta and bound 3X (ACFGU18 footnote 14), m = 4 n log2 q |
| `mife/fh_ipe_pairing` | reviewed | adapted | ACFGU18 Fig. 12 (k = 1) | one r per key shared by all slots; per-slot FH-IPFE as in mcfe/fh_ipe (LV16 padding of App. B); asymmetric pairing (default BLS12-381); keys carry a random id, never a digest of y |
| `mife/fh_multi_ipe_pairing` | reviewed | adapted | DOT18 Sec. 3.1 (bounded, k-LIN, k = sec_level) | n >= 2 enforced (zero sharing r_i); no Remark B.1 SKE layer, so hiding assumes every slot contributes a ciphertext; slot i's client holds B^_i and can decode its own slot of a key; keys are group elements only, decryption does not need y |
| `mife/ddh_threshold` | reviewed | adapted | ACFGU18 Fig. 4 + ABDP15 + Shamir (TAPFed setting) | the functional key (d_i, z) is Shamir-shared mod q; any t servers interpolate in the exponent (threshold-ElGamal style), as mcfe/ddh_threshold |
| `mife/lwe_threshold` | experimental | ad-hoc | MIFE-LWE + Shamir + smudging | MIFE layer as published; the threshold layer (statistical flooding, derandomised noise, enlarged q, clients' HMAC tags on ciphertexts so that servers only decrypt ciphertexts made by the clients) is our own |
| `mcfe/ddh` | reviewed | faithful | CDGPP18 |  |
| `mcfe/ddh_ristretto255` | unreviewed | unknown | CDGPP18 in ristretto255 | ristretto255 (RFC 9496, about 126-bit security: Pollard rho in a group of order 2^252) instead of a 3072-bit Z_p^* subgroup; labels hashed with RFC 9380 hash_to_ristretto255; group arithmetic in Rust (curve25519-dalek) with the native extension, else in pure Python and not constant-time; new code, not yet reviewed |
| `mcfe/lwe` | reviewed | adapted | ABMPSW20 Fig. 4, selective parameters | signed messages and weights (K = 2nXY + 1, centred decoding); t is a rounded continuous Gaussian |
| `mcfe/paillier` | reviewed | faithful | ABMPSW20 Fig. 3 | D_{Z,sigma} realised as a high-precision rounded Gaussian |
| `mcfe/damgard` | reviewed | faithful | ABMPSW20 Fig. 2 (D_k-MDDH) | k = mddh_k (default 1, where it equals CDGPP18) |
| `mcfe/ring_lwe` | reviewed | adapted | BMMS21 Sec. 6 MCFE-compiler (identity-based) | functional keys are bound to a label; ndarray values are packed into one ciphertext per chunk, each chunk with its own label pad; parameters as mife/ring_lwe (widths for bound 3X) |
| `mcfe/fullysec_lwe` | reviewed | adapted | ABMPSW20 Fig. 4, adaptive parameters | signed messages and weights (K = 2nXY + 1, centred decoding); t is a rounded continuous Gaussian; m0 grows with n log2(2X + 1) |
| `mcfe/fh_multi_ipe_pairing` | reviewed | adapted | AGT21b Sec. 6.2 (as recapped in NPS24 Sec. 2) | FH-IPFE = ACFGU18 Fig. 12 (n = 1, SXDH) with LV16 padding; asymmetric pairing (default BLS12-381); keys carry a random id, never a digest of y |
| `mcfe/fh_multi_ipe_pairing_threshold` | experimental | ad-hoc | AGT21b FH-MCFE + Shamir | FH-MCFE layer as published; the key exponents Shamir-shared in the exponent ([e^(k)]_2, [z^(k)]_T), partial decryptions in G_T and labelled, client-tagged ciphertexts are our own; FH-IPFE = ACFGU18 Fig. 12 (n = 1, SXDH) with LV16 padding; asymmetric pairing (default BLS12-381); keys carry a random id, never a digest of y |
| `mcfe/fh_multi_ipe_pairing_decentralized` | reviewed | adapted | NPS24 Fig. 1 (selective FH-DMCFE) | explicit tag-f, each client's key depends on (tag-f, y_i) only and is derived deterministically; one key per client and tag-f (enforced per client object); FH-IPFE = ACFGU18 Fig. 12 (n = 1, SXDH) with LV16 padding; asymmetric pairing (default BLS12-381); keys carry a random id, never a digest of y |
| `mcfe/ddh_threshold` | reviewed | adapted | CDGPP18 + Shamir (TAPFed setting) | the aggregated key is Shamir-shared; any t servers interpolate in the exponent (threshold-ElGamal style) |
| `mcfe/ddh_decentralized` | experimental | ad-hoc | ABMPSW20 Fig. 14 (MCFE1 = CDGPP18) + carry mask | Fig. 14 is stated for MCFEs without a discrete log in decryption; contributions are reduced mod q and masked by q rho_i (rho_i from a per-client key and H(y)) so that the carry of their sum stays hidden -- our own argument; MCFE2 = ABMPSW20 Fig. 3 (DCR); the zero-sum MCFE2 keys (pairwise masks) are sampled in-process |
| `mcfe/lwe_threshold` | experimental | ad-hoc | ABMPSW20 Fig. 4 + Shamir + flooding | MCFE layer as published; the threshold layer and its LWE-based flooding argument are our own (see the module docstring) |
| `mcfe/lwe_decentralized` | reviewed | adapted | ABMPSW20 Fig. 14 (MCFE1 = Fig. 4) | MCFE2 = ABMPSW20 Fig. 3 (DCR); the zero-sum MCFE2 keys (pairwise masks) are sampled in-process |
| `mcfe/ring_lwe_threshold` | experimental | ad-hoc | BMMS21 identity-based MCFE + Shamir + smudging | MCFE layer as published; the threshold layer (statistical flooding, derandomised noise, enlarged q, clients' HMAC tags on ciphertexts) is our own |
| `mcfe/ring_lwe_decentralized` | reviewed | adapted | BMMS21 Sec. 6 identity-based DMCFE (ABKW19 + GDSum) | GDSum realised as pairwise PRF zero-sum masks (its full specification is only in the ePrint version); pairwise secrets sampled in-process |
| `mcfe/paillier_decentralized` | reviewed | adapted | ABMPSW20 Fig. 14 (MCFE1 = Fig. 3) | MCFE2 = ABMPSW20 Fig. 3 (DCR); the zero-sum MCFE2 keys (pairwise masks) are sampled in-process |
| `quadratic/quad` | reviewed | adapted | Gay20 Fig. 4 over Fig. 8 (CPA) | signed inputs (discrete log in [-n m B^3, n m B^3]); CPA variant (no QANIZK); asymmetric pairing (default BLS12-381) |
| `quadratic/sgp` | reviewed | adapted | DSGPP18 Fig. 2 | public key pk = (g1^s, g2^t) as in the paper; signed inputs with one bound B for x, y and F (discrete log in [-n^2 B^3, n^2 B^3]); generic bilinear group model |
| `quadratic/miqfe` | reviewed | adapted | AGT22 Sec. 5 (MIFE for quadratic functions) | cross-user terms only (Def. 2.9); per-user encryption keys made of master ciphertexts; building blocks TAO20 FH-IPFE, Fig. 9 and Fig. 1 as in the paper; PRF = HMAC-SHA512; pos-security (every user contributes); k = mddh_k >= 2 |

## Security Profiles

Every key generator accepts a `security_profile` entry in its config
([`pyfe4ai/schemes/profiles.py`](pyfe4ai/schemes/profiles.py)). The default,
`"toy"`, keeps the small default parameters -- fast, and **not secure**; a
config without the entry behaves exactly as before. `"standard"` fills in
parameters aiming at 128-bit classical security and rejects explicit values
below them with `FEValidationError`:

```python
kg = SIFEKeyGenerator({"eta": 3, "security_profile": "standard"})  # 3072-bit group
```

| Family | `standard` parameters | Measured cost (one core, 2 clients) |
|---|---|---|
| Discrete log: DDH, Damgård / MDDH (SIFE, MIFE, MCFE, tMIFE, tMCFE) | `sec_param = modulus_length = 3072`: RFC 3526 MODP group 15 (NIST SP 800-57 Part 1: 3072 bits for 128-bit security) | setup < 0.1 s; encrypt and aggregate < 1 s |
| `mcfe/ddh_decentralized` | as above; key shares (masked with `stat_param` = 128 bits against carries) are combined under a DCR modulus of about 6410 bits | setup ≈ 2 min (two 3207-bit safe primes); key share 0.7 s |
| DCR: Paillier (SIFE, MIFE, MCFE) | `bit_length = 1536`: two 1536-bit safe primes, N ≈ 3072 bits, generated for every key generator (a cached modulus could have known factors) | setup a few seconds (two 1536-bit safe primes) |
| Ring-LWE (SIFE, MIFE, MCFE, tMCFE, dMCFE) | κ = `sec_param` = 128 and `ring_n = "auto"`: the smallest n in {1024, …, 32768} whose derived q (including the threshold flooding) meets a lattice-estimator bound for Ring-LWE with Gaussian secret and errors (`ring_lwe_utils.RING_LWE_128_LOG_Q`, rows by width 2–16; regenerate with `tools/estimate_ring_lwe.py`), read at the smallest RLWE width of the scheme's proof -- about 8 with the BMMS21 Sec. 4.2 widths, giving log2 q ≤ 31 / 57 / 111 / 218 / 435 / 872 (the Homomorphic Encryption Standard's 27 / 54 / 109 / 218 / 438 / 881 is about 125 bits by the same estimator); an explicit `ring_n` is checked against the same bound | Python backend: SIFE / MIFE (eta = 10, X = Y = 50): n = 4096; one ciphertext holds up to 4096 vectors at the same cost, encrypt 0.4–0.5 s (0.12 ms per vector when full), decrypt 0.04 s (SIFE) / 0.08 s (MIFE). MCFE, 10 000 gradient elements (X = 10^4, Y = 10): `ring_lwe` n = 4096, encrypt 0.26 s per client, aggregate 0.28 s; `ring_lwe_threshold` (3 servers, t = 2) n = 16384, encrypt 0.84 s per client, key share 0.21 s and partial decryption 0.67 s per server, combine 0.02 s; `ring_lwe_decentralized` n = 4096, encrypt 0.21 s per client, key share 0.02 s per client and label, decrypt 0.18 s. With the Rust extension the FL path is far faster (MCFE 10^4 elements: about 3.5 ms per client and 9.5 ms to aggregate; tMCFE with its 212-bit q: 20 ms per client, 211 ms per partial decryption), see `benchmarks/PERFORMANCE.md`. Functional keys of the MCFE variants are bound to a label (round) |
| Adaptively secure Ring-LWE: `sife/fullysec_ring_lwe` (BMMS21 Sec. 5) | as Ring-LWE, plus `m = "auto"`: the smallest number m ≥ 5 of key polynomials whose q meets the same bound (the paper's m = 3 admits no modulus under its own constraints, see the module docstring); widths from Theorems 1–2 and Appendix C.1 with complexity leveraging over (2X+1)^(2·eta) | eta = 4, X = Y = 4: n = 4096, m = 17, setup 1.1 s, encrypt 1.1 s, decrypt 0.6 s; eta = 10, X = Y = 50: n = 8192, m = 8, setup 9 s, encrypt 2.4 s, decrypt 0.5 s |
| Dense LWE: the ABMPSW20 MCFE family (`mcfe/lwe`, `fullysec_lwe`, `lwe_threshold`, `lwe_decentralized`) | `lwe_n = "auto"`: the smallest n0 in {4096, 4608, …, 16384} whose derived q meets a [lattice-estimator](https://github.com/malb/lattice-estimator) bound for LWE(n0, q, ⌈2√n0⌉) with a uniform secret -- the assumption of the scheme's proof (`lwe_utils.DENSE_LWE_128_LOG_Q`, e.g. log2 q ≤ 118 / 173 / 227 for n0 = 4096 / 6144 / 8192; regenerate with `tools/estimate_dense_lwe.py`); an explicit `lwe_n` is checked against the same table. No public matrix: a label is hashed to n0 + m0 values. `lwe_decentralized` combines key shares under a DCR modulus of ≥ 3072 bits, so its key derivation is not post-quantum | n0 = 7168–8192 for small bounds (stat_param 128); about 7 ms per element and client to encrypt and 6–8 ms per element to aggregate (one label per element); dMCFE key share ≈ 90 s per client |
| LWE with a public matrix: ABDP15 (`sife/lwe`, `mife/lwe`) | `lwe_n = "auto"`: the smallest n in {512, 576, …, 6144} whose (q, σ) meets a lattice-estimator bound on log2(q/σ) for the wide-error LWE of the public key A·S + E (`lwe_utils.WIDE_LWE_128_LOG_RATIO`, rows by σ; regenerate with `tools/estimate_wide_lwe.py`). A (m ≈ n·log2 q rows) is published as a 32-byte seed and expanded with SHAKE-128. `matrix_mode` (key generator and crypto configs, forwarded by the ML adapter): `"memory"` keeps A expanded, `"stream"` expands it again for every batch | n = 896 / 1088 for X = Y = 50 / 256 and eta = 10 (A: 6·10^7 / 1.1·10^8 entries); MIFE is sized as if its vectors had dimension n·eta (ACFGU18, footnote 14), e.g. n = 1024 for 100 slots of dimension eta = 1 with X = Y = 50 (1216 for eta = 10). Encryptor alone, n = 1088: `"memory"` 0.10 s per encryption and 2.7 GB, `"stream"` 3.4 s and 0.5 GB; the ndarray helpers encrypt in batches of 64 (eta = 1, n = 768: 4 vs 21 ms per element). The public key is m × eta per slot (32 MB of JSON for eta = 10), so keep eta small. `mife/lwe_threshold` (experimental) is sized the same way at its flooding-enlarged q: n ≈ 6000 for X = Y = 50 with stat_param 128 (A: 7.8·10^9 entries, more than 400 GB in memory), so `standard` fixes `matrix_mode = "stream"`: key generation 38 min (8.5 GB), one encryption 572 s, a batch of 64 encryptions 679 s (10.6 s each, 5.2 GB), partial decryption 13 ms |
| Pairing: FH-IPE, FH-MCFE family, quadratic FE | BLS12-381 (Rust extension; the `toy` default too): about **120–126-bit** security after the exTNFS attacks, below the target; a `FESecurityWarning` says so | `report` benchmark profile: SGP 30 ms, Quad 48 ms, MCFE FH-Multi-IPE 47 ms end to end; a pairing 0.8 ms |
| Refused | `sife/fullysec_lwe`, `mife/fullysec_lwe` (ALS16), `mcfe/paillier_decentralized` | `FEValidationError`: no parameter set that is both secure and usable. ALS16 at a 128-bit dimension (lattice estimator, n = 3328–4352 for X = Y ≤ 50 and eta ≤ 10, with m = 4n·log2 q as its Theorem 2 requires): A has 1–2·10^10 entries and a ciphertext 2.9–4.5·10^6 coordinates of 219–258 bits (80–145 MB); its LWE-to-mheLWE reduction loses about n^6 in the noise ratio, so the parameters cannot be tightened -- for adaptive security use `sife/fullysec_ring_lwe`, otherwise the Ring-LWE scheme of the same family. dMCFE-Paillier needs a DCR modulus above 15000 bits for combining key shares |

The statistical parameter `stat_param` (rounding and leftover-hash slack,
flooding, masks) is 128 under `standard`, a statistical distance of
`2^-128` per sample; lower explicit values are refused. The profile sizes the
hardness assumptions only; the other caveats under
[Known Issues](#known-issues--security-notice) (research prototype, no
side-channel hardening, trusted parameter cache) still apply.

## Repository Layout

- `pyfe4ai/`: installable Python package
  - `schemes/`: core cryptographic scheme implementations
    - `sife/`: single-input families
    - `mife/`: multi-input families
    - `mcfe/`: multi-client families
    - `quadratic/`: quadratic functional encryption families
    - `status.py`: review status of every registered scheme (the table above)
    - `profiles.py`: `security_profile` (`"toy"` / `"standard"`)
    - `threshold_utils.py`: Shamir sharing, client MAC tags and partial-decryption checks of the threshold schemes
    - `abdp15_lwe.py`: shared ABDP15 parameters and seeded public matrix of SIFE- and MIFE-LWE
  - `utils/`: utility modules
    - `crypto_constants.py`, `crypto_utils.py`: group generation and constants
    - `lwe_utils.py`, `ring_lwe_utils.py`: LWE / Ring-LWE helpers with NTT
    - `lwe_matrix.py`: seeded public LWE matrices and exact numpy products with them
    - `dlog_solver.py`: in-memory dlog table + BSGS discrete-log recovery
    - `ml_adapter.py`: ML integration (encrypted inference, FL aggregation)
    - `quantization.py`: float ↔ integer quantization toolkit
    - `exceptions.py`: `FEError` exception hierarchy
    - `validation.py`: strict integer checks and rounding of scheme inputs
    - `sampling_utils.py`, `matrix_utils.py`, `modular_utils.py`: math helpers
    - `pairing_backend.py`, `pairing_utils.py`, `bls12_381.py`: pairing layer (BLS12-381 from the Rust extension, other curves from charm-crypto)
  - `logger.py`, `config.yaml`: logging configuration
- `docs/`: Sphinx API documentation (see [API Documentation](#api-documentation))
- `tests/`: end-to-end and regression tests (`sife/`, `mife/`, `mcfe/`, `quadratic/`)
- `examples/`: minimal runnable usage examples
- `benchmarks/`: lightweight benchmark scripts
- `tools/`: maintenance scripts (`estimate_dense_lwe.py`, `estimate_wide_lwe.py`, `estimate_ring_lwe.py` regenerate the LWE and Ring-LWE security tables with SageMath and the lattice estimator)
- `site/`: project landing page (published with the docs to GitHub Pages)
- `environment.yml`: reproducible Conda environment
- `Dockerfile.pairing`: Docker-based pairing-enabled test environment

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

## Running Tests

```bash
python -m pytest -q                # all tests
python -m pytest -m core -q        # core FE/IPFE smoke lane only
python -m pytest -m extended -q    # pairing-heavy & quadratic families only
```

The `core` lane is the main public release lane. The `extended` lane covers
pairing-heavy and quadratic families that remain publicly available but are kept
outside the minimal smoke path.

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
docker build -f Dockerfile.pairing -t pyfe4ai-pairing .
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
| Honest-but-curious decryption servers | all threshold schemes (tMIFE, tMCFE) | Fewer than `t` servers learn nothing beyond the result, but partial decryptions carry no proof of correctness: one server that deviates changes the result without notice. `combine_decrypt` accepts partial decryptions for several enrolled sets (a list of `{sid: partial}` mappings) and raises `FESchemeError` if they disagree; this detects one faulty server (not which one) if every server is left out of some set, e.g. the `t + 1` subsets of size `t` of `t + 1` servers. tMCFE-DDH and tMCFE-FH-Multi-IPE servers decrypt only ciphertexts under the label they are asked for (`share_decrypt(..., label)`). |
| One ciphertext per client and label | all MCFE / DMCFE schemes | As in the papers, a client must not encrypt two different inputs under one label (or tag); the scheme classes do not enforce it (the ML adapter refuses a second encryption per client and label within one wrapper, and FH-DMCFE client objects a second key per `tag_f`). |
| Rounded Gaussian by default | the LWE, Ring-LWE and DCR schemes | The proofs sample `D_{Z,sigma}`; the default sampler rounds a continuous Gaussian, about `0.02 / sigma^2` from it in statistical distance per sample (about 2^-33 at the Ring-LWE `sigma1`), and is not constant-time. `sampler="facct"`, `gaussian_sampler("facct")` or `PYFE4AI_GAUSSIAN_SAMPLER=facct` (see `pyfe4ai.utils.sampling_utils`) samples `D_{Z,sigma}` exactly, about 2^-185 per sample, at a few microseconds per sample in Python; it is a reference for a native constant-time sampler, not constant-time itself. |
| Toy default parameters | all | `sec_param` is a modulus bit length (default 128), not a security level; LWE dimensions default to 32–64 and the Ring-LWE dimension to 32; the statistical parameter `stat_param` defaults to 64. All far below any real security level (a key column of SIFE-LWE at its defaults falls to plain LLL in minutes). Pass `security_profile="standard"` for 128-bit-oriented parameters -- except for pairings (BLS12-381, about 120–126-bit) and the refused ALS16 / dMCFE-Paillier schemes; see [Security Profiles](#security-profiles). |
| Parameter cache in the working directory | key generators that cache parameters (most; not MCFE `lwe`, `fullysec_lwe`, `lwe_threshold`, the FH-MCFE family, MIFE `fh_ipe_pairing` and quadratic `quad` and `miqfe`) | Public parameters are written to and re-loaded from `./config/authority/...`. A cached file is used only if it matches the configuration; discrete-log groups are checked for primality, size, safe-prime structure, generator order and (at RFC 3526 sizes) for being the RFC prime; DCR moduli for size, compositeness and small factors (their factors are never stored); all other cached values are re-derived on load or expanded from a stored seed (LWE and Ring-LWE parameters and public matrices, the Paillier base, the Damgard `h`); under `standard` the lattice schemes re-check their moduli against the 128-bit tables. An unreadable file is regenerated, and files are written atomically. No check can tell whether someone knows the factors of a DCR modulus, so under `standard` the Paillier moduli and the dMCFE `N2` are generated for every key generator, never loaded or stored; under `toy` a planted modulus with known factors is still used: run only in a directory that nobody else can write to. |
| The ML adapter plays every party | `utils/ml_adapter.py` | `FESchemeWrapper` runs the key generator, so it holds the master key and acts as clients, authority and decryptor in one trust domain; use it for experiments, not as a deployment. By default `aggregate_gradients` needs a ciphertext and a non-zero weight for every enrolled client (`allow_partial=True` relaxes this), a client encrypts at most once per label and a label is aggregated with one weight vector; these checks are per wrapper object, and they do not stop very unequal weights (e.g. `(W, 1)` with `W` above twice the largest gradient entry), which separate two clients' gradients in one aggregate -- as with any inner-product FE, the weights are the key holder's choice. |

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
