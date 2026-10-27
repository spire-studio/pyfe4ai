<p align="center">
  <img src="assets/branding/logo-simple.png" alt="PyFE4AI" width="360">
</p>

<p align="center">
  <a href="https://github.com/spire-studio/pyfe4ai/releases"><img src="https://img.shields.io/github/v/release/spire-studio/pyfe4ai?color=green" alt="Latest release"></a>
  <a href="https://github.com/spire-studio/pyfe4ai/actions/workflows/tests.yml"><img src="https://github.com/spire-studio/pyfe4ai/actions/workflows/tests.yml/badge.svg?branch=main" alt="Tests"></a>
  <a href="https://github.com/spire-studio/pyfe4ai/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://github.com/spire-studio/pyfe4ai/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-orange" alt="License: Apache 2.0"></a>
  <a href="https://spire-studio.github.io/pyfe4ai/docs/"><img src="https://img.shields.io/badge/docs-Sphinx-blueviolet?logo=readthedocs&logoColor=white" alt="Documentation"></a>
</p>

<p align="center">
  <a href="https://spire-studio.github.io/pyfe4ai/"><strong>Project page</strong></a> ·
  <a href="https://spire-studio.github.io/pyfe4ai/docs/"><strong>Documentation</strong></a> ·
  <a href="https://github.com/spire-studio/pyfe4ai/blob/main/CHANGELOG.md"><strong>Changelog</strong></a> ·
  <a href="https://github.com/spire-studio/pyfe4ai/issues"><strong>Issues</strong></a>
</p>

# PyFE4AI

<a id="key-features"></a>
PyFE4AI is a research-oriented Python library of functional encryption (FE)
for trustworthy AI. In functional encryption, a functional key reveals one
chosen function of encrypted data and nothing else about it: here an inner
product `<x, y>` with the key's weights `y`, a weighted sum over several
clients' vectors, or a quadratic form. PyFE4AI implements 42 inner-product
and quadratic FE schemes built on published constructions -- single-input,
multi-input, multi-client, decentralized, threshold and function-hiding,
under DDH, DCR, LWE, Ring-LWE and pairings -- behind one entry point,
`pyfe4ai.create()`, with an adapter for federated gradient aggregation and
encrypted linear inference, and an optional Rust core.

> [!WARNING]
> **Experimental research prototype, not audited — not recommended for
> production use.** The default parameters are toy-sized and not secure
> (see [Security Profiles](#security-profiles)), nothing is hardened against
> side channels, and interfaces, configuration options, defaults and
> serialized formats (keys, threshold key shares, ciphertexts, partial
> decryptions, parameter records, pickles) can change substantially between
> versions, without a deprecation period. When upgrading, regenerate keys and
> key shares rather than reusing them: this version refuses the threshold key
> shares of v0.3.0 (see the
> [upgrade guide](https://spire-studio.github.io/pyfe4ai/docs/upgrading.html)).
> Read the [Scheme Status](#scheme-status) and
> [Known Issues](#known-issues--security-notice) before relying on any
> privacy property.

**Contents:** [Quick Start](#quick-start) · [Usage](#usage) ·
[Implemented Schemes](#implemented-schemes) · [Scheme Status](#scheme-status) ·
[Security Profiles](#security-profiles) · [Performance](#performance) ·
[Known Issues](#known-issues--security-notice) ·
[Pairing-Based Schemes](#pairing-based-schemes) · [Development](#development) ·
[Documentation](#documentation) · [Citing](#citing)

## Quick Start

PyFE4AI is installed from a clone of this repository (Python 3.10 or later):

```bash
git clone https://github.com/spire-studio/pyfe4ai.git
cd pyfe4ai
conda env create -f environment.yml   # Python and gmpy2 from conda-forge
conda activate pyfe4ai
pip install -e ".[test]"              # editable, with pytest
python -m pytest -m core -q           # the tests that need no pairing backend
```

With a Rust toolchain (`cargo`) on the `PATH`, `pip install` also builds the
optional native extension `pyfe4ai._rust_native` (the first build downloads
its crates from crates.io; `python tools/build_rust.py` rebuilds it,
`PYFE4AI_NO_NATIVE=1` skips it). It runs the Ring-LWE, dense-LWE and
threshold hot paths with the same outputs as the Python code, and provides
ristretto255 and BLS12-381, the default group of the pairing-based schemes.
Without it the library runs its Python implementation, about 5 to 35 times
slower on the lattice schemes, and says so once with an `FEWarning`;
`pyfe4ai.backend_info()` reports what runs, and `PYFE4AI_BACKEND=python`
forces the Python implementation. Other pairing curves need charm-crypto (see
[Pairing-Based Schemes](#pairing-based-schemes)). Details:
[Installation](https://spire-studio.github.io/pyfe4ai/docs/install.html) and
[Environment variables](https://spire-studio.github.io/pyfe4ai/docs/environment.html).

## Usage

Every scheme is built from its id with `pyfe4ai.create()`, which returns the
scheme's key generator (built from the config, not yet set up) and its crypto
class, whose objects encrypt (with `pp` and `sk`) and decrypt (with `pp` and
a functional key):

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

`pyfe4ai.list_schemes()` lists the scheme ids (`pyfe4ai.list_schemes("mcfe",
capabilities="threshold")` the threshold MCFEs) and
`pyfe4ai.get_scheme("mcfe/ring_lwe")` gives a scheme's classes, kind,
capabilities and review status. The classes are also importable by name,
named after the id (`from pyfe4ai.schemes.sife import SIFEDDH,
SIFEDDHKeyGenerator`; `mcfe/ring_lwe_threshold` is `ThresholdMCFERingLWE`).
The documentation's
[Quick start](https://spire-studio.github.io/pyfe4ai/docs/quickstart.html)
continues with several clients, federated learning and real parameters,
[Choosing a scheme](https://spire-studio.github.io/pyfe4ai/docs/choosing.html)
explains the families, and
[`examples/`](examples/README.md) has a runnable script per family.

## Implemented Schemes

| Family | Instantiations | Paper(s) |
|--------|---------------|----------|
| **SIFE** (single-input) | DDH, DDH-Dynamic, LWE | [ABDP15](https://eprint.iacr.org/2015/017.pdf) (PKC '15) |
| | DDH over ristretto255 *(unreviewed)* | ABDP15 in the ristretto255 group ([RFC 9496](https://www.rfc-editor.org/rfc/rfc9496)) |
| | Damgård-DDH, FullySec-LWE, Paillier | [ALS16](https://eprint.iacr.org/2015/608.pdf) (CRYPTO '16) |
| | Ring-LWE | [BMMS21](https://eprint.iacr.org/2021/046.pdf) (PKC '22), Sec. 4 |
| | FullySec-Ring-LWE (adaptive) | BMMS21, Sec. 5 |
| | FH-IPE *(pairing)* | [KLMMRW](https://eprint.iacr.org/2016/440.pdf) (SCN '18) |
| | Partial-FH-IPE *(pairing)* | [Gay20](https://eprint.iacr.org/2020/093.pdf) (PKC '20) |
| **MIFE** (multi-input) | DDH, Damgård-DDH, LWE, FullySec-LWE, Paillier | [ACFGU18](https://eprint.iacr.org/2017/972.pdf) (CRYPTO '18), Fig. 4 + ALS16 / ABDP15 |
| | DDH over ristretto255 *(unreviewed)* | ACFGU18 Fig. 4 + ABDP15 in ristretto255 |
| | Ring-LWE | ACFGU18 Fig. 4 as in [BMMS21](https://eprint.iacr.org/2021/046.pdf), Sec. 6 |
| | FH-IPE *(pairing)* | ACFGU18, Fig. 12 |
| | FH-Multi-IPE *(pairing)* | [DOT18](https://eprint.iacr.org/2018/061.pdf) (PKC '18), Sec. 3.1 |
| **MCFE** (multi-client) | DDH | [CDGPP18](https://eprint.iacr.org/2017/989.pdf) (ASIACRYPT '18) |
| | DDH over ristretto255 *(unreviewed)* | CDGPP18 in ristretto255, labels hashed with [RFC 9380](https://www.rfc-editor.org/rfc/rfc9380) |
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

> ¹ The threshold variants add a secret-sharing layer on top of the MIFE or
> MCFE scheme; there is no separate paper for each combination. tMIFE-DDH and
> tMCFE-DDH share the published functional key and interpolate it in the
> exponent; the layers of tMIFE-LWE, tMCFE-LWE, tMCFE-Ring-LWE (noise
> flooding, replicated key sharing for `t < s`) and tMCFE-FH-Multi-IPE are our
> own (*experimental*). All of them assume honest-but-curious decryption
> servers; see
> [Threshold decryption and the ML adapter](https://spire-studio.github.io/pyfe4ai/docs/threshold_ml.html).

## Scheme Status

Every registered scheme carries a review status, declared with the scheme in
the registry [`pyfe4ai/core/registry.py`](pyfe4ai/core/registry.py) and
available at runtime as `pyfe4ai.get_scheme_status("family/variant")`.
Instantiating a key generator or crypto object of a known-insecure scheme
emits a `pyfe4ai.FESecurityWarning`.

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
schemes; none is known to be insecure. Experimental: the threshold layers of
`mife/lwe_threshold`, `mcfe/lwe_threshold`, `mcfe/ring_lwe_threshold` and
`mcfe/fh_multi_ipe_pairing_threshold`, and the carry mask of
`mcfe/ddh_decentralized`. Unreviewed: the DDH schemes over ristretto255
(`*/ddh_ristretto255`, selected with `group="ristretto255"`), new code.

<details>
<summary><strong>Status, construction and deviations of every scheme</strong> (42 rows, rendered by <code>pyfe4ai.core.status.markdown_table()</code>; a test keeps it in sync)</summary>

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

</details>

## Security Profiles

Every key generator accepts a `security_profile` entry in its config
([`pyfe4ai/core/profiles.py`](pyfe4ai/core/profiles.py)). The default,
`"toy"`, keeps the small default parameters -- fast, and **not secure**.
`"standard"` fills in parameters aiming at 128-bit classical security and
rejects explicit values below them with `FEValidationError`:

```python
import pyfe4ai

kg, SIFEDDH = pyfe4ai.create("sife/ddh", {"eta": 3, "security_profile": "standard"})  # 3072-bit group
```

Key generators keep their parameters in a parameter cache, one record per
scheme type: setting up another configuration of the same scheme (here
`standard` after the toy example above) replaces the record and says so with
an `FEWarning`. Give each configuration its own `cache_dir` to keep both (see
[the parameter cache](https://spire-studio.github.io/pyfe4ai/docs/environment.html#the-parameter-cache)).

| Family | `standard` |
|---|---|
| Discrete log: DDH, Damgård / MDDH | the 3072-bit RFC 3526 MODP group 15 |
| DDH over ristretto255 (`*/ddh_ristretto255`, unreviewed) | the group is fixed (about 126-bit: Pollard rho in a group of order about 2^252); the profile sets nothing |
| DCR: Paillier | N ≈ 3072 bits, two 1536-bit safe primes generated for every key generator |
| Ring-LWE (SIFE, MIFE, MCFE, tMCFE, dMCFE, adaptive SIFE) | `ring_n = "auto"`: the smallest ring dimension whose modulus meets a lattice-estimator bound for 128 bits |
| Dense LWE (`mcfe/lwe`, `fullysec_lwe`, `lwe_threshold`, `lwe_decentralized`) | `lwe_n = "auto"` from a lattice-estimator table for the LWE of the scheme's proof |
| LWE with a public matrix (`sife/lwe`, `mife/lwe`) | `lwe_n = "auto"`, read at the error width of the LWE problem the proof reduces to; `matrix_mode` trades memory for time |
| Pairings: FH-IPE, FH-MCFE family, quadratic FE | BLS12-381: about **120–126-bit** after the exTNFS attacks, below the target; a `FESecurityWarning` says so |
| Refused (`FEValidationError`) | `sife/fullysec_lwe`, `mife/fullysec_lwe` (ALS16), `mife/lwe_threshold`, `mcfe/paillier_decentralized`: no parameter set that is both secure and usable |

The statistical parameter `stat_param` (rounding and leftover-hash slack,
flooding, masks) is 128 under `standard`; lower explicit values are refused.
The profile sizes the hardness assumptions only; the caveats under
[Known Issues](#known-issues--security-notice) still apply. The
documentation's [Security & Status](https://spire-studio.github.io/pyfe4ai/docs/security.html)
and [Configuration options](https://spire-studio.github.io/pyfe4ai/docs/configuration.html)
pages list every option under each profile.

<details>
<summary><strong>Parameters and measured costs per family</strong></summary>

| Family | `standard` parameters | Measured cost (one thread; 2 clients unless stated) |
|---|---|---|
| Discrete log: DDH, Damgård / MDDH (SIFE, MIFE, MCFE, tMIFE, tMCFE) | `sec_param = modulus_length = 3072`: RFC 3526 MODP group 15 (NIST SP 800-57 Part 1: 3072 bits for 128-bit security) | setup < 0.1 s; encrypt and aggregate < 1 s |
| `mcfe/ddh_decentralized` | as above; key shares (masked with `stat_param` = 128 bits against carries) are combined under a DCR modulus of about 6410 bits | setup ≈ 2 min (two 3207-bit safe primes); key share 0.7 s |
| DCR: Paillier (SIFE, MIFE, MCFE) | `bit_length = 1536`: two 1536-bit safe primes, N ≈ 3072 bits, generated for every key generator (a cached modulus could have known factors) | setup a few seconds (two 1536-bit safe primes) |
| Ring-LWE (SIFE, MIFE, MCFE, tMCFE, dMCFE) | κ = `sec_param` = 128 and `ring_n = "auto"`: the smallest n in {1024, …, 32768} whose derived q (including the threshold flooding) meets a lattice-estimator bound for Ring-LWE with Gaussian secret and errors (`ring_lwe_utils.RING_LWE_128_LOG_Q`, rows by width 2–16; regenerate with `tools/params/estimate_ring_lwe.py`), read at the smallest RLWE width of the scheme's proof -- about 8 with the BMMS21 Sec. 4.2 widths, giving log2 q ≤ 31 / 57 / 111 / 218 / 435 / 872 (the Homomorphic Encryption Standard's 27 / 54 / 109 / 218 / 438 / 881 is about 125 bits by the same estimator); an explicit `ring_n` is checked against the same bound | Python backend, SIFE / MIFE (eta = 10, X = Y = 50): n = 4096; one ciphertext holds up to 4096 vectors at the same cost, encrypt 0.4–0.5 s (0.12 ms per vector when full), decrypt 0.04 s (SIFE) / 0.08 s (MIFE). Federated rounds of 10^4 gradient elements, Python backend / Rust extension (`benchmarks/run.py fl-compare`: one thread, 3 clients, precision 2, i.e. X = 100, Y = 1): `ring_lwe` (n = 4096, JSON ciphertexts, the default) 345 / 18.2 ms per client and 157 / 7.5 ms to aggregate; `ring_lwe_threshold` with its defaults (binary ciphertexts, a 212-bit RNS q of four factors, split key shares; n = 8192, s = 3, t = 2) 493 / 26.4 ms per client, 274 / 8.0 ms per partial decryption and 686 / 27.8 ms for every party of the round on one machine; the offline phase, binary ciphertexts, more threads, other threshold options and the decentralized MCFE are in `benchmarks/PERFORMANCE.md`. Functional keys of the MCFE variants are bound to a label (round) |
| Adaptively secure Ring-LWE: `sife/fullysec_ring_lwe` (BMMS21 Sec. 5) | as Ring-LWE, plus `m = "auto"`: the smallest number m ≥ 5 of key polynomials whose q meets the same bound (the paper's m = 3 admits no modulus under its own constraints, see the module docstring); widths from Theorems 1–2 and Appendix C.1 with complexity leveraging over (2X+1)^(2·eta); exact FACCT Gaussians by default (`gaussian_sampler`); experimental `ndarray_layout = "packed"`: up to n vectors per ciphertext, leveraging over the span of the differences | eta = 4, X = Y = 4: n = 4096, m = 17, setup 0.25 s, encrypt 0.12 s, decrypt 66 ms; eta = 10, X = Y = 50: n = 8192, m = 8, setup 1.0 s, encrypt 0.31 s, decrypt 0.13 s; packed eta = 1, X = 100, Y = 1: n = 4096, m = 12, 4096 elements per ciphertext (182 bytes each), setup 75 ms, encrypt 79 ms, decrypt 46 ms per ciphertext (Rust extension, exact sampler) |
| Dense LWE: the ABMPSW20 MCFE family (`mcfe/lwe`, `fullysec_lwe`, `lwe_threshold`, `lwe_decentralized`) | `lwe_n = "auto"`: the smallest n0 in {4096, 4608, …, 16384} whose derived q meets a [lattice-estimator](https://github.com/malb/lattice-estimator) bound for LWE(n0, q, ⌈2√n0⌉) with a uniform secret -- the assumption of the scheme's proof (`lwe_utils.DENSE_LWE_128_LOG_Q`, e.g. log2 q ≤ 118 / 173 / 227 for n0 = 4096 / 6144 / 8192; regenerate with `tools/params/estimate_dense_lwe.py`); an explicit `lwe_n` is checked against the same table. No public matrix: a label is hashed to n0 + m0 values. `lwe_decentralized` combines key shares under a DCR modulus of ≥ 3072 bits, so its key derivation is not post-quantum | n0 = 7168–8192 for small bounds (stat_param 128); Python backend: about 2.7 ms per element and client to encrypt and 2.8 ms per element to aggregate (543 / 551 ms for 200 elements, one label per element); with the Rust extension 92.6 / 100.1 ms for 200 elements (client / aggregator, 3 clients); dMCFE key share ≈ 90 s per client |
| LWE with a public matrix: ABDP15 (`sife/lwe`, `mife/lwe`) | `lwe_n = "auto"`: the smallest n in {512, 576, …, 6144} whose (q, σ′) meets a lattice-estimator bound on log2(q/σ′) for wide-error LWE with a uniform secret (`lwe_utils.WIDE_LWE_128_LOG_RATIO`, rows by width; regenerate with `tools/params/estimate_wide_lwe.py`), read at σ′ = σ/(1 + B√eta), the error width of the LWE problem that the security proof reduces to (B = 2X, or 4X per MIFE slot; ABDP15 full version, Thm. 4.4 with the width derived in the proof of its Thm. 6.6 = Bourse 2017, Thm. 8.2.2; its Sec. 6.2 parameter list states the stronger (ℓB + 1)σ′), not at the public key's own σ, which gives 17–67% larger dimensions for 8 ≤ X = Y ≤ 1000; an explicit `lwe_n` is checked the same way. A (m ≈ n·log2 q rows) is published as a 32-byte seed and expanded with SHAKE-128. `matrix_mode` (key generator and crypto configs, forwarded by the ML adapter): `"memory"` keeps A expanded, `"stream"` expands it again for every batch | n = 1280 / 1536 for X = Y = 50 / 256 and eta = 10 (A: 1.3·10^8 / 2.2·10^8 entries; 896 / 1088 when the table was read at σ); MIFE is sized as if its vectors had dimension n·eta (ACFGU18, footnote 14), e.g. n = 1408 for 100 slots of dimension eta = 1 with X = Y = 50 (1664 for eta = 10). Encryptor alone, n = 1536: `"memory"` 0.24 s per encryption and 5.3 GB, `"stream"` 6.9 s and 0.7 GB (0.11 s and 3.4 s at the former n = 1088); the ndarray helpers encrypt in batches of 64 (eta = 1, n = 1088: 6.6 vs 44 ms per element). Key generation (setup, the noise of the keys sampled exactly) 8.3 s / 17 s for X = Y = 50 / 256, 47 s for the 100 MIFE slots. The public key is m × eta per slot (46 MB of JSON for eta = 10), so keep eta small. `mife/lwe_threshold` (experimental) is refused, see below |
| Pairing: FH-IPE, FH-MCFE family, quadratic FE | BLS12-381 (Rust extension; the `toy` default too): about **120–126-bit** security after the exTNFS attacks, below the target; a `FESecurityWarning` says so | `report` benchmark profile: SGP 30 ms, Quad 48 ms, MCFE FH-Multi-IPE 47 ms end to end |
| Refused | `sife/fullysec_lwe`, `mife/fullysec_lwe` (ALS16), `mife/lwe_threshold`, `mcfe/paillier_decentralized` | `FEValidationError`: no parameter set that is both secure and usable. ALS16 at a 128-bit dimension (lattice estimator, n = 3328–4352 for X = Y ≤ 50 and eta ≤ 10, with m = 4n·log2 q as its Theorem 2 requires): A has 1–2·10^10 entries and a ciphertext 2.9–4.5·10^6 coordinates of 219–258 bits (80–145 MB); its LWE-to-mheLWE reduction loses about n^6 in the noise ratio, so the parameters cannot be tightened -- for adaptive security use `sife/fullysec_ring_lwe`, otherwise the Ring-LWE scheme of the same family. tMIFE-LWE: read at σ′, the LWE of its flooding-enlarged q (stat_param 128, two clients) needs a dimension at the very end of the table (5888–6144, A: 6.5–7.8·10^9 entries) for a few small bounds, e.g. X = Y ∈ {1, 2, 3, 8, 10} at eta = 1, and beyond it for most others (X = Y = 4–6, X = Y ≥ 12, most eta ≥ 2; about 6500 for X = Y = 50, q of 217 bits; read at σ it took 5952, with key generation 38 min and one encryption 572 s in `"stream"` mode); use `mife/ddh_threshold`, or `mcfe/ring_lwe_threshold` with labels. The opt-in KL-divergence accounting of the flooding (`flooding_accounting = "kl"`, t = s and an explicit `max_partial_decryptions` budget, a weaker guarantee: about stat_param − 6.85 bits in the bit-security sense of Micciancio and Walter, a total statistical distance of about 2^-65 at stat_param = 128, for the first `max_partial_decryptions` partial decryptions of every server) would bring it to about 4864 at σ′ for s = t = 3 and 2^40 flooded values (A: 4.1·10^9 entries, `"stream"` mode), but `standard` refuses tMIFE-LWE with either accounting; "kl" applies to it under `toy`. dMCFE-Paillier needs a DCR modulus above 15000 bits for combining key shares |

</details>

## Performance

The same code on its two backends: the Rust extension and the Python
implementation compute the same values (the test suite compares them bit for
bit). One federated-learning round, from the *Native and Python backends*
table of [`benchmarks/PERFORMANCE.md`](benchmarks/PERFORMANCE.md), which
gives the commands: Apple M2 Max, one thread, `security_profile="standard"`,
three clients, gradients with two decimal digits (bounds X = 100, Y = 1),
medians of interleaved runs:

| Workload | Ciphertexts | Python backend | Rust backend |
|---|---|---:|---:|
| MCFE Ring-LWE (`mcfe/ring_lwe`), 10^4 elements: client / aggregator | JSON (default) | 345 / 157 ms | 18.2 / 7.5 ms |
| Threshold MCFE (`mcfe/ring_lwe_threshold`), 10^4 elements, `s = 3, t = 2`, defaults: client / partial decryption per server / every party of the round on one machine | binary (default) | 493 / 274 / 686 ms | 26.4 / 8.0 / 27.8 ms |
| Dense-LWE MCFE (`mcfe/lwe`), 200 elements: client / aggregator | JSON (default) | 543 / 551 ms | 92.6 / 100.1 ms |

`PYFE4AI_THREADS=k` lets one native call use `k` threads.
`benchmarks/PERFORMANCE.md` also covers the offline / online phases, the
ciphertext formats, several threads and the other families.

## Known Issues / Security Notice

No registered scheme is currently known to be insecure (should one be found,
the [Scheme Status](#scheme-status) table marks it *known-insecure* and its
key generator emits a `pyfe4ai.FESecurityWarning`). The following
limitations remain:

| Issue | Affected | In short |
|---|---|---|
| Toy default parameters | all schemes | not secure; pass `security_profile="standard"` (pairings stay at about 120–126 bits) |
| No side-channel hardening | all schemes | CPython and big-integer arithmetic are not constant-time, nor is the pure-Python ristretto255 fallback |
| No labels in multi-input FE | all MIFE schemes, quadratic `miqfe` | several inputs per client under one key reveal their differences; use MCFE for federated learning |
| Constructions of our own | tMIFE-LWE, tMCFE-LWE, tMCFE-Ring-LWE, tMCFE-FH-Multi-IPE, the dMCFE-DDH carry mask | no published security proof (*experimental*) |
| Honest-but-curious decryption servers | all threshold schemes | partial decryptions carry no proof of correctness; store and transmit the `sharing_id` of tMCFE-Ring-LWE key shares and partials |
| One ciphertext per client and label | all MCFE / DMCFE schemes | a client must not encrypt two inputs under one label; the scheme classes do not enforce it |
| Rounded Gaussian sampler by default | ALS16, the DCR schemes, ABDP15 LWE under `toy` | select the exact sampler with `"facct"` |
| Trusted parameter cache | most key generators | use a cache directory that nobody else can write to |
| The ML adapter plays every party | `pyfe4ai.ml` | it holds the master key: for experiments, not deployments |

<details>
<summary><strong>Details of each limitation</strong></summary>

| Issue | Affected schemes | Consequence |
|---|---|---|
| No labels in multi-input FE | all MIFE schemes and quadratic `miqfe` | By definition, the holder of a functional key may combine any ciphertext of each slot. Encrypting several inputs per client under one key epoch -- e.g. a gradient element by element -- therefore reveals the differences between that client's values. The ML adapter refuses MIFE for gradient aggregation; use MCFE (labelled ciphertexts) for federated learning. |
| Constructions of our own | tMIFE-LWE, tMCFE-LWE, tMCFE-Ring-LWE, tMCFE-FH-Multi-IPE; the dMCFE-DDH carry mask | The threshold layers (noise flooding, replicated or Shamir key sharing, sharing in the exponent) and the carry mask of `mcfe/ddh_decentralized` are our own and have no published security proof; see the module docstrings. |
| Honest-but-curious decryption servers | all threshold schemes (tMIFE, tMCFE) | Fewer than `t` servers learn nothing beyond the result, but partial decryptions carry no proof of correctness: one server that deviates changes the result without notice. `combine_decrypt` accepts partial decryptions for several enrolled sets (a list of `{sid: partial}` mappings) and raises `FESchemeError` if they disagree; this detects one faulty server (not which one) if every server is left out of some set, e.g. the `t + 1` subsets of size `t` of `t + 1` servers. tMCFE-DDH and tMCFE-FH-Multi-IPE servers decrypt only ciphertexts under the label they are asked for (`share_decrypt(..., label)`). tMCFE-Ring-LWE key shares and partial decryptions carry a `sharing_id`: store and transmit it with them. A combiner refuses partials of two sharings of one key (`MixedSharingsError`), which would otherwise decode to a wrong value inside the decoding bound; partials without an id combine only with each other, so a storage or transport layer that drops the id everywhere would let two sharings be mixed unnoticed. |
| One ciphertext per client and label | all MCFE / DMCFE schemes | As in the papers, a client must not encrypt two different inputs under one label (or tag); the scheme classes do not enforce it (the ML adapter refuses a second encryption per client and label within one wrapper, and FH-DMCFE client objects a second key per `tag_f`). |
| Rounded Gaussian by default outside the exact-default schemes | ALS16 (`fullysec_lwe`), the DCR schemes, and under `toy` ABDP15 SIFE / MIFE LWE (and the keys of tMIFE-LWE) | The proofs sample `D_{Z,sigma}`; these schemes follow the library-wide sampler, which by default rounds a continuous Gaussian: in double precision up to `2^40` (shared Box-Muller radii, ties to even, nothing beyond 8.57 sigma: 2^-25..2^-4 from i.i.d. `D_{Z,sigma}` per key or ciphertext per the October 2026 crypto review) and with MPFR at `bits(sigma) + 320` bits beyond. `sampler="facct"`, `gaussian_sampler("facct")`, `set_gaussian_sampler("facct")` or `PYFE4AI_GAUSSIAN_SAMPLER=facct` select the exact sampler for them (about 2^-185 per sample). The Ring-LWE MIFE / MCFE family, the dense-LWE MCFE keys `t` and SIFE FullySec Ring-LWE sample exactly by default and record it in their public parameters; for them the process-wide choice and the environment variable do not apply, and `standard` refuses `"rounded"`. ABDP15 (`sife/lwe`, `mife/lwe`; `mife/lwe_threshold` is refused under `standard`) does the same for its public keys' noise under `standard`, and under `toy` (also the keys of `mife/lwe_threshold`) follows its config's `gaussian_sampler`, else the library-wide choice; its public parameters record the sampler either way. The threshold flooding always uses the MPFR rounded sampler. Neither sampler is constant-time. |
| Toy default parameters | all | `sec_param` is a modulus bit length (default 128), not a security level; LWE dimensions default to 32–64 and the Ring-LWE dimension to 32; the statistical parameter `stat_param` defaults to 64. All far below any real security level (a key column of SIFE-LWE at its defaults falls to plain LLL in minutes). Pass `security_profile="standard"` for 128-bit-oriented parameters -- except for pairings (BLS12-381, about 120–126-bit) and the refused ALS16, tMIFE-LWE and dMCFE-Paillier schemes; see [Security Profiles](#security-profiles). |
| Parameter cache | key generators that cache parameters (most; not MCFE `lwe`, `fullysec_lwe`, `lwe_threshold`, the FH-MCFE family, MIFE `fh_ipe_pairing` and quadratic `quad` and `miqfe`) | Public parameters are written to and re-loaded from `<cache>/authority/...`, where `<cache>` is the `cache_dir` entry of the key generator's config, else `PYFE4AI_CACHE_DIR`, else the per-user cache directory (`~/Library/Caches/pyfe4ai` on macOS, `$XDG_CACHE_HOME/pyfe4ai` or `~/.cache/pyfe4ai` on Linux, `%LOCALAPPDATA%\pyfe4ai\Cache` on Windows). A cached file is used only if it matches the configuration; discrete-log groups are checked for primality, size, safe-prime structure, generator order and (at RFC 3526 sizes) for being the RFC prime; DCR moduli for size, compositeness and small factors (their factors are never stored); all other cached values are re-derived on load or expanded from a stored seed (LWE and Ring-LWE parameters and public matrices, the Paillier base, the Damgard `h`); under `standard` the lattice schemes re-check their moduli against the 128-bit tables. There is one record per scheme type (`authority/SIFE/param.json`, `authority/MCFE_RING_LWE/param.json`, ...; RNS moduli in `param-rns.json`): a key generator whose configuration does not match the cached record (other dimensions, bounds or profile -- e.g. when several examples or configurations run one after another) regenerates it with new parameters and seeds, with an `FEWarning` ("the parameter record ... does not match this configuration ...: regenerated ..."); an unreadable file is regenerated the same way, and files are written atomically. Keys and ciphertexts already made keep working with the public parameters they were made with, but setting up the earlier configuration again draws new parameters: give each configuration its own `cache_dir` to keep both. The DDH crypto objects keep their discrete-log tables under `<cache>/crypto/`. No check can tell whether someone knows the factors of a DCR modulus, so under `standard` the Paillier moduli and the dMCFE `N2` are generated for every key generator, never loaded or stored; under `toy` a planted modulus with known factors is still used: use a cache directory that nobody else can write to. |
| The ML adapter plays every party | `pyfe4ai.ml` | `FESchemeWrapper` runs the key generator, so it holds the master key and acts as clients, authority and decryptor in one trust domain; use it for experiments, not as a deployment. For `mcfe/ring_lwe_threshold` it also runs the decryption servers (`partial_decrypt_gradients`, `combine_partial_decryptions` show a deployment's messages, and a `PreparedAggregation` holds the servers' label key shares), so there the threshold protects nothing. By default `aggregate_gradients` needs a ciphertext and a non-zero weight for every enrolled client (`allow_partial=True` relaxes this), a client encrypts at most once per label and a label is aggregated with one weight vector; these checks are per wrapper object, and they do not stop very unequal weights (e.g. `(W, 1)` with `W` above twice the largest gradient entry), which separate two clients' gradients in one aggregate -- as with any inner-product FE, the weights are the key holder's choice. A pickled or copied wrapper (`multiprocessing`, Ray) carries the master key and every client's key: ship a worker its client's own crypto object (the scheme class built from `wrapper.pp` and `wrapper.kg.get_private_keys(nid)`), not the wrapper. |

</details>

Fixed issues are listed in [`SECURITY.md`](SECURITY.md) and
[`CHANGELOG.md`](CHANGELOG.md).

## Pairing-Based Schemes

The function-hiding schemes (`sife/fh_ipe_pairing`,
`sife/part_fh_ipe_pairing`, `mife/fh_ipe_pairing`, the FH-Multi-IPE MIFE,
MCFE, threshold and decentralized MCFE) and quadratic FE (`quadratic/sgp`,
`quad`, `miqfe`) need a pairing group. They run over an asymmetric (type-3)
group, BLS12-381 from the Rust extension by default; their constructions
assume SXDH or a generic asymmetric bilinear group, so symmetric groups
(`SS512`, `SS1024`) are rejected with `FEValidationError`. Up to v0.3.0 the
default was `MNT224`, whose keys and ciphertexts BLS12-381 cannot read.

Other curves (`pairing_group_param`, e.g. `MNT224` or `BN254`) need
`charm-crypto-framework`, which is built from source and needs GMP, OpenSSL
and [PBC 1.0.0](https://crypto.stanford.edu/pbc/) first (e.g.
`brew install gmp openssl@3 pbc` on macOS; on Linux, build PBC as in
[`docker/Dockerfile.pairing`](docker/Dockerfile.pairing)):

```bash
pip install -e ".[test,pairing]"
PYFE4AI_TEST_PAIRING_GROUP=MNT224 python -m pytest -m extended -q   # the pairing tests on MNT224
```

The Docker image has all of them (it does not build the Rust extension, so
select a charm curve for the pairing tests):

```bash
docker build -f docker/Dockerfile.pairing -t pyfe4ai-pairing .
docker run --rm -e PYFE4AI_TEST_PAIRING_GROUP=MNT224 pyfe4ai-pairing
```

## Development

<a id="running-tests"></a>
**Tests.** `python -m pytest -q` runs everything; `-m core` the tests that
need no pairing backend, `-m extended` the pairing-based and quadratic
schemes, `-m "not slow"` all but the about 20 slowest, `-m threshold` (or another
group) one directory of `tests/`. With the Rust extension built, run the
suite on both backends and the Rust core's own tests:

```bash
PYFE4AI_BACKEND=native python -m pytest -q
PYFE4AI_BACKEND=python python -m pytest -q   # the reference implementation
(cd rust/core && cargo test --release && cargo build --release)
```

CI runs the core lane without the extension and, in the `rust-native` job,
`cargo test` and the full suite on the native backend (Python 3.10 and 3.13)
and on the Python backend (Python 3.10); the `full-research` job (manual
dispatch, before a release) adds charm-crypto and the pairing tests on
MNT224. The marks and test groups are described in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

<a id="examples"></a>
**Examples and benchmarks.** [`examples/`](examples/README.md) has a minimal
runnable script per family (`python examples/sife_minimal.py`), all with toy
parameters; the benchmarks run through `python benchmarks/run.py <command>`
([`benchmarks/README.md`](benchmarks/README.md)).

<a id="naming-convention"></a>
**Naming.** A scheme `family/variant` lives in
`pyfe4ai/schemes/<family>/<variant>.py` (`damgard_ddh.py` for `damgard`), and
its classes are named after the id: `Threshold` or `Decentralized` for those
variants, then the family, then the rest of the variant in CamelCase, and
`KeyGenerator` for the key generator (`mcfe/ring_lwe_threshold`:
`ThresholdMCFERingLWE`, `ThresholdMCFERingLWEKeyGenerator`).

<a id="repository-layout"></a>
<details>
<summary><strong>Repository layout</strong></summary>

- `pyfe4ai/`: installable Python package
  - `core/`: what every scheme shares
    - `registry.py`: the registry of the schemes (classes, kind, review status, capabilities, refusals) and `create()`
    - `base.py`: key-generator and crypto base classes, the parameter cache
    - `config.py`: the config class of every scheme and the rules that combine options
    - `formats/`, `versions.py`: the serialized objects and their version markers
    - `ids.py`: the rules for server and client ids (`lst_sid`, `lst_nid`: a `str` or an `int`, one type per list)
    - `status.py`: review status of every registered scheme (the table above)
    - `profiles.py`: `security_profile` (`"toy"` / `"standard"`)
    - `one_time.py`: one-time precomputations of the offline / online encryption
    - `forksafe.py`: the library's one fork handler (what a forked child renews; threads, copies and the process-wide state: `docs/threading.rst`)
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
- `docs/`: Sphinx documentation (see [Documentation](#documentation))
- `tests/`: end-to-end and regression tests, by topic (`api/`, `config/`, `params/`, `backend/`, `native/`, `sampling/`, `lattice/`, `primitives/`, `threshold/`, `offline/`, `state/`, `ml/`, `hardening/`), the schemes by family (`sife/`, `mife/`, `mcfe/`, `quadratic/`) and the crypto-review regressions (`review/`); pytest marks select groups (see [`CONTRIBUTING.md`](CONTRIBUTING.md))
- `examples/`: minimal runnable usage examples
- `benchmarks/`: the benchmarks, one command each of `benchmarks/run.py` (`python benchmarks/run.py --help`)
- `tools/`: `build_rust.py` (builds the native extension); `gen_site_matrix.py` (the scheme matrix of the project page, from the registry); `params/` (`estimate_dense_lwe.py`, `estimate_wide_lwe.py`, `estimate_ring_lwe.py`, `estimate_rns_modulus.py` regenerate the LWE and Ring-LWE security tables with SageMath and the lattice estimator)
- `site/`: project landing page (published with the docs to GitHub Pages)
- `pyproject.toml`: package metadata and the dependencies (extras `test`, `pairing`, `doc`)
- `environment.yml`: Conda environment (Python and the conda-forge packages); the dependencies come from `pyproject.toml` (`pip install -e ".[test]"`)
- `docker/Dockerfile.pairing`: Docker-based pairing-enabled test environment

</details>

## Documentation

<a id="api-documentation"></a>
The documentation, at **<https://spire-studio.github.io/pyfe4ai/docs/>**, is
rebuilt on every push to `main`:
[Installation](https://spire-studio.github.io/pyfe4ai/docs/install.html) ·
[Quick start](https://spire-studio.github.io/pyfe4ai/docs/quickstart.html) ·
[Choosing a scheme](https://spire-studio.github.io/pyfe4ai/docs/choosing.html) ·
[Security & Status](https://spire-studio.github.io/pyfe4ai/docs/security.html) ·
[Configuration options](https://spire-studio.github.io/pyfe4ai/docs/configuration.html) ·
[Environment variables and the parameter cache](https://spire-studio.github.io/pyfe4ai/docs/environment.html) ·
[Threshold decryption and the ML adapter](https://spire-studio.github.io/pyfe4ai/docs/threshold_ml.html) ·
[Threads, processes and copies](https://spire-studio.github.io/pyfe4ai/docs/threading.html) ·
[Glossary](https://spire-studio.github.io/pyfe4ai/docs/glossary.html) ·
[Upgrading from 0.3.0](https://spire-studio.github.io/pyfe4ai/docs/upgrading.html) ·
[API reference](https://spire-studio.github.io/pyfe4ai/docs/api/index.html).
To build it locally: `pip install -e ".[doc]"`, then `cd docs && make html`
(the pages land in `docs/_build/html/`).

## Disclaimer

PyFE4AI supports research and proof-of-concept implementations; it has
**not** undergone a security audit and **should not be used in production**
or wherever real-world security guarantees are required. Its implementations
are not hardened against side channels (timing, cache, power), the parameters
of the examples and tests are chosen for speed, not strength, and the
parameter cache is trusted (see [Known Issues](#known-issues--security-notice)).

## Citing

If you use PyFE4AI in your research, please cite it (a technical report is in
preparation and will be added here once published):

```bibtex
@software{pyfe4ai2026,
  author  = {{Spire Studio}},
  title   = {{PyFE4AI}: Python-based Functional Encryption
             for {AI} Security and Privacy},
  year    = {2026},
  version = {0.4.0},
  license = {Apache-2.0},
  url     = {https://github.com/spire-studio/pyfe4ai}
}
```

Citation metadata is also in [`CITATION.cff`](CITATION.cff) -- GitHub's
**"Cite this repository"** button in the sidebar gives APA and BibTeX.

## License

Apache License 2.0; see [`LICENSE`](LICENSE).

## Project Policies

- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security reporting (privately, not in public issues): [`SECURITY.md`](SECURITY.md)
- Code of conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
