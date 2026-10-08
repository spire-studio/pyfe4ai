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

For pairing-based FE families (and the full test suite, `python -m pytest -q`),
additionally install:

```bash
pip install -e ".[test,pairing]"
```

## Usage

```python
from pyfe4ai import SIFE, SIFEKeyGenerator

x = [2, 1, 3]
y = [4, 5, 6]

# Key generation. sec_param is the bit length of the DDH modulus, not a
# security level: 128 keeps the demo fast but is NOT secure (use >= 2048).
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

- **Multiple FE families** — DDH, LWE, Ring-LWE, FullySec-LWE, Paillier, and Damgård instantiations across SIFE, MIFE, and MCFE
- **Threshold & decentralized** — `tMIFE`, `tMCFE`, `dMCFE` for distributed trust settings
- **Function-hiding** — pairing-based FH-IPE and FH-Multi-IPE with full inner-product privacy
- **Quadratic FE** — SGP (secret-key) and Quad (public-key) quadratic functional encryption
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
| | Ring-LWE | [BMMS21](https://eprint.iacr.org/2021/046.pdf) (PKC '22) |
| | FH-IPE *(pairing)* | [KLMMRW](https://eprint.iacr.org/2016/440.pdf) (SCN '18) |
| | Partial-FH-IPE *(pairing)* | [Gay20](https://eprint.iacr.org/2020/093.pdf) (PKC '20) |
| **MIFE** (multi-input) | DDH, Damgård-DDH, LWE, FullySec-LWE, Paillier | [ACFGU18](https://eprint.iacr.org/2017/972.pdf) (CRYPTO '18), Fig. 4 + ALS16 / ABDP15 |
| | Ring-LWE | ACFGU18 Fig. 4 as in [BMMS21](https://eprint.iacr.org/2021/046.pdf), Sec. 6 |
| | FH-IPE *(pairing)* | ACFGU18, Fig. 12 |
| | FH-Multi-IPE *(pairing)* | [DOT18](https://eprint.iacr.org/2018/061.pdf) (PKC '18) ¹ |
| **MCFE** (multi-client) | DDH | [CDGPP18](https://eprint.iacr.org/2017/989.pdf) (ASIACRYPT '18) |
| | Damgård-DDH (MDDH), Paillier (DCR), LWE, FullySec-LWE | [ABMPSW20](https://eprint.iacr.org/2020/788.pdf) (SCN '20), Figs. 2–4 |
| | Ring-LWE | BMMS21, Sec. 6 (identity-based MCFE) |
| | FH-Multi-IPE *(pairing)* | [AGT21b](https://doi.org/10.1007/978-3-030-90453-1_8) (TCC '21), Sec. 6.2 |
| **Decentralized** | dMCFE-DDH, dMCFE-LWE, dMCFE-Paillier | ABMPSW20, App. D, Fig. 14 (DMCFE from MCFE) |
| | dMCFE-Ring-LWE | BMMS21, Sec. 6 ([ABKW19](https://eprint.iacr.org/2019/020.pdf) + GDSum) |
| | dMCFE-FH-Multi-IPE *(pairing)* | [NPS24](https://doi.org/10.62056/andkp2fgx) (IACR CiC '24), Fig. 1 |
| **Threshold** | tMIFE (DDH, LWE), tMCFE (DDH, LWE, Ring-LWE, FH-Multi-IPE) | [Xu+24](https://doi.org/10.1109/TDSC.2024.3350206) (IEEE TDSC '24) ² |
| **Quadratic** | SGP (secret-key), Multi-Input SGP | [DSGPP18](https://eprint.iacr.org/2018/206.pdf) (ePrint) ¹ |
| | Quad (public-key) | [Gay20](https://eprint.iacr.org/2020/093.pdf) (PKC '20), Fig. 4 |

> ¹ Not yet checked against the cited paper (status *unreviewed*, see
> [Scheme Status](#scheme-status)). The other rows implement the cited
> construction, with the deviations listed in the status table.
>
> ² Threshold variants add a Shamir secret-sharing layer on top of the
> corresponding MIFE or MCFE scheme; there is no separate standalone paper
> for each combination. tMCFE-DDH shares the single aggregated functional
> key of CDGPP18 and tMIFE-DDH the MIFE-DDH key, both interpolated in the
exponent. tMIFE-LWE, tMCFE-LWE and
> tMCFE-Ring-LWE additionally flood partial decryptions with noise, and
> tMCFE-FH-Multi-IPE shares the FH-MCFE key exponents and combines partial
> decryptions in `G_T`; these threshold layers are our own (status
> *experimental*, see the module docstrings).

> **⚠️ Not every scheme has been reviewed.** Check the
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

Currently 31 reviewed, 4 experimental and 3 unreviewed out of 38
registered schemes; none is known to be insecure.

| Scheme | Status | Fidelity | Construction | Notes |
|---|---|---|---|---|
| `sife/ddh` | reviewed | faithful | ABDP15 Construction 1 (DDH) | signed x, y; decryption bound dlog_bound (default 10^(precision + 2)) |
| `sife/lwe` | reviewed | adapted | ABDP15 Constructions 2 + 3 (LWE) | k = 0 (no r-ary decomposition); signed X/Y-bounded inputs with centred decoding; GoFE parameter recipe; noise is a rounded continuous Gaussian |
| `sife/paillier` | reviewed | faithful | ALS16 (DCR) | D_{Z,sigma} realised as a high-precision rounded Gaussian |
| `sife/damgard` | reviewed | faithful | ALS16 DDH (ACFGU18 Fig. 7, k = 1) | h = g^a with a discarded |
| `sife/ring_lwe` | reviewed | adapted | BMMS21 Sec. 4 (selective) + Remark 1 batching | signed messages and weights (p = 2lXY + 2, centred decoding); one ciphertext packs up to ring_n vectors; single NTT prime q; ring_n defaults to the toy value 32 |
| `sife/fullysec_lwe` | reviewed | adapted | ALS16 LWE (ACFGU18 Fig. 8) | signed inputs (K = 2 eta X Y + 1, centred decoding); GoFE parameter recipe; D_{Z,sigma} realised as a rounded Gaussian |
| `sife/ddh_dynamic` | reviewed | adapted | ABDP15 Construction 1, variable length | vectors of length <= eta handled as zero padding (prefix keys, truncated ciphertexts) |
| `sife/fh_ipe_pairing` | reviewed | adapted | KLMMRW18 Sec. 3 | vectors padded to (x, 1, 0) / (y, 0, 1), so eta = 1 and zero vectors are safe; keys in G1, ciphertexts in G2; generic bilinear group model; the encryption key is master-secret equivalent |
| `sife/part_fh_ipe_pairing` | reviewed | adapted | Gay20 Fig. 8 (CPA, k = 1) | M is a random full-rank public matrix (subspace_dim < eta), published with V^T M in the clear; Enc' is run by the key generator; no QANIZK |
| `mife/ddh` | reviewed | adapted | ACFGU18 Fig. 4 + ABDP15 (DDH) | ACFGU18 instantiates Fig. 4 with ALS16; here the selectively secure ABDP15 IPFE in every slot |
| `mife/lwe` | reviewed | adapted | ACFGU18 Fig. 4 + ABDP15 (LWE) | ACFGU18 instantiates Fig. 4 with ALS16; here the selectively secure ABDP15 IPFE (same two-step decryption); slots share A |
| `mife/paillier` | reviewed | adapted | ACFGU18 Fig. 4 + ALS16 (DCR, Fig. 9) | slots share N and g; D_{Z,sigma} realised as a high-precision rounded Gaussian |
| `mife/damgard` | reviewed | adapted | ACFGU18 Fig. 4 + ALS16 (DDH, Fig. 7) | slots share the group and h (the matrix A of Fig. 7) |
| `mife/ring_lwe` | reviewed | adapted | BMMS21 Sec. 6 MIFE-compiler (ACFGU18) | slots share a; parameters sized for dimension n * eta |
| `mife/fullysec_lwe` | reviewed | adapted | ACFGU18 Fig. 4 + ALS16 (LWE, Fig. 8) | slots share A; parameters derived for dimension n * eta and bound 3X |
| `mife/fh_ipe_pairing` | reviewed | adapted | ACFGU18 Fig. 12 (k = 1) | one r per key shared by all slots; per-slot FH-IPFE as in mcfe/fh_ipe (LV16 padding of App. B); asymmetric pairing (default MNT224) |
| `mife/fh_multi_ipe_pairing` | unreviewed | unknown | DOT18 |  |
| `mife/ddh_threshold` | reviewed | adapted | ACFGU18 Fig. 4 + ABDP15 + Shamir (TAPFed setting) | the functional key (d_i, z) is Shamir-shared mod q; any t servers interpolate in the exponent (threshold-ElGamal style), as mcfe/ddh_threshold |
| `mife/lwe_threshold` | experimental | ad-hoc | MIFE-LWE + Shamir + smudging | MIFE layer as published; the threshold layer (statistical flooding, derandomised noise, enlarged q) is our own |
| `mcfe/ddh` | reviewed | faithful | CDGPP18 |  |
| `mcfe/lwe` | reviewed | adapted | ABMPSW20 Fig. 4, selective parameters | signed messages and weights (K = 2nXY + 1, centred decoding); t is a rounded continuous Gaussian |
| `mcfe/paillier` | reviewed | faithful | ABMPSW20 Fig. 3 | D_{Z,sigma} realised as a high-precision rounded Gaussian |
| `mcfe/damgard` | reviewed | faithful | ABMPSW20 Fig. 2 (D_k-MDDH) | k = mddh_k (default 1, where it equals CDGPP18) |
| `mcfe/ring_lwe` | reviewed | adapted | BMMS21 Sec. 6 MCFE-compiler (identity-based) | functional keys are bound to a label; ndarray values are packed into one ciphertext per chunk, each chunk with its own label pad |
| `mcfe/fullysec_lwe` | reviewed | adapted | ABMPSW20 Fig. 4, adaptive parameters | signed messages and weights (K = 2nXY + 1, centred decoding); t is a rounded continuous Gaussian; m0 grows with n log2(2X + 1) |
| `mcfe/fh_multi_ipe_pairing` | reviewed | adapted | AGT21b Sec. 6.2 (as recapped in NPS24 Sec. 2) | FH-IPFE = ACFGU18 Fig. 12 (n = 1, SXDH) with LV16 padding; asymmetric pairing (default MNT224), symmetric groups rejected |
| `mcfe/fh_multi_ipe_pairing_threshold` | experimental | ad-hoc | AGT21b FH-MCFE + Shamir | FH-MCFE layer as published; the Shamir-shared key exponents and partial decryptions in G_T are our own; FH-IPFE = ACFGU18 Fig. 12 (n = 1, SXDH) with LV16 padding; asymmetric pairing (default MNT224), symmetric groups rejected |
| `mcfe/fh_multi_ipe_pairing_decentralized` | reviewed | adapted | NPS24 Fig. 1 (selective FH-DMCFE) | tag-f is bound to the whole weight vector and keys are derived deterministically (one key per client and tag-f); FH-IPFE = ACFGU18 Fig. 12 (n = 1, SXDH) with LV16 padding; asymmetric pairing (default MNT224), symmetric groups rejected |
| `mcfe/ddh_threshold` | reviewed | adapted | CDGPP18 + Shamir (TAPFed setting) | the aggregated key is Shamir-shared; any t servers interpolate in the exponent (threshold-ElGamal style) |
| `mcfe/ddh_decentralized` | reviewed | adapted | ABMPSW20 Fig. 14 (MCFE1 = CDGPP18) | MCFE2 = ABMPSW20 Fig. 3 (DCR); the zero-sum MCFE2 keys (pairwise masks) are sampled in-process |
| `mcfe/lwe_threshold` | experimental | ad-hoc | ABMPSW20 Fig. 4 + Shamir + flooding | MCFE layer as published; the threshold layer and its LWE-based flooding argument are our own (see the module docstring) |
| `mcfe/lwe_decentralized` | reviewed | adapted | ABMPSW20 Fig. 14 (MCFE1 = Fig. 4) | MCFE2 = ABMPSW20 Fig. 3 (DCR); the zero-sum MCFE2 keys (pairwise masks) are sampled in-process |
| `mcfe/ring_lwe_threshold` | experimental | ad-hoc | BMMS21 identity-based MCFE + Shamir + smudging | MCFE layer as published; the threshold layer (statistical flooding, derandomised noise, enlarged q) is our own |
| `mcfe/ring_lwe_decentralized` | reviewed | adapted | BMMS21 Sec. 6 identity-based DMCFE (ABKW19 + GDSum) | GDSum realised as pairwise PRF zero-sum masks (its full specification is only in the ePrint version); pairwise secrets sampled in-process |
| `mcfe/paillier_decentralized` | reviewed | adapted | ABMPSW20 Fig. 14 (MCFE1 = Fig. 3) | MCFE2 = ABMPSW20 Fig. 3 (DCR); the zero-sum MCFE2 keys (pairwise masks) are sampled in-process |
| `quadratic/quad` | reviewed | adapted | Gay20 Fig. 4 over Fig. 8 (CPA) | signed inputs (discrete log in [-n m B^3, n m B^3]); CPA variant (no QANIZK); asymmetric pairing (default MNT224) |
| `quadratic/sgp` | unreviewed | unknown | DSGPP18 |  |
| `quadratic/multi_input_sgp` | unreviewed | unknown | DSGPP18, multi-input |  |

## Repository Layout

- `pyfe4ai/`: installable Python package
  - `schemes/`: core cryptographic scheme implementations
    - `sife/`: single-input families
    - `mife/`: multi-input families
    - `mcfe/`: multi-client families
    - `quadratic/`: quadratic functional encryption families
  - `utils/`: utility modules
    - `crypto_constants.py`, `crypto_utils.py`: group generation and constants
    - `lwe_utils.py`, `ring_lwe_utils.py`: LWE / Ring-LWE helpers with NTT
    - `dlog_solver.py`: in-memory dlog table + BSGS discrete-log recovery
    - `ml_adapter.py`: ML integration (encrypted inference, FL aggregation)
    - `quantization.py`: float ↔ integer quantization toolkit
    - `exceptions.py`: `FEError` exception hierarchy
    - `sampling_utils.py`, `matrix_utils.py`, `modular_utils.py`: math helpers
    - `pairing_backend.py`, `pairing_utils.py`: charm-crypto pairing layer
  - `logger.py`, `config.yaml`: logging configuration
- `docs/`: Sphinx API documentation (see [API Documentation](#api-documentation))
- `tests/`: end-to-end and regression tests (`sife/`, `mife/`, `mcfe/`, `quadratic/`)
- `examples/`: minimal runnable usage examples
- `benchmarks/`: lightweight benchmark scripts
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

The schemes in the following files depend on
`charm-crypto-framework`:

- `pyfe4ai/schemes/sife/fh_ipe_pairing.py`
- `pyfe4ai/schemes/sife/part_fh_ipe_pairing.py`
- `pyfe4ai/schemes/mife/fh_ipe_pairing.py`
- `pyfe4ai/schemes/mife/fh_multi_ipe_pairing.py`
- `pyfe4ai/schemes/mcfe/fh_ipe.py`
- `pyfe4ai/schemes/mcfe/fh_multi_ipe_pairing.py`
- `pyfe4ai/schemes/mcfe/fh_multi_ipe_pairing_threshold.py`
- `pyfe4ai/schemes/mcfe/fh_multi_ipe_pairing_decentralized.py`
- `pyfe4ai/schemes/quadratic/sgp.py`
- `pyfe4ai/schemes/quadratic/multi_input_sgp.py`
- `pyfe4ai/schemes/quadratic/quad.py`

All of them run over an asymmetric (type-3) pairing group, `MNT224` by
default (`pairing_group_param`, e.g. `BN254`); their constructions assume
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
| Unreviewed schemes | MIFE `fh_multi_ipe_pairing` (DOT18); quadratic `sgp`, `multi_input_sgp` (DSGPP18) | Not yet checked against their papers: no defect is known, but none was looked for systematically either. |
| No labels in multi-input FE | all MIFE schemes | By definition, the holder of a functional key may combine any ciphertext of each slot. Encrypting several inputs per client under one key epoch -- e.g. a gradient element by element -- therefore reveals the differences between that client's values. The ML adapter refuses MIFE for gradient aggregation; use MCFE (labelled ciphertexts) for federated learning. |
| Experimental threshold layers | tMIFE-LWE, tMCFE-LWE, tMCFE-Ring-LWE, tMCFE-FH-Multi-IPE | The threshold layers (noise flooding, Shamir sharing in the exponent) are our own and have no published security proof; see the module docstrings. |
| One ciphertext per client and label | all MCFE / DMCFE schemes | As in the papers, a client must not encrypt two different inputs under one label (or tag); the library does not enforce it. |
| Toy default parameters | all | `sec_param` is a modulus bit length (default 128), not a security level; LWE dimensions default to 16–64 and the Ring-LWE dimension to 32; the statistical parameter `stat_param` defaults to 64. All far below any real security level (a key column of SIFE-LWE at its defaults falls to plain LLL in minutes); nothing here estimates concrete hardness. |
| Unvalidated parameter cache in the working directory | key generators that cache parameters (most; not MCFE `lwe`, `fullysec_lwe`, `lwe_threshold`, the FH-MCFE family, MIFE `fh_ipe_pairing` and quadratic `quad`) | Parameters (public values only; no trapdoors since the Paillier fix) are written to and re-loaded from `./config/authority/...` without structural validation; a planted `param.json` (e.g. a 5-bit group) is accepted. Run only in a directory that nobody else can write to. The pairing dlog cache (`./config/crypto/...`) is likewise trusted if its generators match: a planted table gives wrong results and an inflated bound makes decryption loop for a very long time. |
| Float fusion weights in `aggregate_gradients` | `utils/ml_adapter.py` | Float weights are quantised but the result is not rescaled (e.g. weights 0.5 give 100× the expected sum in decimal mode). Pass integer weights. |

Fixed since v0.2.0 (unreleased, see [`CHANGELOG.md`](CHANGELOG.md)): every
scheme previously marked *known-insecure* now implements a published
construction or, for four threshold variants, an *experimental* layer on
top of one. MCFE `paillier`, `lwe`, `fullysec_lwe` and `damgard` follow
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
serialised into a non-member point. MIFE HybridAlpha was removed.

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
  version = {0.2.0},
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
