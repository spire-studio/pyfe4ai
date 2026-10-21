# Example Guide

This folder contains small runnable entry points for the main scheme families.
All examples use the toy default parameters (fast, **not secure**); pass
`"security_profile": "standard"` in a key generator config for 128-bit-oriented
parameters (see *Security Profiles* in the top-level README).

## Core Examples

- `sife_minimal.py`: baseline DDH-style single-input example
- `mife_minimal.py`: baseline DDH-style multi-input example
- `mcfe_threshold_demo.py`: threshold DDH-style multi-client example

## LWE Examples

- `lwe_minimal.py`: minimal `SIFE/LWE`
- `mife_lwe_minimal.py`: minimal `MIFE/LWE`
- `mcfe_lwe_minimal.py`: minimal `MCFE/LWE`
- `mcfe_lwe_decentralized_demo.py`: decentralized `dMCFE/LWE`
- `mcfe_lwe_threshold_demo.py`: threshold `tMCFE/LWE` (experimental threshold layer)
- `mcfe_ring_lwe_threshold_demo.py`: threshold `tMCFE/Ring-LWE` (experimental threshold layer):
  federated rounds with the default split key shares (a weight key share per
  server once, a label key share per round), then the same through the ML
  adapter with ciphertexts and partial decryptions sent as JSON
- `mife_lwe_threshold_demo.py`: threshold `tMIFE/LWE` (experimental threshold layer)

## ML / Privacy-Preserving AI

- `ml_encrypted_inference.py`: encrypted linear inference `<x, w>` with
  `SIFE/DDH` through the ML adapter
- `ml_fl_aggregation.py`: federated gradient aggregation with `MCFE/DDH`
  through the ML adapter (one wrapper plays every party: for experiments only)

## Pairing-Based Examples (need `charm-crypto-framework`)

Install it with `pip install -e ".[pairing]"` (needs GMP, OpenSSL and PBC
first) or run the examples in the `docker/Dockerfile.pairing` image; see *Quick Start*
and *Pairing-Based Schemes* in the top-level README.

- `mife_fh_multi_ipe_minimal.py`: `MIFE/FHMultiIPE` example
- `mcfe_fh_multi_ipe_minimal.py`: `MCFE/FHMultiIPE` example
- `mcfe_fh_multi_ipe_decentralized_demo.py`: decentralized `dMCFE/FHMultiIPE` example
- `mcfe_fh_multi_ipe_threshold_demo.py`: threshold `tMCFE/FHMultiIPE` example (experimental threshold layer)
- `quadratic_sgp_minimal.py`: minimal `Quadratic/SGP` example for `x^T F y`
- `quadratic_quad_minimal.py`: minimal `Quadratic/Quad` example for public-key style `x^T F y`

Run any example from the repository root after `pip install -e .` (see the
top-level README), for example:

```bash
conda run -n pyfe4ai python examples/mife_lwe_minimal.py
```

The benchmarks run through `python benchmarks/run.py <command>`; see
`benchmarks/README.md` for the commands.
