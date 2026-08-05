# Example Guide

This folder contains small runnable entry points for the main scheme families.

## Core Examples

- `sife_minimal.py`: baseline DDH-style single-input example
- `mife_minimal.py`: baseline DDH-style multi-input example
- `mife_fh_multi_ipe_minimal.py`: pairing-based `MIFE/FHMultiIPE` example
- `mcfe_fh_multi_ipe_minimal.py`: pairing-based `MCFE/FHMultiIPE` example
- `mcfe_fh_multi_ipe_decentralized_demo.py`: decentralized pairing-based `dMCFE/FHMultiIPE` example
- `mcfe_fh_multi_ipe_threshold_demo.py`: threshold pairing-based `tMCFE/FHMultiIPE` example
- `mcfe_threshold_demo.py`: threshold DDH-style multi-client example

## LWE Examples

- `lwe_minimal.py`: minimal `SIFE/LWE`
- `mife_lwe_minimal.py`: minimal `MIFE/LWE`
- `mcfe_lwe_minimal.py`: minimal `MCFE/LWE`
- `mcfe_lwe_decentralized_demo.py`: decentralized `dMCFE/LWE`
- `mcfe_lwe_threshold_demo.py`: threshold `tMCFE/LWE`
- `mife_lwe_threshold_demo.py`: threshold `tMIFE/LWE`

## Quadratic Examples

- `quadratic_sgp_minimal.py`: minimal `Quadratic/SGP` example for `x^T F y`
- `quadratic_quad_minimal.py`: minimal `Quadratic/Quad` example for public-key style `x^T F y`

Run any example from the repository root, for example:

```bash
conda run -n pyfe4ai python examples/mife_lwe_minimal.py
```

Benchmark entry points live in `benchmarks/`; see `benchmarks/README.md` for a
short index.
