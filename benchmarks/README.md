# Benchmark Guide

This folder contains lightweight end-to-end benchmark entry points for major
scheme families in the repository.

## Available Scripts

- `benchmark_lwe_family.py`: compares `SIFE/MIFE/MCFE/dMCFE/tMCFE` over the
  repository's LWE family.
- `benchmark_fh_multi_ipe_family.py`: compares `MIFE/MCFE/dMCFE/tMCFE` over the
  pairing-based `FHMultiIPE` family.
- `benchmark_quadratic_family.py`: compares the repository's two quadratic FE
  prototypes, `Quadratic/SGP` and `Quadratic/Quad`.
- `benchmark_sife_instantiations.py`: compares `SIFE` instantiations such as
  `DDH`, `LWE`, `FullySecLWE`, `RingLWE`, `Paillier`, and `DamgardDDH` under a
  shared single-input workload.
- `benchmark_mife_instantiations.py`: compares core `MIFE` instantiations under
  a shared multi-input workload.
- `benchmark_mcfe_instantiations.py`: compares core label-aware `MCFE`
  instantiations under a shared multi-client workload.
- `benchmark_dmcfe_instantiations.py`: compares decentralized `MCFE`
  instantiations under a shared workload.
- `benchmark_tmife_instantiations.py`: compares threshold `MIFE`
  instantiations.
- `benchmark_tmcfe_instantiations.py`: compares threshold `MCFE`
  instantiations.
- `benchmark_ndarray_helpers.py`: compares representative `ndarray` helper
  pipelines across `SIFE`, `FHMultiIPE`, and quadratic FE families.

## Recommended Usage

- Start with `--profile small --rounds 1` for a smoke run.
- Use `--profile medium` as the default table-building workload for
  non-pairing families.
- Use `--profile large` when you want a more expansion-oriented comparison.
- Use `--profile report` for heavier pairing-oriented `FH` and `Quadratic`
  runs, especially when you want more stable numbers for plots or tables.
- `report` is not meant to be the largest possible workload. It is the
  repository's "presentation-quality" profile: stable enough for tables, but
  still intended to finish in a reasonable amount of time.
- Use `--warmup 1` or higher when you want to hide one-time setup noise before
  recording timed rounds.
- Use `--seed N` when you want reproducible benchmark sampling across runs.
- Use `--output json` when you want to aggregate results with scripts or
  notebooks; the default `csv` output is easier to inspect directly in the
  terminal.
- Use `benchmark_ndarray_helpers.py --group sife|fh|quadratic` to keep
  comparisons fair and runs short.
- Use `--shape ROWSxCOLS` only when you intentionally want to override the
  profile's standard array workload.

## Methodology

- Use family benchmarks first when you want to understand the extra cost of
  moving from `SIFE` to `MIFE`, `MCFE`, `dMCFE`, or `tMCFE` under the same
  cryptographic line.
- Use instantiation benchmarks when you want to compare different backends for
  the same capability level, for example `SIFE/DDH` versus `SIFE/LWE`.
- Keep pairing-based and non-pairing families in separate tables whenever you
  want a fairer comparison.
- Each benchmark script performs a built-in correctness check before recording a
  timing sample, so the output is meant to be both a performance snapshot and a
  validation pass.
- Reported sizes are coarse serialized sizes from JSON-friendly public
  parameters, keys, ciphertexts, and partial-share bundles; they are useful for
  relative comparison, not for protocol-level wire formats.
- All benchmark scripts now emit the same top-level schema. Fields that do not
  apply to a scheme family are left blank so that outputs can be merged into a
  single table more easily.
- `*_total_bytes` columns describe the total serialized footprint for the whole
  object family in one benchmark run.
- `*_avg_bytes` columns describe the per-client or per-share average footprint
  when a benchmark naturally contains multiple clients or partial decryptions.
- `ndarray` helper benchmarks fix the output shape and report element throughput
  in addition to total time, which makes helper-oriented comparisons easier to
  read.

## Suggested Commands

```bash
conda run -n pyfe4ai python benchmarks/benchmark_sife_instantiations.py --profile small --rounds 1 --warmup 1
conda run -n pyfe4ai python benchmarks/benchmark_mife_instantiations.py --profile medium --output json
conda run -n pyfe4ai python benchmarks/benchmark_mcfe_instantiations.py --profile medium
conda run -n pyfe4ai python benchmarks/benchmark_dmcfe_instantiations.py --profile medium
conda run -n pyfe4ai python benchmarks/benchmark_tmife_instantiations.py --profile medium
conda run -n pyfe4ai python benchmarks/benchmark_tmcfe_instantiations.py --profile medium
conda run -n pyfe4ai python benchmarks/benchmark_lwe_family.py --profile medium
conda run -n pyfe4ai python benchmarks/benchmark_fh_multi_ipe_family.py --profile report
conda run -n pyfe4ai python benchmarks/benchmark_quadratic_family.py --profile report
conda run -n pyfe4ai python benchmarks/benchmark_ndarray_helpers.py --group sife --profile medium
conda run -n pyfe4ai python benchmarks/benchmark_ndarray_helpers.py --group fh --profile report --shape 2x2 --rounds 1
conda run -n pyfe4ai python benchmarks/benchmark_ndarray_helpers.py --group quadratic --profile report --shape 2x2 --rounds 1
```

Run a benchmark from the repository root, for example:

```bash
conda run -n pyfe4ai python benchmarks/benchmark_quadratic_family.py
```
