# Benchmark Guide

All benchmarks run through one entry point, `benchmarks/run.py`, with one
command per benchmark:

```bash
python benchmarks/run.py --help               # the commands
python benchmarks/run.py sife --help          # the options of one command
python benchmarks/run.py sife --profile small --rounds 1 --warmup 1
```

Run it from the repository with `pyfe4ai` installed (`pip install -e .`), or
with `PYTHONPATH` set to the repository. Each command keeps the options and
the output format of the script it replaces (the scripts were merged in
0.4.0; the code of each command is in `benchmarks/commands/`, the shared
helpers in `benchmarks/_common.py`).

## Commands

| Command | Former script | What it measures |
|---|---|---|
| `sife` | `benchmark_sife_instantiations.py` | `SIFE` instantiations such as `DDH`, `LWE`, `FullySecLWE`, `RingLWE`, `Paillier` and `DamgardDDH` under a shared single-input workload (not the adaptive `sife/fullysec_ring_lwe`) |
| `mife` | `benchmark_mife_instantiations.py` | core `MIFE` instantiations under a shared multi-input workload |
| `mcfe` | `benchmark_mcfe_instantiations.py` | core label-aware `MCFE` instantiations under a shared multi-client workload |
| `dmcfe` | `benchmark_dmcfe_instantiations.py` | decentralized `MCFE` instantiations under a shared workload |
| `tmife` | `benchmark_tmife_instantiations.py` | threshold `MIFE` instantiations |
| `tmcfe` | `benchmark_tmcfe_instantiations.py` | threshold `MCFE` instantiations |
| `lwe-family` | `benchmark_lwe_family.py` | `SIFE/MIFE/MCFE/dMCFE/tMCFE` over the repository's LWE family |
| `fh-family` | `benchmark_fh_multi_ipe_family.py` | `MIFE/MCFE/dMCFE/tMCFE` over the pairing-based `FHMultiIPE` family |
| `quadratic` | `benchmark_quadratic_family.py` | the two single-input quadratic FE schemes, `Quadratic/SGP` and `Quadratic/Quad` (the multi-input `quadratic/miqfe` is not benchmarked yet) |
| `ndarray` | `benchmark_ndarray_helpers.py` | representative `ndarray` helper pipelines across `SIFE`, `FHMultiIPE` and quadratic FE (`--group sife\|fh\|quadratic\|all`) |
| `ntt` | `benchmark_ntt.py` | schoolbook (`O(n^2)`) and NTT (`O(n log n)`) negacyclic polynomial multiplication, as used by the Ring-LWE schemes |
| `sampling` | `benchmark_sampling.py` | the default rounded Gaussian sampler and the exact `"facct"` sampler per width (`--e2e`: one MCFE Ring-LWE round, `security_profile="standard"`, with each) |
| `fl-compare` | `bench_fl_compare.py` | the FL aggregation round (`encrypt_gradient` + `aggregate_gradients`) per scheme family, with backend, threads, ciphertext format and offline phase as options; appends JSON records for before / after comparisons. Headline results are in [`PERFORMANCE.md`](PERFORMANCE.md) |
| `fl-coverage` | `fl_path_coverage.py` | not a timing benchmark: runs the ML-adapter FL path for every registry entry with that variant's own test configuration and reports `usable` / `unusable` (with the reason) / `unverified` (pairing entries without a backend for the default group); `--output json` for machine-readable output |
| `replicated-sharing` | `bench_replicated_sharing.py` | the replicated key sharing of the LWE threshold layers (`replicated`, `replicated-seeded`): dealing, a server's piece sums and its partial decryption with a cold cache, one JSON line per configuration (positional arguments `KIND S T SHARING [REPS]`) |
| `offline-potential` | `bench_offline_potential.py` | per-element split of the DDH and DCR MCFE into label-dependent offline work and the online rest |
| `fullysec-packing` | `bench_fullysec_packing.py` | the elementwise and packed layouts of `sife/fullysec_ring_lwe` |

The `fh-family` and `quadratic` commands, and `ndarray --group
fh|quadratic|all`, need a pairing backend: the Rust extension for BLS12-381
(the default), or `charm-crypto-framework` for the curves chosen with
`--pairing-group` (e.g. `MNT224`; see *Quick Start* in the top-level README,
or the Docker image); everything else runs without either. All commands use
the toy default parameters unless stated otherwise. Without the Rust
extension the schemes run their Python implementation (about 5 to 35 times
slower on the lattice schemes) and say so once with an `FEWarning`;
`pyfe4ai.backend_info()` reports what is in use, and `fl-compare` stores
`pyfe4ai.backend.describe()` with its records.

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
- Use `run.py ndarray --group sife|fh|quadratic` to keep
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
python benchmarks/run.py sife --profile small --rounds 1 --warmup 1
python benchmarks/run.py mife --profile medium --output json
python benchmarks/run.py mcfe --profile medium
python benchmarks/run.py dmcfe --profile medium
python benchmarks/run.py tmife --profile medium
python benchmarks/run.py tmcfe --profile medium
python benchmarks/run.py lwe-family --profile medium
python benchmarks/run.py fh-family --profile report
python benchmarks/run.py quadratic --profile report
python benchmarks/run.py ndarray --group sife --profile medium
python benchmarks/run.py ndarray --group fh --profile report --shape 2x2 --rounds 1
python benchmarks/run.py ndarray --group quadratic --profile report --shape 2x2 --rounds 1
python benchmarks/run.py fl-compare --families ring_lwe --elements 10000 --json out.json
python benchmarks/run.py fl-coverage
```

With conda, prefix each command with `conda run -n pyfe4ai`.
