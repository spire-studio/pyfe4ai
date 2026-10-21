# Contributing

Thanks for your interest in contributing to `pyfe4ai`.

This project is currently a research-oriented prototype, so contributions that
improve reproducibility, documentation, test stability, and implementation
clarity are especially welcome.

## Before You Start

- Open an issue first for large changes, new schemes, or API-breaking updates.
- Keep pull requests focused. Smaller, reviewable changes are preferred.
- If a change is based on a paper, include the exact reference in the PR
  description and, when possible, in the module docstring.

## Development Setup

Create the recommended environment:

```bash
conda env create -f environment.yml
conda activate pyfe4ai
pip install -e ".[test]"           # the package itself, so examples import it
```

For the pairing-based schemes, also run `pip install -e ".[test,pairing]"`.
`charm-crypto-framework` is built from source and needs GMP, OpenSSL and PBC
1.0.0 first (see *Quick Start* in `README.md`); `docker/Dockerfile.pairing` builds an
image with all of them:

```bash
docker build -f docker/Dockerfile.pairing -t pyfe4ai-pairing .
docker run --rm pyfe4ai-pairing
```

## Running Tests

Run all tests, or the core release lane (selected by marks, see below):

```bash
python -m pytest -q
python -m pytest -m core -q
```

Without `charm-crypto-framework` the pairing tests on other curves than
BLS12-381 are skipped; run them in the Docker image above before changing
pairing-based or quadratic schemes. Before a pull request run the suite also
with `PYFE4AI_BACKEND=python` (the reference implementation the native code
must match bit for bit) and, if Rust changed, `cargo test` in `rust/core`.

### Tests

The tests are grouped by topic, one directory each:

| Directory | What |
|---|---|
| `tests/api/` | `pyfe4ai.create()`, the registry and its derived tables, the scheme status, exceptions and warnings, the benchmark commands |
| `tests/config/` | typed configurations, client lists, the parameter cache, the serialized records and their versions |
| `tests/params/` | security profiles, the Ring-LWE parameters, the wide-error LWE bounds |
| `tests/backend/` | the backend interface (both implementations, the static boundary of the extension), `backend_info()`, the missing-extension notice |
| `tests/native/` | the extension and the `rust/core` C ABI against the Python reference (NTTs, DRBG, SIFE, dMCFE, dense LWE, ristretto255) |
| `tests/sampling/` | the Gaussian samplers (rounded, FACCT, wide) and their native kernels, the per-scheme sampler |
| `tests/lattice/` | NTTs, seeded LWE matrices, plaintext encodings, the Ring-LWE paths, binary ciphertexts, the RNS modulus |
| `tests/primitives/` | hashing into `Z_q` and the label hashes, discrete logs, ristretto255, BLS12-381 and the pairing groups |
| `tests/threshold/` | sharings (Shamir, replicated, seeded), key shares and their binding, threshold binary / RNS ciphertexts, flooding and its accounting |
| `tests/offline/` | offline / online phases and their one-time use |
| `tests/state/` | process-wide state, fork safety, the caches of the objects |
| `tests/ml/` | the ML adapter and quantization |
| `tests/hardening/` | input validation and decryption hardening across schemes, review fixes outside `tests/review/` |
| `tests/sife/`, `tests/mife/`, `tests/mcfe/`, `tests/quadratic/` | the schemes, one module per scheme |
| `tests/review/` | regression tests of the crypto reviews, one per finding (kept as the review wrote them) |

Helpers shared by several groups stay where they are: `tests/pairing_group.py`
(the pairing group of the tests), `tests/log_config.py` (logging of the
tests), and the byte streams and fixtures of
`tests/native/test_native_backend.py` that the native comparisons import.

`tests/conftest.py` marks every test (the marks are registered in
`pyproject.toml`, and `--strict-markers` refuses others):

- its group, the directory's name: `-m threshold`, `-m "api or config"`,
  `-m "not review"` (except `tests/native/`, see `native`);
- `core` or `extended`: `extended` needs charm-crypto (pairing schemes,
  pairing-group checks, quadratic FE), `core` is the rest -- the CI lane
  without the extension runs `-m core`;
- `native`: the test exercises the native code (the extension or the
  `rust/core` C ABI) and is skipped or not generated without it -- a `skipif`
  whose reason names the extension, a `"native"` backend parameter, a pairing
  module on BLS12-381; mark a test that skips itself in its body with
  `@pytest.mark.native` (with `tests.pairing_group.GROUP_MARKS` /
  `needs_group` if it needs the tests' pairing group; the review tests are
  listed in `NATIVE_IN_REVIEW` in `tests/conftest.py`). With the extension,
  `-m "not native"` runs exactly the tests that also run without it;
- `slow`: 2 s or more on an M2 Max in one of the five configurations (listed
  in `SLOW` in `tests/conftest.py`; about two thirds of the run time).

```bash
python -m pytest -m "not slow" -q          # the quick run
python -m pytest -m "not native" -q        # what runs without the extension
python -m pytest -m native -q              # after changing Rust code
python -m pytest -m "threshold and not slow" -q
python -m pytest tests/threshold -q        # the same group by path
```

A new test module goes into the directory of its topic (with nothing else
to register); a new directory needs its mark in `pyproject.toml`. Tests
must not depend on the working directory: read files relative to
`Path(__file__)` (`parents[2]` is the repository from `tests/<group>/`), and
use `tmp_path` and `PYFE4AI_CACHE_DIR` (the conftest gives the run a
temporary parameter cache).

Without the extension the library warns once per process that it falls back
to Python (an `FEWarning`, `pyfe4ai.backend.MISSING_EXTENSION_NOTICE`); the
conftest gives that notice at the start of the run, so that it does not land
in an arbitrary test (or fail one under `-W error`).
`tests/backend/test_backend_info.py` checks it in fresh interpreters.

### Benchmarks

`python benchmarks/run.py --help` lists the benchmark commands and
`python benchmarks/run.py <command> --help` their options; the code of each
command is in `benchmarks/commands/` (`main(argv, prog)`, registered in
`benchmarks/commands/__init__.py`), the shared helpers in
`benchmarks/_common.py`. A new benchmark is a new module there and an entry
in `COMMANDS`; `tests/api/test_registry.py` checks that every command has
its module and `--help`. Record measurements with the build and selection
(`pyfe4ai.backend.describe()`, as `fl-compare` does, or
`pyfe4ai.backend_info()`).

## Pull Request Guidelines

- Add or update tests for behavior changes whenever feasible.
- Do not mix unrelated refactors with functional changes.
- Update `README.md` if user-facing setup or usage changes.
- Preserve existing naming and module organization unless there is a strong
  reason to change it.
- Prefer clear code over clever code. This repository is easier to maintain
  when implementations stay close to the underlying papers.
- Add an entry to the *Unreleased* section of `CHANGELOG.md`.

### Adding or changing a scheme

- Cite the paper with the figure or section you implement, in the module
  docstring and in the PR, and list every deviation from it.
- Register the scheme in `SCHEMES` (`pyfe4ai/core/registry.py`): its id
  `family/variant`, kind, review status, capabilities and, if the
  `standard` profile refuses it, the reason; name its classes after the id
  (see the registry's docstring) and add its config class to
  `pyfe4ai/core/config.py`. The status table, the profiles' kinds, the ML
  adapter and the benchmark scripts read the registry, and
  `tests/api/test_registry.py` checks it against the code.
- A construction of our own, with no published scheme to check it against,
  gets the status *experimental*; one checked against its paper is
  *reviewed*, with fidelity *faithful* or *adapted* (list the adaptations).
- Regenerate the README *Scheme Status* table from
  `pyfe4ai.core.status.markdown_table()` instead of editing it by hand
  (`tests/api/test_scheme_status.py` compares them), and add the scheme to the
  *Implemented Schemes* table and to `docs/api/`.
- Say how `security_profile="standard"` sizes its parameters (or why it is
  refused) and add a test to `tests/params/test_security_profiles.py`.

### Adding or changing a backend operation

The scheme and primitive layers reach the Rust extension only through the
backend interface (`pyfe4ai/backend/interface.py`): they call
`pyfe4ai.backend.ops()` and `pyfe4ai.backend.ring(q, ring_n)`, and use
only the interface of what they return; `tests/backend/test_backend_interface.py`
fails on any other use (an import of `pyfe4ai._rust_native` or of the
implementation modules, also by `import_module`, `__import__` or
`sys.modules`; `ext()`, `.cap`; a `getattr` on the backend or its full path
`pyfe4ai.backend.x`; a member of a ring outside `interface.Ring`, ...). To
accelerate a computation:

- Declare the operation in `interface.py` (on `Backend`, or on `Ring` if it
  works in `Z_q[x] / (x^n + 1)`), with its contract: outputs, the
  randomness of each implementation (source, bytes per sample, order), the
  errors, and which inputs are secret.
- Implement the reference in `python_impl.py` (`PythonBackend` /
  `PythonRing`), moving the Python code the scheme ran, with its sources of
  randomness (or taking them as arguments), and make the scheme call the
  operation.
- Implement it in `native_impl.py` (`NativeBackend` / `RingContext`):
  check what the kernel takes (moduli, widths, value types, sizes) and
  return `super()` (the reference) otherwise; draw random bytes with
  `random_input` and `seed_kwargs` (the tests' hooks) and threads from
  `pyfe4ai.backend._threads`; map the kernel's errors to the reference's;
  wipe secret copies in the Rust code.
- Where the scheme's Python steps differ from one native call (another
  order of checks, element by element), keep both step sequences and let
  the scheme ask the backend which applies (`Ring.native`,
  `Backend.batched_layers`, ...), as the decryption and threshold kernels
  do; document the scheme steps as the reference.
- Add a case to `tests/backend/test_backend_interface.py`: the same inputs (and
  random bytes) through both implementations, the same outputs, and a spy
  that the native kernel ran.

### Caches, locks and other state

Keep a cache on the object that owns it (a key generator, a crypto object, a
server), created on first use. If it may hold native objects, leave it out
of the object's `__getstate__` (pickles and copies then rebuild it); never
change what an existing class pickles without a reason in the CHANGELOG
(other versions load those pickles). Module-level state is reserved for what
every object shares (selections, public tables, the threshold store locks):
`tests/state/test_global_state.py` refuses a new module-level cache, lock,
registry or lazily set global unless it is allow-listed there with its
reason and listed on `docs/threading.rst`. A process-wide lock is renewed in
a forked child with `pyfe4ai.core.forksafe.renew_lock`, other per-process
state is reset with `call_in_child` or `track`; nothing else calls
`os.register_at_fork`.

## Reporting Bugs

When filing a bug report, please include:

- operating system and architecture
- Python version
- how the environment was created
- the exact command you ran
- the full traceback or failure output
- whether the problem affects only pairing-based schemes or also core schemes
- the pyfe4ai version or commit, the scheme (`family/variant`) and the
  `security_profile`

## Research and Security Notes

- This repository is intended for research and experimentation.
- The code should not be treated as production-ready cryptographic software.
- For sensitive security reports, please follow the process in `SECURITY.md`
  instead of opening a public issue.
