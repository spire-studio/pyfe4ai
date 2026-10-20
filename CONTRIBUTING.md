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

Run the core test suite (auto-selected via `conftest.py` markers):

```bash
python -m pytest -m core -q
```

Run all tests:

```bash
python -m pytest -q
```

Without `charm-crypto-framework` the pairing tests are skipped; run them in
the Docker image above before changing pairing-based or quadratic schemes.

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
  `tests/test_registry.py` checks it against the code.
- A construction of our own, with no published scheme to check it against,
  gets the status *experimental*; one checked against its paper is
  *reviewed*, with fidelity *faithful* or *adapted* (list the adaptations).
- Regenerate the README *Scheme Status* table from
  `pyfe4ai.core.status.markdown_table()` instead of editing it by hand
  (`tests/test_scheme_status.py` compares them), and add the scheme to the
  *Implemented Schemes* table and to `docs/api/`.
- Say how `security_profile="standard"` sizes its parameters (or why it is
  refused) and add a test to `tests/test_security_profiles.py`.

### Adding or changing a backend operation

The scheme and primitive layers reach the Rust extension only through the
backend interface (`pyfe4ai/backend/interface.py`): they call
`pyfe4ai.backend.ops()` and `pyfe4ai.backend.ring(q, ring_n)`, and
`tests/test_backend_interface.py` fails on any other use (an import of
`pyfe4ai._rust_native` or of the implementation modules, `ext()`, `.cap`,
...). To accelerate a computation:

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
- Add a case to `tests/test_backend_interface.py`: the same inputs (and
  random bytes) through both implementations, the same outputs, and a spy
  that the native kernel ran.

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
