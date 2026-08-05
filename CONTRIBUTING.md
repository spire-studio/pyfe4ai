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

Note: pairing-based schemes like `fh_ipe_pairing.py` and the quadratic
families depend on `charm-crypto-framework`, which may require extra system
libraries or a Linux/Docker-based environment depending on the platform.

## Pull Request Guidelines

- Add or update tests for behavior changes whenever feasible.
- Do not mix unrelated refactors with functional changes.
- Update `README.md` if user-facing setup or usage changes.
- Preserve existing naming and module organization unless there is a strong
  reason to change it.
- Prefer clear code over clever code. This repository is easier to maintain
  when implementations stay close to the underlying papers.

## Reporting Bugs

When filing a bug report, please include:

- operating system and architecture
- Python version
- how the environment was created
- the exact command you ran
- the full traceback or failure output
- whether the problem affects only pairing-based schemes or also core schemes

## Research and Security Notes

- This repository is intended for research and experimentation.
- The code should not be treated as production-ready cryptographic software.
- For sensitive security reports, please follow the process in `SECURITY.md`
  instead of opening a public issue.
