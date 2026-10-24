---
name: Bug report
about: Report a reproducible bug in the project
title: "[Bug] "
labels: bug
assignees: ""
---

<!-- Security vulnerabilities: please do not file them here; see SECURITY.md
     (private report: Security tab -> Report a vulnerability). -->

## Summary

Describe the problem briefly.

## Environment

- pyfe4ai version or commit (`python -c "import pyfe4ai; print(pyfe4ai.__version__)"`, `git rev-parse --short HEAD`):
- `python -c "import pyfe4ai; print(pyfe4ai.backend_info())"` (Rust extension, threads, sampler, cache directory):
- OS and architecture:
- Python version:
- Environment setup (conda `environment.yml`, pip, Docker `docker/Dockerfile.pairing`):
- `charm-crypto-framework` version and `pairing_group_param` (pairing schemes only):

## Scheme

- Scheme (`family/variant`, e.g. `mcfe/ring_lwe`):
- `security_profile` (`toy` or `standard`):
- Key generator / crypto config (without secret keys):

## Reproduction

A minimal script or the failing test:

```python
# minimal reproduction
```

```bash
# exact command, e.g. python -m pytest -m core -q tests/mcfe/test_ring_lwe.py
```

## Expected behavior

Describe what you expected to happen.

## Actual behavior

Paste the traceback, error message, or failing test output.

## Additional context

- Does this affect only pairing-based schemes? (`python -m pytest -m core -q` runs the tests that need no pairing backend.)
- Does it reproduce on the latest `main`?
