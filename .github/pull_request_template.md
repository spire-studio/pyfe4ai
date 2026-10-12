## Summary

- What does this PR change?
- Why is the change needed?

## Breaking changes

- Formats, APIs or config keys that change (or "none").

## Validation

- [ ] Added or updated tests where appropriate
- [ ] Ran the core lane locally (`python -m pytest -m core -q`)
- [ ] Ran the pairing tests (`python -m pytest -q` with `charm-crypto`, e.g. in the `Dockerfile.pairing` image) if pairing-based or quadratic schemes changed
- [ ] Added an entry to the *Unreleased* section of `CHANGELOG.md`
- [ ] Updated docs/README if user-facing behavior changed (for scheme changes: `status.py`, the README *Scheme Status* table via `markdown_table()`, `profiles.py`)

## Notes

- Paper or reference (with figure / section):
- Any follow-up work:
