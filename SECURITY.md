# Security Policy

## Scope

This repository contains research-oriented cryptographic prototype code. It is
not currently positioned as production-ready security software.

That said, responsible reporting of security issues is still appreciated,
especially for:

- cryptographic implementation mistakes
- unsafe defaults
- key material leakage
- insecure serialization or parameter handling
- dependency-related vulnerabilities

## Security Model

The schemes protect inputs against the parties named in their papers; the
library adds no defence against active network attackers and assumes
authenticated channels between parties. Decryption servers of the threshold
schemes are honest-but-curious: partial decryptions carry no proof of
correctness (redundant enrolled sets can detect a deviating server). Default
parameters are toy-sized; use `security_profile="standard"`, which aims at
128-bit security except for pairings (BN254, about 100-bit) and refuses the
ALS16 and dMCFE-Paillier schemes. See "Known
Issues / Security Notice" in the README for the full list of limitations.

## Supported Versions

At the moment, only the latest state of the `main` branch should be considered
supported for security fixes. The latest release, v0.2.0, still contains
schemes that have since been found insecure (see its Known Issues and the
*Unreleased* section of `CHANGELOG.md`); use `main` until the next release.

## Reporting a Vulnerability

Please do not open a public GitHub issue for undisclosed security
vulnerabilities.

Instead, report it privately through GitHub's private vulnerability
reporting: open the repository's **Security** tab and choose **Report a
vulnerability**, or go directly to
<https://github.com/spire-studio/pyfe4ai/security/advisories/new>. Only the
maintainers can see the report.

When reporting, please include:

- a clear description of the issue
- affected files and scheme(s) (`family/variant`, e.g. `mcfe/ring_lwe`) and
  the `security_profile` used
- the pyfe4ai version or commit
- steps to reproduce
- impact assessment
- any proof-of-concept or patch suggestion, if available

You can expect the report to be reviewed and triaged, but response times may
vary because this repository is maintained as a research project rather than a
commercial service.

## Disclosure

Please allow reasonable time for investigation and remediation before public
disclosure.

If the issue is confirmed and fixed, a public note can be added later through a
commit, release note, or repository advisory process.
