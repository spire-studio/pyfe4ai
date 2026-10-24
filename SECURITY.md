# Security Policy

## Status

PyFE4AI is an **experimental research prototype under active development**.
It has not undergone a security audit and is **not recommended for production
use**. Interfaces, defaults and serialized formats (keys, threshold key
shares, ciphertexts, partial decryptions, parameter records, pickles) can
change substantially between versions; regenerate keys and key shares when
upgrading, as the upgrade notes in `CHANGELOG.md` describe.

## Scope

Responsible reporting of security issues is appreciated, especially for:

- cryptographic implementation mistakes, in the Python code or in the Rust
  extension (`rust/`)
- unsafe defaults
- key material leakage
- insecure serialization or parameter handling
- dependency-related vulnerabilities

## Security Model

- **What is protected.** The schemes protect inputs against the parties named
  in their papers. Every registered scheme carries a review status
  (`pyfe4ai.get_scheme_status(family, variant)`, README *Scheme Status*):
  four threshold layers and the dMCFE-DDH carry mask are our own constructions
  (*experimental*, no published proof), and the DDH schemes over ristretto255
  are not yet reviewed.
- **Network and parties.** The library adds no defence against active network
  attackers and assumes authenticated channels between parties.
- **Threshold schemes.** Decryption servers are honest-but-curious: partial
  decryptions carry no proof of correctness (redundant enrolled sets can
  detect a deviating server). In the LWE threshold layers
  (`mife/lwe_threshold`, `mcfe/ring_lwe_threshold`, `mcfe/lwe_threshold`) the
  key is shared with a replicated sharing for `t < s` (Shamir sharing leaks,
  see below) and the partial decryptions are flooded with derandomised noise;
  key shares are bound to their public parameters. Key shares and partial
  decryptions of `mcfe/ring_lwe_threshold` carry a `sharing_id` that must be
  stored and transmitted with them: combiners refuse partials of two sharings
  of one key, but partials without an id combine only with each other, so a
  layer that drops the id on all sides would let two sharings be mixed and
  decode to a wrong sum.
- **Parameters.** Default parameters are toy-sized and not secure. Use
  `security_profile="standard"`, which aims at 128-bit security except for
  the pairing-based schemes (BLS12-381, see below) and refuses the ALS16
  schemes (`sife/fullysec_lwe`, `mife/fullysec_lwe`), `mife/lwe_threshold`
  and `mcfe/paillier_decentralized`, which have no parameter set that is both
  secure and usable. Parameter records cached in the parameter cache
  directory (the per-user cache directory, or `PYFE4AI_CACHE_DIR`) are
  checked or re-derived on load, but a DCR modulus with known factors cannot
  be detected under `toy`: use a cache directory nobody else can write to.
- **Sampling and randomness.** The lattice MIFE / MCFE schemes sample exact
  discrete Gaussians (FACCT) by default; ALS16, the DCR schemes and ABDP15
  under `toy` follow the library-wide sampler, which rounds a continuous
  Gaussian unless `"facct"` is selected. The Python code reads randomness from
  the operating system; the Rust extension expands one 32-byte OS seed per
  call with ChaCha20 (not an SP 800-90A DRBG, which matters only for FIPS
  validation).
- **Side channels.** Nothing is hardened against timing, cache or power
  analysis: CPython and big-integer arithmetic are not constant-time, and
  neither is the pure-Python ristretto255 fallback.
- **ML adapter.** `FESchemeWrapper` runs the key generator and holds the
  master key (for `mcfe/ring_lwe_threshold` it also runs the decryption
  servers): one trust domain, meant for experiments, not for deployments.

See "Known Issues / Security Notice" in the README for the full list of
limitations.

## Known Issues

- **BLS12-381 is below the 128-bit target.** The pairing-based schemes default
  to BLS12-381, which gives about 120-126 bits of security after the exTNFS
  attacks on its 381-bit field (Barbulescu and Duquesne, J. Cryptol. 2019).
  Under `standard` their key generators emit a `FESecurityWarning` saying so.
  No strict 128-bit pairing curve is offered.
- **Experimental threshold layers** (tMIFE-LWE, tMCFE-LWE, tMCFE-Ring-LWE,
  tMCFE-FH-Multi-IPE) and the dMCFE-DDH carry mask have no published security
  proof; see their module docstrings.
- The remaining limitations (no labels in multi-input FE, one ciphertext per
  client and label in MCFE, the parameter cache, the ML adapter's trust
  domain) are listed in the README.

## Fixed Issues in Released Versions

- **v0.3.0 and earlier: Lagrange-multiplier leak in the threshold LWE layers**
  (`mife/lwe_threshold`, `mcfe/ring_lwe_threshold`, `mcfe/lwe_threshold`
  with `t < s`; severity High, practical impact Medium). These versions
  Shamir-shared the functional key for every `t < s`. For `s >= 3` and
  `t < s`, some products of Lagrange coefficients are non-integers `r / d`,
  and an honest partial decryption then reveals the decryption noise modulo
  `d`, whatever the flooding width -- to `t - 1` corrupted servers, for some
  `(s, t)` to fewer, and for some (e.g. `t = 1`, `(5, 2)`, `(7, 2)`) to the
  aggregator alone. At the `standard` parameters the leak is partial (at most
  about 25 bits of the noise per ciphertext for `s <= 10`), and no key or
  plaintext recovery has been shown. `t = s` and the DDH and pairing
  threshold layers are not affected. Fixed after v0.3.0 by a replicated key
  sharing, the default for `t < s`; Shamir sharing with `t < s` is refused
  under `standard`. **Re-key:** with `t < s` every partial decryption under
  v0.3.0 key shares still leaks, so finish with v0.3.0 only the rounds that
  must finish under the old keys, then issue new client keys and key shares
  with the fixed version (it refuses v0.3.0 key shares). The README's *Known Issues* and the
  `CHANGELOG.md` give the details. No separate security advisory was
  published.
- **Also fixed after v0.3.0** (see the Security section of `CHANGELOG.md`):
  two concurrent `encrypt_gradient` calls of the ML adapter for one client
  and label both succeeded, which revealed the difference of two gradients;
  the double-precision rounded Gaussian sampler was far from the discrete
  Gaussians the lattice proofs assume, and the MPFR Gaussians of the DCR
  schemes and of the threshold flooding were 2 to 19 bits short of their
  budgets; the Ring-LWE MIFE / MCFE widths were sized for messages of norm
  `X` instead of `3X`; the ABDP15 LWE dimensions were read at the public
  key's error width instead of the reduction's; threshold key shares were not
  bound to their public parameters. No attack is known for the parameter
  issues.
- **v0.2.0 and earlier** contain schemes that were found insecure; they were
  fixed in v0.3.0 (see its section of `CHANGELOG.md`).

## Supported Versions

Security fixes go into `main` and the next release; only the latest release
(currently v0.3.0) is supported. v0.2.0 and earlier contain schemes that have
since been found insecure (see the 0.3.0 section of `CHANGELOG.md`); upgrade
to v0.3.0 or later.

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
- the pyfe4ai version or commit, and the output of
  `pyfe4ai.backend_info()` (whether the Rust extension was built and used,
  threads, Gaussian sampler, parameter cache directory)
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
