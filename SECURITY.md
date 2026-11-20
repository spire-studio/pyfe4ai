# Security Policy

## Status

PyFE4AI is a **research library**. Its schemes are reviewed internally
against their papers (each has a review status), but it has **not undergone
an external security audit** and is **not recommended for production use**;
the [audit package](https://spire-studio.github.io/pyfe4ai/docs/audit.html) is the entry point for one. The default `toy` parameters are not
secure, and nothing is guaranteed to be constant-time (the documentation's
*Side channels* section lists what was checked).

From 1.0 on, the schemes with status *reviewed* form the **stable** tier:
their public API, configuration options and serialized records (keys, key
shares, ciphertexts, partial decryptions, parameter records) stay backward
compatible within 1.x, with a deprecation before any removal in 2.0. The one
exception is a security fix: any release may refuse a scheme, an option or a
kind of record found insecure, with a security advisory and an upgrade note.
Experimental, unreviewed and known-insecure schemes, experimental options and
capabilities (such as the threshold key issuance `distributed_keygen`), the
identity directory `pyfe4ai.ac` and its records, and pickles are not covered
(see `docs/compatibility.rst`).

## Scope

Responsible reporting of security issues is appreciated, especially for:

- cryptographic implementation mistakes, in the Python code or in the Rust
  extension (`rust/`)
- unsafe defaults
- key material leakage
- insecure serialization or parameter handling
- dependency-related vulnerabilities

## Security Model

The [threat model](https://spire-studio.github.io/pyfe4ai/docs/threat_model.html)
gathers, family by family, who is trusted, what each scheme reveals by design
and what is out of scope; the points below are its summary.

- **What is protected.** The schemes protect inputs against the parties named
  in their papers. Every registered scheme carries a review status
  (`pyfe4ai.get_scheme_status(family, variant)`, README *Scheme Status*).
  Six are *experimental*, our own constructions with no published proof:
  the threshold layers of `mife/lwe_threshold`, `mcfe/lwe_threshold`,
  `mcfe/ring_lwe_threshold` and `mcfe/fh_multi_ipe_pairing_threshold`, the
  carry mask of `mcfe/ddh_decentralized`, and the bounded-key scheme
  `bcfe/gc_ristretto255`. The DDH schemes over ristretto255 are not yet
  reviewed. The attribute-based `sife/cp_ab_pairing` and
  `sife/kp_ab_pairing` were checked against their papers (*reviewed*, which is
  not a security audit). Two schemes are *known-insecure*:
  `mcfe/kp_ab_pairing` (NPP22 Sec. 5.2, checked against the paper; the
  construction fails with one corrupted client) and
  `mcfe/kp_ab_pairing_per_client` (the version in which each client has an
  attribute set of its own, Sec. 5.4). Their key generators and crypto objects
  emit a `FESecurityWarning`, and `security_profile="standard"` refuses them, as
  it refuses every known-insecure scheme (see Known Issues).
- **Network and parties.** The library adds no defence against active network
  attackers and assumes authenticated channels between parties; the threshold
  key issuance also needs them to be confidential (below).
- **Experimental features of 0.6 and 0.7.** Their security arguments are the
  project's own, reviewed and re-checked within the project, neither published
  nor externally reviewed, and they hold only under these conditions:
  - `bcfe/gc_ristretto255` assumes an **honest encryptor**: a malicious one can
    garble another circuit or cause a selective failure and learn bits of a key
    holder's `w`, so do not send results, or whether a decryption worked, back
    to encryptors. `max_keys` bounds the keys issued in all, not a collusion
    (any number of issued keys may collude); key holders learn their own `w`;
    the labels are classical (harvest now, decrypt later).
  - The **threshold key issuance** of `sife/cp_ab_pairing`
    (`distributed_keygen`) needs mutually **authenticated, confidential
    channels** (mTLS or an equivalent) between the requester and each key
    server: whoever reads the `t` partial keys of a request, or replays it,
    gets the key. It assumes static corruption of fewer than `t` servers, and
    the approval rule must decide on the weights `y`, not only the attributes.
  - The **identity directory** (`pyfe4ai.ac`) is **key escrow**: the key centre
    holds every identity's master key and authenticates nobody. Ciphertexts
    name their identity in the clear, and a directory is checked only against
    an `expected_digest` the encryptor obtained over a trusted channel.
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
  secure and usable, and every known-insecure scheme. Parameter records cached
  in the parameter cache directory (the per-user cache directory, or
  `PYFE4AI_CACHE_DIR`) are checked or re-derived on load, but a DCR modulus
  with known factors cannot be detected under `toy`: use a cache directory
  nobody else can write to.
- **Sampling and randomness.** The lattice MIFE / MCFE schemes sample exact
  discrete Gaussians (FACCT) by default; ALS16, the DCR schemes and ABDP15
  under `toy` follow the library-wide sampler, which rounds a continuous
  Gaussian unless `"facct"` is selected. The Python code reads randomness from
  the operating system; the Rust extension expands one 32-byte OS seed per
  call with ChaCha20 (not an SP 800-90A DRBG, which matters only for FIPS
  validation).
- **Side channels.** Nothing is hardened against timing, cache or power
  analysis: CPython and big-integer arithmetic are not constant-time, neither
  is the pure-Python ristretto255 fallback, and the BLS12-381 group of the
  native extension (arkworks) multiplies by scalars in variable time, also by
  secret scalars. In the native lattice core only the paths listed in the
  documentation's *Side channels* section were checked, which is not a
  guarantee. Among them, the exact Gaussian sampler (FACCT) does not branch
  on the values it returns (since 1.0; the samples are unchanged): it
  branches on the values it draws only at its decisions (a redraw of `y`, a
  rejected candidate, `-0`, and about once in 2^99 tests the deferral to the
  exact threshold), so its time follows the number of candidates and the
  order of its decisions, which do not depend on the values it returns, and
  replaying one sample's decisions with other values did not change it
  measurably (the code of 0.4.0 to 0.5.0 differed by about 4 %). In the
  native garbling of `bcfe/gc_ristretto255`, the shape of the time
  distribution depends on the secret seed, while the means stay within the
  control; the mechanism was not found, and whether this can be exploited
  was not evaluated (see
  [Side channels](https://spire-studio.github.io/pyfe4ai/docs/security.html#side-channels)).
  The library does not set the processor's data-independent-timing mode
  (DIT).
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
- **`mcfe/kp_ab_pairing` is known-insecure: a corrupted client opens every
  functional key.** In NPP22 Sec. 5.2 Fig. 3 the encryption key of client `i`
  holds the first two rows of its basis `H_i` (times `p_i`) as vectors of
  `Z_q`. Applied to the `G2` coordinates of `k_{i,ipfe}` of any functional key,
  without a pairing and without the policy, they give `[p_i <S, y>]_2` and
  `[p_i <U, y>]_2`. With a key of weight on that client only, which gives
  `[p_i]_2`, the client then computes `<x, y>` of any other key from the
  ciphertexts of a round, also when the round's attributes do not satisfy the
  key's policy: an honest client's input for `y = e_h`. Both keys are
  admissible under Def. 9, so the security notion of Def. 12 fails with one
  corrupted client; the proof of Thm. 14 (App. B.4, step G4 to G5) misses this
  opening. The policy therefore does not hold against a decryptor that
  colludes with a client, and the ML adapter's policy-gated aggregation, which
  runs on this scheme, is for experiments only. `mcfe/kp_ab_pairing_per_client`
  has the same rows and is affected as well; `sife/kp_ab_pairing` (Sec. 4) is
  not. The test `test_a_corrupted_client_opens_every_key_and_reads_an_honest_input`
  reproduces it. A key-side mask of the first two coordinates stops this
  computation, but no proof of a masked construction is known, and the library
  does not implement one.
- **`mcfe/kp_ab_pairing_per_client` is known-insecure.** In the construction of
  NPP22 Sec. 5.4 (Nguyen, Phan and Pointcheval, ePrint 2022/215, revision of
  2022-09-18) each client encrypts for an attribute set of its own, and the
  paper adds a fifth coordinate to the vectors `c_{i,ipfe}` and `k_{i,ipfe}`
  against a mix-and-match attack on keys of different policies. The product of
  the two fifth coordinates, `theta_i d_{K,i}`, depends on the client and the
  key but not on the tag. So two keys of one `y` that open neither of two rounds
  (two tags: two labels, or two elements of one array), where the attribute set
  of each client satisfies one of the keys in both rounds, give the difference
  of the two rounds' inner products, `<x - x', y>`. The adversary encrypts
  under the challenge tag and under a second tag, which the paper's game
  allows, and is admissible under its Def. 9 since neither key opens either
  round, so Thm. 25 (App. B.5) as stated fails. The attack reads the difference
  of two inner products for each pair of keys of one `y`: with pairs for several
  `y` (the unit vectors) it gives `x - x'` coordinate by coordinate, and the
  `x_i` themselves when one round is known. `mcfe/kp_ab_pairing` (Sec. 5.2,
  one attribute set per tag) is not affected by this attack (it has the one
  above). The scheme is kept for
  functionality and performance experiments and as the record of the attack;
  the test `test_two_tags_and_two_keys_read_the_difference_of_two_rounds`
  reproduces it, and the security page of the documentation states it in full.
  The fixed-tag MIFE variant of NPP22 (Remark 16) is not provided: it has one
  ciphertext per slot, so this attack does not reach it, but its proof rests on
  the same Thm. 25.
- **Experimental threshold layers** (tMIFE-LWE, tMCFE-LWE, tMCFE-Ring-LWE,
  tMCFE-FH-Multi-IPE), the dMCFE-DDH carry mask and `bcfe/gc_ristretto255`
  have no published security proof; see their module docstrings, and the
  Security Model above for the conditions of the bounded-key scheme, the
  threshold key issuance and the identity directory.
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
  threshold layers are not affected. Fixed in v0.4.0 by a replicated key
  sharing, the default for `t < s`; Shamir sharing with `t < s` is refused
  under `standard`. **Re-key:** with `t < s` every partial decryption under
  v0.3.0 key shares still leaks, so finish with v0.3.0 only the rounds that
  must finish under the old keys, then issue new client keys and key shares
  with the fixed version (it refuses v0.3.0 key shares). The upgrade guide of the
  documentation and `CHANGELOG.md` give the details. No separate security advisory was
  published.
- **Also fixed in v0.4.0** (see the Security section of `CHANGELOG.md`):
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
(currently v1.0.0) is supported. v0.3.0 and earlier are affected by the
Lagrange-multiplier leak of the threshold LWE layers described above, and
v0.2.0 and earlier contain schemes that were found insecure; upgrade to
v1.0.0 (the fixes are in v0.4.0 and later) and re-key as the upgrade notes in
`CHANGELOG.md` describe.

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
