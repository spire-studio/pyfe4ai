# Performance of the FL aggregation path, per scheme family

Measurements of branch `perf/native-core` (2026-10-17) against v0.3.0 and
against homomorphic encryption (TenSEAL 0.3.18, i.e. Microsoft SEAL) on the
same workload. Raw numbers: `results/perf_native_core_2026-10-17.json`.

## Workload and method

`bench_fl_compare.py`: every one of 3 clients encrypts a gradient of
10^4 values in `[-1, 1]` with two decimal digits (`encrypt_gradient`); the
aggregator derives the functional key for the round label and decrypts the
sum with weight 1 (`aggregate_gradients`). `security_profile="standard"`,
`bound_x = 100`, `bound_y = 1`, as in the report's aggregation example. For
MCFE over Ring-LWE this gives `ring_n = 4096` and a 77-bit `q` (u128 words
in the native code); `bound_x = 2`, precision 0 gives a 60-bit `q` (u64
words). Client time is one client's encryption, aggregator time includes the
key derivation (`dkgen`). Medians of 9 rounds after one warm-up round; one
thread unless stated. `--offline` precomputes the client masks and the
aggregator's key before each round and reports the online phase.

Machines: *host* = Apple M2 Max, macOS, Python 3.13, clang; *docker* = the
Linux aarch64 VM of Docker Desktop on the same machine (12 CPUs), PyFE4AI in
the `pyfe4ai-pairing` image (Python 3.10, GCC 13), TenSEAL in `pyfe4ai-he`
(Python 3.11, `whitepaper/scripts/he_baseline.py`, means of 9 rounds). The
FE / HE comparison uses the docker numbers only.

## MCFE over Ring-LWE (BMMS21), 10^4 elements, 3 clients

| docker, 77-bit q | client ms | aggregator ms | of which dkgen | ct B/element |
| --- | ---: | ---: | ---: | ---: |
| v0.3.0 (Python) | 229.7 | 367.5 | 41.8 | 66.0 |
| this branch, Python backend | 135.9 | 211.1 | 35.8 | 66.0 |
| native, JSON ciphertexts | 4.58 | 10.76 | 5.18 | 66.0 |
| native, binary ciphertexts | 3.41 | 9.21 | 5.33 | 24.6 |
| native, binary, 4 threads | 1.38 | 4.46 | 2.30 | 24.6 |
| native, binary, online after `--offline` | 0.23 | 3.60 | - | 24.6 |
| native, binary, online, 4 threads | 0.21 | 2.02 | - | 24.6 |
| TenSEAL CKKS, N = 8192 | 7.54 | 2.59 | - | 99.4 |
| TenSEAL BFV, N = 4096 | 2.91 | 1.22 | - | 26.6 |

Offline work per round (docker, one thread): 3.3 ms per client, 5.2 ms for
the aggregator (key terms of 4 labels for 3 clients).

| host, 77-bit q | client ms | aggregator ms |
| --- | ---: | ---: |
| v0.3.0 | 197.5 | 280.0 |
| Python backend | 120.7 | 159.0 |
| native JSON / binary | 4.01 / 3.24 | 8.83 / 7.15 |
| native binary, 4 threads | 1.29 | 3.50 |
| native binary, online / offline | 0.21 / 3.13 | 2.79 / 4.25 |

60-bit q (docker): v0.3.0 217.1 / 346.0 ms, native binary 2.01 / 5.06 ms
(4 threads 0.86 / 2.74 ms).

10^5 elements (docker, 25 chunks): native binary client 28.4 ms (1 thread),
8.7 ms (4), 5.4 ms (8); aggregator 56.7 ms (30.3 ms dkgen), 21.3 ms, 15.2 ms,
online after `--offline` with 8 threads 0.76 / 6.1 ms. TenSEAL CKKS 64.4 /
22.7 ms, BFV 25.3 / 10.9 ms.

Speedup over v0.3.0 (docker, one thread): client 50x (JSON) to 67x
(binary), aggregator 34x to 40x; with the offline phase the online latency
is 1000x (client) and 100x (aggregator) lower.

### Where the time goes (native, 77-bit q, per 4096-coefficient chunk)

| kernel | host ms | docker ms |
| --- | ---: | ---: |
| forward / inverse NTT (u128 words, Shoup + Harvey) | 0.12 / 0.11 | 0.12 / 0.11 |
| forward NTT, u64 words (60-bit q) | 0.027 | 0.028 |
| label pad: SHAKE-256 of 106 KB + reduction | 0.25 | 0.27 |
| OS random bytes for the samples (86 KB) | 0.32 | 0.14 |
| 4096 decimal strings out (JSON format) | 0.16 | 0.18 |

Client: 3 NTTs, one label pad, 3 x 4096 Gaussians and their random bytes per
chunk. Aggregator: per chunk one forward NTT per client and one inverse;
per round one label pad per client for the label and every chunk.

### What remains slower than HE, and why

* Key derivation (aggregator, 4.2 to 5.3 ms of 7 to 9 ms): `(chunks + 1) x
  clients` label pads `H(u'_i, l)` per round, each a SHAKE-256 expansion of
  `ring_n (bits(q) + 128)` bits. Scheme-inherent (identity-based MCFE binds
  keys to labels); labels are known in advance, so `prepare_aggregation`
  moves it before the round.
* Decryption: one forward NTT per client and chunk (`sum_i ct1_i sk_i` with
  a different key per client) against one decryption of the summed HE
  ciphertext. Scheme-inherent; it grows with the number of clients.
* u128 arithmetic for `q > 2^62`: an NTT costs 4x the u64 one. See *RNS*.
* JSON ciphertexts: 0.36 ms of formatting per chunk on the client and the
  parsing on the aggregator; `ct_format="binary"` removes both and is 2.7x
  smaller (24.6 B per element, BFV 26.6, CKKS 99.4).
* Trust model differs: HE needs a decryptor holding the secret key that does
  not collude with the aggregator; the FE aggregator holds a key for the
  weighted sum only, and the key generator is a trusted party.

### Rust backend

`rust/core` (no dependencies) ports these kernels; its C ABI lets
`tests/test_rust_core.py` check them against the Python code bit for bit.
`pyfe4ai._rust_native` (PyO3, `rust/native`) binds it with the interface
of the C++ module; both pass the same bit-identity tests, and the Rust one
is preferred when both are built (`PYFE4AI_NATIVE_IMPL`). Host, 10^4
elements, one thread, same run: client 2.94 ms (C++ 3.38 ms, binary format;
the difference is the DRBG below), aggregator 7.94 ms (C++ 7.42 ms). Kernel
timings: NTT 0.10 / 0.028 ms (77 / 60-bit q), label pad 0.26 / 0.18 ms.

### Sampling randomness (DRBG)

The Rust backend expands one 32-byte OS seed per tensor call with ChaCha20
(`rand_chacha`), chunk `c` reading stream `c`; seed, generator states and
sample bytes are overwritten when the call ends and nothing persists across
calls (a fork or VM snapshot can only duplicate the call in progress).
Before, every sample byte came from the OS (`getentropy` on macOS: 256 bytes
per call, 0.32 ms per 4096-coefficient chunk; `getrandom` on Linux:
0.14 ms). ChaCha20 is a standard stream cipher, not an SP 800-90A DRBG,
which matters only for FIPS validation; the distribution is unchanged (the
same Box-Muller consumption, checked by tests with an injected seed).

| host (macOS), Rust, one thread | OS bytes | DRBG |
| --- | ---: | ---: |
| client, 10^4 elements, JSON / binary | 4.41 / 3.44 ms | 3.90 / 2.94 ms |
| client offline phase (precompute), 10^4 | 3.37 ms | 2.79 ms |
| client online phase, 10^4, binary | 0.23 ms | 0.23 ms |
| client, 10^5 elements, binary | 28.96 ms | 25.16 ms |
| client offline phase, 10^5 | 28.27 ms | 24.02 ms |

Linux: pending (the Rust toolchain is not installed in the Linux container).

### RNS (not adopted; needs a decision)

With `q` a product of two NTT-friendly primes of about 39 bits, every NTT
becomes two u64 NTTs (2 x 0.027 ms instead of 0.12 ms). Estimated from the
kernel timings: client -17% (binary 3.4 -> about 2.9 ms), aggregator online
-25% (3.6 -> about 2.8 ms). It changes the parameters (composite `q`;
BMMS21's analysis and the parameter derivation assume a prime) and the
binary format, so it needs a security review first; the 60-bit-q rows show
what u64 arithmetic gives.

## Dense LWE: MCFE of ABMPSW20 (`mcfe/lwe`), 200 elements

`n0 + m0 = 7724`, 187-bit `q`. One LWE sample per element: the label of every
element is hashed to `Z_q^7724` (309 KB of SHAKE-256) and multiplied with the
client's key. `lwe_matrix.py` (float64 limbs + BLAS) serves the seeded-matrix
ABDP15 schemes, not this one.

| docker | client ms | aggregator ms |
| --- | ---: | ---: |
| v0.3.0 | 1240 | 1249 |
| Python backend (cached keys, `mpz.from_bytes`) | 604 | 609 |
| native, 1 thread | 148 | 167 |
| native, 4 threads | 40 | 49 |

Host: 1021 / 1030 -> 550 / 557 (Python) -> 140 / 147 (native) -> 38 / 45 ms
(4 threads). Per element 0.7 ms per client natively, of which 0.44 ms is the
SHAKE-256 expansion; 10^4 elements take about 7 s per client and thread.

## DDH over ristretto255 (`*/ddh_ristretto255`, new, unreviewed)

The DDH constructions in ristretto255 (RFC 9496) instead of a 3072-bit
`Z_p^*` subgroup, selected with `group="ristretto255"`. MCFE, 3 clients,
host, per element:

| | client | aggregator | ciphertext |
| --- | ---: | ---: | ---: |
| 3072-bit group (Python on gmpy2) | 10.75 ms | 11.10 ms | 936 B |
| ristretto255, pure Python | 3.47 ms | 4.56 ms | 75 B |
| ristretto255, Rust, 1 thread | 0.069 ms | 0.139 ms | 75 B |
| ristretto255, Rust, 4 threads | 0.019 ms | 0.044 ms | 75 B |

10^4 elements with 8 threads: client 107 ms, aggregator 284 ms -- still one
group element per value (no packing), about 30 times the Ring-LWE MCFE.

## DDH, DCR and pairing families (design note)

No native port: 99.6% (client) and 97.6% (aggregator) of the DDH time is in
`gmpy2.powmod` (GMP), and DCR is likewise GMP-bound; docker measurements of
v0.3.0 and this branch agree within noise (DDH 229 / 223 vs 216 / 233 ms,
Paillier 794 / 768 vs 737 / 765 ms per 20 elements).

`bench_offline_potential.py` (host, per element, standard profile):

| | client: full / offline / online ms | aggregator: full / offline / online ms |
| --- | --- | --- |
| DDH (3072-bit group, CDGPP18) | 11.0 / 10.2 / 0.019 | 10.7 / 10.2 / 0.42 |
| DCR (3072-bit N, ABMPSW20) | 35.1 / 35.1 / 0.004 | 35.1 / 35.0 / 0.034 |

* **Offline precomputation.** The client mask `[u_l]^s` (DDH; two 3072-bit
  exponentiations of 5.1 ms) or `H(l)^s mod N^2` (DCR; a 7684-bit exponent)
  depends only on the label and the key; online encryption is then
  `g^x` times the mask (small `x`) or `(1 + x N)` times it, 500x to 9000x
  less. The aggregator's `[u_l]^d` or `H(l)^{-k}` depends on the label and
  the weights; online it multiplies the ciphertexts and, for DDH, solves the
  bounded dlog (0.42 ms). The outputs are the same (the script checks it);
  the total work is not: at 10^4 elements a DDH client spends 100 s and a DCR
  client 350 s per round offline. Implementable without changing outputs,
  like the Ring-LWE `precompute`.
* **Multi-exponentiation** (`u_0^{s_0} u_1^{s_1}` in one pass): about 1.3x on
  GMP's `mpn` layer, but gmpy2 has no such primitive and a Python-level
  Straus loop over gmpy2 products is slower than two `powmod` calls. Not
  worth it without a native GMP binding.
* **Elliptic-curve DDH** (e.g. ristretto255). Measured with the installed
  libsodium (PyNaCl 1.6.0, edwards25519 group, the arithmetic ristretto255
  uses): variable-base scalar multiplication 0.057 ms (vs 5.1 ms for the
  3072-bit `powmod`, 90x), fixed-base 0.012 ms, addition 0.010 ms, hash to
  the group 0.016 ms: about 0.18 ms per element for encryption, 0.02 ms
  online with masks. A new instantiation with its own security argument and
  dlog table; still one group element per value (32 bytes, vs 936 B per
  element now), so 1.8 s per client for 10^4 elements. Needs a decision.
* **Pairing backend.** Charm (docker): MNT224 pairing 2.1 ms, G2
  exponentiation 2.7 ms; BN254 pairing 12.6 ms. `mcfe/fh_multi_ipe_pairing`
  takes 16 to 22 ms per element per client and 550 to 730 ms per element
  for the aggregator (toy MNT224 / standard BN254). A Rust backend on
  BLS12-381 (arkworks, blst, or RustPFE, Apache-2.0, which already has PyO3
  bindings) would also lift the curve to about 128-bit security; published
  BLS12-381 pairings take well under 1 ms on this class of CPU (not
  measured here, no Rust toolchain installed). A new curve and backend:
  needs a decision.

### Pairing backend (planned)

A Rust BLS12-381 backend (arkworks `ark-bls12-381`, or `blst`) behind the
interface the pairing schemes use today -- `PairingGroup(name)`, `random`,
`init`, `pair`, `serialize` / `deserialize`, `order`, elements with `*`, `**`
and `==` -- so that `PairingGroup("BLS12-381")` comes from Rust and the Charm
curves stay available as fallback until the results agree. Native batch
paths where the schemes spend their time: products of pairings (one final
exponentiation) and the bounded discrete log in G_T with a baby-step table
built once per (generator, bound) and shared. BLS12-381 gives about 120-126
bits after the exTNFS attacks, below the 128-bit target, so the `standard`
profile would keep a `FESecurityWarning`; a strict 128-bit curve
(BLS12-446/461, BN462) has no mature Rust crate. Which one the user wants
is open.

