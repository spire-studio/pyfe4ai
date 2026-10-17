# Performance of the FL aggregation path, per scheme family

Measurements of branch `perf/native-core` (2026-10-17) against v0.3.0 and
against homomorphic encryption (TenSEAL 0.3.18, i.e. Microsoft SEAL) on the
same workload. Raw numbers: `results/perf_native_core_2026-10-17.json`
(first session: a C++ prototype of the native backend, host and docker) and
`results/perf_rust_linux_2026-10-17.json` (the Rust backend in docker, DRBG
and BLS12-381 measurements); the threshold Ring-LWE MCFE in 256-bit words
(branch `perf/native-tmcfe-wide-q`): `results/perf_tmcfe_wide_q_2026-10-17.json`.
The C++ prototype was removed once the Rust backend matched it (same kernels,
same outputs, the same tests); its numbers stay here as measured.

## Native coverage (status)

| family | native (Rust) | still Python |
| --- | --- | --- |
| MCFE Ring-LWE (`ring_lwe`, `ring_lwe_decentralized`) | ndarray encryption, offline / online phases, key derivation, aggregation | single-vector label pad; setup; dMCFE key shares and their combination |
| MCFE Ring-LWE threshold | any `q` below 2^253 (the flooded `q` of `standard` has 212 bits: 256-bit words): ndarray encryption, key derivation, Shamir shares of the key, partial decryptions (parsing, digests, NTT-domain products), combination | the flooding noise (`discrete_gaussian_bigint`, mpfr Box-Muller: 74% of a partial decryption), ciphertext tags (JSON + HMAC), setup; see below |
| MIFE Ring-LWE | encryption, key derivation, decryption (no ndarray helpers) | setup |
| SIFE Ring-LWE, SIFE FullySec Ring-LWE | -- | everything (NTT in Python) |
| MCFE dense LWE (`lwe`, `fullysec_lwe`, `lwe_decentralized`) | label hashing and dot products of ndarray encryption / decryption | rounding, decoding, key derivation, single vectors; dMCFE DCR key shares |
| MCFE LWE threshold | ndarray encryption (as `lwe`) | partial decryptions (label hashed again in Python per element), flooding, combination, Shamir shares |
| ABDP15 SIFE / MIFE LWE, MIFE LWE threshold, ALS16 | -- (ABDP15 encryption: numpy matrix products) | decryption, keys, flooding, Shamir shares |
| DDH over ristretto255 | MCFE ndarray encryption and decryption with dlog in one call each; group operations | MIFE / SIFE call per group operation |
| DDH over Z_p^*, Damgård, Paillier | -- | everything (gmpy2 `powmod`, Python dlog tables); see the design note below |
| Pairing schemes, quadratic FE | BLS12-381 group operations, `multi_pair`, products of powers and powers of one base (`multiexp`, `pow_many`), batch deserialisation, G_T dlog, hash to G1 / G2 | matrix algebra mod p, Shamir shares of tMCFE keys; the G_T dlog table is built once per base and bound |

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
FE / HE comparison uses the docker numbers only. The Rust backend was built
in a throwaway container of the same, unchanged `pyfe4ai-pairing` image with
a rustup minimal toolchain (rustc 1.99.1, aarch64-unknown-linux-gnu) and the
crates of the host's cache, offline. Docker timings drift by about 10% over
a day; numbers compared with each other come from one session.

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

*native* in these tables is the C++ prototype (first session); the Rust
backend in docker, with the prototype and TenSEAL in one later session, is
under *Rust backend* below (Rust binary: client 3.51 ms, aggregator 9.49 ms).

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
`pyfe4ai._rust_native` (PyO3, `rust/native`) binds it; it replaced the C++
prototype after passing the same bit-identity tests and the comparison
below (`pip install` builds it with cargo when a Rust toolchain is present).

Docker, 10^4 elements, commit `eb9cc29`, one session (medians of 3 runs of
9 rounds; TenSEAL measured right after):

| docker, 77-bit q, same session | client ms | aggregator ms | of which dkgen |
| --- | ---: | ---: | ---: |
| Rust, JSON ciphertexts | 4.37 | 12.07 | 5.19 |
| Rust, binary | 3.51 | 9.49 | 4.89 |
| Rust, binary, 4 threads | 1.54 | 4.83 | 2.14 |
| Rust, binary, online after `--offline` | 0.25 | 4.51 | - |
| Rust, binary, online, 4 threads | 0.26 | 2.34 | - |
| C++ prototype, JSON / binary | 5.35 / 3.95 | 12.65 / 10.30 | 5.98 / 5.87 |
| C++ prototype, binary, online | 0.28 | 4.32 | - |
| TenSEAL CKKS, N = 8192 | 8.01 | 3.01 | - |
| TenSEAL BFV, N = 4096 | 3.30 | 1.31 | - |

Offline phase: Rust 3.50 ms per client and 5.27 ms for the aggregator
(4 threads: 1.31 / 2.19 ms), C++ 3.86 / 6.07 ms. 10^5 elements, Rust,
binary: client 30.4 / 8.5 / 5.2 ms with 1 / 4 / 8 threads, aggregator 60.7
(dkgen 31.4) / 23.5 / 17.3 ms, online after `--offline` with 8 threads 0.91 /
7.6 ms; TenSEAL CKKS 66.8 / 23.2 ms, BFV 26.2 / 11.4 ms.

Rust is faster than C++ everywhere except the online decryption of the
aggregator (+4% in docker, +9% on the host: 3.46 vs 3.18 ms), which spends
its time in moving data between Python and the kernels (the kernels
themselves are on par: forward NTT 0.095 ms, one decryption chunk of 3
clients 0.58 ms). `eb9cc29` removed most of that gap (it was +20%): integers
are read without Python-level `operator.index` / `%` and without the
`OverflowError` that every negative key coefficient used to raise and drop,
and binary ciphertexts are decoded per chunk without the GIL instead of
being copied.

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

Client, 10^4 elements, Rust, one thread; `8f09fc9` (OS bytes), `2b47eea`
(the DRBG alone) and `a722be3` (the DRBG and faster string output),
interleaved, medians of 3 runs of 9 rounds:

| | macOS 8f09fc9 | macOS 2b47eea | macOS a722be3 | docker 8f09fc9 | docker 2b47eea | docker a722be3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JSON | 4.94 ms | 4.44 ms | 4.19 ms | 4.22 ms | 4.26 ms | 3.92 ms |
| binary | 3.76 ms | 3.34 ms | 3.30 ms | 3.28 ms | 3.37 ms | 3.34 ms |
| offline phase (precompute) | 3.78 ms | 3.16 ms | 3.15 ms | 3.22 ms | 3.20 ms | 3.20 ms |
| online phase, binary | 0.28 ms | 0.30 ms | 0.28 ms | 0.21 ms | 0.21 ms | 0.21 ms |

The DRBG saves 0.4 to 0.6 ms per 10^4 elements on macOS and nothing on
Linux, where `getrandom` is already cheap (10^5 elements, docker: binary
27.4 vs 28.1 ms, offline phase 26.4 vs 26.2 ms, within noise). The JSON
difference between `2b47eea` and `a722be3` is the string output, not the
DRBG.

### RNS (not adopted; needs a decision)

With `q` a product of two NTT-friendly primes of about 39 bits, every NTT
becomes two u64 NTTs (2 x 0.027 ms instead of 0.12 ms). Estimated from the
kernel timings: client -17% (binary 3.4 -> about 2.9 ms), aggregator online
-25% (3.6 -> about 2.8 ms). It changes the parameters (composite `q`;
BMMS21's analysis and the parameter derivation assume a prime) and the
binary format, so it needs a security review first; the 60-bit-q rows show
what u64 arithmetic gives.

## Threshold MCFE over Ring-LWE (`mcfe/ring_lwe_threshold`), 10^4 elements

The flooded modulus (statistical parameter 64 on top of BMMS21's `q`) has
212 bits at `ring_n` 8192 under `standard`, beyond the 125-bit u128 words, so
the scheme ran entirely in Python. `rust/core` now has a 256-bit word (four
u64 limbs): Montgomery products with R = 2^256, Shoup products for the
twiddles (`q < 2^253` keeps 4q below 2^256 for the lazy butterflies);
`AnyRing` picks u64 / u128 / 256-bit words by the size of `q`. With it the
MCFE layer (encryption, key terms) runs natively, and three kernels take the
threshold layer: `shamir_share` (the key's coefficients, drawn as
`SystemRandom.randrange(0, q)` draws them, from the per-call DRBG),
`share_inputs` / `share_partial` (a partial decryption's parsing, the JSON
text that `ciphertexts_digest` hashes, and `w = lambda (Delta z + sum_i ct1_i
sk_i) + e`, `c_y` in the NTT domain) and `share_combine`. The flooding noise
is still drawn by `discrete_gaussian_bigint` (mpfr at `bits(sigma_f) + 128`
= 324 bits; a native sampler would not reproduce its rounding), now with one
call per polynomial instead of one per coefficient (the same draws, 24%
faster); tags are serialised once for all servers.

Host (Apple M2 Max, Python 3.13), 3 clients, `s = 3`, `t = 2`, weight 1,
`bound_x = 100`, precision 2; `bench_fl_compare.py --families ring_lwe
ring_lwe_threshold`, before (`50e3188`) and after interleaved, medians of 2
runs (3 / 5 rounds each); "aggregator" is key shares + both partial
decryptions + combination on one machine:

| `ring_lwe_threshold`, 212-bit q, ring_n 8192 | client ms | key shares (2 servers) | partial, per server | combine | aggregator ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| before (`q` beyond the native words: Python) | 213.5 | 275.9 | 528.0 | 18.0 | 1347.9 |
| after, Python backend | 204.6 | 276.0 | 478.7 | 17.9 | 1251.8 |
| after, native | **19.6** | **41.3** | **211.2** | **4.3** | **468.3** |
| after, native, 4 threads | 13.7 | 32.3 | 211.8 | 4.3 | 461.2 |
| `ring_lwe` (77-bit q), before / after | 3.66 / 3.46 | -- | -- | -- | 9.19 / 8.90 |

One 8192-coefficient chunk of a partial decryption takes 103 ms: the
flooding noise 76 ms (Python), the tag checks 7.5 ms (Python),
`share_partial` 8.1 ms (three forward NTTs, one inverse, reading the noise)
and `share_inputs` 6.8 ms. Kernels at 212 bits (`cargo run --release
--example kernels`): NTT 0.97 / 0.83 ms forward / inverse (77-bit q, 4096:
0.11 / 0.09 ms), encryption of 10^4 elements 10.9 ms (two chunks), one
decryption chunk of 3 clients 5.3 ms. The client's other 9 ms are the
ciphertexts' Python strings and the tags (one JSON serialisation per chunk,
one HMAC per server and chunk). A faster partial
decryption needs a different flooding sampler (an exact discrete Gaussian
at this width, natively) or a smaller `sigma_f`; both change the
distribution and need a review.

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
* **Elliptic-curve DDH.** Done: `*/ddh_ristretto255`, see *DDH over
  ristretto255* above (still one group element per value).
* **Pairing backend.** Done: BLS12-381 in Rust, see the next section.

## Pairing families on BLS12-381 (Rust)

`PairingGroup("BLS12-381")` comes from the Rust extension (arkworks 0.6,
`pyfe4ai.utils.bls12_381`) with charm's element interface, and is the
default of every pairing scheme under both profiles; other curve names still
come from charm. Native paths: products of pairings with one final
exponentiation (`pairing_backend.multi_pair`, used by every decryption), the
bounded discrete log in G_T with the baby-step table kept per (base, bound),
RFC 9380 hash to G1 / G2 (BLS12381G1/G2_XMD:SHA-256_SSWU_RO_; SHA-256 in
`rust/core`, checked against the RFC vectors). SGP decryption pairs `a_i`
with `prod_j b_j^{F_ij}` (2 n + 1 pairings instead of 2 n^2 pairings and n^2
G_T exponentiations), for every backend. Elements cross the boundary as
Python objects; serialisation is the compressed encoding (48 / 96 / 576
bytes for G1 / G2 / G_T, base64 behind charm's `"<type>:"` prefix), checked
for subgroup membership on deserialisation. Not constant-time.

Per operation (charm in the Docker image, Linux VM on the same Mac; Rust on
the host; all three in docker in the table after the next):

| | BLS12-381 (Rust) | MNT224 (charm) | BN254 (charm) |
| --- | ---: | ---: | ---: |
| pairing | 0.91 ms | 2.36 ms | 13.7 ms |
| product of 8 pairings | 2.61 ms | 19.4 ms | 110.8 ms |
| G1 / G2 / G_T exponentiation | 0.08 / 0.39 / 0.48 ms | 0.34 / 3.21 / 0.87 ms | 0.44 / 0.87 / 3.16 ms |
| G1 / G2 deserialisation | 0.06 / 0.11 ms | 0.04 / 7.8 ms | 0.03 / 0.06 ms |
| G_T dlog, bound 10^4 (table cached) | 0.05 ms | 1.33 ms | 4.81 ms |

End to end, one setup + key + encryption + decryption round
(`benchmark_quadratic_family.py`, `benchmark_fh_multi_ipe_family.py`,
`--rounds 2`):

| scheme | profile | v0.3 code, MNT224 (charm) | this branch, BLS12-381 (Rust) | |
| --- | --- | ---: | ---: | ---: |
| Quadratic SGP | small / report | 179 / 320 ms | 14 / 36 ms | 13x / 9x |
| Quadratic Quad | small / report | 302 / 506 ms | 32 / 67 ms | 9.5x / 7.6x |
| MIFE FH-Multi-IPE | small / report | 406 / 425 ms | 24 / 25 ms | 17x |
| MCFE FH-Multi-IPE | small / report | 889 / 889 ms | 79 / 82 ms | 11x |
| dMCFE FH-Multi-IPE | small / report | 909 / 945 ms | 80 / 79 ms | 12x |
| tMCFE FH-Multi-IPE | small / report | 1680 / 1565 ms | 137 / 136 ms | 12x |

All in docker (one container with charm and the Rust extension, `report`
profile, end to end):

| scheme | v0.3 code, MNT224 (charm) | BLS12-381 (Rust) | this branch, MNT224 / BN254 (charm) |
| --- | ---: | ---: | ---: |
| Quadratic SGP | 334 ms | 38 ms | 273 / 201 ms |
| Quadratic Quad | 538 ms | 69 ms | 554 / 537 ms |
| MIFE FH-Multi-IPE | 361 ms | 26 ms | 373 / 437 ms |
| MCFE FH-Multi-IPE | 947 ms | 86 ms | 998 / 1032 ms |
| dMCFE FH-Multi-IPE | 1016 ms | 87 ms | 936 / 1018 ms |
| tMCFE FH-Multi-IPE | 1726 ms | 151 ms | 1725 / 1833 ms |

Per operation in docker: pairing 0.96 ms (MNT224 2.63, BN254 14.3), product
of 8 pairings 3.3 ms (19.9 / 113.6), G2 exponentiation 0.41 ms (3.23 /
0.90). The FL path of `bench_fl_compare.py --families pairing` (MCFE
FH-Multi-IPE, `standard` profile, 5 elements, 3 clients): BN254 client 115
ms, aggregator 3703 ms -> BLS12-381 24 ms and 186 ms; ciphertexts 858 ->
1198 B per element (48-byte compressed G1 points against BN254's 33).

This branch on charm (same scripts, `--pairing-group`): MNT224 162 / 301 /
381 / 923 / 943 / 1657 ms and BN254 133 / 323 / 423 / 974 / 981 / 1756 ms
(small profile, same order), i.e. the scheme changes alone do not move the
charm numbers; the gain is the backend. The time left was in native group
operations (pairing products, checked G2 deserialisation, G2
exponentiations in key generation); the exponentiations are batched since,
see the next section.

### Multi-scalar multiplication (branch `perf/native-bls-msm`)

`pairing_backend.multiexp(bases, exps)` computes `prod_i b_i^{e_i}` in one
call (`bls_msm`), `pow_many(base, exps)` the powers of one base with one
table (`bls_pow_many`), and `deserialize_many(group, encodings)` decodes a
list in one call, equal encodings once (the 368 entries of Quad's `[M]_1`
have 21 distinct encodings); on charm groups they are the loops they
replace. The schemes use them for every vector: FH-IPE / FH-Multi-IPE
encryption and keys (MIFE, MCFE, dMCFE, tMCFE, with the tMCFE key shares as
one batch of powers of `g2`), SIFE FH-IPE and partial FH-IPE, SGP, Quad
(public key decoded once per encryptor object) and MIQFE (G_T products in
decryption). Keys and ciphertexts are unchanged: FH-IPE outputs with seeded
randomness are byte for byte those of the previous code, and the decryption
tests pass on BLS12-381 and on charm MNT224. Methods (`rust/native/src/msm.rs`):

* G1 / G2: Straus' interleaved wNAF (window 5) over the GLV decomposition
  `k = k1 + lambda k2` (`|k_i| ~ 2^128`), so the 128 doublings are shared by
  all terms; arkworks' Pippenger from 512 terms (its MSM does not use the
  endomorphism, and neither did its G2 scalar multiplication, plain
  double-and-add over 255 bits: a single G2 exponentiation is the one-term
  case and 2.3x faster).
* One base, many exponents: a table of `d 2^(w j) P` and its GLV images,
  each power `~2 * 129 / w` additions without doublings: the NAF (`w = 1`)
  below 12 exponents, windows of 4 / 5 / 6 bits from 12 / 32 / 96.
* G_T: `p = u (mod r)`, so the Frobenius map is `f -> f^u` on G_T, and
  `k = sum_{i<4} k_i |u|^i` (`k_i < 2^64`) turns `f^k` into a four-term
  product over 64-bit exponents (64 cyclotomic squarings instead of 255).
* G_T deserialisation tests membership by `f^(p^4) f = f^(p^2)` (cyclotomic
  subgroup) and `f^p = f^u` (Scott, ePrint 2021/1130; `gcd(Phi_12(p), p - u)
  = r` for BLS12-381, checked) instead of arkworks' `f^r = 1`; the tests
  reject elements of the cyclotomic subgroup outside G_T built in Python.
* `e(g1, g2)` of the FH-IPE, Quad and MIQFE contexts is computed when first
  needed (encryptors never need it), and Python ints of up to 64 bits cross
  into Rust without a call back into Python.

Per operation, host (M2 Max), both builds in one session, best of 5:

| | `perf/native-core` | this branch | |
| --- | ---: | ---: | ---: |
| G1 / G2 / G_T exponentiation | 0.079 / 0.367 / 0.492 ms | 0.058 / 0.159 / 0.215 ms | 1.4x / 2.3x / 2.3x |
| product of 4 powers, G1 / G2 / G_T | 0.34 / 1.63 / 2.02 ms | 0.14 / 0.39 / 0.75 ms | 2.4x / 4.1x / 2.7x |
| product of 16 powers, G1 / G2 / G_T | 1.44 / 6.52 / 8.24 ms | 0.47 / 1.33 / 2.79 ms | 3.0x / 4.9x / 3.0x |
| 16 powers of one base, G1 / G2 / G_T | 1.43 / 6.47 / 8.15 ms | 0.64 / 1.95 / 3.47 ms | 2.2x / 3.3x / 2.3x |
| G_T deserialisation | 0.81 ms | 0.078 ms | 10x |
| G1 / G2 deserialisation | 0.056 / 0.110 ms | 0.055 / 0.110 ms | -- |
| pairing / product of 8 pairings | 0.81 / 2.56 ms | 0.81 / 2.56 ms | -- |

End to end (`report` profile, `--rounds 2`; the two builds alternated twice
on the host, the faster run of each):

| scheme | `perf/native-core` | this branch | |
| --- | ---: | ---: | ---: |
| Quadratic SGP | 35.4 ms | 30.0 ms | 1.18x |
| Quadratic Quad | 64.7 ms | 47.8 ms | 1.35x |
| MIFE FH-Multi-IPE | 24.2 ms | 18.1 ms | 1.34x |
| MCFE FH-Multi-IPE | 79.2 ms | 46.6 ms | 1.70x |
| dMCFE FH-Multi-IPE | 79.4 ms | 48.3 ms | 1.64x |
| tMCFE FH-Multi-IPE | 138.4 ms | 70.2 ms | 1.97x |

Setup: Quad 15.8 -> 5.9 ms, the FH-Multi-IPE MCFE family 2.3-3.1 -> 1.2 ms.
Against v0.3 on charm MNT224 (table above): SGP 11x, Quad 11x, MIFE 23x,
MCFE 19x, dMCFE 20x, tMCFE 22x. Raw numbers:
`results/perf_native_bls_msm_2026-10-17.json`. Not repeated in docker: the
`pyfe4ai-pairing` image has no Rust toolchain.

Where the time goes now (one round, profiled): in the FH-Multi-IPE family,
products of pairings 35% (the Miller loops of the 17-pair FH-IPE
decryptions), checked deserialisation 25% (a G2 point costs a square root
in Fq2 and the `[x] P` subgroup test, ~0.11 ms, batched or not), powers of
one base 22% (tMCFE's 153 key-share powers of `g2`, FH-IPE key generation)
and products of powers 7%. In SGP and Quad, the G_T dlog takes 45%: its
baby-step table (16 sqrt(2 bound + 1) entries, ~17 ms at the report bounds)
is built per base and bound and pays off only over many decryptions, while
the benchmark decrypts once per fresh setup.

BLS12-381 gives about 120-126 bits after the exTNFS attacks (Barbulescu and
Duquesne, J. Cryptol. 2019), below the 128-bit target, so the `standard`
profile keeps a `FESecurityWarning`; a strict 128-bit curve (BLS12-446/461,
BN462) has no mature Rust crate. The `toy` default moved from MNT224 to
BLS12-381 too: pairing keys and ciphertexts of v0.3 are not compatible.
