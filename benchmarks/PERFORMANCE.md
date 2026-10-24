# Performance

Headline timings of PyFE4AI on the federated-learning aggregation path, with
the native backend (the Rust extension `pyfe4ai._rust_native`) against the
Python backend and v0.3.0. Measured on an Apple M2 Max (Python 3.13), one
thread unless stated, `security_profile="standard"`, three clients, medians
of interleaved runs; timings vary with the machine and its load. The commands
of `benchmarks/run.py` reproduce them (see `README.md`), for example
`python benchmarks/run.py fl-compare --help`.

## MCFE over Ring-LWE (`mcfe/ring_lwe`), 10^4 gradient elements

The ciphertext format matters for the aggregator: `mcfe/ring_lwe` writes JSON
ciphertexts by default, and `ct_format="binary"` (crypto config, or
`FESchemeWrapper(..., crypto_options={"ct_format": "binary"})`) selects the
compact binary format.

| | ciphertexts | client | aggregator |
| --- | --- | ---: | ---: |
| v0.3.0 (Python) | JSON | 197.5 ms | 280.0 ms |
| native, 1 thread | JSON (default) | 18.2 ms | 7.5 ms |
| native, 1 thread | binary | 17.95 ms | 6.24 ms |
| native, 4 threads | binary | 6.21 ms | 3.62 ms |
| native, online phase after `precompute_gradient` / `prepare_aggregation` | binary | 0.25 ms | 3.16 ms |
| Python backend (`PYFE4AI_BACKEND=python`) | JSON (default) | 345 ms | 157 ms |
| Python backend | binary | 360 ms | 164 ms |

- The offline phase takes 17.8 ms per client, most of it exact discrete
  Gaussian sampling (FACCT, the default under `standard`), and 2.9 ms for
  the aggregator (the label-dependent key terms).
- Ciphertexts take 24.6 B per element with `ct_format="binary"` (68.4 B as
  JSON).
- The opt-in `ring_modulus="rns"` brings the aggregator to 5.69 ms and its
  online phase to 2.65 ms (binary ciphertexts).
- On the Python backend most of the client's time is exact sampling in
  Python.
- The JSON rows of the native and Python backends were measured again for
  the 0.4.0 release (interleaved, three runs of `fl-compare`); the binary
  rows come from the October 2026 integration measurements.

## Other families (native, one thread)

| workload | v0.3.0 or Python path | current |
| --- | ---: | ---: |
| threshold MCFE (`mcfe/ring_lwe_threshold`), 10^4 elements, `s = 3, t = 2` (replicated sharing; defaults: binary ciphertexts, RNS modulus, split key shares): client / key shares per round / partial decryption per server / aggregator | 213.5 / 275.9 / 528 / 1348 ms | 26.7 / 10.7 / 8.5 / 29.1 ms |
| ... `t = 3` (Shamir): client / key shares / partial / aggregator | | 26.7 / 13.6 / 8.0 / 39.4 ms |
| ... with the options of earlier versions (`ct_format="json"`, `ring_modulus="prime"`, `key_shares="full"`), `t = 2` / `t = 3` | | 39.1 / 40.9 / 65.4 / 177.0 ms; 39.2 / 43.3 / 55.3 / 215.7 ms |
| decentralized MCFE (`mcfe/ring_lwe_decentralized`), 10^4 elements, binary ciphertexts: client / key share per client / aggregator | | 17.7 / 3.28 / 5.45 ms |
| dense-LWE MCFE (`mcfe/lwe`), 200 elements: client / aggregator | 1021 / 1030 ms | 93.1 / 101.4 ms |
| DDH MCFE, per element: client / aggregator / ciphertext | 10.75 / 11.10 ms, 936 B (3072-bit group) | 0.069 / 0.139 ms, 75 B (`ddh_ristretto255`) |
| SIFE FullySec Ring-LWE, `eta = 10, X = Y = 50`: setup / encrypt / decrypt | | 1.02 s / 0.31 s / 131 ms |
| ... packed layout, `eta = 1`, 4096 elements per ciphertext: setup / encrypt / decrypt | | 75 / 79 / 46 ms, 182 B per element |
| pairing schemes, one round of setup, keys, encryption and decryption: SGP / Quad / MCFE FH-Multi-IPE | 320 / 506 / 889 ms (charm MNT224) | 30 / 48 / 47 ms (BLS12-381) |

- The threshold MCFE's Python-path column was measured at the same 212-bit
  `q` before the native 256-bit words. With its defaults the ciphertexts take
  71 B per element (222 as JSON) and a server's key share per round 0.89 /
  0.44 MB at `t = 2` / `t = 3` (6.75 / 3.38 MB as whole key shares), plus the
  weight part once (15 / 18 ms for the `t` servers).
- The v0.3.0 pairing timings come from charm in the Docker image on the same
  Mac.

## Costs of the security defaults

- **Exact sampling.** FACCT replaces the rounded double-precision sampler for
  the Ring-LWE MIFE / MCFE family, the dense-LWE MCFE keys, SIFE FullySec and
  the ABDP15 public keys under `standard`. It is most of the MCFE client's
  18 ms, which the offline phase moves out of the round.
- **ABDP15 at the reduction width `sigma'`.** Key generation of `sife/lwe` and
  `mife/lwe` under `standard` takes about twice as long as before:
  - `sife/lwe`, `eta = 10`, `X = Y = 50`: `lwe_n` 1280, setup 8.3 s;
  - `sife/lwe`, `X = Y = 256`: `lwe_n` 1536, setup 17.2 s;
  - `mife/lwe` with 100 slots: `lwe_n` 1408, setup 47 s.

What runs natively, and what still runs in Python, is listed in the
docstring of `pyfe4ai.backend`.
