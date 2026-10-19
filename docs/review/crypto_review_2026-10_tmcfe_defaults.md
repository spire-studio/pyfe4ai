# Review round 5: threshold MCFE defaults and the ML adapter's servers (October 2026)

Review only: `perf/tmcfe-defaults` (5c07202) and `main` (1e3b1b8) are
unchanged. This branch (`review/crypto-2026-10-tmcfe-defaults`, from 5c07202)
adds `tests/review/test_tmcfe_defaults_review.py`.

Reviewed: `1e3b1b8..5c07202` (a35079d defaults, ML adapter, tests and docs;
1015701 benchmarks; 5c07202 figures). No critical, high or medium finding.

Checked and holding:

* defaults under `standard` (q of 212-213 bits, four factors of 53-54 bits)
  and `toy` (three factors, e.g. 142 bits as 3 x 48), every factor above `s`,
  native 256-bit RNS ring, records in `param-rns.json`; 224 toy configurations
  whose prime key generation works also work with the RNS default;
* the options of earlier versions (`ct_format="json"`, `ring_modulus="prime"`,
  `key_shares="full"`) and old data: b51989f's archived inputs give the
  archived SHA-256 of dealing, partials, combination and native encryption;
  b51989f key shares and ciphertexts are served with b51989f's partials; a
  b51989f `param.json` is loaded by the earlier options and left untouched by
  the defaults;
* mixing: ciphertexts, key shares, weight parts and partials of four
  deployments (defaults; JSON/prime/full; binary/prime/full; JSON/RNS/split)
  fed to each other, label parts without or with another weight part, weight
  parts as key shares: every case refused with a library error and the new
  messages, both backends; 84 malformed weight / label parts under the
  defaults refused with `FEKeyError`;
* the review tests' edits pin the options each test was written for (the
  configurations they run are those of 1e3b1b8); the archived probes p1-p11
  and the re-check probes r2-r5 give the outputs of 1e3b1b8 with those options
  (p5 differs as `MIN_RNS_FACTORS` intends);
* the ML adapter sends no key material (EncryptedGradient / ThresholdPartial
  JSON scanned for key-share polynomials, MAC and noise keys), parses
  `ThresholdPartial` strictly (20 malformed dicts: `ValueError`), gives each
  server its weight part once per weights and a label part per round, and
  checks `servers=`;
* I-a, I-c, I-d as described; the state version refuses 1e3b1b8's pickles.

## Findings and their tests

| Id | Severity | Where | Test |
| --- | --- | --- | --- |
| R5-1 | low (availability, fails closed) | `utils/ml_adapter.py` `partial_decrypt_gradients`: a prepared threshold aggregation is not completed by a pickled / deep-copied wrapper (its servers are new, without weight parts) nor after the server's `WEIGHT_SHARES_KEPT` store dropped the weight part: `FEKeyError`, although the wrapper holds the key generator | `test_r5_1_a_prepared_aggregation_completes_in_a_copy_of_the_wrapper`, `test_r5_1_a_prepared_aggregation_survives_the_servers_weight_part_eviction` |
| R5-2 | info (availability, fails closed) | `ring_lwe_threshold.py` `_label_parts`: only a newly made label part is checked against the weights' current share set; a cached one is returned from an entry that another thread evicted right after `_weight_entry` returned it | `test_r5_2_a_cached_label_part_is_of_the_current_share_set` |

Further informational items are in the round-5 report (records).
