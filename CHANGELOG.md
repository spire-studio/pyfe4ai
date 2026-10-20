# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor bumps may break
compatibility).

## [Unreleased]

An optional native backend in Rust, the fixes of a cryptographic code review
of October 2026 -- among them a leak in the threshold LWE layers of v0.3.0 --,
exact Gaussian sampling by default, and faster defaults for the threshold MCFE
over Ring-LWE. PyFE4AI remains an experimental research prototype under
active development: interfaces, defaults and serialized formats change between
versions, and it is not recommended for production use. **Several formats are
not compatible with v0.3.0, and modules moved (Code organization); read the
upgrade notes first.**

### Code organization (breaking)

- **Modules moved** (no aliases remain at the old paths; algorithms,
  parameters and serialized formats are unchanged). The package is now in
  layers: `pyfe4ai.core` (base classes and the parameter cache, the
  configurations, the serialized objects and their versions, security
  profiles, review status, one-time precomputations, exceptions),
  `pyfe4ai.schemes` (the families, plus `schemes.common` and
  `schemes.threshold`), `pyfe4ai.primitives` (lattices, sampling, hashing,
  groups, pairings), `pyfe4ai.backend` (native or Python implementation)
  and `pyfe4ai.ml` (the ML adapter). `pyfe4ai` exports what it exported
  before; the configuration types are the config classes of
  `pyfe4ai.core.config` (see Formats, versions and configurations), no
  longer re-exported by `pyfe4ai.schemes`. Update imports as follows:

  | Old | New |
  |---|---|
  | `pyfe4ai.utils.native_backend` | `pyfe4ai.backend` |
  | `pyfe4ai.utils.ml_adapter` | `pyfe4ai.ml` (defined in `ml.wrapper`, `ml.serialization`, `ml.gradients`, `ml.threshold` and `ml.inference`; private names such as `_SCHEME_REGISTRY` from their module, here `pyfe4ai.ml.wrapper`) |
  | `pyfe4ai.utils.quantization` | `pyfe4ai.ml.quantization` |
  | `pyfe4ai.utils.exceptions`, the exceptions re-exported by `pyfe4ai.utils` | `pyfe4ai.core.exceptions` |
  | `pyfe4ai.utils.crypto_constants` | `pyfe4ai.core.constants` |
  | `pyfe4ai.utils.crypto_utils`, its hashes (`hash_to_qr_group`, `hash_to_zn2_star`, `paillier_base`, `hash_to_zq_vector`, `hash_to_zq_material`, `zq_rejection_parameters`, `LABEL_HASHES`, `check_label_hash`, ...) | `pyfe4ai.primitives.hashing` |
  | `pyfe4ai.utils.crypto_utils`, its groups (`generate_group_primes`, `group_generator_fe`, `group_generator_threshold_fe`, `is_dl_group`, `is_dcr_modulus`, `RFC3526_PRIMES`, ...) | `pyfe4ai.primitives.groups.integer_groups` |
  | `pyfe4ai.utils.sampling_utils` | `pyfe4ai.primitives.sampling` |
  | `pyfe4ai.utils.ring_lwe_utils`, `.lwe_utils`, `.lwe_matrix`, `.matrix_utils` | `pyfe4ai.primitives.lattice.ring_lwe_utils`, `.lwe_utils`, `.lwe_matrix`, `.matrix_utils` (the ciphertext byte codecs: `pyfe4ai.core.formats.binary`, see Formats, versions and configurations) |
  | `pyfe4ai.utils.ec_group`, `.ristretto255`, `.modular_utils`, `.dlog_solver` | `pyfe4ai.primitives.groups.ec_group`, `.ristretto255`, `.modular_utils`, `.dlog_solver` |
  | `pyfe4ai.utils.pairing_backend`, `.pairing_utils`, `.bls12_381` | `pyfe4ai.primitives.pairing.pairing_backend`, `.pairing_utils`, `.bls12_381` |
  | `pyfe4ai.utils.validation` | `pyfe4ai._validation` (internal) |
  | `pyfe4ai.schemes.ipfe` | `pyfe4ai.core.base` |
  | `pyfe4ai.schemes.status`, `pyfe4ai.schemes.profiles` | `pyfe4ai.core.status`, `pyfe4ai.core.profiles` (the profile rules, `resolve` and its tables: `pyfe4ai.core.config`) |
  | `pyfe4ai.schemes.one_time`, `pyfe4ai.schemes.random_offline` | `pyfe4ai.core.one_time` |
  | `pyfe4ai.schemes.abdp15_lwe`, `.ddh_base`, `.ec_ddh_base` | `pyfe4ai.schemes.common.abdp15_lwe`, `.ddh_base`, `.ec_ddh_base` |
  | `pyfe4ai.schemes.config_types` (TypedDicts) | `pyfe4ai.core.config` (config classes; see Formats, versions and configurations) |
  | `pyfe4ai.schemes.threshold_utils` | split into `pyfe4ai.schemes.threshold.sharing`, `.binding`, `.codec`, `.store` and `.roles` (names unchanged; see the next table) |
  | `pyfe4ai.schemes.flooding_accounting` | `pyfe4ai.schemes.threshold.kl` |
  | `pyfe4ai.logger` (`init_logging_config`), `pyfe4ai/config.yaml` | removed (see Logging) |

  The threshold schemes share these modules (`schemes.threshold`):
  `sharing` (Shamir, replicated and seed-expanded replicated sharing, dealt
  and added up for an enrolled set), `binding` (what key shares,
  ciphertexts and partial decryptions are bound to: `pp_binding`, the
  binding fields of each threshold LWE scheme, weights, client MAC tags),
  `codec` (encodings
  of key material, threshold ciphertexts and partial decryptions, strict
  parsing), `store` (stores and caches of key-share parts and sums, their
  locks), `roles` (server and combiner logic: checks, the flooding noise,
  combination) and `kl`. `mcfe/ring_lwe_threshold` keeps the scheme's key
  generation and encryption and decryption flow; `mife/lwe_threshold` and
  `mcfe/lwe_threshold` use the same modules where they duplicated code (the
  replicated dealing, the server checks, the noise seed, the Gaussian
  flooding sample). Names moved (those of `mcfe.ring_lwe_threshold` other
  than the private constants were added after v0.3.0 and are listed for code
  written against the development branch):

  | Old | New |
  |---|---|
  | `threshold_utils`: `eval_poly`, `share_secret`, `lagrange_coefficient_at_zero`, `lagrange_at_zero`, `center_mod`, `share_points`, `validate_threshold`, `replicated_sets`, `replicated_assignment`, `share_replicated`, `pieces_held`, `SEED_BYTES`, `piece_layout`, `expand_seeded_piece`, `share_replicated_seeded`, `seeded_pieces_held`, `share_for_enrolled` | `pyfe4ai.schemes.threshold.sharing` |
  | `threshold_utils`: `SHARING_SCHEMES`, `validate_sharing`, `resolve_sharing` | `pyfe4ai.core.config` |
  | `threshold_utils`: `pp_binding`, `check_pp_binding`, `fusion_weight_id`, `check_fusion_weight`, `client_server_mac_keys`, `ciphertext_tag`, `tag_ciphertext`, `check_ciphertext_tags`, `tag_ciphertext_message`, `check_ciphertext_tag_messages` | `pyfe4ai.schemes.threshold.binding` |
  | `threshold_utils`: `check_served_sharing` | `pyfe4ai.core.versions` |
  | `threshold_utils`: `strict_mpz`, `poly_kinds`, `uniform_material`, `binary_material`, `ciphertexts_digest` | `pyfe4ai.schemes.threshold.codec` |
  | `threshold_utils`: `SHARE_CACHE_SIZE`, `share_cache` | `pyfe4ai.schemes.threshold.store` |
  | `threshold_utils`: `validate_enrolled`, `validate_partial_decryptions`, `partial_groups`, `combine_consistently`, `combine_elementwise` | `pyfe4ai.schemes.threshold.roles` |
  | `threshold_utils`: `LEGACY_SHAMIR_POLICIES`, `legacy_shamir_policy` | `pyfe4ai.core.versions` |
  | `mcfe.ring_lwe_threshold`: `MixedSharingsError`, `MIXED_SHARINGS`, `CT_FORMATS`, `pp_ct_format` | `pyfe4ai.core.versions` |
  | `mcfe.ring_lwe_threshold`: `binding_fields` | `pyfe4ai.schemes.threshold.binding.tmcfe_ring_lwe_binding_fields` |
  | `mcfe.ring_lwe_threshold`: `DEFAULT_CT_FORMAT`, `DEFAULT_RING_MODULUS`, `KEY_SHARES`, `FLOODINGS`, `DEFAULT_FLOODING` | `pyfe4ai.core.config` |
  | `mcfe.ring_lwe_threshold`: `WEIGHT_SHARINGS_KEPT`, `LABEL_PARTS_KEPT` | `pyfe4ai.schemes.threshold.store` (set them there; reading or setting them on `mcfe.ring_lwe_threshold` raises `AttributeError` rather than being ignored) |
  | `mcfe.ring_lwe_threshold`: `binary_ciphertext_digests`, `binary_tag_message`, `binary_request_digests` | `pyfe4ai.schemes.threshold.codec` |
  | `mcfe.ring_lwe_threshold`: `uniform_flooding_poly`, `uniform_flooding_material`; `mcfe.lwe_threshold.flooding_noise` | `pyfe4ai.schemes.threshold.roles` |
  | `mcfe.ring_lwe_threshold`: `check_rns_servers` | `pyfe4ai.schemes.threshold.sharing` |
  | the construction ids of `mife/lwe_threshold`, `mcfe/lwe_threshold` and `mcfe/ring_lwe_threshold`, and the latter's state version (private constants) | `pyfe4ai.core.versions`: `TMIFE_LWE_*`, `TMCFE_LWE_*`, `TMCFE_RING_LWE_*` (the same values) |
  | the binding fields and their defaults of `mcfe/ring_lwe_threshold` (private constants) | `pyfe4ai.schemes.threshold.binding.TMCFE_RING_LWE_BINDING_DEFAULTS` |

  Objects pickled by an earlier version that refer to a moved module (for
  example a pickled `FESchemeWrapper`, which holds a `QuantizationConfig`)
  do not load. The serialized keys, shares, ciphertexts and partial
  decryptions (JSON and binary) are unchanged.
- **Formats, versions and configurations** (no change to any serialized
  byte, accepted or refused input, or exception):
  - `pyfe4ai.core.formats` declares every serialized object -- parameter
    records, public parameters, encryption keys, functional keys, key
    shares (whole, weight and label parts, replicated pieces), ciphertexts,
    partial decryptions, the ML wire objects -- as a `Record`: its fields in
    wire order, what each holds and what an absent field means. A record is
    a typed view of the object's dict itself (`Record.from_dict(d).to_dict()
    is d`; reading a field is the dict read, with its exception), and the
    schemes build and read their objects through them. The byte codecs of
    the Ring-LWE ciphertexts are in `core.formats.binary`.
  - `pyfe4ai.core.versions` holds every version marker and compatibility
    decision in one place: parameter-record ids, the threshold construction
    ids, the sharing, flooding, label-hash and ciphertext-format markers with
    what their absence means, sharing ids, key-share parts, the flooding
    noise version and the state versions, with the checks that accept,
    refuse or regenerate (the schemes call them where they checked before).
  - `pyfe4ai.core.config`: a key generator's config is converted at entry
    into the config class of its scheme (`MCFERingLWEConfig`,
    `ThresholdMCFERingLWEConfig`, ...; a `dict` subclass, so dict configs keep
    working) and completed by its security profile there; each class
    declares its options with their defaults and accepted values
    (`config.ring_n` reads an option with its default). The page
    *Configuration options* of the documentation lists every scheme's
    options, their defaults under each profile and the refused combinations,
    rendered from these declarations. The `TypedDict`s of
    `schemes.common.config_types` are replaced by these classes, which keep
    their names where a class remains (`DDHConfig(sec_param=128, eta=3)`
    still builds a config); for static typing they are `dict` subclasses,
    not `TypedDict`s.

  | Old | New |
  |---|---|
  | `pyfe4ai.schemes.config_types` (v0.3.0; `schemes.common.config_types` in development builds): `FEBaseConfig`; `DDHConfig`, `DamgardDDHConfig`, `LWEConfig`, `RingLWEConfig`, `PaillierConfig`, `PairingIPEConfig`, `CryptoConfig`; `MCFELWEConfig`, `MCFERingLWEConfig`, `ThresholdMCFELWEConfig`, `ThresholdMCFERingLWEConfig`, `QuadraticQuadConfig`, `QuadraticSGPConfig`, `MultiInputQuadraticConfig` | `pyfe4ai.core.config`: `SchemeConfig`; the same names |
  | the same module: `PairingMultiIPEConfig`, `PartFHIPEConfig`, `ThresholdDDHConfig`, `ThresholdLWEConfig`, `ThresholdPairingMultiIPEConfig` | `pyfe4ai.core.config`: `MIFEFHMultiIPEConfig`, `SIFEPartFHIPEConfig`, `ThresholdConfig` (and `ThresholdMIFEConfig`, `ThresholdMCFEConfig`), `ThresholdSharingConfig` (and `ThresholdMIFELWEConfig`), `ThresholdMCFEFHMultiIPEConfig` |
  | `pyfe4ai.schemes.profiles` (v0.3.0; `core.profiles` in development builds): `resolve`, `STANDARD_PAIRING_GROUP`, `_STANDARD_MINIMA`, `_STANDARD_FIXED`, `_REFUSED` | `pyfe4ai.core.config`: `resolve`, `STANDARD_PAIRING_GROUP`, `STANDARD_MINIMA`, `STANDARD_FIXED`, `REFUSED` (and `STANDARD_PRESETS`) |
  | `pyfe4ai.utils.ring_lwe_utils.canonical_ring_ciphertext` (v0.3.0) | `pyfe4ai.core.formats.binary.canonical_ring_ciphertext` |
  | `pyfe4ai.schemes.abdp15_lwe.CONSTRUCTION` (v0.3.0); the record ids of the other schemes (private constants) | `pyfe4ai.core.versions`: `SIFE_LWE_RECORD`, `MIFE_LWE_RECORD`, `SIFE_FULLYSEC_LWE_RECORD`, ..., `DMCFE_PAILLIER_RECORD` (the same values) |
  | added after v0.3.0 (development builds): `threshold.kl`: `PER_SAMPLE`, `KL`, `FLOODING_ACCOUNTINGS`, `flooding_config`, `check_kl_threshold` | `pyfe4ai.core.config` |
  | added after v0.3.0: `primitives.lattice.ring_lwe_utils`: `encode_binary_ciphertext`, `decode_binary_ciphertext`, `encode_threshold_ciphertext`, `decode_threshold_ciphertext`, `threshold_ciphertext_from_format1`, `threshold_ciphertext_header`, `encode_values`, `decode_values` | `pyfe4ai.core.formats.binary` |
  | added after v0.3.0: `primitives.lattice.ring_lwe_utils.CT_FORMATS`; `primitives.sampling`: `FLOODING_NOISE_VERSION`, `check_flooding_version`; `mcfe.ring_lwe`: `pp_label_hash`, `ciphertext_label_hash`, `check_ciphertext_label_hashes`; `mcfe.ring_lwe_threshold.LEGACY_FLOODING` | `pyfe4ai.core.versions` |
- **Logging**: the library no longer configures logging or writes log
  files. `pyfe4ai.logger.init_logging_config()` and the packaged
  `config.yaml` (console and rotating files under `./logs/`) are removed;
  the `pyfe4ai` logger has a `NullHandler`, and the modules log to
  `logging.getLogger(__name__)` as before. Configure logging in the
  application (e.g. `logging.basicConfig(level=logging.DEBUG)`); without
  it, the library's log messages are no longer printed by Python's
  last-resort handler. What an application should see is therefore also
  issued as a warning, `pyfe4ai.core.exceptions.FEWarning` (a
  `UserWarning`, not an `FESecurityWarning`): a cached parameter record that
  no longer matches its configuration or is unreadable and is regenerated
  (with new parameters and seeds), a request answered with `None` (an
  unknown client or server id, a decryption out of range or with a wrong
  key, partial decryptions that disagree on the ciphertexts), and a slower
  fallback (an RNS modulus of the threshold Ring-LWE MCFE with more factors
  than the native ring, a modulus without an NTT root). Each warning is
  attributed to the application's call (the first caller outside pyfe4ai),
  so the default filter shows it once per call site, and its text holds no
  caller-supplied value (an unknown id is in the log record only). A
  warnings filter by module therefore matches the application's module: a
  filter such as `ignore::pyfe4ai.core.exceptions.FEWarning:pyfe4ai.*` no
  longer matches the library's `FEWarning`s, while one on the
  application's own module does; filters by category are unaffected, and
  the attribution of `FESecurityWarning`s is unchanged. PyYAML is no longer
  a dependency.
- **Parameter cache location**: key generators keep their parameter
  records in a per-user cache directory instead of `./config/` of the
  working directory: `~/Library/Caches/pyfe4ai` on macOS,
  `$XDG_CACHE_HOME/pyfe4ai` (`~/.cache/pyfe4ai`) on Linux and other POSIX
  systems, `%LOCALAPPDATA%\pyfe4ai\Cache` on Windows
  (`pyfe4ai.core.base.default_cache_dir()`). The `PYFE4AI_CACHE_DIR`
  environment variable overrides it, and a `cache_dir` entry in a key
  generator's config (also through the ML wrapper's key-generator config)
  overrides both. Below that directory the layout and the records are
  unchanged (`authority/<scheme type>/param.json`, `param-rns.json`, ...):
  to keep an existing cache, move `./config/authority` there or set
  `PYFE4AI_CACHE_DIR=./config`. The cache is trusted (see the README's
  known issues): it must not be writable by others.
- **Repository layout**: `Dockerfile.pairing` moved to
  `docker/Dockerfile.pairing` (`docker build -f docker/Dockerfile.pairing
  -t pyfe4ai-pairing .`); the parameter estimators moved to
  `tools/params/` (`tools/params/estimate_ring_lwe.py`, ...);
  `requirements.txt` is removed: `pyproject.toml` is the only list of
  dependencies (`pip install -e ".[test]"`), and `environment.yml` keeps
  the conda-specific parts (Python, pip, gmpy2 from conda-forge).

### Breaking changes and upgrade notes

- **Re-key the threshold LWE layers** (`mife/lwe_threshold`,
  `mcfe/ring_lwe_threshold`, `mcfe/lwe_threshold`). This release refuses
  every key share issued by v0.3.0 (`FEKeyError`): the derandomised flooding
  noise has a new derivation (decryption keys carry `noise_version = 2`; new
  construction ids), and key shares are bound to their public parameters
  (`pp_binding`, see Security). For `t < s` the key is also shared
  differently (replicated sharing, see Security) under new parameter
  records, so `setup()` draws a new public matrix or polynomial. To upgrade:
  1. finish the rounds encrypted under the old keys with v0.3.0 (the leak
     described under Security remains for those rounds);
  2. run this release's key generator (`setup()`, `get_private_keys`,
     `get_decryption_keys`) and hand out the new client keys and key shares;
  3. discard the old key shares.
- **Shamir sharing with `t < s` is refused** by the key generators of these
  layers under `standard` and warns (`FESecurityWarning`) under `toy`.
  Servers refuse Shamir key shares with `t < s` (`FESchemeError`) unless
  their crypto config has `legacy_shamir="warn"`, which never applies to
  public parameters of the `standard` profile and has no environment
  variable. `"shamir"` stays the default for `t = s`, with unchanged outputs.
- **New defaults of `mcfe/ring_lwe_threshold`** (both profiles):
  * binary ciphertexts and partial decryptions (`ct_format="binary"`);
  * an RNS modulus (`ring_modulus="rns"`): `q` is a product of at least three
    NTT primes below `2^62` (four of about 54 bits under `standard`), each
    above `s`;
  * split key shares (`key_shares="split"`): a server first receives its
    share of the label-independent part of the key once per weight vector
    (`get_weight_key_share` at the key generator, `add_weight_key_share` at
    the server; `has_weight_key_share`), then per round only the shares of
    the label terms, which `get_decryption_keys` returns unless
    `key_part="full"`. The key generator keeps the sharings of the last 64
    weight vectors (`threshold.store.WEIGHT_SHARINGS_KEPT`); weights issued again after that
    get a new share set, and their servers need the new weight part;
  * uniform flooding of the partial decryptions (`flooding="uniform"`, noise
    on `[-2^tau, 2^tau)` with `tau = bits(B) + stat_param`, read from the
    request's seeded SHAKE-256 stream): the same statistical distance per
    coefficient as the Gaussian, and `q` of 212 instead of 215 bits under
    `standard` (what it gives up against the Gaussian beyond that bound is in
    the module docstring).

  `ct_format="json"`, `ring_modulus="prime"`, `key_shares="full"` and
  `flooding="gaussian"` select the formats of v0.3.0 (JSON ciphertexts, a
  prime `q`, whole key shares, Gaussian flooding); re-keying is still
  required (above). Cached v0.3.0 parameter records are re-derived; RNS
  records are kept in `param-rns.json`, beside the prime's `param.json`.
  Mixed formats are refused with a message that names the fix (a label part
  without its weight part, JSON ciphertexts under binary public parameters,
  and the other way round). The opt-in `sharing="replicated-seeded"` issues
  whole key shares.
- **Sharing ids** (`mcfe/ring_lwe_threshold`). Every sharing the key
  generator issues -- whole key shares, or the label parts of one label --
  has a random `sharing_id`, which the partial decryptions carry. Partials of
  two sharings of one key would decode to a wrong value inside the decoding
  bound rather than fail, so combiners refuse them
  (`pyfe4ai.core.versions.MixedSharingsError`:
  fetch the label parts again for all servers; the ML adapter's
  `aggregate_gradients` does this once by itself). A label part asked for
  again after it left the key generator's cache (the last 16 labels per
  weight vector, `threshold.store.LABEL_PARTS_KEPT`) is a new sharing. **Store and transmit
  the `sharing_id` with every key share and partial decryption**: partials
  without one combine only with each other, so a layer that drops it on all
  sides would let two sharings be mixed unnoticed.
- **MCFE label hash** (`mcfe/ring_lwe` and its threshold and decentralized
  variants; the dense-LWE MCFE family `mcfe/lwe`, `fullysec_lwe`,
  `lwe_threshold`, `lwe_decentralized`): labels are hashed to `Z_q` by
  rejection sampling from SHAKE128 (`label_hash="shake128-rejection"`, the
  new default), exactly uniform and cheaper, so **MCFE ciphertexts and
  functional-key terms differ from v0.3.0**. Data without the field -- that
  of v0.3.0 -- is read as `"shake256-wide"`, the v0.3.0 hash, so v0.3.0
  public parameters, keys and ciphertexts keep working together;
  `label_hash="shake256-wide"` gives new setups the old hash. The mode is
  recorded in the public parameters, client keys, functional keys, key shares
  and ciphertext layers, and data of different modes is refused instead of
  decrypting to wrong sums.
- **ABDP15 LWE dimensions** (`sife/lwe`, `mife/lwe`; see Security): records
  cached for `lwe_n = "auto"`, in any profile, no longer match their
  re-derivation and are regenerated on load with a new seed for `A`. Keys,
  public parameters and ciphertexts made under the old record stop matching
  it, and the only signs are a logged error and an `FEWarning`: **generate
  keys again after upgrading.** Records for an explicit `lwe_n` (such as the `toy` default 32)
  are kept; under `standard` an explicit `lwe_n` that meets only the old rule
  is refused. Under `toy`, `mife/lwe_threshold` with `lwe_n = "auto"` picks a
  larger dimension (3456 -> 3904 for `X = Y = 20`, `stat_param = 64`) and
  raises where nothing up to 6144 fits.
- **Ring-LWE MIFE / MCFE parameters** (`mife/ring_lwe`, `mcfe/ring_lwe` and
  its threshold and decentralized variants) are sized for messages of norm
  `3X` (see Security): cached parameters are re-derived (construction
  `acfgu18-bmms21/v5`), and `q` grows by 3 to 4 bits.
- **Pairing groups:** every pairing-based scheme defaults to BLS12-381 from
  the Rust extension, under both profiles. Toy keys, public parameters and
  ciphertexts of earlier versions (MNT224) cannot be read on it; pass
  `pairing_group_param="MNT224"`, with charm-crypto installed, to keep using
  them. Under `standard` BLS12-381 replaces BN254 (about 100-bit) and still
  emits a `FESecurityWarning` (about 120-126-bit security). Without the Rust
  extension the pairing-based schemes need charm-crypto and an explicit
  group.
- **Exact Gaussian sampling by default** (FACCT, see Security) for the
  Ring-LWE MIFE / MCFE family (keys and encryption noise), the dense-LWE
  MCFE keys `t`, `sife/fullysec_ring_lwe` and, under `standard`, the
  public-key noise of ABDP15 (`sife/lwe`, `mife/lwe`). These schemes record
  `gaussian_sampler` in their public parameters and ignore the process-wide
  choice (`set_gaussian_sampler`, `PYFE4AI_GAUSSIAN_SAMPLER`); `standard`
  refuses `gaussian_sampler="rounded"` for them, `toy` still accepts it.
  Formats are unchanged; public parameters without the field (earlier
  versions) give FACCT to new encryptions.
- **Refused under `standard`**, besides the ALS16 schemes and
  `mcfe/paillier_decentralized`: `mife/lwe_threshold` (no parameter set is
  both secure and usable, see Security), an explicit `sharing="shamir"` with
  `t < s`, `gaussian_sampler="rounded"` for the schemes above, and ABDP15
  dimensions below the new rule. Both profiles refuse `stat_param` above 256
  and thresholds that are not integers (also `t=True`).
- **Pickles are not portable across versions.** Keep pickled scheme objects
  within one version (e.g. for multiprocessing or Ray); the threshold
  Ring-LWE MCFE key generator, clients and servers refuse a state pickled by
  another version with a clear error (create the object again).
- The threshold LWE layers record `threshold_construction` and
  `security_profile` in their public parameters, and servers refuse
  constructions they do not know or of another sharing. The `ddh` key
  generators refuse a `group` other than `"modp"` instead of ignoring it
  (ristretto255 is the `ddh_ristretto255` variant, see Added).

### Security

Fixes from a cryptographic code review of the schemes and of the new native
code (October 2026), and from the follow-up checks of the fixes.

- **Threshold LWE layers: Lagrange-multiplier leak; v0.3.0 and earlier are
  affected** (`mife/lwe_threshold`, `mcfe/ring_lwe_threshold`,
  `mcfe/lwe_threshold` with `t < s`; severity High, practical impact
  Medium).
  * **Mechanism.** These layers Shamir-shared the functional key for every
    `t < s`. An honest partial decryption then differs from its simulation
    (from the result and a coalition's key shares) by `M nu`, where `nu` is
    the decryption noise and `M` a product of Lagrange coefficients. For
    `s >= 3` and `t < s`, `M = r / d` is not an integer for some corrupted
    set, enrolled set and server, and `d` times the partial minus the
    computable part reveals `nu mod d`, whatever the flooding width.
  * **Who can read it.** With `s = 3, t = 2`, server 2 corrupted and enrolled
    set {1, 3}, `nu mod 4` was read from every tMIFE-LWE partial and every
    tMCFE-Ring-LWE coefficient. Combining enrolled sets raises the modulus
    (144 at `(5, 3)`, about `2^25` at `(10, 6)`, at most `2^25.5` for
    `s <= 10`). Fewer than `t - 1` corrupted servers can suffice, and for
    many `(s, t)` none: the aggregator alone reads `nu` modulo `(s - 1)!` for
    `t = 1`, 6 at `(5, 2)` and 120 at `(7, 2)`. At `(2, 1)` no residue leaks;
    integer multipliers above 1 (up to 4 at `(3, 2)`) multiply the
    statistical distance of the flooding.
  * **Consequences.** With `nu` exact, the tMIFE-LWE functional key would
    follow from the partials by linear algebra. At the `standard` parameters
    the leak is partial -- at most about 25 bits of `nu` per ciphertext for
    `s <= 10`, against noise bounds of about `2^67` (tMIFE-LWE) and `2^71`
    (tMCFE-Ring-LWE) -- and no key or plaintext recovery has been shown. In
    tMCFE-LWE the multipliers narrow the error of its LWE argument, up to
    9-fold at `(3, 2)` and 869-fold at `(5, 3)`.
  * **Not affected:** `t = s`, and the DDH and pairing threshold layers,
    which interpolate exactly in the exponent.
  * **Fix.** For `t < s` the key is shared additively over the
    `(t - 1)`-subsets of the servers (`sharing="replicated"`, the new
    default): piece `T` is held by the servers outside `T` and added by the
    smallest enrolled one, every multiplier is 0 or 1 and the flooding
    argument holds, with `q`, `ring_n` and `lwe_n` unchanged. A server holds
    `C(s - 1, t - 1)` key pieces instead of one share (126 at `(10, 5)`); the
    opt-in `sharing="replicated-seeded"` (see Added) stores at most one full
    piece. Re-key as described in the upgrade notes; the README's Known
    Issues has the details.
- **Exact discrete Gaussians.** The double-precision rounded sampler (shared
  Box-Muller radii, ties to even, nothing beyond 8.57 sigma) is about
  `2^-12` to `2^-4` from i.i.d. `D_{Z,sigma}` per ciphertext and `2^-25` to
  `2^-15` per key, against the `2^-128` budgets of the BMMS21 and ABMPSW20
  proofs. The schemes listed in the upgrade notes now sample with FACCT
  (Zhao, Steinfeld and Sakzad 2020; about `2^-185` per sample). ALS16, the
  DCR schemes and ABDP15 under `toy` still follow the library-wide sampler,
  rounded by default (README, Known Issues).
- **ABDP15 LWE schemes sized by the proof that covers them** (`sife/lwe`,
  `mife/lwe`). Theorem 2 of the PKC 2015 paper, cited so far, does not cover
  noisy LWE keys. The proof is that of the full version (ePrint 2015/017,
  Oct. 2015: Thm. 4.4 with Thms. 6.6 and 6.7), written out in F. Bourse's PhD
  thesis (ENS 2017, Thm. 4.3.5 with Thms. 8.2.2, 8.2.3, 8.3.2). It reduces to
  LWE with error width `sigma' = sigma / (1 + B sqrt(eta))` (`B = 2X`, and
  `4X` per slot of `mife/lwe` in the library's ACFGU18 composition), and
  `standard` now reads the 128-bit lattice-estimator table at `sigma'`
  instead of `sigma`: `lwe_n` grows by 17 to 67 percent for
  `8 <= X = Y <= 1000`, e.g. 896 -> 1280 for `eta = 10`, `X = Y = 50`, and
  1024 -> 1408 for 100 MIFE slots of `eta = 1`; an encryption costs about
  twice as much. No attack is known: the old dimensions still meet the
  estimator for an attack on the public key alone.
- **`mife/lwe_threshold` is refused under `standard`**, like ALS16 and
  dMCFE-Paillier: at `sigma'` its flooding-enlarged `q` needs a dimension at
  the very end of the 128-bit table (5888 to 6144) for a few small bounds and
  beyond it for most others. Use `mife/ddh_threshold`, or
  `mcfe/ring_lwe_threshold` with labels. The opt-in KL accounting (see
  Added) would bring it within the table, but the refusal stays with it.
- **Ring-LWE MIFE / MCFE sized for `3X`.** The ACFGU18 MIFE compiler needs a
  single-input scheme secure for messages of norm `3X` (Fig. 4, footnote 12).
  Up to v0.3.0 the BMMS21 widths were sized for `X`, which met the condition
  only for nine or more clients (with three clients `sigma1` was 0.58 of the
  required width). No attack is known.
- **Threshold key shares bound to their public parameters.** Every key share
  of the threshold LWE layers carries `pp_binding`, a digest of the
  public-parameter fields its partial decryptions depend on (modulus and
  dimension, servers and threshold, flooding distribution, width and
  accounting, label hash, and the formats of the threshold MCFE), and a
  server refuses a share bound to other parameters; the derandomised
  flooding noise is seeded with the digest as well. A share served under
  parameters that differ in one of these fields would answer a request a
  second time under another derivation; for the label hash of the dense
  tMCFE-LWE that gave an exact linear relation on the key share.
- **Wide Gaussians at `bits(sigma) + 320` bits of MPFR precision**
  (`BIGINT_GUARD_BITS`): the DCR keys and masks and all threshold flooding
  were 2 to 19 bits short of their budgets at `+ 128`. The threshold
  Gaussian flooding always uses this sampler, recorded in the decryption
  keys.
- **ML adapter: one ciphertext per client and label under threads** (since
  v0.3.0). `encrypt_gradient` checked `(nid, label)` before encrypting and
  recorded it afterwards, so two concurrent calls for one client and label
  both succeeded and an aggregator could learn `gA - gB`. The pair is now
  reserved under a lock first. A pickled or copied wrapper refuses the pairs
  used so far but does not learn of later ones, so send each client's
  gradient for a label through one wrapper. A pickled wrapper also carries
  the master key and every client's key: ship a worker its client's own
  crypto object (built from `wrapper.pp` and
  `wrapper.kg.get_private_keys(nid)`), not the wrapper.
- **`XofRandom` counters** are claimed atomically: two threads sharing an
  instance could draw the same SHAKE block.
- **The rounded Python sampler no longer uses `SystemRandom.gauss`**, which
  kept the second Box-Muller value in a global: a future secret coefficient
  stayed in memory between calls, and parent and child drew the same sample
  after `fork()`.
- **New code hardened during the review** (no released version affected):
  the precomputations of the offline / online APIs are strictly one-time
  (claimed under a lock, refused in another process, invalidated in a
  forked child, not copyable or picklable); the native code wipes secret
  rows, pads, key pieces and noise material on every path, errors included,
  takes the exact sampler's constant at full precision and checks its
  tables, checks sizes (`ValueError` instead of a panic) and never waits for
  a lock while holding the GIL; binary threshold ciphertexts have a unique
  encoding (values `>= q` refused, tags over the public parameters' digest
  and the bytes), so a client cannot redraw the flooding noise of a key part
  by re-encoding a ciphertext; servers and combiners refuse public
  parameters changed in place and malformed key shares (`FEKeyError`);
  BLS12-381 and ristretto255 decoders accept only canonical encodings, and
  their discrete-log caches are bounded in size.

### Performance

Measured on an Apple M2 Max, one thread, `security_profile="standard"`,
three clients, with the Rust extension (`benchmarks/PERFORMANCE.md` has the
method and more rows):

| workload | v0.3.0 (Python) | this release |
| --- | ---: | ---: |
| MCFE Ring-LWE, 10^4 gradient elements: client / aggregator | 197.5 / 280.0 ms | 17.95 / 6.24 ms |
| ... online, after `precompute_gradient` / `prepare_aggregation` | | 0.25 / 3.16 ms |
| threshold MCFE Ring-LWE, 10^4 elements, `s = 3, t = 2`: client / key shares per round / partial decryption per server / aggregator | 213.5 / 275.9 / 528 / 1348 ms ¹ | 26.7 / 10.7 / 8.5 / 29.1 ms |
| dense-LWE MCFE (`mcfe/lwe`), 200 elements: client / aggregator | 1021 / 1030 ms | 93.1 / 101.4 ms |
| DDH MCFE, per element: client / aggregator / ciphertext | 10.75 / 11.10 ms, 936 B | 0.069 / 0.139 ms, 75 B ² |
| SGP / Quad / MCFE FH-Multi-IPE, one round | 320 / 506 / 889 ms ³ | 30 / 48 / 47 ms |

¹ The Python path at the same 212-bit `q`. ² `mcfe/ddh_ristretto255`
against the 3072-bit `Z_p^*` group. ³ charm-crypto on MNT224 (Docker, same
Mac) against BLS12-381.

- **Native backend** (`pyfe4ai._rust_native`, see Added): the Ring-LWE MIFE
  / MCFE encryption, decryption and key derivation of a whole tensor per
  call, with the GIL released and optional threads (4 threads: MCFE client
  6.2 ms, aggregator 3.6 ms); the threshold MCFE (256-bit words for moduli
  up to `2^253`, the RNS ring, Shamir and replicated dealing, partial
  decryption and combination); the dense-LWE MCFE label sums; the exact
  FACCT sampler (0.45 to 1.1 us per sample instead of 7 to 12 us) and the
  wide MPFR Gaussians (0.55 to 2 us instead of 8 to 12 us); ristretto255; and
  BLS12-381 pairings (0.9 ms; charm MNT224 2.4 ms) with multi-scalar
  multiplication for every product of powers. The outputs are those of the
  Python code, which the tests check bit for bit; unsupported parameters
  fall back to Python.
- **Threshold MCFE defaults**: ciphertexts of 71 instead of 222 bytes per
  element, a server's key shares per round of 0.89 / 0.44 MB at
  `t = 2` / `t = 3` instead of 6.75 / 3.38 MB (plus the weight part once).
  With the options of v0.3.0 the same round takes 39.1 / 40.9 / 65.4 /
  177.0 ms on the native backend.
- **Replicated sharing natively**: key shares at `(s, t) = (3, 2)` 272 -> 43
  ms, at `(10, 5)` 26.9 -> 2.2 s (tMCFE Ring-LWE, 10^4 elements).
- **Offline / online phases** (see Added) move the randomness and label
  masks out of the round: online, 20 elements, MCFE-DDH / Damgård / Paillier
  clients 212 / 212 / 717 -> 0.67 / 0.67 / 0.39 ms.
- **Cheaper label hash** (SHAKE128 with rejection): MCFE Ring-LWE aggregator
  7.40 -> 5.87 ms. With the opt-in RNS modulus of the MIFE / MCFE / dMCFE
  Ring-LWE schemes the aggregator takes 5.69 ms and its online phase 2.65 ms
  (3.16 ms with the prime).
- **Costs of the security defaults.** FACCT is most of the MCFE client's 18
  ms (the offline phase moves it out of the round); on the Python backend the
  same client takes 360 ms. ABDP15 key generation under `standard` takes
  about twice as long (`sife/lwe`, `eta = 10`, `X = Y = 50`: `lwe_n` 1280,
  setup 8.3 s; `mife/lwe` with 100 slots: 47 s).
- Python path: `NTT(a)`, `NTT(pk)` and `NTT(sk_y)` cached and decryption
  products summed in the NTT domain (same outputs); dense-LWE MCFE keys
  parsed once per object.

### Added

- **Native backend** in Rust and PyO3 (`pyfe4ai._rust_native`, sources in
  `rust/`; `rust/core` has no dependencies). `pip install` builds it when
  `cargo` is on the `PATH` (in a source checkout also `python
  tools/build_rust.py`; `PYFE4AI_NO_NATIVE=1` skips it); without it the
  package runs its Python implementation. `pyfe4ai.backend`
  selects `"auto"` (default), `"python"` or `"native"` per process
  (`set_backend`), per block (`with backend(...)`) or with
  `PYFE4AI_BACKEND`, and the threads of one call (`set_threads`,
  `PYFE4AI_THREADS`, default 1); `describe()` reports the build for
  benchmark records. Native sampling reads one 32-byte seed from the
  operating system per call and expands it with ChaCha20 (a standard stream
  cipher, not an SP 800-90A DRBG, which matters only for FIPS validation);
  no generator state outlives a call.
- **BLS12-381** (`pyfe4ai.primitives.pairing.bls12_381`, behind charm's `PairingGroup`
  interface; `group.hash` is RFC 9380 hash_to_curve),
  `pairing_backend.multiexp`, `pow_many` and `deserialize_many`;
  `gt_dlog(..., cache=False)`. charm-crypto is needed only for other curves.
- **DDH schemes over ristretto255** (`sife/`, `mife/`,
  `mcfe/ddh_ristretto255`, or `group="ristretto255"` with the `ddh` variants
  in `FESchemeWrapper`; status *unreviewed*): RFC 9496, labels hashed with
  RFC 9380 `hash_to_ristretto255`, about 126-bit security; curve25519-dalek in
  the extension, a pure-Python fallback (not constant-time) without it.
- **ML adapter for the threshold MCFE** (`mcfe/ring_lwe_threshold`):
  `FESchemeWrapper` runs the decryption servers in-process;
  `prepare_aggregation` and `aggregate_gradients` take `servers=` (default
  the first `t` of `lst_sid`) and hand each server its label key share per
  round and its weight key share once per weights. The steps are also
  separate functions, `partial_decrypt_gradients` (one server) and
  `combine_partial_decryptions`, whose `ThresholdPartial` has `to_dict` /
  `from_dict` (binary values as base64, layout checked).
  `EncryptedGradient.to_dict` writes chunk fields holding bytes as base64
  strings named in the layer's `base64` list. A `PreparedAggregation` holds
  key material (the servers' label key shares). The adapter refuses
  `flooding_accounting="kl"` and the other threshold variants.
- **Offline / online encryption and key preparation**:
  `MCFERingLWE.precompute(label, shapes)` and `encrypt_lst_ndarray(...,
  precomputed=...)`; `precompute(label, shapes)` and `prepare_key(dk, label,
  shapes)` for MCFE-DDH, -Damgård and -Paillier and their decentralized
  variants; `precompute(shapes)` for SIFE- / MIFE-Paillier and MIFE-DDH; ML
  adapter `precompute_gradient`, `encrypt_gradient(precomputed=...)`,
  `prepare_aggregation` and `aggregate_gradients(prepared=...)`. The same
  ciphertexts as the one-shot calls. A precomputation can be used once
  (`pyfe4ai.core.one_time`).
- `ct_format="binary"` for the Ring-LWE MIFE / MCFE ciphertexts (crypto
  config; `FESchemeWrapper(crypto_options=...)`): a versioned compact
  encoding, 37 % of the JSON size; JSON stays their default, and decryption
  reads both.
- Opt-in **RNS modulus** for the Ring-LWE MIFE, MCFE and decentralized MCFE
  (`ring_modulus="rns"`: `q` a product of two NTT primes of the prime's bit
  length; same ciphertext formats, `q_factors` in the public parameters).
  Security review in `docs/rns_modulus_review.md`.
- Opt-in **seed-expanded replicated sharing** for the threshold LWE layers
  (`sharing="replicated-seeded"`): every key piece but one is expanded from
  a secret 32-byte seed with SHAKE-256, so a server stores at most one full
  piece -- the largest tMCFE-Ring-LWE key share under `standard` (3 clients,
  `ring_n` 8192) at `(10, 5)` is 3.4 MB instead of 428 MB, dealt in 2.0
  instead of 30 s. Its key shares hide the key computationally (SHAKE-256
  keyed by the seeds as a PRF) instead of perfectly; full replicated sharing
  stays the default.
- Opt-in **KL accounting of the threshold flooding** for `mife/lwe_threshold`
  and `mcfe/ring_lwe_threshold` with `t = s`
  (`flooding_accounting="kl"`, a mandatory `max_partial_decryptions` budget
  `Q`, and a `PartialDecryptionBudget` per server that the deployment keeps
  across restarts; `pyfe4ai.schemes.threshold.kl`). It gives a weaker
  guarantee -- about `stat_param - 6.85` bits, for the first `Q` partial
  decryptions of every server -- for a smaller `q`: 178 bits for
  `mcfe/ring_lwe_threshold` under `standard` with `N = s Q k = 2^53`
  flooded coefficients (Gaussian flooding, which `"kl"` uses), against 212
  bits with the default uniform flooding. `t < s` and `mcfe/lwe_threshold`
  refuse it.
- Opt-in **packed layout for `sife/fullysec_ring_lwe`**
  (`ndarray_layout="packed"`, experimental: the leveraging argument is the
  project's own, independently checked, not externally reviewed): one
  ciphertext carries up to `ring_n` inner products, e.g. 4096 elements at 182
  bytes each for `eta = 1`, `X = 100`.
- **Exact sampler API**: `pyfe4ai.primitives.sampling.discrete_gaussian_facct`,
  `sampler="facct"` on `discrete_gaussian_matrix` / `discrete_gaussian_bigint`,
  the `gaussian_sampler()` context manager, `set_gaussian_sampler()` and
  `PYFE4AI_GAUSSIAN_SAMPLER=facct` for the schemes that follow the
  library-wide choice; output platform-independent for a given bit source
  (known-answer tests); not constant-time. `discrete_gaussian_bigint` takes
  an optional higher `precision`.
- `examples/mcfe_ring_lwe_threshold_demo.py`; benchmark scripts
  `bench_fl_compare.py`, `bench_offline_potential.py`,
  `bench_replicated_sharing.py`, `bench_fullysec_packing.py` and
  `benchmark_sampling.py`, with `benchmarks/PERFORMANCE.md`;
  `tools/params/estimate_rns_modulus.py`.
- Regression tests of the October 2026 review in `tests/review/`.
- CI: a `rust-native` job builds the extension and runs `cargo test` and the
  full suite on the native backend (Python 3.10 and 3.13) and on the Python
  backend (Python 3.10).

### Changed

- **`sife/fullysec_ring_lwe` samples exact Gaussians** for its keys and
  encryption noise in both layouts (`gaussian_sampler = "rounded"` restores
  the previous draws). With the rounded sampler its keys were about `2^-47`
  to `2^-40.5` from `D_{Z,sigma1}` for an ideal rounded Gaussian, and about
  `2^-12` with the double-precision path, terms its adaptive proof does not
  budget.
- The pairing-scheme tests run on `PYFE4AI_TEST_PAIRING_GROUP` (default
  BLS12-381) and skip when no backend implements it.
- The NTT root search requires `root^n = -1 (mod q)`, which selects the same
  roots for every prime.
- Discrete-log tables are keyed by value (the old string keys are still
  read).
- Documentation claimed more than the code: the flooding argument of tMCFE
  (Ring-LWE) and tMIFE (LWE) is statistical per answer and gives no bound
  for many answers to varied ciphertexts of a corrupted client, and the
  per-sample `0.02 / sigma^2` figure of the rounded sampler holds for the
  ideal rounded Gaussian only, not for its double-precision implementation.

### Fixed

- `EncryptedGradient.to_dict()` failed with `KeyError: ()` on the packed
  ciphertext layers of the Ring-LWE MCFE schemes (`mcfe/ring_lwe` and its
  threshold and decentralized variants), and `from_dict()` could not rebuild
  them. Packed layers are written as `{"packed": {"shape", "first_chunk",
  "chunks"}}` (the per-element format is unchanged), and `from_dict()` checks
  the packed layout.


## [0.3.0] — 2026-10-13

Privacy fixes: every scheme that was insecure in v0.2.0 now implements a
published construction (four threshold variants add an *experimental*
threshold layer of our own on top of one, and dMCFE-DDH an *experimental*
carry mask), and every scheme carries a
review status. **Keys, ciphertexts and partial decryptions of the affected
schemes are not compatible with v0.2.0.** Read the README *Scheme Status*
table and *Known Issues* before relying on any privacy property.

### Security

- **MCFE-Paillier** implements ABMPSW20 (Abdalla et al., SCN 2020) Fig. 3:
  `c = (1 + N)^x H(l)^s mod N^2` with a full-domain hash into `Z_{N^2}^*`,
  `s ~ D_{Z,sigma}` with `sigma > sqrt(kappa + 2n log2(2X)) N^{5/2}`, and the
  single aggregated key `sum y s`. Previously a single ciphertext revealed a
  client's `<x_i, y_i>` to the key holder, and keys were brute-forceable.
- **MCFE-LWE / MCFE-FullySec-LWE** implement ABMPSW20 Fig. 4
  (`c = round_q0(Z . H(l) + floor(q/K) x)`, aggregated key `sum y Z`) with
  parameters derived from the Appendix C.3 constraints (selective and
  adaptive choice of `m0` respectively). Previously the key holder learned
  `x_i mod l` and `x_i` up to a few candidates.
- **MCFE-Damgard** implements ABMPSW20 Fig. 2 (`D_k`-MDDH, `k = mddh_k`,
  default 1, where it coincides with CDGPP18). Previously two ciphertexts of
  one client under two labels revealed its plaintexts.
- **tMCFE-DDH** is CDGPP18 with the aggregated key Shamir-shared over `Z_q`;
  any `t` servers interpolate `[H(l)]^d` in the exponent. Previously the
  combiner could strip each client's MD5-scalar label mask.
- **tMCFE-LWE** is ABMPSW20 Fig. 4 with a Shamir-shared key and
  noise-flooded partial decryptions (each is an LWE sample under the
  server's share instead of an exact linear function of it). The noise is
  derived from a per-server secret, the label and the server's Lagrange
  coefficient for the enrolled set, so repeated requests cannot be
  averaged. The threshold layer is our own, so the
  scheme's status is *experimental*.
- **Paillier parameter caches** (SIFE, MIFE, MCFE, dMCFE) no longer store
  the primes `p, q`: the factorization decrypts SIFE ciphertexts directly
  and MCFE ciphertexts from two labels. Caches written by earlier versions
  are regenerated with a fresh `N`.
- **Labels**: ndarray element labels were plain JSON strings, so a chosen
  user label could reproduce an element's label (two ciphertexts under one
  label leak `x - x'`); user and element labels are now hashed in disjoint
  domains (`pyfe4ai.schemes.mcfe.labels`).
- Threshold key shares record a digest of their fusion weights and servers
  refuse ciphertexts presented under other weights.
- **SIFE-Paillier** samples keys from `D_{Z,sigma}`, `sigma > sqrt(kappa)
  N^{5/2}` (ALS16); keys used to be drawn from a range small enough to
  brute-force from the public key.
- **MIFE** Damgard, Paillier, LWE, FullySec-LWE and Ring-LWE follow
  ACFGU18 (CRYPTO 2018) Fig. 4: each slot encrypts `x_i + u_i mod L` under
  its single-input IPFE with a one-time pad `u_i`, the functional key adds
  `z = sum <u_i, y_i>`, and decryption decodes only the sum (ALS16 Figs. 7, 9
  and 8, ABDP15, and BMMS21 Sec. 6 as the per-slot IPFEs). Previously the
  key holder learned every client's `<x_i, y_i>`. MIFE-Paillier keys are
  Gaussian (they used to be brute-forceable).
- **tMIFE-LWE** Shamir-shares the padded MIFE-LWE key and floods each
  partial decryption with `D_{Z, 2^kappa B}` (`B` bounds the decryption
  noise; `q` enlarged), derandomised per request (*experimental*).
- **dMCFE-DDH, dMCFE-LWE, dMCFE-Paillier** follow ABMPSW20 App. D.1, Fig. 14
  (DMCFE from MCFE): encryption and decryption are those of the reviewed
  MCFE, and each client publishes an MCFE2 (DCR, Fig. 3) encryption of its
  key contribution under the label `H(y)`; zero-sum MCFE2 keys make the
  shares combine to exactly the MCFE1 functional key. Previously the
  combiner could strip each client's label mask, and dMCFE-Paillier keys
  were brute-forceable.
- **MCFE-Ring-LWE** implements BMMS21 (PKC 2022) Sec. 6, the
  identity-based MCFE-compiler: clients encrypt `x_i + u_i + H(u'_i, l)` and
  the key for `(y, l)` carries the label's pad term, so keys are bound to a
  label. **dMCFE-Ring-LWE** follows BMMS21's identity-based DMCFE (ABKW19 +
  GDSum, realised with pairwise PRF zero-sum masks). **tMCFE-Ring-LWE**
  Shamir-shares the identity-based key and floods partial decryptions
  (*experimental*). Previously all three masked inputs with a small public
  label scalar.
- **FH-MCFE** implements AGT21b (TCC 2021) Sec. 6.2 as recapped in NPS24
  (IACR CiC 2024) Sec. 2: one function-hiding IPFE per client
  (`pyfe4ai.schemes.mcfe.fh_ipe`, ACFGU18 Fig. 12 with `n = 1` and LV16
  padding), ciphertexts carry `H1(tag)` and every functional key embeds a
  fresh zero sharing. **dMCFE-FH-Multi-IPE** implements NPS24 Fig. 1
  (selective FH-DMCFE): the zero sharing lives in the encryption keys and
  each client derives its own key from an explicit `tag-f` and its own
  weights (`H2(tag-f)`), deterministically, one key per client and `tag-f`.
  **tMCFE-FH-Multi-IPE** Shamir-shares the FH-MCFE key exponents (servers
  hold their shares as group elements); partial decryptions are combined in
  `G_T` (*experimental*). Previously all three masked inputs with a public label
  scalar and ran over the symmetric group `SS512`, where SXDH fails; the
  family now defaults to `MNT224` and rejects symmetric groups.
- **All pairing-based schemes** (SIFE FH-IPE and partial FH-IPE, MIFE
  FH-IPE and FH-Multi-IPE, quadratic Quad, SGP and MIQFE, the FH-MCFE
  family) default to the asymmetric group `MNT224` instead of the symmetric
  `SS512` and reject symmetric groups: their constructions assume SXDH or a
  generic asymmetric bilinear group, and DDH is easy in a symmetric one.
- **Review of the remaining schemes** against their papers:
  - SIFE-DDH, SIFE-DDH-Dynamic and MIFE-DDH published ciphertext components
    as the integer product of two residues mod `p`; trial division (or
    factoring) recovered the plaintext from the ciphertext alone. Every
    element is now reduced mod `p`, and keys mod `q` (a key with a large
    weight used to reveal the master secret / the MIFE pads).
  - SIFE partial-FH-IPE (Gay20 Fig. 8) handed the master secret key
    `(b, U, V)` to every encryptor, so any client decrypted everything and
    recovered every key's weights. `get_private_keys` now returns `None` and
    `Enc'` is run by the key generator (`sec_encrypt`). The default subspace
    dimension equalled `eta` (a square `M` reveals `V`); it is now 1 (see
    below) and `M` is resampled until it has full rank.
  - SIFE FH-IPE (KLMMRW18) had no privacy for `eta = 1` (`B* = 1`) and let
    anyone recognise zero vectors; vectors are now padded to `(x, 1, 0)`,
    `(y, 0, 1)`.
  - MIFE FH-IPE ran independent per-slot instances and decoded every slot,
    revealing each `<x_i, y_i>` and allowing slot keys of different
    functional keys to be recombined; it now implements ACFGU18 Fig. 12.
  - MIFE in federated learning: MIFE has no labels, so one functional key
    combines any ciphertexts of a client and reveals differences between
    its values. `encrypt_gradient` / `aggregate_gradients` refuse MIFE
    schemes (use MCFE).
  - tMIFE-DDH is rebuilt on MIFE-DDH with the functional key Shamir-shared
    mod `q` (as tMCFE-DDH); key shares are bound to their weights.
  - Quadratic Quad now implements Gay20 Fig. 4 (it encrypted `x (x) y` with
    an IPFE before): ciphertexts of `4n + 2m + 4` G1 and `m` G2 elements.
  - Gaussian samples wider than `2^40` (FullySec-LWE keys and noise) were
    rounded doubles whose low bits were all zero; they now use the
    arbitrary-precision sampler.
  - charm serialises the identity of G1 into a point outside the group;
    `pyfe4ai.utils.pairing_backend.PairingGroup` writes it explicitly.
- **Independent verification** of the rebuilt schemes found:
  - Function-hiding keys (FH-MCFE, FH-DMCFE, tMCFE-FH, MIFE FH-IPE)
    carried `weights_id`, an unsalted SHA-256 of `y`: any key holder could
    brute-force `y` and link keys for the same function. Keys now carry a
    random `key_id` (FH-DMCFE shares only `tag_f`); decryption no longer
    needs or checks `y`, and threshold servers do not need it either.
    Clients receive their slot key without the outer `A'` (only key
    generation uses it).
  - SIFE partial-FH-IPE keys reveal `M^T y`, which for small
    `eta - subspace_dim` usually determines a bounded integer `y`; the
    default `subspace_dim` is now 1 and the leakage is documented.
  - tMIFE-DDH applied key shares to unchecked ciphertext elements: an
    element outside the order-q subgroup leaked one bit of a server's share
    through the Legendre symbol. Servers (and MIFE-DDH decryption) now
    require subgroup elements. All threshold servers refuse a key share
    issued to another server.
- **DOT18 and DSGPP18 reviews:**
  - Quadratic SGP handed the master secret key `(s, t)` to every encryptor;
    as in DSGPP18 Fig. 2 the public key is now `(g1^s, g2^t)`, encryption
    uses it only and `get_private_keys` returns `None`. Floats were
    truncated silently; inputs are validated.
  - MIFE FH-Multi-IPE (DOT18 Sec. 3.1) produced deterministic keys for
    `n = 1` with `k = 1`; it now requires `n >= 2`, `k >= 1` and
    `len(lst_nid) == n`, checks that every slot is present, and its
    decryption helpers no longer ask for `y`.
  - Quadratic MI-SGP had no published basis (per-client copies of SGP): the
    key holder learned every client's `x_i^T F_i y_i` and could mix sub-keys.
    It is replaced by `quadratic/miqfe`, the MIFE for quadratic functions of
    Agrawal, Goyal, Tomida (TCC 2022, Sec. 5): per-user encryption keys made
    of master ciphertexts (a corrupted user learns its own key, not the master
    secret), function
    class `sum_{i<k} c[(i,j,k,l)] x_i[j] x_k[l]` (cross-user terms), built from
    the TAO20 FH-IPFE and the paper's SK-MCFE schemes (Figs. 1 and 9).
- **SIFE-Paillier and MIFE-Paillier** decryption returned whatever the
  decryption equation produced. The ciphertext `ct0 = 1 + N`, `ct1 = 1`
  turns `ct0^{-k}` into `-k mod N`, so anyone shown the output of a
  decryption (e.g. an inference client) learnt the functional key.
  Decryption now checks ciphertext lengths and ranges and returns `None`
  unless the result is a valid encoding within `+-eta X Y`
  (`modular_utils.check_zn2_elements`, `decode_paillier`).
- **dMCFE-DDH** reduced every client's key contribution mod `q` before the
  MCFE2 layer, whose decryption is the *integer* sum: anyone holding the
  published shares read the carry (`0..n-1`) of that sum, a comparison
  oracle on the clients' secrets. ABMPSW20 states Fig. 14 for MCFEs without
  a discrete log in decryption only. Contributions are now masked with
  `q rho_i`, `rho_i < 2^(stat_param + bits(n))` derived from a new per-client
  key and `H(y)`; the scheme becomes *experimental* (our own argument).
- **dMCFE-LWE** reduced every contribution mod `q` as well, so the integer
  sum also told how many `y_i t_i` were negative in each coordinate. The
  contributions are now sent over the integers, as the functional key
  `sum y_i Z_i in Z^(n0 + m0)` of ABMPSW20 Fig. 4 / Fig. 14.
- **SIFE-LWE and SIFE-Ring-LWE** encoded a plaintext as `trunc(x q / p)`
  instead of `floor(q / p) x`. The two differ by a small term that is not
  linear in `x`, while ABDP15's proof relies on the linear ciphertext
  homomorphism of its PKE and BMMS21's last game on `floor(q/K)` times the
  message; the residue was hidden only statistically by the ciphertext
  noise. Both now use `floor(q / p) x mod q` (`center_lwe_vector`,
  `center_matrix`); found while an independent review of the new
  `sife/fullysec_ring_lwe` flagged the same encoding there. Ciphertexts
  change; decryption is unaffected.
- **tMIFE-LWE** (experimental) enlarged `q` for its flooding noise but kept
  the `m` of the base modulus, so the binary `r` of a ciphertext had about
  half the length the leftover hash lemma needs for `A^T r` to be close to
  uniform (toy defaults: `m = 753` where `2139` is needed); ABDP15's
  semantic-security argument then fails. `m` and `q` are now iterated to a
  fixed point; cached parameters are regenerated.
- **tMIFE-LWE** (experimental) servers decrypted any ciphertexts they were
  handed. A partial decryption is linear in `ct0`, so a combiner submitting
  `ct0 = c e_j` for chosen `c` read the shared functional key off the
  partials coordinate by coordinate (the flooding hides a fixed noise term,
  not a chosen multiple of the key). Clients now tag every ciphertext with
  one HMAC-SHA256 per server, under per-(client, server) keys sampled by
  `setup`, and a server refuses ciphertexts whose tag for it does not verify.
  A corrupted client can still choose its own slot's ciphertext, which
  reveals only that slot's key part.
- **tMCFE-Ring-LWE** (experimental) servers decrypted any ciphertexts they
  were handed; a partial decryption is linear in `ct1`, so a combiner
  recovered the functional-key parts `S_i y_i` exactly in a few queries
  with chosen `ct1`. As for tMIFE-LWE, clients now tag every ciphertext (and
  every ndarray chunk) for every server, over the (chunk) label and the
  ciphertext; servers refuse untagged, altered or other-label ciphertexts.
- **tMCFE-FH-Multi-IPE** (experimental) servers decrypted whatever round the
  combiner submitted (FH ciphertexts hide their label). Ciphertexts now
  carry the encoded label, tagged together with the ciphertext for every
  server, and `share_decrypt` takes the label the server agrees to decrypt.
- **tMIFE-LWE and tMCFE-Ring-LWE** seeded the flooding noise with the JSON of
  the request, so the same ciphertexts with an extra field, another server's
  tag or (Ring-LWE) coefficients written as `v + q` drew fresh noise, and
  averaging such partials removes it. The noise is now seeded with a digest
  of the values the key part depends on (`ct0` for tMIFE-LWE; `ct1` and the
  label or chunk for tMCFE-Ring-LWE), so neither the encoding, nor extra
  fields, nor the other half of a ciphertext re-tagged by a corrupted client
  draws fresh noise (`threshold_utils.ciphertexts_digest`); the enrolled set
  enters only through the server's Lagrange coefficient (two sets with the
  same coefficient drew different noise), and tMCFE-Ring-LWE's noise keys
  no longer change when a key is re-shared for another `n_chunks`. A
  corrupted client varying its own slot still draws fresh noise (averaging
  needs about `2^(2 stat_param)` partials; documented); tMCFE-Ring-LWE
  also refuses ciphertexts of the wrong shape
  (`ring_lwe_utils.canonical_ring_ciphertext`). Tags use the shared
  `threshold_utils.client_server_mac_keys`, `tag_ciphertext` and
  `check_ciphertext_tags`.
- **tMCFE-DDH** servers decrypted whatever label the submitted ciphertexts
  carried (the key is label-independent), so they could not restrict
  decryption to the rounds they mean to serve: a combiner could have any
  round decrypted at any time. `share_decrypt` now takes the label the
  server agrees to decrypt and refuses ciphertexts under any other.
- **tMCFE-FH-Multi-IPE** (experimental) handed every server Shamir shares of
  the FH-MCFE key exponents `e_i` in `Z_p`. The key hides `y` only as
  `[e_i]_2`: `t` colluding servers (one if `t = 1`) interpolated `e_i` and
  solved for the weights. Servers now hold their shares in the exponent
  (`[e^(k)]_2`, `[z^(k)]_T`) and raise their pairing product to their
  Lagrange coefficient; `t` servers interpolate the FH-MCFE key itself.
- **dMCFE-FH-Multi-IPE** used `tag-f = (tag_f, H(y))` over all clients'
  weights, so a client's key share depended on the others' weights: a
  corrupted client derived its share for a guess of the honest clients'
  weights and learnt whether the guess was right from whether decryption
  succeeded. As in NPS24 Fig. 1, a client's share now depends on an explicit
  `tag_f` and its own `y_i` only; a client object refuses a second weight
  vector under a `tag_f` it already used.
- **ML adapter**: `FESchemeWrapper` holds the master key, and
  `aggregate_gradients` issued a key for any weights, so weights `(1, 0)` or
  a subset of the clients decrypted one client's gradient; labels could be
  reused. It now needs a ciphertext and a non-zero weight for every enrolled
  client (`allow_partial=True` relaxes this), refuses a second encryption by
  a client under one label and a second weight vector for one label, and is
  documented as a single-trust-domain simulation.
- **`standard` profile**: the statistical parameter `stat_param` kept its
  default 64 and accepted any explicit value (`stat_param = 1` gave
  flooding at statistical distance about 1/2), although the profile claims
  128 bits and refuses weaker explicit values. It is now 128 (the `kappa` of
  the Ring-LWE schemes) and lower values are refused: a distance of
  `2^-128` per sample instead of `2^-64`. ABMPSW20 moduli grow by about 64
  bits (`mcfe/lwe` picks `n0` of about 7200 to 8200 instead of 4600 to
  5600), tMCFE-Ring-LWE moves to `ring_n = 16384`, and tMIFE-LWE to
  `lwe_n` of about 6000, for which `WIDE_LWE_128_LOG_RATIO` now extends to
  `n = 6144` (same estimator and method).
- **`standard` profile**: a config that was a `Mapping` but not a `dict`
  (e.g. `MappingProxyType`) skipped the profile while the key generator
  recorded `security_profile = "standard"` (`sife/ddh` ran on a 128-bit
  group). Every `Mapping` is now resolved.
- **Parameter cache**: cached discrete-log groups and DCR moduli were used
  without any check (a planted 5-bit group or a prime "RSA modulus" was
  accepted, also under `standard`). DL groups are now checked for primality,
  size, a safe-prime structure (a cached cofactor `r` with a small `q`
  passed a first version of the check), the order of their generators and,
  at RFC 3526 sizes, for being the RFC prime; DCR moduli for size,
  compositeness and prime factors below 2^16, Paillier bases for not being
  `1 mod N` (`crypto_utils.is_dl_group`, `is_dcr_modulus`). Everything else
  a cache holds is now either re-derived on load or expanded from a stored
  seed: the Ring-LWE parameters and public polynomial `a`
  (`ring_lwe_utils.seeded_polynomial`; `a = 0` used to be accepted), the
  ABDP15 / tMIFE-LWE parameters including `sigma_f` (`lwe_n = 1`,
  `sigma_q = 0`, `m = 8` or `sigma_f = 1` used to be accepted), the ALS16
  parameters and matrix `A`, the Paillier base `g = H(seed)^(2N)`
  (`crypto_utils.paillier_base`; `g = N^2 - 1` used to reveal the
  plaintexts) and the Damgard element `h`, hashed into the group (a planted
  `h = g` had a known discrete logarithm).
  A file that fails, or does not parse (e.g. after an interrupted write), is
  regenerated, and files are written atomically. Since no check can tell
  whether someone knows the factors of a DCR modulus, the `standard` profile
  never loads or stores the Paillier moduli and the dMCFE `N2` (they are
  generated for every key generator); `toy` keeps the cache.
- **Ring-LWE parameters (SIFE, MIFE, MCFE, tMCFE, dMCFE)**: `bmms21_parameters`
  cites BMMS21 Sec. 4.2 but used `sigma = 1` and `sigma1 = 2 sqrt(l) X`,
  which miss its smoothing conditions (the mhe-RLWE condition of G1 -> G2
  needs `sigma sqrt(1 - ...) > eta(Z^(n + n l))`, about 5.6, and had 0.71;
  G3 -> G4 needs `sigma' >= sqrt(2) eta(Z^n)` and had `sigma' = 1`). The
  widths now follow Sec. 4.2 (`ring_lwe_utils.bmms21_selective_parameters`),
  which adds about 10 bits to `q`. The `standard` profile checked the
  moduli against the Homomorphic Encryption Standard's bound, which assumes
  errors of width 3.2 (the standard SIFE-Ring-LWE had `sigma1 = 2`) and which
  the lattice estimator puts at about 125 bits; it now uses
  `RING_LWE_128_LOG_Q`, estimated with the same tool and rule as the LWE
  tables for widths 2 to 16 and `n = 1024` to `32768`, read at the smallest
  width among the RLWE problems of the proof (about 8). Cached parameters
  are regenerated.
- **Threshold schemes** assume honest-but-curious servers; this is now stated
  in the README, `SECURITY.md` and `threshold_utils`. Nothing proves a
  partial decryption correct, so `combine_decrypt` and
  `decrypt_lst_ndarray_ct` of every threshold scheme accept partial
  decryptions for several enrolled sets and raise `FESchemeError` if their
  results disagree (`threshold_utils.combine_consistently`).

### Added

- `security_profile` config key for every key generator
  (`pyfe4ai.schemes.profiles`). `"toy"` (default) leaves configs unchanged,
  so all defaults stay as they were. `"standard"` presets 128-bit-oriented
  sizes: a 3072-bit DL group (RFC 3526 MODP group 15), a Paillier `N` of
  about 3072 bits (`bit_length = 1536`), Ring-LWE `kappa = 128` with `ring_n = "auto"`
  (smallest dimension whose derived `q` meets `RING_LWE_128_LOG_Q`, a
  lattice-estimator table read at the smallest RLWE width of the scheme's
  proof, re-checked after derivation, flooding included)
  and the BN254 pairing group (about 100-bit, with a `FESecurityWarning`).
  The ABMPSW20 MCFE-LWE family (`mcfe/lwe`, `fullysec_lwe`, `lwe_threshold`,
  `lwe_decentralized`) gets `lwe_n = "auto"`: the smallest dimension whose
  derived `q` meets `lwe_utils.DENSE_LWE_128_LOG_Q`, a lattice-estimator
  table for LWE(n0, q, ceil(2 sqrt(n0))) with a uniform secret (the
  assumption of the scheme's proof), regenerated by
  `tools/estimate_dense_lwe.py`; under `standard` its decentralised variant
  combines key shares under a DCR modulus of at least 3072 bits.
  Explicit values below the profile, and the schemes without a usable
  secure parameter set (SIFE / MIFE `fullysec_lwe` and dMCFE-Paillier),
  raise `FEValidationError`.
- ABDP15 LWE IPFE (`sife/lwe`, `mife/lwe`) under `standard`: `lwe_n = "auto"`
  picks the smallest dimension whose `(q, sigma)` meets
  `lwe_utils.WIDE_LWE_128_LOG_RATIO`, a lattice-estimator table of the
  largest `log2(q / sigma)` per dimension and error width for the LWE problem
  of the public key `A S + E` (regenerated by `tools/estimate_wide_lwe.py`);
  explicit dimensions are checked against it. New config key `matrix_mode`
  (key generators and crypto objects; forwarded by the ML adapter):
  `"memory"` keeps the expanded public matrix (n = 1088: 0.1 s per
  encryption, 2.7 GB), `"stream"` expands it for every batch (3.4 s, 0.5 GB).
  `pyfe4ai.utils.lwe_matrix` (`SeededMatrix`, `LimbMatrix`),
  `pyfe4ai.schemes.abdp15_lwe`, `sampling_utils.rand_bit_matrix`.
  tMIFE-LWE gets the same `lwe_n = "auto"` at its flooding-enlarged `q`
  (about 6000 for `X = Y = 50` with `stat_param = 128`; the table covers
  dimensions up to 6144), and `standard` fixes its `matrix_mode` to
  `"stream"` (its `A` would take more than 400 GB in memory): key generation
  38 minutes, one encryption 572 s, 10.6 s each in batches of 64. `"memory"` mode refuses a matrix larger than the
  machine's physical memory.
- `sife/fullysec_ring_lwe`: the adaptively secure Ring-LWE IPFE of BMMS21
  (Bermudo Mera, Karmakar, Marc, Soleimanian, PKC 2022, Sec. 5), the ring
  counterpart of ALS16. The paper takes `m = 3` key polynomials, but under
  its own constraints (Appendix C.1: the Ring-LHL width `sigma'` grows like
  `q^(2/m)`, the decryption noise like `q^(4/m)`) no modulus exists for
  `m <= 4`; `m = "auto"` takes the smallest `m >= 5` that admits one (within
  the 128-bit Ring-LWE bound under `standard`). Widths follow Theorems 1 and 2
  and Appendix C.1 for signed inputs, with complexity leveraging over
  `(2X + 1)^(2 eta)`; one vector per ciphertext. `standard`: eta = 10,
  X = Y = 50 gives n = 8192, m = 8 (encrypt 2.4 s, decrypt 0.5 s).
  `ring_lwe_utils.bmms21_adaptive_parameters`, `bmms21_adaptive_log2_q`,
  `bmms21_adaptive_choice`.
- `lwe_n = "auto"` for the MCFE-LWE family; `lwe_utils.DENSE_LWE_128_LOG_Q`,
  `check_dense_lwe`, `dense_lwe_n`, `resolve_lwe_n`;
  `profiles.STANDARD_DCR_PRIME_BITS`; `modular_utils.pow_signed_many`
  (fixed-base exponentiation of one base to many exponents).
- `ring_n = "auto"`; `ring_lwe_utils.RING_LWE_128_LOG_Q` (regenerated by
  `tools/estimate_ring_lwe.py`), `ring_lwe_bound`, `check_ring_lwe`,
  `ring_lwe_auto_n`, `resolve_ring_n`, `bmms21_selective_parameters`;
  `crypto_utils.RFC3526_PRIMES`.
- `pyfe4ai.utils.validation.int_vector`: scheme inputs must be integers
  (integral floats accepted); non-integral values raise
  `FEValidationError` instead of being truncated.
- `pyfe4ai.utils.ring_lwe_utils.bmms21_parameters`, `check_ring_n` and
  `DEFAULT_RING_N`; `FHIPE.keygen(..., r=...)`.
- `dlog_bound` crypto option for SIFE-DDH, MIFE-DDH, MCFE-DDH and
  tMCFE-DDH (results beyond the default `10^(precision + 2)` decrypt to
  `None`).
- `pyfe4ai.schemes.status` (`SchemeStatus`, `get_scheme_status`, exported
  from `pyfe4ai`): status (`reviewed` / `experimental` / `unreviewed` /
  `known-insecure`), fidelity, reference and issue for all registered
  schemes; key generators and crypto objects of known-insecure schemes emit
  `pyfe4ai.FESecurityWarning`; `FESchemeWrapper.status`;
  `markdown_table()` renders the README table, which a test keeps in sync.
- `XofRandom` (SHAKE-256 bit source for derandomised sampling) and an `rng`
  argument for `discrete_gaussian_bigint`.
- `discrete_gaussian_bigint` (arbitrary-width rounded Gaussian),
  `hash_to_zn2_star`, `hash_to_zq_vector`, `abmpsw20_lwe_parameters`,
  `round_to_q0`, `decode_rounded_inner_product`.
- Config keys `stat_param` (ABMPSW20 schemes, SIFE-Paillier; default 64) and
  `mddh_k` (MCFE-Damgard).
- `pyfe4ai.utils.pairing_backend.check_asymmetric` (detects symmetric
  groups by pairing two `G1` elements) and `DEFAULT_PAIRING_GROUP`.
- `pyfe4ai.schemes.mcfe.fh_ipe` (`FHIPE`),
  `pyfe4ai.schemes.mcfe.decentralize` (MCFE2 and zero-sum keys for
  Fig. 14), `labels.chunk_label`, `MCFERingLWE.key_chunks` (the ML adapter
  passes the resulting `n_chunks` when requesting Ring-LWE keys).
- `tag_f` argument of the dMCFE-FH-Multi-IPE key derivation.

### Changed (breaking)

- tMCFE-DDH: `share_decrypt(dict_ct, credentials, dk, lst_sid_enrolled,
  label)`, `compute_lst_ndarray_ct(..., lst_sid_enrolled, label)` and
  `decrypt_lst_ndarray_ct(partials, label)` take the label to decrypt;
  `combine_decrypt(partials, label=None)` checks it when given.
- Exports and registry: `HybridAlphaMIFE` and `HybridAlphaMIFEKeyGenerator`
  are removed (see *Removed*); `MultiInputQuadraticSGP` and
  `MultiInputQuadraticSGPKeyGenerator` are replaced by `MultiInputQuadratic`
  and `MultiInputQuadraticKeyGenerator`, the ML adapter's registry key
  `("quadratic", "multi_input_sgp")` by `("quadratic", "miqfe")` and
  `config_types.MultiInputSGPConfig` by `MultiInputQuadraticConfig`. The
  static method `ThresholdMIFE.L` is removed (use
  `threshold_utils.lagrange_coefficient_at_zero`).
- tMCFE-FH-Multi-IPE: key shares hold `k` as `G2` elements and `z` as a `GT`
  element (serialised) instead of integers, plus `mac_keys`; client keys
  carry `mac_keys`; ciphertexts are `{ct, label, tags}`;
  `share_decrypt(dict_ct, credentials, dk, lst_sid_enrolled, label)`,
  `compute_lst_ndarray_ct(..., lst_sid_enrolled, label)`,
  `decrypt_lst_ndarray_ct(partials, label)`, `combine_decrypt(partials,
  label=None)`; partial decryptions carry `label`.
- tMCFE-Ring-LWE: client keys and key shares carry `mac_keys`, ciphertexts
  and ndarray chunks `tags`.
- dMCFE-FH-Multi-IPE: `derive_function_decryption_key_share(weights, tag_f)`
  takes this client's weights (or a dict, of which only its entry is used)
  and a required non-empty `tag_f`.
- tMIFE-LWE: client keys carry `mac_keys` (`{sid: key}`), key shares
  `mac_keys` (`{nid: key}`) and ciphertexts `tags` (`{sid: tag}`);
  ciphertexts without valid tags are refused.
- MCFE-Paillier: `sk = {s}`, `dk = {k}`, `ct = {c}`. MCFE-LWE /
  FullySec-LWE: `sk = {z}`, `dk = {sk_y}`, `ct = {c}`; the config keys
  `bound_u` and `label_modulus` are ignored and parameters are no longer
  cached in `./config`. MCFE-Damgard: `sk = {s}`, `dk = {d}`, `ct = {c}`, no
  `h` in the public parameters.
- tMCFE-DDH: `sk = {s}`, `dk = {d, sid, lst_sid}` (label-independent),
  `ct = {c, label}`, partial decryption `{c_y, u_d, label, sid,
  lst_sid_enrolled}`; key shares also carry `weights_id`. tMCFE-LWE:
  `dk = {sk_y, sid, lst_sid, weights_id, noise_key}`, partial decryption
  `{w, c_y, label, sid, lst_sid_enrolled}`, validated like the DDH threshold
  scheme. The `label` fields hold the encoded label.
- The ndarray helpers of all rebuilt MCFE schemes encrypt each element
  under its own `ElementLabel` (one ciphertext per slot and label); labels
  are hashed through `encode_label`, which also changes MCFE-DDH
  ciphertexts relative to v0.2.0.
- Decrypting under a wrong label returns `None` for MCFE-Paillier and
  MCFE-Damgard and an arbitrary in-range value for the LWE schemes.
- MIFE Damgard / Paillier / LWE / FullySec-LWE / Ring-LWE: client keys
  carry the pad `u`; functional keys carry `z` (`{k1, k2, z}`, `{k, z}`,
  `{sk_y, z}`); decryption needs a ciphertext for every slot of the key.
  tMIFE-LWE key shares carry `noise_key`.
- dMCFE-DDH / -LWE / -Paillier: key shares are MCFE2 ciphertexts
  `{c, nid, weights_id}`, every client must contribute (with zero weights
  if absent), and the combined key is the MCFE1 key.
- MCFE / dMCFE / tMCFE-Ring-LWE: functional keys are bound to a label
  (`credentials` need `label`, optionally `n_chunks`;
  `derive_function_decryption_key_share(fusion_weight, label, n_chunks)`);
  ndarray ciphertexts pack values into chunks of `ring_n` coefficients.
- FH-MCFE family: client keys `{imsk}` (dMCFE adds `t_tilde`, `seed`),
  functional keys `{dk: {nid: {k, z}}, weights_id}`; tMCFE key shares
  `{k, z, sid, lst_sid, weights_id}` and partial decryptions
  `{w, ct_digest, sid, lst_sid_enrolled}`. The config keys `sec_level`,
  `u_bound` and `label_modulus` are ignored and parameters are no longer
  cached.
- Decryption keys of the SIFE schemes (DDH, LWE, Damgard, FullySec-LWE,
  Ring-LWE) carry their weights `y`, and decryption raises
  `FESchemeError` for other weights; MIFE/FH keys carry `weights_id`.
  Ciphertext lengths are checked.
- SIFE-FullySec-LWE: `K = 2 eta X Y + 1`, encoding `floor(q / K) x`
  (caches carry a construction tag and are regenerated).
- SIFE-Ring-LWE / MIFE / MCFE: `ring_n` defaults to 32 (it was claimed to be
  derived but always was 32) and must be a power of two.
- MIFE-DDH: client keys `{h, u}` (no secret slot key), `ct = {t, c}`, keys
  must cover every slot. tMIFE-DDH: shares `{d, z, sid, lst_sid,
  weights_id}`, partial decryptions `{c_y, w, ct_digest, sid,
  lst_sid_enrolled}`. MIFE FH-IPE: `sk = {imsk}`, `ct = {ct}`,
  `dk = {dk, z, weights_id}`, no parameter cache. Quad: `ct = {ct_x, ct_y,
  ct}`, `dk = {f, k}`, no parameter cache.
- SIFE FH-IPE ciphertexts and keys have `eta + 2` components.
- Pairing-based schemes: `pairing_group_param` defaults to `MNT224`;
  `SS512` and `SS1024` raise `FEValidationError`. Parameters, keys and
  ciphertexts created over `SS512` cannot be reused (cached parameters are
  regenerated).

### Changed

- SIFE-LWE, MIFE-LWE and tMIFE-LWE publish the matrix `A` as a 32-byte seed
  (`pp["A_seed"]` replaces the `m x lwe_n` digit matrix `pp["A"]`), expanded
  with SHAKE-128, and encrypt with exact numpy products over 31-bit limbs
  (whole batches in the ndarray helpers); `q` is the largest prime with the
  derived bit length. Public parameters, keys and cached parameters are not
  compatible with earlier versions.
- `hash_to_zq_vector` cuts one SHAKE-256 stream instead of calling SHAKE-256
  once per coordinate (hashing a label of MCFE-LWE at n0 = 8192 drops from
  about 12 ms to 3.6 ms). Its outputs change, so ciphertexts of `mcfe/lwe`,
  `fullysec_lwe`, `lwe_threshold`, `lwe_decentralized`, `mcfe/ring_lwe` and
  `ring_lwe_decentralized` are not compatible with earlier versions.
- MCFE2 key shares of the decentralised MCFE schemes (`ddh_decentralized`,
  `lwe_decentralized`, `paillier_decentralized`) raise one label hash to all
  key coordinates with a fixed-base method (about 4 times faster at 3072
  bits; same outputs).
- Safe primes are generated by sieving candidates `q` and `2q + 1` against
  the primes below 2^16 before the primality tests: a 1536-bit safe prime
  takes seconds instead of tens of seconds (all Paillier setups and the
  dMCFE modulus benefit). The tests themselves are unchanged.
- `group_generator_fe` (DL schemes) returns the RFC 3526 group for 2048- and
  3072-bit requests instead of generating a fresh safe prime; smaller
  sizes, including the default, are generated as before.

### Removed

- `quadratic/multi_input_sgp` (replaced by `quadratic/miqfe`, see above).
- `mife/ddh_hybrid_alpha` (HybridAlpha): it duplicated MIFE-DDH without the
  paper's inference-prevention filter, and the published design leaks
  individual updates (MIFE without labels; a stateless filter that allows
  keys for `S` and `S` minus one client).
- `dlog_solver.PairingDLogCacheMixin`, which had no users (it trusted a
  dlog table read from `./config/crypto`), and the unused helpers
  `crypto_utils.md5_hash` and `group_generator_paillier`.
- The `jsonpickle` dependency (never imported).
- The unused modules `schemes/lwe_base.py`, `schemes/ring_lwe_base.py` and
  `schemes/paillier_base.py` and the helpers `label_scalar_from_hash` and
  `decode_fullysec_lwe_inner_product`.
- The repository-root `config.yaml`, an unused copy of the bundled
  `pyfe4ai/config.yaml`.

### Build

- `Dockerfile.pairing`: the deprecated `continuumio/miniconda3` base is
  replaced by `condaforge/miniforge3:26.5.3-0`, pinned by digest; the PBC
  1.0.0 archive is checked against its SHA-256 before it is built; the image
  runs as an unprivileged user. `environment.yml` uses the `conda-forge`
  channel.

### Fixed

- `benchmarks/benchmark_ndarray_helpers.py` ran every group before applying
  `--group`, so `--group sife` failed without `charm-crypto`, and its warm-up
  rounds ran after the timed ones. It now runs the warm-up and then only the
  selected groups.
- The ndarray helpers truncated `x 10^precision` toward zero (`0.57` became
  `0.56` at precision 2) and turned NaN and infinities into `-2^63`. They now
  round to nearest and refuse non-finite or too large values
  (`validation.scale_to_int`).
- Ring-LWE MIFE / MCFE / dMCFE and FH-MCFE keys truncated fractional weights
  (`0.5` became `0`), and `threshold_utils.fusion_weight_id` gave `1.5` and
  `1` the same digest; weights now go through `validation.int_vector`.
  Ring-LWE MIFE / MCFE plaintext rows of different lengths were cut or
  padded silently and are now refused; so are ciphertexts of the wrong shape
  (short polynomials were truncated by `zip`) and ndarray layers whose
  `first_chunk` is negative or beyond the key (a negative index picked chunk
  keys from the end of the list).
- MIFE-LWE sized its noise for `eta`-dimensional vectors, but decryption adds
  the noise of all `n` slots, so the decryption margin shrank with
  `sqrt(n)`. As in ACFGU18 (footnote 14), the ALS16 / ABDP15 parameters are
  now chosen as if the vectors had dimension `n * eta`; cached parameters are
  regenerated.
- The ABDP15 plaintext modulus `p` was random in its bit range, so
  `lwe_n = "auto"` could pick different dimensions on different runs (the
  README's `n = 832` for `eta = 1` was one such draw; the deterministic
  choice is 768).
  `p` is now the smallest prime of that size.
- SIFE / MIFE FullySec-LWE (ALS16) used `m = 1.01 n log2 q`; ALS16 Theorem 2
  needs `m >= 4 n log2 q`. The recipe now uses `4 n log2 q` (also inside
  the width derivation), sizes the row norm `B_tau` of `tau` with `sqrt(m)`
  instead of `sqrt(n)` (with the larger `m`, the old bound let about 0.1% of
  extreme decryptions fail at toy sizes), uses at least the 12-sigma tail
  cut for its `omega(sqrt(log n))` (`log2 n` alone left dimensions below
  4096 a few sigma of margin) and takes the largest prime of the
  derived size (a random draw could exceed it by one bit); cached parameters
  are regenerated. The `standard` refusal now quotes the resulting sizes
  (`n = 3328` to `4352`, 80 to 145 MB per ciphertext).
- `sife/fullysec_ring_lwe` with an explicit `ring_n` and `m = "auto"` under
  `standard` took `m = 5`, whose `q` exceeds the bound, and refused; it now
  takes the smallest `m` that meets the bound.
- SIFE-Paillier truncated fractional plaintexts and weights (`[1.7, 2.9]`
  encrypted as `[1, 2]`, a weight `0.5` became `0`); the same `abs(int(v))`
  checks were in MIFE/MCFE LWE, Paillier and Damgard, dMCFE-LWE /
  -Paillier and partial FH-IPE. They now refuse non-integral values
  (`validation.strict_int`), and so do MCFE-DDH key derivation, encryption
  and decryption (`gp.mpz(2.5)` was 2); rationals are checked exactly
  (`Fraction(2^60 + 1, 2)` passed through a float). `scale_to_int` refuses a
  negative precision;
  a key generator config that is not a `Mapping` is refused (it could skip
  the security profile).
- MCFE-DDH and tMCFE-DDH put the plaintext into the message of the
  exception for a malformed plaintext, and `int_vector` the offending value;
  messages now name only the problem (and the type).
- SIFE FH-IPE decryption required the weights `y` although its keys hide
  them; `fusion_weight` is now optional (checked if given).
- The logging configuration is read with `yaml.safe_load` semantics
  (`SafeLoader`); dependencies have lower bounds (`gmpy2>=2.2` for
  `prev_prime`, `numpy>=1.22`, `PyYAML>=5.4`).
- The ndarray combiners of the threshold schemes crashed with a `TypeError`
  (or returned NaN, tMCFE-LWE) when an element's partial decryptions
  disagreed, and took the output shape from the first server; they now
  check that all servers' outputs have one structure (lists of arrays) and
  raise `FESchemeError` for malformed outputs and for an element that does
  not decrypt (`threshold_utils.combine_elementwise`).
- The ndarray helpers of the other schemes did the same for an element that
  does not decrypt (result outside the decryption bound, or key and
  ciphertext do not match): a `TypeError` in most schemes, a silent NaN in
  MCFE-Ring-LWE, MIFE FH-Multi-IPE, `quad` and `sgp`. They now raise
  `FESchemeError` naming the element (`ipfe.decoded_layer`), as
  `ml_adapter.compute_linear` does for a result that does not decrypt.
- SIFE-Paillier `decrypt_lst_ndarray_ct` decoded inline without the checks
  of `decrypt` (ciphertexts in `Z_{N^2}^*`, result in range), so a wrong key
  gave wrapped values instead of an error; it now calls `decrypt` per
  element.
- ML adapter: `aggregate_gradients` on a SIFE wrapper returned the first
  client's result when given several; it now refuses more than one. Fractional
  weights were quantised but the result was not scaled back (weights `0.5`
  gave `10^precision` times the sum); the result is now rescaled, and
  fractional weights without a precision, or not representable at it, are
  refused. The policy does not stop very unequal weights, which separate
  two clients' gradients in one aggregate; this is now documented.
  `EncryptedGradient.from_dict` allocated whatever shape it was given; it now
  bounds the shape and requires one entry per element.
- SIFE-FullySec-LWE decoded `+-eta X Y` with a random sign (`K` was
  `2 eta X Y`). SIFE-Damgard could not decrypt in an instance holding the
  encryption key. SIFE-LWE's ndarray helper skipped the `bound_x` check.
  The MIFE Ring-LWE key generator overwrote the SIFE parameter cache.
- Ring-LWE plaintext modulus (SIFE, MIFE, MCFE): `p` was exactly
  `2 l X Y`, so inner products of `+-l X Y` were ambiguous; it is now
  `2 l X Y + 2` (parameter caches are regenerated).
- Safe primes below 64 bits could have one bit too many (`next_prime`
  stepped past the size, about 1% of 16-bit draws), which made a test fail
  intermittently.
- Key generators accepted `n` and a `lst_nid` of another length; bounds,
  noise widths and moduli were then sized for `n` slots (MCFE-LWE with
  `n = 2` and 10 clients decrypted 160 instead of 810). They now raise
  `FEValidationError`, as threshold schemes do for repeated server ids.

### Documentation

- README: *Scheme Status* table, rewritten *Known Issues*, the scheme table
  cites the implemented constructions (ACFGU18 Fig. 4, ABMPSW20, BMMS21
  Sec. 6, AGT21b, NPS24, ACFGU18 Fig. 12, Gay20 Fig. 4); the ALS16 link
  pointed to a different paper (ePrint 2016/011, now 2015/608); landing
  page matrix updated; the docs index no longer claims a uniform
  `setup / keygen / encrypt / decrypt` API.
- README and docs: the usage example's advice (adding `standard` next to an
  explicit `sec_param = 128` is refused), `sife/fullysec_ring_lwe` in the
  scheme table and features, the landing page's scheme count (39), the size
  of the dMCFE-DDH modulus (about 6410 bits with the carry mask), the docs index's group size
  (3072 bits); a Sphinx page for `pyfe4ai.utils.validation`; `config_types`
  lists `stat_param`, `matrix_mode`, `m`, `lwe_n = "auto"` and the crypto
  config's `dlog_bound`; the benchmarks README lists `benchmark_ntt.py`.
- Tests: a bare `pytest` from the repository root works without an install
  (`pythonpath`); the core lane no longer collects pairing tests
  (`test_fh_ipe.py`, `test_pairing_groups.py`); the ACFGU18 pad test no
  longer fails when a padded slot hits the plain value by chance (about 0.3%
  of runs for MIFE FullySec-LWE); the README status table is compared in full
  with `status.markdown_table()`.
- Docs site: a *Security & Status* page whose scheme table and counts are
  rendered from `pyfe4ai.schemes.status` at build time; `GLOSSARY.md`, which
  was never rendered and named config keys that no longer exist
  (`poly_degree`, `group`), is now `glossary.rst`, corrected and extended
  (`security_profile`, `stat_param`, `matrix_mode`, `label`, `tag_f`,
  `mac_keys`); every class is documented once instead of twice; the
  ML-adapter and quantization docstrings render (they were NumPy style),
  as do 16 over-indented `Args:` sections and 57 bullet items; placeholder
  docstrings are replaced; nitpicky build warnings went from 171 to 0.
- Landing page: experimental schemes are marked in the matrix, MIQFE is
  listed, the ML-adapter card no longer claims to wrap any scheme, a
  security-profile card, the toy-parameter warning in the snippet, how to
  install `charm-crypto`, social-preview tags, a navigation bar that fits
  phones, and WCAG AA contrast.
- README, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue and
  PR templates: vulnerabilities are reported through GitHub private
  vulnerability reporting; the prerequisites of `charm-crypto`; a checklist
  for adding a scheme; LWE dimensions default to 32–64 (not 16–64); the
  examples and benchmarks READMEs list every script and say which need
  `charm-crypto`.

## [0.2.0] — 2026-10-07

Correctness and security fixes. **Not compatible with v0.1.0 keys or
ciphertexts for MCFE-DDH and the DDH threshold schemes; ciphertexts of the
label-based LWE / Ring-LWE / FH-Multi-IPE MCFE schemes produced by v0.1.0 do
not decrypt correctly with v0.2.0 (label scalar mapping changed).** Several multi-client
schemes still have known privacy issues — read
[Known Issues / Security Notice](README.md#known-issues--security-notice)
before using them.

### Security

- **MCFE-DDH** (`pyfe4ai.MCFE`) re-implemented following CDGPP18: the label is
  hashed to a vector of group elements (SHAKE-256 into the quadratic-residue
  subgroup), every client slot has an independent key, and the functional key
  is the aggregate `d = Σ y_i · s_i`. Previously the label entered as a public
  MD5 scalar and keys carried per-client components, so a functional-key holder
  could recover an individual client's plaintexts. Because CDGPP18 ciphertexts
  are deterministic per label, the ndarray helpers derive a distinct label per
  element.
- **MIFE-DDH**: every input slot now has its own IPFE master key; previously all
  clients shared `w` / `g_a` and one client could unmask another's ciphertext.
- **DDH threshold** (tMIFE-DDH, tMCFE-DDH): Shamir share points start at 1 and
  sharing is done over the group order (point 0 used to be the secret itself,
  so server `sid_0` alone could decrypt); share and combine require at least
  `t` shares from the enrolled set, reject duplicate or unknown partial
  decryptions, and validate `1 ≤ t ≤ s`.
- Discrete-log tables for integer groups are built in memory and no longer
  loaded from disk (the pickle-based cache in the working directory could run
  arbitrary code).
- LWE-family label scalars never map to 0 (about 1 in `label_modulus` labels,
  12.5% with the default of 8, used to remove the mask entirely). The structural leakage of LWE label masking remains; see
  Known Issues.

### Fixed

- MCFE-DDH decryption for `eta ≥ 2` (returned `None`).
- DDH threshold Lagrange coefficients are computed modulo the group order
  instead of with float division and truncation (enrolled subsets with
  non-integer coefficients, e.g. `{sid_1, sid_3}`, failed); `setup` no longer raises `IndexError` and combining no longer
  hard-codes divisors for `eta ≥ 2`.
- `get_decryption_keys` of the five decentralized MCFE schemes raises
  `NotImplementedError` instead of returning the exception class; the ML
  adapter reports unsupported threshold/decentralized variants clearly.
- Pairing and quadratic test modules skip cleanly when `charm-crypto` is not
  installed.

### Changed (breaking)

- MCFE-DDH formats: `sk = {"s"}`, `dk = {"d": [d0, d1]}`, `ct = {"c"}`.
- MCFE-DDH ndarray helpers use per-element derived labels
  (`json.dumps([label, layer, index])`).
- MCFE-DDH `get_decryption_keys` raises `FEValidationError` on an unknown
  client id instead of silently skipping it; `decrypt` validates the length of
  `d`, the ciphertext range, and that ciphertext and weight lengths match.
- Threshold `L()` takes a `modulus` argument; partial decryptions carry `sid`
  and `lst_sid_enrolled`, which `combine_decrypt` validates.
- LWE label scalar mapping changed from `h mod M` to `(h mod (M − 1)) + 1`
  (centered, never 0), so v0.1.0 ciphertexts of the label-based LWE /
  Ring-LWE / FH-Multi-IPE MCFE schemes no longer decrypt; keys are unaffected.
  `label_modulus < 2` raises `FEValidationError`.
- `dlog_solver.load_or_build_dlog_table` ignores its `filepath` argument:
  integer-group tables are no longer read from or written to `./config/crypto`,
  and existing cache files there are ignored.

### Added

- `benchmarks/fl_path_coverage.py`: enumerates federated-learning path support
  over all 39 registry entries (`--output json` available).
- Regression tests for `eta ≥ 2`, `t < s` subsets that exclude `sid_0`,
  single-server rejection, duplicate shares, independent client keys, label
  binding, and the dlog cache.
- README *Known Issues / Security Notice*.

### Documentation

- `sec_param` is documented as a modulus bit length, not a security level.
- README bibliography: TAPFed DOI, venues for Gay20, DOT18 and the Ring-LWE
  construction, complete list of pairing-dependent modules.
- Landing page aligned with the Known Issues.

## [0.1.0] — 2026-08-05

Initial public release.

[Unreleased]: https://github.com/spire-studio/pyfe4ai/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/spire-studio/pyfe4ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/spire-studio/pyfe4ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/spire-studio/pyfe4ai/releases/tag/v0.1.0
