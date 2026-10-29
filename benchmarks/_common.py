from __future__ import annotations

import base64
import contextlib
import json
import os
import random
import re
import statistics
import sys
import tempfile

try:
    import numpy as np
except Exception:  # pragma: no cover - benchmark helper fallback
    np = None


#: ``--security-profile`` of the running benchmark (see :func:`parse_args`).
_SECURITY_PROFILE = "toy"

#: Size parameters the benchmarks set for the toy profile; under another
#: profile they are left out so that the profile chooses them.
_TOY_SIZE_KEYS = (
    "sec_param", "modulus_length", "bit_length", "lwe_n", "ring_n",
    "pairing_group_param", "stat_param",
)


def parse_args(parser, argv=None):
    """Add ``--security-profile``, parse the command line and select the profile."""
    parser.add_argument(
        "--security-profile", choices=["toy", "standard"], default="toy",
        help="key generator security_profile; 'standard' replaces the toy sizes "
        "(sec_param, modulus_length, bit_length, lwe_n, ring_n, pairing group, "
        "stat_param) by the profile's and skips the schemes it refuses",
    )
    args = parser.parse_args(argv)
    set_security_profile(args.security_profile)
    return args


def set_security_profile(profile: str) -> None:
    """Select the profile of :func:`keygen_config` and :func:`run_cases`, for
    a command that parses its own ``--security-profile``."""
    global _SECURITY_PROFILE
    _SECURITY_PROFILE = profile


def keygen_config(config: dict) -> dict:
    """Key generator config for the selected security profile."""
    if _SECURITY_PROFILE == "toy":
        return config
    out = {k: v for k, v in config.items() if k not in _TOY_SIZE_KEYS}
    out["security_profile"] = _SECURITY_PROFILE
    return out


def run_cases(thunks) -> list[dict]:
    """Run benchmark callables (each returning a row or a list of rows),
    skipping the schemes the selected security profile refuses."""
    from pyfe4ai.core.exceptions import FEValidationError

    rows = []
    for thunk in thunks:
        try:
            result = thunk()
        except ImportError as exc:  # pairing-based scheme without a backend for its group
            if "charm" not in str(exc) and "pairing" not in str(exc):
                raise
            print("skipped: {}".format(str(exc).splitlines()[0]), file=sys.stderr)
            continue
        except FEValidationError as exc:
            if _SECURITY_PROFILE == "toy" or "parameter set" not in str(exc):
                raise
            print("skipped under '{}': {}".format(_SECURITY_PROFILE, exc), file=sys.stderr)
            continue
        rows.extend(result if isinstance(result, list) else [result])
    return rows


BENCHMARK_HEADER = [
    "scheme",
    "profile",
    "setup_mean_sec",
    "keygen_mean_sec",
    "client_key_export_mean_sec",
    "encrypt_mean_sec",
    "share_mean_sec",
    "dkgen_mean_sec",
    "partial_decrypt_mean_sec",
    "combine_mean_sec",
    "combine_decrypt_mean_sec",
    "decrypt_mean_sec",
    "compute_mean_sec",
    "end_to_end_mean_sec",
    "end_to_end_median_sec",
    "end_to_end_std_sec",
    "shape",
    "layers",
    "output_elements",
    "elements_per_sec",
    "pp_total_bytes",
    "sk_total_bytes",
    "client_sk_total_bytes",
    "client_sk_avg_bytes",
    "dk_total_bytes",
    "combined_dk_total_bytes",
    "share_bundle_total_bytes",
    "share_bundle_avg_bytes",
    "partial_bundle_total_bytes",
    "partial_bundle_avg_bytes",
    "ct_total_bytes",
    "ct_avg_bytes",
]


def size_bytes(obj: object) -> int:
    """Size of *obj* as JSON, bytes values (binary formats) counted at their length."""
    raw = []

    def default(value):
        if isinstance(value, (bytes, bytearray)):
            raw.append(len(value))
            return ""
        return str(value)

    return len(json.dumps(obj, sort_keys=True, default=default).encode("utf-8")) + sum(raw)


def size_summary(values: list[int]) -> tuple[int, int]:
    if not values:
        return 0, 0
    total = int(sum(values))
    avg = int(round(total / len(values)))
    return total, avg


def stats(values: list[float]) -> tuple[float, float, float]:
    mean_v = statistics.mean(values)
    median_v = statistics.median(values)
    std_v = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_v, median_v, std_v


def profile_value(profile: str, mapping: dict[str, object]):
    if profile not in mapping:
        raise ValueError(f"unsupported profile: {profile}")
    return mapping[profile]


def resolve_rounds(default_rounds: int, profile: str, mapping: dict[str, int], override: int | None) -> int:
    if override is not None:
        return override
    if profile in mapping:
        return mapping[profile]
    return default_rounds


def resolve_shape(default_shape: tuple[int, int], profile: str, mapping: dict[str, tuple[int, int]], override: tuple[int, int] | None) -> tuple[int, int]:
    if override is not None:
        return override
    if profile in mapping:
        return mapping[profile]
    return default_shape


def prepare_benchmark(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def normalize_row(result: dict[str, object], header: list[str] | None = None) -> dict[str, object]:
    if header is None:
        header = BENCHMARK_HEADER
    row = {key: result.get(key, "") for key in header}
    if _SECURITY_PROFILE != "toy" and row.get("profile"):
        row["profile"] = "{}/{}".format(row["profile"], _SECURITY_PROFILE)
    return row


def emit_results(results: list[dict[str, object]], output: str = "csv", header: list[str] | None = None) -> None:
    if header is None:
        header = BENCHMARK_HEADER
    rows = [normalize_row(result, header) for result in results]
    if output == "json":
        print(json.dumps(rows, indent=2, sort_keys=False, default=str))
        return

    print(",".join(header))
    for row in rows:
        values = []
        for key in header:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        print(",".join(values))


_DECIMAL = re.compile(r"^-?[0-9]+$")
_CHARM = re.compile(r"^[0-9]:[A-Za-z0-9+/]+=*$")


def compact_bytes(obj) -> int:
    """Payload of a ciphertext in a compact binary encoding.

    Counts every integer (also decimal strings, as the schemes serialise them)
    at ``ceil(bits / 8)`` bytes, every charm group element (``"t:base64"``) at
    its compressed encoding, other strings at their UTF-8 length, and nothing
    for keys, lists and dicts. A lower bound for a wire format without
    framing; ``size_bytes`` gives the JSON size.
    """
    if isinstance(obj, dict):
        return sum(compact_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(compact_bytes(v) for v in obj)
    if np is not None and isinstance(obj, np.ndarray):
        return sum(compact_bytes(v) for v in obj.ravel())
    if isinstance(obj, bool) or obj is None:
        return 1
    if isinstance(obj, int):
        return max(1, (abs(obj).bit_length() + 7) // 8)
    if isinstance(obj, float):
        return 8
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, str):
        if _DECIMAL.match(obj):
            return max(1, (abs(int(obj)).bit_length() + 7) // 8)
        if _CHARM.match(obj):
            return len(base64.b64decode(obj[2:]))
        return len(obj.encode("utf-8"))
    try:  # gmpy2 mpz and other integer-like values
        return max(1, (abs(int(obj)).bit_length() + 7) // 8)
    except (TypeError, ValueError):
        return len(str(obj).encode("utf-8"))


@contextlib.contextmanager
def fresh_cache():
    """Run in a new temporary directory that is also the parameter cache
    (``PYFE4AI_CACHE_DIR``), so that no cached parameters are found; restores
    the working directory and the variable afterwards."""
    cwd, previous = os.getcwd(), os.environ.get("PYFE4AI_CACHE_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        os.environ["PYFE4AI_CACHE_DIR"] = tmp
        try:
            yield tmp
        finally:
            os.chdir(cwd)
            if previous is None:
                del os.environ["PYFE4AI_CACHE_DIR"]
            else:
                os.environ["PYFE4AI_CACHE_DIR"] = previous


def with_clients(config: dict, clients: int | None) -> dict:
    """*config* with ``clients`` clients (``n`` and ``lst_nid``), if given."""
    if not clients or "lst_nid" not in config:
        return config
    return dict(config, n=clients, lst_nid=["nid_{}".format(i) for i in range(clients)])
