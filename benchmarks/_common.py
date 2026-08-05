from __future__ import annotations

import json
import random
import statistics

try:
    import numpy as np
except Exception:  # pragma: no cover - benchmark helper fallback
    np = None


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
    return len(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


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
    return {key: result.get(key, "") for key in header}


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
