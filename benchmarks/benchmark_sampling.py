"""Benchmark the Gaussian samplers: ``"rounded"`` (default) vs ``"facct"`` (exact).

Times ``--count`` samples per width (default 4096, one Ring-LWE polynomial)
with each sampler; ``--e2e`` also times one MCFE Ring-LWE round through the
ML adapter (10^4 elements, 3 clients, ``security_profile="standard"``).
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from unittest import mock

from _common import emit_results, prepare_benchmark, stats
from pyfe4ai.primitives.sampling import (
    discrete_gaussian_bigint,
    discrete_gaussian_matrix,
    gaussian_sampler,
)

# widths of the schemes: Ring-LWE sigma1 / sigma2 / sigma3 (aggregation.py
# parameters), a small LWE-style width, and a DCR-style big-integer width
WIDTHS = [3.2, 1.21e4, 1.49e10, 2.11e10, 2 ** 200]
SAMPLERS = ["rounded", "facct"]
ROUNDS = 5

HEADER = ["case", "sampler", "sigma", "count", "rounds", "mean_sec", "median_sec", "std_sec", "us_per_sample",
          "slowdown"]


def _draw(sigma, count: int, sampler: str):
    """The entry point the schemes use for this width."""
    if sigma > 2 ** 40:
        return discrete_gaussian_bigint(sigma, count, sampler=sampler)
    return discrete_gaussian_matrix(1, count, sigma, sampler=sampler)


def bench_widths(count: int, rounds: int) -> list[dict]:
    results = []
    for sigma in WIDTHS:
        base = None
        for sampler in SAMPLERS:
            _draw(sigma, 16, sampler)  # warm caches
            times = []
            for _ in range(rounds):
                t0 = time.perf_counter()
                _draw(sigma, count, sampler)
                times.append(time.perf_counter() - t0)
            mean, median, std = stats(times)
            base = base or median
            results.append({
                "case": "sample", "sampler": sampler, "sigma": float(sigma), "count": count, "rounds": rounds,
                "mean_sec": mean, "median_sec": median, "std_sec": std,
                "us_per_sample": median / count * 1e6, "slowdown": median / base,
            })
    return results


def bench_e2e(rounds: int) -> list[dict]:
    import numpy as np

    from pyfe4ai.ml import FESchemeWrapper, aggregate_gradients, encrypt_gradient

    clients = ["c0", "c1", "c2"]
    results, base = [], None
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        cache = mock.patch.dict(os.environ, {"PYFE4AI_CACHE_DIR": os.path.join(tmp, "config")})
        cache.start()  # parameter caches go to tmp/config
        try:
            for sampler in SAMPLERS:
                with gaussian_sampler(sampler):
                    wrapper = FESchemeWrapper("mcfe", "ring_lwe", {
                        "n": 3, "lst_nid": clients, "eta": 1, "bound_x": 100, "bound_y": 1,
                        "security_profile": "standard"}, precision=2)
                    rng = np.random.default_rng(0)
                    times = []
                    for r in range(rounds + 1):
                        grads = {c: rng.uniform(-1, 1, 10_000).round(2) for c in clients}
                        t0 = time.perf_counter()
                        enc = {c: encrypt_gradient(g, wrapper, nid=c, label="r%d" % r) for c, g in grads.items()}
                        elapsed = (time.perf_counter() - t0) / len(clients)
                        total = aggregate_gradients(enc, wrapper, {c: 1 for c in clients}, label="r%d" % r)
                        assert np.allclose(total, sum(grads.values()))
                        if r:
                            times.append(elapsed)
                mean, median, std = stats(times)
                base = base or median
                results.append({
                    "case": "mcfe/ring_lwe client encrypt (10^4)", "sampler": sampler, "sigma": "",
                    "count": 10_000, "rounds": rounds, "mean_sec": mean, "median_sec": median, "std_sec": std,
                    "us_per_sample": "", "slowdown": median / base,
                })
        finally:
            cache.stop()
            os.chdir(cwd)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--e2e", action="store_true", help="also time an MCFE Ring-LWE round")
    parser.add_argument("--output", choices=["csv", "json"], default="csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_benchmark(args.seed)
    results = bench_widths(args.count, args.rounds)
    if args.e2e:
        results.extend(bench_e2e(args.rounds))
    emit_results(results, output=args.output, header=HEADER)


if __name__ == "__main__":
    main()
