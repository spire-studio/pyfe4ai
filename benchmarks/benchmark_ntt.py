"""Benchmark NTT vs naive polynomial multiplication in Ring-LWE.

Compares O(n²) ``poly_mul_negacyclic`` with O(n log n) ``poly_mul_ntt``
across increasing polynomial degrees.
"""

from __future__ import annotations

import argparse
import time

import gmpy2 as gp

from _common import emit_results, prepare_benchmark, stats
from pyfe4ai.utils.ring_lwe_utils import (
    _find_primitive_root,
    poly_mul_negacyclic,
    poly_mul_ntt,
)

# NTT-friendly primes:  q ≡ 1 (mod 2n)
# Each entry: (ring_n, q) where q-1 is divisible by 2*ring_n.
NTT_PARAMS: list[tuple[int, int]] = [
    (16, 97),       # 96 / 32 = 3
    (32, 193),      # 192 / 64 = 3
    (64, 257),      # 256 / 128 = 2
    (128, 769),     # 768 / 256 = 3
    (256, 7681),    # 7680 / 512 = 15
    (512, 12289),   # 12288 / 1024 = 12
    (1024, 12289),  # 12288 / 2048 = 6
]

ROUNDS = 5

HEADER = [
    "method",
    "ring_n",
    "q",
    "rounds",
    "mean_sec",
    "median_sec",
    "std_sec",
    "speedup",
]


def _random_poly(n: int, q: gp.mpz) -> list[gp.mpz]:
    import random
    return [gp.mpz(random.randint(0, int(q) - 1)) for _ in range(n)]


def bench_pair(ring_n: int, q_int: int, rounds: int) -> list[dict]:
    q = gp.mpz(q_int)
    root = _find_primitive_root(q, 2 * ring_n)
    assert root is not None, f"No NTT root for n={ring_n}, q={q_int}"

    a = _random_poly(ring_n, q)
    b = _random_poly(ring_n, q)

    # Warm up
    poly_mul_negacyclic(a, b, q)
    poly_mul_ntt(a, b, q, root)

    naive_times = []
    ntt_times = []

    for _ in range(rounds):
        t0 = time.perf_counter()
        poly_mul_negacyclic(a, b, q)
        naive_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        poly_mul_ntt(a, b, q, root)
        ntt_times.append(time.perf_counter() - t0)

    naive_mean, naive_median, naive_std = stats(naive_times)
    ntt_mean, ntt_median, ntt_std = stats(ntt_times)
    speedup = naive_mean / ntt_mean if ntt_mean > 0 else float("inf")

    return [
        {
            "method": "naive",
            "ring_n": ring_n,
            "q": q_int,
            "rounds": rounds,
            "mean_sec": naive_mean,
            "median_sec": naive_median,
            "std_sec": naive_std,
            "speedup": 1.0,
        },
        {
            "method": "NTT",
            "ring_n": ring_n,
            "q": q_int,
            "rounds": rounds,
            "mean_sec": ntt_mean,
            "median_sec": ntt_median,
            "std_sec": ntt_std,
            "speedup": speedup,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rounds", type=int, default=ROUNDS, help="timing rounds per (n, q) pair"
    )
    parser.add_argument(
        "--max-n", type=int, default=1024, help="max polynomial degree to test"
    )
    parser.add_argument("--output", choices=["csv", "json"], default="csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_benchmark(args.seed)

    results: list[dict] = []
    for ring_n, q_int in NTT_PARAMS:
        if ring_n > args.max_n:
            break
        results.extend(bench_pair(ring_n, q_int, args.rounds))

    emit_results(results, output=args.output, header=HEADER)


if __name__ == "__main__":
    main()
