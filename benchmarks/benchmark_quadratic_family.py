"""Family benchmark for the repository's quadratic FE prototypes."""

from __future__ import annotations

import argparse
import time

from _common import BENCHMARK_HEADER
from _common import emit_results
from _common import prepare_benchmark
from _common import profile_value
from _common import size_bytes
from _common import stats
from pyfe4ai.schemes.quadratic.quad import QuadraticQuad
from pyfe4ai.schemes.quadratic.quad import QuadraticQuadKeyGenerator
from pyfe4ai.schemes.quadratic.sgp import QuadraticSGP
from pyfe4ai.schemes.quadratic.sgp import QuadraticSGPKeyGenerator

ROUNDS = 3

def _bench_sgp(cfg: dict, x: list[int], y: list[int], f_matrix: list[list[int]], rounds: int) -> dict:
    setup_times = []
    end_to_end_times = []
    pp_sizes = []
    sk_sizes = []
    dk_sizes = []
    ct_sizes = []

    for _ in range(rounds):
        kg = QuadraticSGPKeyGenerator(cfg)
        t0 = time.perf_counter()
        kg.setup()
        setup_times.append(time.perf_counter() - t0)
        pp = kg.get_public_parameters()
        sk = kg.get_private_keys()
        enc = QuadraticSGP({"id": "nid_default", "keys": {"pp": pp, "sk": sk}})
        dec = QuadraticSGP({"id": "sid_0", "keys": {"pp": pp}})

        t0 = time.perf_counter()
        ct = enc.encrypt({"x": x, "y": y})
        dk = kg.get_decryption_keys("sid_0", credentials={"function_matrix": f_matrix})
        result = dec.decrypt(ct, dk)
        end_to_end_times.append(time.perf_counter() - t0)
        expected = sum(x[i] * sum(f_matrix[i][j] * y[j] for j in range(len(y))) for i in range(len(x)))
        if result != expected:
            raise AssertionError("Quadratic/SGP benchmark produced an unexpected result")

        pp_sizes.append(size_bytes(pp))
        sk_sizes.append(size_bytes(sk))
        dk_sizes.append(size_bytes(dk))
        ct_sizes.append(size_bytes(ct))

    return {
        "scheme": "Quadratic/SGP",
        "setup_mean_sec": stats(setup_times)[0],
        "end_to_end_mean_sec": stats(end_to_end_times)[0],
        "end_to_end_median_sec": stats(end_to_end_times)[1],
        "end_to_end_std_sec": stats(end_to_end_times)[2],
        "pp_total_bytes": int(sum(pp_sizes) / len(pp_sizes)),
        "sk_total_bytes": int(sum(sk_sizes) / len(sk_sizes)),
        "dk_total_bytes": int(sum(dk_sizes) / len(dk_sizes)),
        "ct_total_bytes": int(sum(ct_sizes) / len(ct_sizes)),
    }


def _bench_quad(cfg: dict, x: list[int], y: list[int], f_matrix: list[list[int]], rounds: int) -> dict:
    setup_times = []
    end_to_end_times = []
    pp_sizes = []
    dk_sizes = []
    ct_sizes = []

    for _ in range(rounds):
        kg = QuadraticQuadKeyGenerator(cfg)
        t0 = time.perf_counter()
        kg.setup()
        setup_times.append(time.perf_counter() - t0)
        pp = kg.get_public_parameters()
        enc = QuadraticQuad({"id": "nid_default", "keys": {"pp": pp}})
        dec = QuadraticQuad({"id": "sid_0", "keys": {"pp": pp}})

        t0 = time.perf_counter()
        ct = enc.encrypt({"x": x, "y": y})
        dk = kg.get_decryption_keys("sid_0", credentials={"function_matrix": f_matrix})
        result = dec.decrypt(ct, dk)
        end_to_end_times.append(time.perf_counter() - t0)
        expected = sum(x[i] * sum(f_matrix[i][j] * y[j] for j in range(len(y))) for i in range(len(x)))
        if result != expected:
            raise AssertionError("Quadratic/Quad benchmark produced an unexpected result")

        pp_sizes.append(size_bytes(pp))
        dk_sizes.append(size_bytes(dk))
        ct_sizes.append(size_bytes(ct))

    return {
        "scheme": "Quadratic/Quad",
        "setup_mean_sec": stats(setup_times)[0],
        "end_to_end_mean_sec": stats(end_to_end_times)[0],
        "end_to_end_median_sec": stats(end_to_end_times)[1],
        "end_to_end_std_sec": stats(end_to_end_times)[2],
        "pp_total_bytes": int(sum(pp_sizes) / len(pp_sizes)),
        "dk_total_bytes": int(sum(dk_sizes) / len(dk_sizes)),
        "ct_total_bytes": int(sum(ct_sizes) / len(ct_sizes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--profile", choices=["small", "medium", "large", "report"], default="small")
    parser.add_argument("--output", choices=["csv", "json"], default="csv")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    prepare_benchmark(args.seed)

    dim = profile_value(args.profile, {"small": 3, "medium": 4, "large": 5, "report": 5})
    quad_m = profile_value(args.profile, {"small": 2, "medium": 3, "large": 4, "report": 4})
    bound = profile_value(args.profile, {"small": 4, "medium": 8, "large": 12, "report": 12})
    x = [1 for _ in range(dim)]
    y_sgp = [1 for _ in range(dim)]
    y_quad = [1 for _ in range(quad_m)]
    f_sgp = [[1 for _ in range(dim)] for _ in range(dim)]
    f_quad = [[1 for _ in range(quad_m)] for _ in range(dim)]

    for _ in range(args.warmup):
        _bench_sgp({"sec_param": 64, "n": dim, "bound": bound, "pairing_group_param": "SS512"}, x, y_sgp, f_sgp, 1)
        _bench_quad({"sec_param": 64, "n": dim, "m": quad_m, "bound": bound, "pairing_group_param": "SS512"}, x, y_quad, f_quad, 1)
    results = [
        _bench_sgp({"sec_param": 64, "n": dim, "bound": bound, "pairing_group_param": "SS512"}, x, y_sgp, f_sgp, args.rounds),
        _bench_quad({"sec_param": 64, "n": dim, "m": quad_m, "bound": bound, "pairing_group_param": "SS512"}, x, y_quad, f_quad, args.rounds),
    ]
    for result in results:
        result["profile"] = args.profile
    emit_results(results, output=args.output, header=BENCHMARK_HEADER)


if __name__ == "__main__":
    main()
