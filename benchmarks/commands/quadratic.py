"""Family benchmark for the repository's quadratic FE prototypes."""

from __future__ import annotations

import argparse
import time

from _common import BENCHMARK_HEADER
from _common import emit_results
from _common import keygen_config
from _common import parse_args
from _common import run_cases
from _common import prepare_benchmark
from _common import profile_value
from _common import size_bytes
from _common import stats
from pyfe4ai.schemes.quadratic.miqfe import QuadraticMIQFE
from pyfe4ai.schemes.quadratic.miqfe import QuadraticMIQFEKeyGenerator
from pyfe4ai.schemes.quadratic.quad import QuadraticQuad
from pyfe4ai.schemes.quadratic.quad import QuadraticQuadKeyGenerator
from pyfe4ai.schemes.quadratic.sgp import QuadraticSGP
from pyfe4ai.schemes.quadratic.sgp import QuadraticSGPKeyGenerator
from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP

ROUNDS = 3

def _bench_sgp(cfg: dict, x: list[int], y: list[int], f_matrix: list[list[int]], rounds: int) -> dict:
    setup_times = []
    end_to_end_times = []
    pp_sizes = []
    dk_sizes = []
    ct_sizes = []

    for _ in range(rounds):
        kg = QuadraticSGPKeyGenerator(keygen_config(cfg))
        t0 = time.perf_counter()
        kg.setup()
        setup_times.append(time.perf_counter() - t0)
        pp = kg.get_public_parameters()
        enc = QuadraticSGP({"id": "nid_default", "keys": {"pp": pp}})
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
        dk_sizes.append(size_bytes(dk))
        ct_sizes.append(size_bytes(ct))

    return {
        "scheme": "Quadratic/SGP",
        "setup_mean_sec": stats(setup_times)[0],
        "end_to_end_mean_sec": stats(end_to_end_times)[0],
        "end_to_end_median_sec": stats(end_to_end_times)[1],
        "end_to_end_std_sec": stats(end_to_end_times)[2],
        "pp_total_bytes": int(sum(pp_sizes) / len(pp_sizes)),
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
        kg = QuadraticQuadKeyGenerator(keygen_config(cfg))
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


def _bench_miqfe(cfg: dict, rounds: int) -> dict:
    """AGT22 multi-input quadratic FE: n users with m values each, all cross-user
    coefficients 1 (blocks (i, k) with i < k), all inputs 1."""
    n, m = cfg["n"], cfg["m"]
    lst_nid = cfg["lst_nid"]
    coeff = [[1 if r // m < c // m else 0 for c in range(n * m)] for r in range(n * m)]
    expected = n * (n - 1) // 2 * m * m
    times = {k: [] for k in ("setup", "keygen", "encrypt", "dkgen", "decrypt", "e2e")}
    sizes = {k: [] for k in ("pp", "sk", "dk", "ct")}
    for _ in range(rounds):
        kg = QuadraticMIQFEKeyGenerator(keygen_config(cfg))
        t0 = time.perf_counter()
        kg.setup()
        times["setup"].append(time.perf_counter() - t0)
        pp = kg.get_public_parameters()
        t0 = time.perf_counter()
        sks = {nid: kg.get_private_keys(nid) for nid in lst_nid}
        times["keygen"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        cts = {nid: QuadraticMIQFE({"keys": {"pp": pp, "sk": sks[nid]}}).encrypt([1] * m) for nid in lst_nid}
        times["encrypt"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        dk = kg.get_decryption_keys(None, credentials={"coefficients": coeff})
        times["dkgen"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        result = QuadraticMIQFE({"keys": {"pp": pp}}).decrypt(cts, dk)
        times["decrypt"].append(time.perf_counter() - t0)
        if result != expected:
            raise AssertionError("Quadratic/MIQFE produced {}, expected {}".format(result, expected))
        times["e2e"].append(sum(times[k][-1] for k in ("setup", "keygen", "encrypt", "dkgen", "decrypt")))
        sizes["pp"].append(size_bytes(pp))
        sizes["sk"].append(sum(size_bytes(v) for v in sks.values()))
        sizes["dk"].append(size_bytes(dk))
        sizes["ct"].append(sum(size_bytes(v) for v in cts.values()))
    mean = {k: stats(v)[0] for k, v in times.items()}
    e2e = stats(times["e2e"])
    return {
        "scheme": "Quadratic/MIQFE",
        "setup_mean_sec": mean["setup"],
        "client_key_export_mean_sec": mean["keygen"],
        "encrypt_mean_sec": mean["encrypt"],
        "dkgen_mean_sec": mean["dkgen"],
        "decrypt_mean_sec": mean["decrypt"],
        "end_to_end_mean_sec": e2e[0],
        "end_to_end_median_sec": e2e[1],
        "end_to_end_std_sec": e2e[2],
        "pp_total_bytes": int(sum(sizes["pp"]) / rounds),
        "client_sk_total_bytes": int(sum(sizes["sk"]) / rounds),
        "dk_total_bytes": int(sum(sizes["dk"]) / rounds),
        "ct_total_bytes": int(sum(sizes["ct"]) / rounds),
    }


def main(argv=None, prog=None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--profile", choices=["small", "medium", "large", "report"], default="small")
    parser.add_argument("--output", choices=["csv", "json"], default="csv")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pairing-group", default=DEFAULT_PAIRING_GROUP,
                        help="BLS12-381 (Rust extension) or a charm-crypto group such as MNT224")
    args = parse_args(parser, argv)
    prepare_benchmark(args.seed)

    dim = profile_value(args.profile, {"small": 3, "medium": 4, "large": 5, "report": 5})
    quad_m = profile_value(args.profile, {"small": 2, "medium": 3, "large": 4, "report": 4})
    bound = profile_value(args.profile, {"small": 4, "medium": 8, "large": 12, "report": 12})
    x = [1 for _ in range(dim)]
    y_sgp = [1 for _ in range(dim)]
    y_quad = [1 for _ in range(quad_m)]
    f_sgp = [[1 for _ in range(dim)] for _ in range(dim)]
    f_quad = [[1 for _ in range(quad_m)] for _ in range(dim)]

    miqfe_n = profile_value(args.profile, {"small": 2, "medium": 2, "large": 3, "report": 3})
    miqfe_m = profile_value(args.profile, {"small": 1, "medium": 2, "large": 2, "report": 2})
    miqfe_cfg = {"n": miqfe_n, "m": miqfe_m, "lst_nid": ["nid_{}".format(i) for i in range(miqfe_n)],
                 "bound_x": bound, "bound_c": 1, "pairing_group_param": args.pairing_group}
    for _ in range(args.warmup):
        _bench_sgp({"sec_param": 64, "n": dim, "bound": bound, "pairing_group_param": args.pairing_group}, x, y_sgp, f_sgp, 1)
        _bench_quad({"sec_param": 64, "n": dim, "m": quad_m, "bound": bound, "pairing_group_param": args.pairing_group}, x, y_quad, f_quad, 1)
    results = run_cases([
        lambda: _bench_sgp({"sec_param": 64, "n": dim, "bound": bound, "pairing_group_param": args.pairing_group}, x, y_sgp, f_sgp, args.rounds),
        lambda: _bench_quad({"sec_param": 64, "n": dim, "m": quad_m, "bound": bound, "pairing_group_param": args.pairing_group}, x, y_quad, f_quad, args.rounds),
        # no warm-up: one MIQFE setup takes seconds
        lambda: _bench_miqfe(miqfe_cfg, args.rounds),
    ])
    for result in results:
        result["profile"] = args.profile
    emit_results(results, output=args.output, header=BENCHMARK_HEADER)

