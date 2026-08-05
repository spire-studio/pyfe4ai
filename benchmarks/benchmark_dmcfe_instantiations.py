"""Cross-instantiation benchmark for decentralized MCFE families."""

from __future__ import annotations

import argparse
import time

from _common import BENCHMARK_HEADER
from _common import emit_results
from _common import prepare_benchmark
from _common import profile_value
from _common import size_bytes
from _common import stats
from pyfe4ai.schemes.mcfe.ddh_decentralized import DecentralizedMCFE
from pyfe4ai.schemes.mcfe.ddh_decentralized import DecentralizedMCFEKeyGenerator
from pyfe4ai.schemes.mcfe.lwe_decentralized import DecentralizedMCFELWE
from pyfe4ai.schemes.mcfe.lwe_decentralized import DecentralizedMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_decentralized import DecentralizedMCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe_decentralized import DecentralizedMCFERingLWEKeyGenerator

ROUNDS = 3


def _run_case(case: dict, rounds: int) -> dict:
    setup_times = []
    encrypt_times = []
    share_times = []
    combine_decrypt_times = []
    end_to_end_times = []
    pp_sizes = []
    sk_totals = []
    share_totals = []
    dk_sizes = []
    ct_totals = []

    for _ in range(rounds):
        kg = case["kg_cls"](case["config"])
        t0 = time.perf_counter()
        kg.setup()
        setup_times.append(time.perf_counter() - t0)
        pp = kg.get_public_parameters()
        sk_by_nid = {nid: kg.get_private_keys(nid) for nid in case["lst_nid"]}

        dct_ct = {}
        actors = {}
        t0 = time.perf_counter()
        for nid in case["lst_nid"]:
            actor = case["crypto_cls"]({"id": nid, "keys": {"pp": pp, "sk": sk_by_nid[nid]}})
            dct_ct[nid] = actor.encrypt(case["x"][nid], case["label"])
            actors[nid] = actor
        encrypt_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        dct_shares = {nid: actors[nid].derive_function_decryption_key_share(case["y"]) for nid in case["lst_nid"]}
        share_times.append(time.perf_counter() - t0)

        dec = case["crypto_cls"]({"id": "sid_0", "keys": {"pp": pp}})
        t0 = time.perf_counter()
        dk = dec.combine_function_decryption_key_share(dct_shares)
        computed = dec.decrypt(dct_ct, dk, case["y"], case["label"])
        combine_decrypt_times.append(time.perf_counter() - t0)

        if computed != case["expected"]:
            raise AssertionError(f"{case['name']} produced {computed}, expected {case['expected']}")

        pp_sizes.append(size_bytes(pp))
        sk_totals.append(sum(size_bytes(sk) for sk in sk_by_nid.values()))
        share_totals.append(sum(size_bytes(share) for share in dct_shares.values()))
        dk_sizes.append(size_bytes(dk))
        ct_totals.append(sum(size_bytes(ct) for ct in dct_ct.values()))
        end_to_end_times.append(
            setup_times[-1] + encrypt_times[-1] + share_times[-1] + combine_decrypt_times[-1]
        )

    setup_mean, _, _ = stats(setup_times)
    encrypt_mean, _, _ = stats(encrypt_times)
    share_mean, _, _ = stats(share_times)
    combine_mean, _, _ = stats(combine_decrypt_times)
    end_mean, end_median, end_std = stats(end_to_end_times)
    clients = len(case["lst_nid"])
    return {
        "scheme": case["name"],
        "setup_mean_sec": setup_mean,
        "encrypt_mean_sec": encrypt_mean,
        "share_mean_sec": share_mean,
        "combine_decrypt_mean_sec": combine_mean,
        "end_to_end_mean_sec": end_mean,
        "end_to_end_median_sec": end_median,
        "end_to_end_std_sec": end_std,
        "pp_total_bytes": int(sum(pp_sizes) / len(pp_sizes)),
        "client_sk_total_bytes": int(sum(sk_totals) / len(sk_totals)),
        "client_sk_avg_bytes": int(sum(sk_totals) / (len(sk_totals) * clients)),
        "share_bundle_total_bytes": int(sum(share_totals) / len(share_totals)),
        "share_bundle_avg_bytes": int(sum(share_totals) / (len(share_totals) * clients)),
        "combined_dk_total_bytes": int(sum(dk_sizes) / len(dk_sizes)),
        "ct_total_bytes": int(sum(ct_totals) / len(ct_totals)),
        "ct_avg_bytes": int(sum(ct_totals) / (len(ct_totals) * clients)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--profile", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--output", choices=["csv", "json"], default="csv")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    prepare_benchmark(args.seed)

    lst_nid = ["nid_0", "nid_1", "nid_2"]
    lwe_n = profile_value(args.profile, {"small": 16, "medium": 24, "large": 32})
    ring_n = profile_value(args.profile, {"small": 16, "medium": 16, "large": 32})
    bound = profile_value(args.profile, {"small": 4, "medium": 8, "large": 12})
    core_x = {nid: [1] for nid in lst_nid}
    core_y = {nid: [1] for nid in lst_nid}
    ring_x = {nid: [[1]] for nid in lst_nid}
    ring_y = core_y

    cases = [
        {
            "name": "dMCFE/DDH",
            "kg_cls": DecentralizedMCFEKeyGenerator,
            "crypto_cls": DecentralizedMCFE,
            "config": {"sec_param": 128, "eta": 1, "n": 3, "s": 1, "lst_nid": lst_nid},
            "label": "bench-dmcfe-ddh",
            "lst_nid": lst_nid,
            "x": core_x,
            "y": core_y,
            "expected": 3,
        },
        {
            "name": "dMCFE/LWE",
            "kg_cls": DecentralizedMCFELWEKeyGenerator,
            "crypto_cls": DecentralizedMCFELWE,
            "config": {"sec_param": 64, "eta": 1, "n": 3, "s": 1, "lst_nid": lst_nid, "lwe_n": lwe_n, "bound_x": bound, "bound_y": bound, "bound_u": 2, "label_modulus": 8},
            "label": "bench-dmcfe-lwe",
            "lst_nid": lst_nid,
            "x": core_x,
            "y": core_y,
            "expected": 3,
        },
        {
            "name": "dMCFE/RingLWE",
            "kg_cls": DecentralizedMCFERingLWEKeyGenerator,
            "crypto_cls": DecentralizedMCFERingLWE,
            "config": {"sec_param": 64, "eta": 1, "n": 3, "s": 1, "lst_nid": lst_nid, "ring_n": ring_n, "bound_x": bound, "bound_y": bound, "bound_u": 1, "label_modulus": 8},
            "label": "bench-dmcfe-ring-lwe",
            "lst_nid": lst_nid,
            "x": ring_x,
            "y": ring_y,
            "expected": [3],
        },
    ]

    results = []
    for case in cases:
        for _ in range(args.warmup):
            _run_case(case, rounds=1)
        result = _run_case(case, rounds=args.rounds)
        result["profile"] = args.profile
        results.append(result)
    emit_results(results, output=args.output, header=BENCHMARK_HEADER)


if __name__ == "__main__":
    main()
