"""Cross-instantiation benchmark for threshold MCFE families."""

from __future__ import annotations

import argparse
import time

from _common import BENCHMARK_HEADER
from _common import emit_results
from _common import prepare_benchmark
from _common import profile_value
from _common import size_bytes
from _common import stats
from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFE
from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFEKeyGenerator
from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWE
from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWEKeyGenerator

ROUNDS = 3


def _run_case(case: dict, rounds: int) -> dict:
    setup_times = []
    encrypt_times = []
    partial_times = []
    combine_times = []
    end_to_end_times = []
    pp_sizes = []
    sk_totals = []
    partial_totals = []
    ct_totals = []

    for _ in range(rounds):
        kg = case["kg_cls"](case["config"])
        t0 = time.perf_counter()
        kg.setup()
        setup_times.append(time.perf_counter() - t0)
        pp = kg.get_public_parameters()
        sk_by_nid = {nid: kg.get_private_keys(nid) for nid in case["lst_nid"]}

        encryptors = {}
        dct_ct = {}
        t0 = time.perf_counter()
        for nid in case["lst_nid"]:
            enc = case["crypto_cls"]({"id": nid, "keys": {"pp": pp, "sk": sk_by_nid[nid]}})
            dct_ct[nid] = enc.encrypt(case["x"][nid], case["label"])
            encryptors[nid] = enc
        encrypt_times.append(time.perf_counter() - t0)

        enrolled = case["lst_sid"][: case["threshold"]]
        dk_credentials = {"fusion_weight": case["y"], "label": case["label"]}
        share_credentials = (
            case["y"] if case.get("share_decrypt_mode") == "fusion_weight_only" else dk_credentials
        )
        dct_partial = {}
        t0 = time.perf_counter()
        for sid in enrolled:
            dk_sid = kg.get_decryption_keys(sid, credentials=dk_credentials)
            dec = case["crypto_cls"]({"id": sid, "keys": {"pp": pp, "dk": dk_sid}})
            dct_partial[sid] = dec.share_decrypt(dct_ct, share_credentials, dk_sid, enrolled)
        partial_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        computed = encryptors[case["lst_nid"][0]].combine_decrypt(dct_partial)
        combine_times.append(time.perf_counter() - t0)

        if computed != case["expected"]:
            raise AssertionError(f"{case['name']} produced {computed}, expected {case['expected']}")

        pp_sizes.append(size_bytes(pp))
        sk_totals.append(sum(size_bytes(sk) for sk in sk_by_nid.values()))
        partial_totals.append(sum(size_bytes(part) for part in dct_partial.values()))
        ct_totals.append(sum(size_bytes(ct) for ct in dct_ct.values()))
        end_to_end_times.append(
            setup_times[-1] + encrypt_times[-1] + partial_times[-1] + combine_times[-1]
        )

    setup_mean, _, _ = stats(setup_times)
    encrypt_mean, _, _ = stats(encrypt_times)
    partial_mean, _, _ = stats(partial_times)
    combine_mean, _, _ = stats(combine_times)
    end_mean, end_median, end_std = stats(end_to_end_times)
    clients = len(case["lst_nid"])
    shares = case["threshold"]
    return {
        "scheme": case["name"],
        "setup_mean_sec": setup_mean,
        "encrypt_mean_sec": encrypt_mean,
        "partial_decrypt_mean_sec": partial_mean,
        "combine_mean_sec": combine_mean,
        "end_to_end_mean_sec": end_mean,
        "end_to_end_median_sec": end_median,
        "end_to_end_std_sec": end_std,
        "pp_total_bytes": int(sum(pp_sizes) / len(pp_sizes)),
        "client_sk_total_bytes": int(sum(sk_totals) / len(sk_totals)),
        "client_sk_avg_bytes": int(sum(sk_totals) / (len(sk_totals) * clients)),
        "partial_bundle_total_bytes": int(sum(partial_totals) / len(partial_totals)),
        "partial_bundle_avg_bytes": int(sum(partial_totals) / (len(partial_totals) * shares)),
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
    lst_sid = ["sid_0", "sid_1", "sid_2"]
    lwe_n = profile_value(args.profile, {"small": 16, "medium": 24, "large": 32})
    ring_n = profile_value(args.profile, {"small": 16, "medium": 16, "large": 32})
    bound = profile_value(args.profile, {"small": 4, "medium": 8, "large": 12})
    dct_x = {nid: [1] for nid in lst_nid}
    dct_y = {nid: [1] for nid in lst_nid}
    ring_x = {nid: [[1]] for nid in lst_nid}

    cases = [
        {
            "name": "tMCFE/DDH",
            "kg_cls": ThresholdMCFEKeyGenerator,
            "crypto_cls": ThresholdMCFE,
            "config": {"sec_param": 128, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": lst_nid, "lst_sid": lst_sid},
            "label": "bench-tmcfe-ddh",
            "share_decrypt_mode": "fusion_weight_only",
            "lst_nid": lst_nid,
            "lst_sid": lst_sid,
            "threshold": 2,
            "x": dct_x,
            "y": dct_y,
            "expected": 3,
        },
        {
            "name": "tMCFE/LWE",
            "kg_cls": ThresholdMCFELWEKeyGenerator,
            "crypto_cls": ThresholdMCFELWE,
            "config": {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": lst_nid, "lst_sid": lst_sid, "lwe_n": lwe_n, "bound_x": bound, "bound_y": bound, "bound_u": 2, "label_modulus": 8},
            "label": "bench-tmcfe-lwe",
            "lst_nid": lst_nid,
            "lst_sid": lst_sid,
            "threshold": 2,
            "x": dct_x,
            "y": dct_y,
            "expected": 3,
        },
        {
            "name": "tMCFE/RingLWE",
            "kg_cls": ThresholdMCFERingLWEKeyGenerator,
            "crypto_cls": ThresholdMCFERingLWE,
            "config": {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": lst_nid, "lst_sid": lst_sid, "ring_n": ring_n, "bound_x": bound, "bound_y": bound, "bound_u": 1, "label_modulus": 8},
            "label": "bench-tmcfe-ring-lwe",
            "lst_nid": lst_nid,
            "lst_sid": lst_sid,
            "threshold": 2,
            "x": ring_x,
            "y": dct_y,
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
