"""Cross-instantiation benchmark for the repository's core MIFE family."""

from __future__ import annotations

import argparse
import time

from _common import BENCHMARK_HEADER
from _common import emit_results
from _common import prepare_benchmark
from _common import profile_value
from _common import size_bytes
from _common import stats
from pyfe4ai.schemes.mife.damgard_ddh import MIFEDamgard
from pyfe4ai.schemes.mife.damgard_ddh import MIFEDamgardKeyGenerator
from pyfe4ai.schemes.mife.ddh import MIFE
from pyfe4ai.schemes.mife.ddh import MIFEKeyGenerator
from pyfe4ai.schemes.mife.fullysec_lwe import MIFEFullySecLWE
from pyfe4ai.schemes.mife.fullysec_lwe import MIFEFullySecLWEKeyGenerator
from pyfe4ai.schemes.mife.lwe import MIFELWE
from pyfe4ai.schemes.mife.lwe import MIFELWEKeyGenerator
from pyfe4ai.schemes.mife.paillier import MIFEPaillier
from pyfe4ai.schemes.mife.paillier import MIFEPaillierKeyGenerator
from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWE
from pyfe4ai.schemes.mife.ring_lwe import MIFERingLWEKeyGenerator

ROUNDS = 3


def _run_case(case: dict, rounds: int) -> dict:
    setup_times = []
    keyexport_times = []
    encrypt_times = []
    dkgen_times = []
    decrypt_times = []
    end_to_end_times = []
    pp_sizes = []
    sk_totals = []
    dk_sizes = []
    ct_totals = []

    for _ in range(rounds):
        kg = case["kg_cls"](case["config"])
        t0 = time.perf_counter()
        kg.setup()
        setup_times.append(time.perf_counter() - t0)
        pp = kg.get_public_parameters()

        t0 = time.perf_counter()
        sk_by_nid = {nid: kg.get_private_keys(nid) for nid in case["lst_nid"]}
        keyexport_times.append(time.perf_counter() - t0)

        dct_ct = {}
        t0 = time.perf_counter()
        for nid in case["lst_nid"]:
            enc = case["crypto_cls"](
                {
                    "id": nid,
                    "precision": case.get("precision", 0),
                    "keys": {"pp": pp, "sk": sk_by_nid[nid]},
                }
            )
            dct_ct[nid] = enc.encrypt(case["x"][nid])
        encrypt_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": case["y"]})
        dkgen_times.append(time.perf_counter() - t0)

        dec = case["crypto_cls"](
            {"id": "sid_0", "precision": case.get("precision", 0), "keys": {"pp": pp}}
        )
        t0 = time.perf_counter()
        computed = dec.decrypt(dct_ct, dk, case["y"])
        decrypt_times.append(time.perf_counter() - t0)

        if computed != case["expected"]:
            raise AssertionError(f"{case['name']} produced {computed}, expected {case['expected']}")

        pp_sizes.append(size_bytes(pp))
        sk_totals.append(sum(size_bytes(sk) for sk in sk_by_nid.values()))
        dk_sizes.append(size_bytes(dk))
        ct_totals.append(sum(size_bytes(ct) for ct in dct_ct.values()))
        end_to_end_times.append(
            setup_times[-1]
            + keyexport_times[-1]
            + encrypt_times[-1]
            + dkgen_times[-1]
            + decrypt_times[-1]
        )

    setup_mean, _, _ = stats(setup_times)
    keyexport_mean, _, _ = stats(keyexport_times)
    encrypt_mean, _, _ = stats(encrypt_times)
    dkgen_mean, _, _ = stats(dkgen_times)
    decrypt_mean, _, _ = stats(decrypt_times)
    end_mean, end_median, end_std = stats(end_to_end_times)
    clients = len(case["lst_nid"])
    return {
        "scheme": case["name"],
        "setup_mean_sec": setup_mean,
        "client_key_export_mean_sec": keyexport_mean,
        "encrypt_mean_sec": encrypt_mean,
        "dkgen_mean_sec": dkgen_mean,
        "decrypt_mean_sec": decrypt_mean,
        "end_to_end_mean_sec": end_mean,
        "end_to_end_median_sec": end_median,
        "end_to_end_std_sec": end_std,
        "pp_total_bytes": int(sum(pp_sizes) / len(pp_sizes)),
        "client_sk_total_bytes": int(sum(sk_totals) / len(sk_totals)),
        "client_sk_avg_bytes": int(sum(sk_totals) / (len(sk_totals) * clients)),
        "dk_total_bytes": int(sum(dk_sizes) / len(dk_sizes)),
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
    eta = profile_value(args.profile, {"small": 1, "medium": 2, "large": 4})
    lwe_n = profile_value(args.profile, {"small": 16, "medium": 24, "large": 32})
    ring_n = profile_value(args.profile, {"small": 16, "medium": 16, "large": 32})
    bit_length = profile_value(args.profile, {"small": 64, "medium": 96, "large": 128})
    bound = profile_value(args.profile, {"small": 4, "medium": 8, "large": 12})

    core_x = {nid: [1 for _ in range(eta)] for nid in lst_nid}
    core_y = {nid: [1 for _ in range(eta)] for nid in lst_nid}
    ring_x = {nid: [[1] for _ in range(eta)] for nid in lst_nid}
    ring_y = core_y

    cases = [
        {
            "name": "MIFE/DDH",
            "kg_cls": MIFEKeyGenerator,
            "crypto_cls": MIFE,
            "config": {"sec_param": 256, "eta": eta, "n": 3, "s": 1, "lst_nid": lst_nid},
            "precision": 3,
            "lst_nid": lst_nid,
            "x": core_x,
            "y": core_y,
            "expected": sum(core_x[nid][i] * core_y[nid][i] for nid in lst_nid for i in range(eta)),
        },
        {
            "name": "MIFE/LWE",
            "kg_cls": MIFELWEKeyGenerator,
            "crypto_cls": MIFELWE,
            "config": {"sec_param": 64, "eta": eta, "n": 3, "s": 1, "lst_nid": lst_nid, "lwe_n": lwe_n, "bound_x": bound, "bound_y": bound},
            "lst_nid": lst_nid,
            "x": core_x,
            "y": core_y,
            "expected": sum(core_x[nid][i] * core_y[nid][i] for nid in lst_nid for i in range(eta)),
        },
        {
            "name": "MIFE/FullySecLWE",
            "kg_cls": MIFEFullySecLWEKeyGenerator,
            "crypto_cls": MIFEFullySecLWE,
            "config": {"sec_param": 64, "eta": eta, "n": 3, "s": 1, "lst_nid": lst_nid, "lwe_n": lwe_n + 8, "bound_x": bound, "bound_y": bound},
            "lst_nid": lst_nid,
            "x": core_x,
            "y": core_y,
            "expected": sum(core_x[nid][i] * core_y[nid][i] for nid in lst_nid for i in range(eta)),
        },
        {
            "name": "MIFE/RingLWE",
            "kg_cls": MIFERingLWEKeyGenerator,
            "crypto_cls": MIFERingLWE,
            "config": {"sec_param": 64, "eta": eta, "n": 3, "s": 1, "lst_nid": lst_nid, "ring_n": ring_n, "bound_x": bound, "bound_y": bound},
            "lst_nid": lst_nid,
            "x": ring_x,
            "y": ring_y,
            "expected": [sum(ring_x[nid][i][0] * ring_y[nid][i] for nid in lst_nid for i in range(eta))],
        },
        {
            "name": "MIFE/Paillier",
            "kg_cls": MIFEPaillierKeyGenerator,
            "crypto_cls": MIFEPaillier,
            "config": {"sec_param": 64, "eta": eta, "n": 3, "s": 1, "lst_nid": lst_nid, "bit_length": bit_length, "bound_x": bound, "bound_y": bound},
            "lst_nid": lst_nid,
            "x": core_x,
            "y": core_y,
            "expected": sum(core_x[nid][i] * core_y[nid][i] for nid in lst_nid for i in range(eta)),
        },
        {
            "name": "MIFE/DamgardDDH",
            "kg_cls": MIFEDamgardKeyGenerator,
            "crypto_cls": MIFEDamgard,
            "config": {"sec_param": 64, "eta": eta, "n": 3, "s": 1, "lst_nid": lst_nid, "modulus_length": bit_length, "bound": bound},
            "lst_nid": lst_nid,
            "x": core_x,
            "y": core_y,
            "expected": sum(core_x[nid][i] * core_y[nid][i] for nid in lst_nid for i in range(eta)),
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
