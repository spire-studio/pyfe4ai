"""Cross-instantiation benchmark for the repository's SIFE family.

This script compares multiple single-input instantiations under a shared
workload shape. It records setup, key export, encryption, functional-key
derivation, and decryption timings together with coarse serialized object
sizes.
"""

from __future__ import annotations

import argparse
import time

from _common import BENCHMARK_HEADER
from _common import emit_results
from _common import prepare_benchmark
from _common import profile_value
from pyfe4ai.schemes.sife.damgard_ddh import SIFEDamgard
from pyfe4ai.schemes.sife.damgard_ddh import SIFEDamgardKeyGenerator
from pyfe4ai.schemes.sife.ddh import SIFE
from pyfe4ai.schemes.sife.ddh import SIFEKeyGenerator
from pyfe4ai.schemes.sife.fullysec_lwe import SIFEFullySecLWE
from pyfe4ai.schemes.sife.fullysec_lwe import SIFEFullySecLWEKeyGenerator
from pyfe4ai.schemes.sife.lwe import SIFELWE
from pyfe4ai.schemes.sife.lwe import SIFELWEKeyGenerator
from pyfe4ai.schemes.sife.paillier import SIFEPaillier
from pyfe4ai.schemes.sife.paillier import SIFEPaillierKeyGenerator
from pyfe4ai.schemes.sife.ring_lwe import SIFERingLWE
from pyfe4ai.schemes.sife.ring_lwe import SIFERingLWEKeyGenerator
from _common import size_bytes
from _common import stats

ROUNDS = 3


def _run_case(case: dict, rounds: int = ROUNDS) -> dict:
    setup_times = []
    keygen_times = []
    encrypt_times = []
    dkgen_times = []
    decrypt_times = []
    end_to_end_times = []
    pp_sizes = []
    sk_sizes = []
    dk_sizes = []
    ct_sizes = []

    for _ in range(rounds):
        kg = case["kg_cls"](case["config"])

        t0 = time.perf_counter()
        kg.setup()
        setup_times.append(time.perf_counter() - t0)

        pp = kg.get_public_parameters()

        t0 = time.perf_counter()
        sk = kg.get_private_keys()
        keygen_times.append(time.perf_counter() - t0)

        encryptor = case["crypto_cls"]({"id": "nid_default", "keys": {"pp": pp, "sk": sk}})
        decryptor = case["crypto_cls"]({"id": "sid_0", "keys": {"pp": pp}})

        t0 = time.perf_counter()
        ct = encryptor.encrypt(case["x"])
        encrypt_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": case["y"]})
        dkgen_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        computed = decryptor.decrypt(ct, dk, case["y"])
        decrypt_times.append(time.perf_counter() - t0)

        if computed != case["expected"]:
            raise AssertionError(f"{case['name']} produced {computed}, expected {case['expected']}")

        pp_sizes.append(size_bytes(pp))
        sk_sizes.append(size_bytes(sk))
        dk_sizes.append(size_bytes(dk))
        ct_sizes.append(size_bytes(ct))
        end_to_end_times.append(
            setup_times[-1]
            + keygen_times[-1]
            + encrypt_times[-1]
            + dkgen_times[-1]
            + decrypt_times[-1]
        )

    setup_mean, _, _ = stats(setup_times)
    keygen_mean, _, _ = stats(keygen_times)
    encrypt_mean, _, _ = stats(encrypt_times)
    dkgen_mean, _, _ = stats(dkgen_times)
    decrypt_mean, _, _ = stats(decrypt_times)
    end_mean, end_median, end_std = stats(end_to_end_times)

    return {
        "scheme": case["name"],
        "setup_mean_sec": setup_mean,
        "keygen_mean_sec": keygen_mean,
        "encrypt_mean_sec": encrypt_mean,
        "dkgen_mean_sec": dkgen_mean,
        "decrypt_mean_sec": decrypt_mean,
        "end_to_end_mean_sec": end_mean,
        "end_to_end_median_sec": end_median,
        "end_to_end_std_sec": end_std,
        "pp_total_bytes": int(sum(pp_sizes) / len(pp_sizes)),
        "sk_total_bytes": int(sum(sk_sizes) / len(sk_sizes)),
        "dk_total_bytes": int(sum(dk_sizes) / len(dk_sizes)),
        "ct_total_bytes": int(sum(ct_sizes) / len(ct_sizes)),
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

    eta = profile_value(args.profile, {"small": 4, "medium": 8, "large": 16})
    lwe_n = profile_value(args.profile, {"small": 16, "medium": 24, "large": 32})
    ring_n = profile_value(args.profile, {"small": 16, "medium": 32, "large": 32})
    sec_param_ddh = profile_value(args.profile, {"small": 64, "medium": 128, "large": 128})
    bit_length = profile_value(args.profile, {"small": 64, "medium": 96, "large": 128})
    bound = profile_value(args.profile, {"small": 4, "medium": 8, "large": 12})

    vector_x = [1 for _ in range(eta)]
    vector_y = [1 for _ in range(eta)]
    ring_x = [[1] for _ in range(eta)]
    ring_y = [1 for _ in range(eta)]

    cases = [
        {
            "name": "SIFE/DDH",
            "kg_cls": SIFEKeyGenerator,
            "crypto_cls": SIFE,
            "config": {"sec_param": sec_param_ddh, "eta": eta},
            "x": vector_x,
            "y": vector_y,
            "expected": sum(a * b for a, b in zip(vector_x, vector_y)),
        },
        {
            "name": "SIFE/LWE",
            "kg_cls": SIFELWEKeyGenerator,
            "crypto_cls": SIFELWE,
            "config": {"sec_param": 64, "eta": eta, "lwe_n": lwe_n, "bound_x": bound, "bound_y": bound},
            "x": vector_x,
            "y": vector_y,
            "expected": sum(a * b for a, b in zip(vector_x, vector_y)),
        },
        {
            "name": "SIFE/FullySecLWE",
            "kg_cls": SIFEFullySecLWEKeyGenerator,
            "crypto_cls": SIFEFullySecLWE,
            "config": {"sec_param": 64, "eta": eta, "lwe_n": lwe_n + 8, "bound_x": bound, "bound_y": bound},
            "x": vector_x,
            "y": vector_y,
            "expected": sum(a * b for a, b in zip(vector_x, vector_y)),
        },
        {
            "name": "SIFE/RingLWE",
            "kg_cls": SIFERingLWEKeyGenerator,
            "crypto_cls": SIFERingLWE,
            "config": {"sec_param": 64, "eta": eta, "ring_n": ring_n, "bound_x": bound, "bound_y": bound},
            "x": ring_x,
            "y": ring_y,
            "expected": [sum(ring_x[row][0] * ring_y[row] for row in range(eta))],
        },
        {
            "name": "SIFE/Paillier",
            "kg_cls": SIFEPaillierKeyGenerator,
            "crypto_cls": SIFEPaillier,
            "config": {"sec_param": 64, "eta": eta, "bit_length": bit_length, "bound_x": bound, "bound_y": bound},
            "x": vector_x,
            "y": vector_y,
            "expected": sum(a * b for a, b in zip(vector_x, vector_y)),
        },
        {
            "name": "SIFE/DamgardDDH",
            "kg_cls": SIFEDamgardKeyGenerator,
            "crypto_cls": SIFEDamgard,
            "config": {"sec_param": 64, "eta": eta, "modulus_length": bit_length, "bound": bound},
            "x": vector_x,
            "y": vector_y,
            "expected": sum(a * b for a, b in zip(vector_x, vector_y)),
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
