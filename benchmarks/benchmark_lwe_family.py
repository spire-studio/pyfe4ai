"""Family benchmark for the repository's LWE-based IPFE line."""

from __future__ import annotations

import argparse
import time

from _common import BENCHMARK_HEADER
from _common import emit_results
from _common import prepare_benchmark
from _common import profile_value
from _common import size_bytes
from _common import stats
from pyfe4ai.schemes.mcfe.lwe import MCFELWE
from pyfe4ai.schemes.mcfe.lwe import MCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.lwe_decentralized import DecentralizedMCFELWE
from pyfe4ai.schemes.mcfe.lwe_decentralized import DecentralizedMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWE
from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWEKeyGenerator
from pyfe4ai.schemes.mife.lwe import MIFELWE
from pyfe4ai.schemes.mife.lwe import MIFELWEKeyGenerator
from pyfe4ai.schemes.sife.lwe import SIFELWE
from pyfe4ai.schemes.sife.lwe import SIFELWEKeyGenerator

ROUNDS = 3


def _bench_sife(cfg: dict, x: list[int], y: list[int], rounds: int) -> dict:
    setup_times = []
    end_to_end_times = []
    pp_sizes = []
    sk_sizes = []
    dk_sizes = []
    ct_sizes = []

    for _ in range(rounds):
        kg = SIFELWEKeyGenerator(cfg)
        t0 = time.perf_counter()
        kg.setup()
        setup_times.append(time.perf_counter() - t0)
        pp = kg.get_public_parameters()
        sk = kg.get_private_keys()
        enc = SIFELWE({"id": "nid_default", "keys": {"pp": pp, "sk": sk}})
        dec = SIFELWE({"id": "sid_0", "keys": {"pp": pp}})

        t0 = time.perf_counter()
        ct = enc.encrypt(x)
        dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
        result = dec.decrypt(ct, dk, y)
        end_to_end_times.append(time.perf_counter() - t0)
        if result != sum(a * b for a, b in zip(x, y)):
            raise AssertionError("SIFE/LWE benchmark produced an unexpected result")

        pp_sizes.append(size_bytes(pp))
        sk_sizes.append(size_bytes(sk))
        dk_sizes.append(size_bytes(dk))
        ct_sizes.append(size_bytes(ct))

    return {
        "scheme": "SIFE/LWE",
        "setup_mean_sec": stats(setup_times)[0],
        "end_to_end_mean_sec": stats(end_to_end_times)[0],
        "end_to_end_median_sec": stats(end_to_end_times)[1],
        "end_to_end_std_sec": stats(end_to_end_times)[2],
        "pp_total_bytes": int(sum(pp_sizes) / len(pp_sizes)),
        "sk_total_bytes": int(sum(sk_sizes) / len(sk_sizes)),
        "dk_total_bytes": int(sum(dk_sizes) / len(dk_sizes)),
        "ct_total_bytes": int(sum(ct_sizes) / len(ct_sizes)),
    }


def _bench_mife(cfg: dict, x: dict[str, list[int]], y: dict[str, list[int]], rounds: int) -> dict:
    setup_times = []
    end_to_end_times = []
    pp_sizes = []
    sk_totals = []
    dk_sizes = []
    ct_totals = []
    clients = len(x)

    for _ in range(rounds):
        kg = MIFELWEKeyGenerator(cfg)
        t0 = time.perf_counter()
        kg.setup()
        setup_times.append(time.perf_counter() - t0)
        pp = kg.get_public_parameters()
        cts = {}
        sk_by_nid = {nid: kg.get_private_keys(nid) for nid in x}

        t0 = time.perf_counter()
        for nid in x:
            enc = MIFELWE({"id": nid, "keys": {"pp": pp, "sk": sk_by_nid[nid]}})
            cts[nid] = enc.encrypt(x[nid])
        dec = MIFELWE({"id": "sid_0", "keys": {"pp": pp}})
        dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
        result = dec.decrypt(cts, dk, y)
        end_to_end_times.append(time.perf_counter() - t0)
        if result != sum(x[nid][i] * y[nid][i] for nid in x for i in range(len(x[nid]))):
            raise AssertionError("MIFE/LWE benchmark produced an unexpected result")

        pp_sizes.append(size_bytes(pp))
        sk_totals.append(sum(size_bytes(sk) for sk in sk_by_nid.values()))
        dk_sizes.append(size_bytes(dk))
        ct_totals.append(sum(size_bytes(ct) for ct in cts.values()))

    return {
        "scheme": "MIFE/LWE",
        "setup_mean_sec": stats(setup_times)[0],
        "end_to_end_mean_sec": stats(end_to_end_times)[0],
        "end_to_end_median_sec": stats(end_to_end_times)[1],
        "end_to_end_std_sec": stats(end_to_end_times)[2],
        "pp_total_bytes": int(sum(pp_sizes) / len(pp_sizes)),
        "sk_total_bytes": int(sum(sk_totals) / len(sk_totals)),
        "sk_avg_bytes": int(sum(sk_totals) / (len(sk_totals) * clients)),
        "dk_total_bytes": int(sum(dk_sizes) / len(dk_sizes)),
        "ct_total_bytes": int(sum(ct_totals) / len(ct_totals)),
        "ct_avg_bytes": int(sum(ct_totals) / (len(ct_totals) * clients)),
    }


def _bench_mcfe(cfg: dict, x: dict[str, list[int]], y: dict[str, list[int]], label: str, rounds: int) -> dict:
    setup_times = []
    end_to_end_times = []
    pp_sizes = []
    sk_totals = []
    dk_sizes = []
    ct_totals = []
    clients = len(x)

    for _ in range(rounds):
        kg = MCFELWEKeyGenerator(cfg)
        t0 = time.perf_counter()
        kg.setup()
        setup_times.append(time.perf_counter() - t0)
        pp = kg.get_public_parameters()
        cts = {}
        sk_by_nid = {nid: kg.get_private_keys(nid) for nid in x}

        t0 = time.perf_counter()
        for nid in x:
            enc = MCFELWE({"id": nid, "keys": {"pp": pp, "sk": sk_by_nid[nid]}})
            cts[nid] = enc.encrypt(x[nid], label)
        dec = MCFELWE({"id": "sid_0", "keys": {"pp": pp}})
        dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
        result = dec.decrypt(cts, dk, y, label)
        end_to_end_times.append(time.perf_counter() - t0)
        if result != sum(x[nid][i] * y[nid][i] for nid in x for i in range(len(x[nid]))):
            raise AssertionError("MCFE/LWE benchmark produced an unexpected result")

        pp_sizes.append(size_bytes(pp))
        sk_totals.append(sum(size_bytes(sk) for sk in sk_by_nid.values()))
        dk_sizes.append(size_bytes(dk))
        ct_totals.append(sum(size_bytes(ct) for ct in cts.values()))

    return {
        "scheme": "MCFE/LWE",
        "setup_mean_sec": stats(setup_times)[0],
        "end_to_end_mean_sec": stats(end_to_end_times)[0],
        "end_to_end_median_sec": stats(end_to_end_times)[1],
        "end_to_end_std_sec": stats(end_to_end_times)[2],
        "pp_total_bytes": int(sum(pp_sizes) / len(pp_sizes)),
        "sk_total_bytes": int(sum(sk_totals) / len(sk_totals)),
        "sk_avg_bytes": int(sum(sk_totals) / (len(sk_totals) * clients)),
        "dk_total_bytes": int(sum(dk_sizes) / len(dk_sizes)),
        "ct_total_bytes": int(sum(ct_totals) / len(ct_totals)),
        "ct_avg_bytes": int(sum(ct_totals) / (len(ct_totals) * clients)),
    }


def _bench_dmcfe(cfg: dict, x: dict[str, list[int]], y: dict[str, list[int]], label: str, rounds: int) -> dict:
    setup_times = []
    end_to_end_times = []
    pp_sizes = []
    sk_totals = []
    share_totals = []
    dk_sizes = []
    ct_totals = []
    clients = len(x)

    for _ in range(rounds):
        kg = DecentralizedMCFELWEKeyGenerator(cfg)
        t0 = time.perf_counter()
        kg.setup()
        setup_times.append(time.perf_counter() - t0)
        pp = kg.get_public_parameters()
        cts = {}
        shares = {}
        sk_by_nid = {nid: kg.get_private_keys(nid) for nid in x}

        t0 = time.perf_counter()
        for nid in x:
            actor = DecentralizedMCFELWE({"id": nid, "keys": {"pp": pp, "sk": sk_by_nid[nid]}})
            cts[nid] = actor.encrypt(x[nid], label)
            shares[nid] = actor.derive_function_decryption_key_share(y)
        dec = DecentralizedMCFELWE({"id": "sid_0", "keys": {"pp": pp}})
        dk = dec.combine_function_decryption_key_share(shares)
        result = dec.decrypt(cts, dk, y, label)
        end_to_end_times.append(time.perf_counter() - t0)
        if result != sum(x[nid][i] * y[nid][i] for nid in x for i in range(len(x[nid]))):
            raise AssertionError("dMCFE/LWE benchmark produced an unexpected result")

        pp_sizes.append(size_bytes(pp))
        sk_totals.append(sum(size_bytes(sk) for sk in sk_by_nid.values()))
        share_totals.append(sum(size_bytes(share) for share in shares.values()))
        dk_sizes.append(size_bytes(dk))
        ct_totals.append(sum(size_bytes(ct) for ct in cts.values()))

    return {
        "scheme": "dMCFE/LWE",
        "setup_mean_sec": stats(setup_times)[0],
        "end_to_end_mean_sec": stats(end_to_end_times)[0],
        "end_to_end_median_sec": stats(end_to_end_times)[1],
        "end_to_end_std_sec": stats(end_to_end_times)[2],
        "pp_total_bytes": int(sum(pp_sizes) / len(pp_sizes)),
        "sk_total_bytes": int(sum(sk_totals) / len(sk_totals)),
        "sk_avg_bytes": int(sum(sk_totals) / (len(sk_totals) * clients)),
        "share_bundle_total_bytes": int(sum(share_totals) / len(share_totals)),
        "share_bundle_avg_bytes": int(sum(share_totals) / (len(share_totals) * clients)),
        "dk_total_bytes": int(sum(dk_sizes) / len(dk_sizes)),
        "ct_total_bytes": int(sum(ct_totals) / len(ct_totals)),
        "ct_avg_bytes": int(sum(ct_totals) / (len(ct_totals) * clients)),
    }


def _bench_tmcfe(cfg: dict, x: dict[str, list[int]], y: dict[str, list[int]], label: str, rounds: int) -> dict:
    setup_times = []
    end_to_end_times = []
    pp_sizes = []
    sk_totals = []
    partial_totals = []
    ct_totals = []
    clients = len(x)
    threshold = cfg["t"]

    for _ in range(rounds):
        kg = ThresholdMCFELWEKeyGenerator(cfg)
        t0 = time.perf_counter()
        kg.setup()
        setup_times.append(time.perf_counter() - t0)
        pp = kg.get_public_parameters()
        cts = {}
        encs = {}
        sk_by_nid = {nid: kg.get_private_keys(nid) for nid in x}

        t0 = time.perf_counter()
        for nid in x:
            enc = ThresholdMCFELWE({"id": nid, "keys": {"pp": pp, "sk": sk_by_nid[nid]}})
            cts[nid] = enc.encrypt(x[nid], label)
            encs[nid] = enc
        credentials = {"fusion_weight": y, "label": label}
        enrolled = cfg["lst_sid"][: cfg["t"]]
        partials = {}
        for sid in enrolled:
            dk_sid = kg.get_decryption_keys(sid, credentials=credentials)
            dec = ThresholdMCFELWE({"id": sid, "keys": {"pp": pp, "dk": dk_sid}})
            partials[sid] = dec.share_decrypt(cts, credentials, dk_sid, enrolled)
        result = encs["nid_0"].combine_decrypt(partials)
        end_to_end_times.append(time.perf_counter() - t0)
        if result != sum(x[nid][i] * y[nid][i] for nid in x for i in range(len(x[nid]))):
            raise AssertionError("tMCFE/LWE benchmark produced an unexpected result")

        pp_sizes.append(size_bytes(pp))
        sk_totals.append(sum(size_bytes(sk) for sk in sk_by_nid.values()))
        partial_totals.append(sum(size_bytes(partial) for partial in partials.values()))
        ct_totals.append(sum(size_bytes(ct) for ct in cts.values()))

    return {
        "scheme": "tMCFE/LWE",
        "setup_mean_sec": stats(setup_times)[0],
        "end_to_end_mean_sec": stats(end_to_end_times)[0],
        "end_to_end_median_sec": stats(end_to_end_times)[1],
        "end_to_end_std_sec": stats(end_to_end_times)[2],
        "pp_total_bytes": int(sum(pp_sizes) / len(pp_sizes)),
        "sk_total_bytes": int(sum(sk_totals) / len(sk_totals)),
        "sk_avg_bytes": int(sum(sk_totals) / (len(sk_totals) * clients)),
        "partial_bundle_total_bytes": int(sum(partial_totals) / len(partial_totals)),
        "partial_bundle_avg_bytes": int(sum(partial_totals) / (len(partial_totals) * threshold)),
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

    eta = profile_value(args.profile, {"small": 2, "medium": 4, "large": 8})
    lwe_n = profile_value(args.profile, {"small": 16, "medium": 24, "large": 32})
    bound = profile_value(args.profile, {"small": 4, "medium": 8, "large": 12})
    x_sife = [1 for _ in range(eta)]
    y_sife = [1 for _ in range(eta)]
    x_multi = {"nid_0": [1 for _ in range(eta)], "nid_1": [1 for _ in range(eta)], "nid_2": [1 for _ in range(eta)]}
    y_multi = {"nid_0": [1 for _ in range(eta)], "nid_1": [1 for _ in range(eta)], "nid_2": [1 for _ in range(eta)]}

    for _ in range(args.warmup):
        _bench_sife({"sec_param": 64, "eta": eta, "lwe_n": lwe_n, "bound_x": bound, "bound_y": bound}, x_sife, y_sife, 1)
        _bench_mife({"sec_param": 64, "eta": eta, "n": 3, "s": 1, "lst_nid": list(x_multi.keys()), "lwe_n": lwe_n, "bound_x": bound, "bound_y": bound}, x_multi, y_multi, 1)
        _bench_mcfe({"sec_param": 64, "eta": eta, "n": 3, "s": 1, "lst_nid": list(x_multi.keys()), "lwe_n": lwe_n, "bound_x": bound, "bound_y": bound, "bound_u": 2, "label_modulus": 8}, x_multi, y_multi, "bench-mcfe-lwe", 1)
        _bench_dmcfe({"sec_param": 64, "eta": eta, "n": 3, "s": 1, "lst_nid": list(x_multi.keys()), "lwe_n": lwe_n, "bound_x": bound, "bound_y": bound, "bound_u": 2, "label_modulus": 8}, x_multi, y_multi, "bench-dmcfe-lwe", 1)
        _bench_tmcfe({"sec_param": 64, "eta": eta, "n": 3, "s": 3, "t": 2, "lst_nid": list(x_multi.keys()), "lst_sid": ["sid_0", "sid_1", "sid_2"], "lwe_n": lwe_n, "bound_x": bound, "bound_y": bound, "bound_u": 2, "label_modulus": 8}, x_multi, y_multi, "bench-tmcfe-lwe", 1)
    results = [
        _bench_sife({"sec_param": 64, "eta": eta, "lwe_n": lwe_n, "bound_x": bound, "bound_y": bound}, x_sife, y_sife, args.rounds),
        _bench_mife({"sec_param": 64, "eta": eta, "n": 3, "s": 1, "lst_nid": list(x_multi.keys()), "lwe_n": lwe_n, "bound_x": bound, "bound_y": bound}, x_multi, y_multi, args.rounds),
        _bench_mcfe({"sec_param": 64, "eta": eta, "n": 3, "s": 1, "lst_nid": list(x_multi.keys()), "lwe_n": lwe_n, "bound_x": bound, "bound_y": bound, "bound_u": 2, "label_modulus": 8}, x_multi, y_multi, "bench-mcfe-lwe", args.rounds),
        _bench_dmcfe({"sec_param": 64, "eta": eta, "n": 3, "s": 1, "lst_nid": list(x_multi.keys()), "lwe_n": lwe_n, "bound_x": bound, "bound_y": bound, "bound_u": 2, "label_modulus": 8}, x_multi, y_multi, "bench-dmcfe-lwe", args.rounds),
        _bench_tmcfe({"sec_param": 64, "eta": eta, "n": 3, "s": 3, "t": 2, "lst_nid": list(x_multi.keys()), "lst_sid": ["sid_0", "sid_1", "sid_2"], "lwe_n": lwe_n, "bound_x": bound, "bound_y": bound, "bound_u": 2, "label_modulus": 8}, x_multi, y_multi, "bench-tmcfe-lwe", args.rounds),
    ]
    for result in results:
        result["profile"] = args.profile
    emit_results(results, output=args.output, header=BENCHMARK_HEADER)


if __name__ == "__main__":
    main()
