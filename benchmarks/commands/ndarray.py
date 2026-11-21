"""Benchmark ndarray helper throughput across representative scheme families."""

from __future__ import annotations

import argparse
import time

import numpy as np

from _common import BENCHMARK_HEADER
from _common import emit_results
from _common import keygen_config
from _common import parse_args
from _common import run_cases
from _common import profile_value
from _common import prepare_benchmark
from _common import resolve_rounds
from _common import resolve_shape
from _common import stats
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPEPairing
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPEPairingKeyGenerator
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_decentralized import DecentralizedMCFEFHMultiIPEPairing
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_decentralized import DecentralizedMCFEFHMultiIPEPairingKeyGenerator
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold import ThresholdMCFEFHMultiIPEPairing
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold import ThresholdMCFEFHMultiIPEPairingKeyGenerator
from pyfe4ai.schemes.mife.fh_multi_ipe_pairing import MIFEFHMultiIPEPairing
from pyfe4ai.schemes.mife.fh_multi_ipe_pairing import MIFEFHMultiIPEPairingKeyGenerator
from pyfe4ai.schemes.quadratic.quad import QuadraticQuad
from pyfe4ai.schemes.quadratic.quad import QuadraticQuadKeyGenerator
from pyfe4ai.schemes.quadratic.sgp import QuadraticSGP
from pyfe4ai.schemes.quadratic.sgp import QuadraticSGPKeyGenerator
from pyfe4ai.schemes.sife.damgard_ddh import SIFEDamgard
from pyfe4ai.schemes.sife.damgard_ddh import SIFEDamgardKeyGenerator
from pyfe4ai.schemes.sife.fullysec_lwe import SIFEFullySecLWE
from pyfe4ai.schemes.sife.fullysec_lwe import SIFEFullySecLWEKeyGenerator
from pyfe4ai.schemes.sife.lwe import SIFELWE
from pyfe4ai.schemes.sife.lwe import SIFELWEKeyGenerator
from pyfe4ai.schemes.sife.paillier import SIFEPaillier
from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP
from pyfe4ai.schemes.sife.paillier import SIFEPaillierKeyGenerator

ROUNDS = 2
SHAPE = (8, 8)
PRECISION = 1
FH_SEC_LEVEL = 2
FH_BOUND_X = 50
FH_BOUND_Y = 4


def _base_array(scale: float) -> np.ndarray:
    rows, cols = SHAPE
    grid = np.arange(rows * cols, dtype=float).reshape(rows, cols)
    return ((grid % 7) * 0.1 + scale).astype(float)


def _bench_sife_helper(name: str, kg_cls, crypto_cls, config: dict, weight: int) -> dict:
    encrypt_times = []
    compute_times = []
    end_to_end_times = []
    elements = int(np.prod(SHAPE))

    payload = [_base_array(0.3)]
    expected = [
        ((payload[0] * pow(10, PRECISION)).astype(int) / pow(10, PRECISION)) * weight
    ]
    fusion_weight = [weight]

    for _ in range(ROUNDS):
        kg = kg_cls(keygen_config(config))
        kg.setup()
        pp = kg.get_public_parameters()
        enc = crypto_cls(
            {
                "id": "nid_default",
                "precision": PRECISION,
                "keys": {"pp": pp, "sk": kg.get_private_keys()},
            }
        )
        dec = crypto_cls({"id": "sid_0", "precision": PRECISION, "keys": {"pp": pp}})
        dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": fusion_weight})

        t0 = time.perf_counter()
        ct = enc.encrypt_lst_ndarray(payload)
        encrypt_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        computed = dec.compute_lst_ndarray_ct(ct, dk=dk, fusion_weight=fusion_weight)
        compute_t = time.perf_counter() - t0

        if not np.allclose(computed[0], expected[0], atol=1e-2):
            raise AssertionError(f"{name} ndarray helper produced an unexpected result")

        encrypt_times.append(encrypt_t)
        compute_times.append(compute_t)
        end_to_end_times.append(encrypt_t + compute_t)

    encrypt_mean, _, _ = stats(encrypt_times)
    compute_mean, _, _ = stats(compute_times)
    end_mean, end_median, end_std = stats(end_to_end_times)
    return {
        "scheme": name,
        "shape": f"{SHAPE[0]}x{SHAPE[1]}",
        "layers": 1,
        "output_elements": elements,
        "encrypt_mean_sec": encrypt_mean,
        "compute_mean_sec": compute_mean,
        "end_to_end_mean_sec": end_mean,
        "end_to_end_median_sec": end_median,
        "end_to_end_std_sec": end_std,
        "elements_per_sec": elements / end_mean if end_mean else 0.0,
    }


def _bench_mife_fh_multi_ipe() -> dict:
    encrypt_times = []
    compute_times = []
    end_to_end_times = []
    elements = int(np.prod(SHAPE))
    cfg = {
        "sec_param": 64,
        "sec_level": FH_SEC_LEVEL,
        "eta": 1,
        "n": 3,
        "s": 1,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "bound_x": FH_BOUND_X,
        "bound_y": FH_BOUND_Y,
        "pairing_group_param": DEFAULT_PAIRING_GROUP,
    }
    dct_y = {"nid_0": [1], "nid_1": [2], "nid_2": [-1]}
    dct_x = {
        "nid_0": [_base_array(0.1)],
        "nid_1": [_base_array(0.2)],
        "nid_2": [_base_array(0.3)],
    }
    expected = sum(
        (np.rint(dct_x[nid][0] * pow(10, PRECISION)) / pow(10, PRECISION))
        * dct_y[nid][0]
        for nid in cfg["lst_nid"]
    )

    for _ in range(ROUNDS):
        kg = MIFEFHMultiIPEPairingKeyGenerator(keygen_config(cfg))
        kg.setup()
        pp = kg.get_public_parameters()
        dct_ct = {}

        t0 = time.perf_counter()
        for nid in cfg["lst_nid"]:
            enc = MIFEFHMultiIPEPairing(
                {
                    "id": nid,
                    "precision": PRECISION,
                    "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
                }
            )
            dct_ct[nid] = enc.encrypt_lst_ndarray(dct_x[nid])
        encrypt_t = time.perf_counter() - t0

        dec = MIFEFHMultiIPEPairing({"id": "sid_0", "precision": PRECISION, "keys": {"pp": pp}})
        dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})

        t0 = time.perf_counter()
        computed = dec.compute_lst_ndarray_ct(dct_ct, dk=dk, fusion_weight=dct_y)
        compute_t = time.perf_counter() - t0

        if not np.allclose(computed[0], expected, atol=1e-2):
            raise AssertionError("MIFE/FHMultiIPE ndarray helper produced an unexpected result")

        encrypt_times.append(encrypt_t)
        compute_times.append(compute_t)
        end_to_end_times.append(encrypt_t + compute_t)

    encrypt_mean, _, _ = stats(encrypt_times)
    compute_mean, _, _ = stats(compute_times)
    end_mean, end_median, end_std = stats(end_to_end_times)
    return {
        "scheme": "MIFE/FHMultiIPE",
        "shape": f"{SHAPE[0]}x{SHAPE[1]}",
        "layers": 1,
        "output_elements": elements,
        "encrypt_mean_sec": encrypt_mean,
        "compute_mean_sec": compute_mean,
        "end_to_end_mean_sec": end_mean,
        "end_to_end_median_sec": end_median,
        "end_to_end_std_sec": end_std,
        "elements_per_sec": elements / end_mean if end_mean else 0.0,
    }


def _bench_mcfe_fh_multi_ipe(name: str, decentralized: bool = False, threshold: bool = False) -> dict:
    encrypt_times = []
    compute_times = []
    end_to_end_times = []
    elements = int(np.prod(SHAPE))
    base_cfg = {
        "sec_param": 64,
        "eta": 1,
        "n": 3,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "bound_x": FH_BOUND_X,
        "bound_y": FH_BOUND_Y,
        "pairing_group_param": DEFAULT_PAIRING_GROUP,
    }
    if threshold:
        base_cfg.update({"s": 3, "t": 2, "lst_sid": ["sid_0", "sid_1", "sid_2"]})
    else:
        base_cfg.update({"s": 1})

    dct_y = {"nid_0": [1], "nid_1": [2], "nid_2": [-1]}
    label = f"bench-{name.lower().replace('/', '-')}"
    dct_x = {
        "nid_0": [_base_array(0.1)],
        "nid_1": [_base_array(0.2)],
        "nid_2": [_base_array(0.3)],
    }
    expected = sum(
        ((dct_x[nid][0] * pow(10, PRECISION)).astype(int) / pow(10, PRECISION))
        * dct_y[nid][0]
        for nid in base_cfg["lst_nid"]
    )

    for _ in range(ROUNDS):
        if decentralized:
            kg = DecentralizedMCFEFHMultiIPEPairingKeyGenerator(keygen_config(base_cfg))
            actor_cls = DecentralizedMCFEFHMultiIPEPairing
        elif threshold:
            kg = ThresholdMCFEFHMultiIPEPairingKeyGenerator(keygen_config(base_cfg))
            actor_cls = ThresholdMCFEFHMultiIPEPairing
        else:
            kg = MCFEFHMultiIPEPairingKeyGenerator(keygen_config(base_cfg))
            actor_cls = MCFEFHMultiIPEPairing
        kg.setup()
        pp = kg.get_public_parameters()

        if threshold:
            dct_ct = {}
            t0 = time.perf_counter()
            for nid in base_cfg["lst_nid"]:
                enc = actor_cls(
                    {
                        "id": nid,
                        "precision": PRECISION,
                        "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
                    }
                )
                dct_ct[nid] = enc.encrypt_lst_ndarray(dct_x[nid], label=label)
            encrypt_t = time.perf_counter() - t0

            credentials = {"fusion_weight": dct_y, "label": label}
            enrolled = ["sid_0", "sid_1"]
            partials = {}
            t0 = time.perf_counter()
            for sid in enrolled:
                dk_sid = kg.get_decryption_keys(sid, credentials=credentials)
                dec = actor_cls(
                    {
                        "id": sid,
                        "precision": PRECISION,
                        "keys": {"pp": pp, "dk": dk_sid},
                    }
                )
                partials[sid] = dec.compute_lst_ndarray_ct(dct_ct, credentials, dk_sid, enrolled, label)
            combiner = actor_cls(
                {
                    "id": base_cfg["lst_nid"][0],
                    "precision": PRECISION,
                    "keys": {"pp": pp, "sk": kg.get_private_keys(base_cfg['lst_nid'][0])},
                }
            )
            computed = combiner.decrypt_lst_ndarray_ct(partials, label)
            compute_t = time.perf_counter() - t0
        elif decentralized:
            dct_ct = {}
            shares = {}
            t0 = time.perf_counter()
            for nid in base_cfg["lst_nid"]:
                actor = actor_cls(
                    {
                        "id": nid,
                        "precision": PRECISION,
                        "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
                    }
                )
                dct_ct[nid] = actor.encrypt_lst_ndarray(dct_x[nid], label=label)
                shares[nid] = actor.derive_function_decryption_key_share(dct_y[nid], "bench")
            encrypt_t = time.perf_counter() - t0

            dec = actor_cls({"id": "sid_0", "precision": PRECISION, "keys": {"pp": pp}})
            dk = dec.combine_function_decryption_key_share(shares)
            t0 = time.perf_counter()
            computed = dec.compute_lst_ndarray_ct(dct_ct, dk=dk, fusion_weight=dct_y, label=label)
            compute_t = time.perf_counter() - t0
        else:
            dct_ct = {}
            t0 = time.perf_counter()
            for nid in base_cfg["lst_nid"]:
                enc = actor_cls(
                    {
                        "id": nid,
                        "precision": PRECISION,
                        "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
                    }
                )
                dct_ct[nid] = enc.encrypt_lst_ndarray(dct_x[nid], label=label)
            encrypt_t = time.perf_counter() - t0

            dec = actor_cls({"id": "sid_0", "precision": PRECISION, "keys": {"pp": pp}})
            dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": dct_y})
            t0 = time.perf_counter()
            computed = dec.compute_lst_ndarray_ct(dct_ct, dk=dk, fusion_weight=dct_y, label=label)
            compute_t = time.perf_counter() - t0

        if not np.allclose(computed[0], expected, atol=1e-2):
            raise AssertionError(f"{name} ndarray helper produced an unexpected result")

        encrypt_times.append(encrypt_t)
        compute_times.append(compute_t)
        end_to_end_times.append(encrypt_t + compute_t)

    encrypt_mean, _, _ = stats(encrypt_times)
    compute_mean, _, _ = stats(compute_times)
    end_mean, end_median, end_std = stats(end_to_end_times)
    return {
        "scheme": name,
        "shape": f"{SHAPE[0]}x{SHAPE[1]}",
        "layers": 1,
        "output_elements": elements,
        "encrypt_mean_sec": encrypt_mean,
        "compute_mean_sec": compute_mean,
        "end_to_end_mean_sec": end_mean,
        "end_to_end_median_sec": end_median,
        "end_to_end_std_sec": end_std,
        "elements_per_sec": elements / end_mean if end_mean else 0.0,
    }


def _bench_quadratic(name: str, quad_cls, kg_cls, config: dict, lhs: list[np.ndarray], rhs: list[np.ndarray], function_matrix: list[list[int]]) -> dict:
    encrypt_times = []
    compute_times = []
    end_to_end_times = []
    elements = int(np.prod(SHAPE))
    expected = sum(
        function_matrix[i][j] * lhs[i] * rhs[j]
        for i in range(len(lhs))
        for j in range(len(rhs))
    )

    for _ in range(ROUNDS):
        kg = kg_cls(keygen_config(config))
        kg.setup()
        pp = kg.get_public_parameters()
        enc = quad_cls({"id": "nid_default", "precision": PRECISION, "keys": {"pp": pp}})
        dec = quad_cls({"id": "sid_0", "precision": PRECISION, "keys": {"pp": pp}})
        dk = kg.get_decryption_keys("sid_0", credentials={"function_matrix": function_matrix})

        t0 = time.perf_counter()
        ct = enc.encrypt_lst_ndarray(lhs, rhs_lst_ndarray=rhs)
        encrypt_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        computed = dec.compute_lst_ndarray_ct(ct, dk=dk)
        compute_t = time.perf_counter() - t0

        if not np.allclose(computed[0], expected, atol=1e-2):
            raise AssertionError(f"{name} ndarray helper produced an unexpected result")

        encrypt_times.append(encrypt_t)
        compute_times.append(compute_t)
        end_to_end_times.append(encrypt_t + compute_t)

    encrypt_mean, _, _ = stats(encrypt_times)
    compute_mean, _, _ = stats(compute_times)
    end_mean, end_median, end_std = stats(end_to_end_times)
    return {
        "scheme": name,
        "shape": f"{SHAPE[0]}x{SHAPE[1]}",
        "layers": 1,
        "output_elements": elements,
        "encrypt_mean_sec": encrypt_mean,
        "compute_mean_sec": compute_mean,
        "end_to_end_mean_sec": end_mean,
        "end_to_end_median_sec": end_median,
        "end_to_end_std_sec": end_std,
        "elements_per_sec": elements / end_mean if end_mean else 0.0,
    }


def main(argv=None, prog=None) -> None:
    global ROUNDS, SHAPE, FH_SEC_LEVEL, FH_BOUND_X, FH_BOUND_Y

    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["small", "medium", "large", "report", "mnist"],
        default="small",
        help="select a standard ndarray workload profile",
    )
    parser.add_argument(
        "--shape",
        default=None,
        help="override output shape as ROWSxCOLS, for example 16x16",
    )
    parser.add_argument(
        "--group",
        choices=["sife", "fh", "quadratic", "all"],
        default="sife",
        help="choose which ndarray-helper family group to benchmark",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="override the number of timing rounds",
    )
    parser.add_argument("--output", choices=["csv", "json"], default="csv")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parse_args(parser, argv)
    prepare_benchmark(args.seed)

    override_shape = None
    if args.shape:
        rows_str, cols_str = args.shape.lower().split("x", maxsplit=1)
        override_shape = (int(rows_str), int(cols_str))

    ROUNDS = resolve_rounds(
        ROUNDS,
        args.profile,
        {"small": 1, "medium": 2, "large": 3, "report": 5, "mnist": 2},
        args.rounds,
    )
    SHAPE = resolve_shape(
        SHAPE,
        args.profile,
        {
            "small": (8, 8),
            "medium": (16, 16),
            "large": (28, 28),
            "report": (6, 6),
            "mnist": (28, 28),
        },
        override_shape,
    )

    sife_lwe_n = profile_value(args.profile, {"small": 16, "medium": 20, "large": 24, "report": 24, "mnist": 24})
    sife_bound = profile_value(args.profile, {"small": 6, "medium": 8, "large": 12, "report": 12, "mnist": 12})
    sife_bit_length = profile_value(args.profile, {"small": 64, "medium": 96, "large": 128, "report": 128, "mnist": 128})
    FH_SEC_LEVEL = profile_value(args.profile, {"small": 2, "medium": 2, "large": 2, "report": 3, "mnist": 2})
    FH_BOUND_X = profile_value(args.profile, {"small": 32, "medium": 40, "large": 50, "report": 50, "mnist": 50})
    FH_BOUND_Y = profile_value(args.profile, {"small": 4, "medium": 4, "large": 6, "report": 6, "mnist": 6})
    quadratic_dim = profile_value(args.profile, {"small": 2, "medium": 2, "large": 3, "report": 3, "mnist": 2})
    quadratic_bound = profile_value(args.profile, {"small": 200, "medium": 300, "large": 500, "report": 500, "mnist": 300})

    def sife_lwe() -> dict:
        return _bench_sife_helper(
            "SIFE/LWE",
            SIFELWEKeyGenerator,
            SIFELWE,
            {"sec_param": 64, "eta": 1, "lwe_n": sife_lwe_n, "bound_x": FH_BOUND_X, "bound_y": 3},
            2,
        )

    def sife_group() -> list[dict]:
        return run_cases([
            sife_lwe,
            lambda: _bench_sife_helper(
                "SIFE/FullySecLWE",
                SIFEFullySecLWEKeyGenerator,
                SIFEFullySecLWE,
                {"sec_param": 64, "eta": 1, "lwe_n": sife_lwe_n + 4, "bound_x": FH_BOUND_X, "bound_y": 3},
                2,
            ),
            lambda: _bench_sife_helper(
                "SIFE/Paillier",
                SIFEPaillierKeyGenerator,
                SIFEPaillier,
                {"sec_param": 64, "eta": 1, "bit_length": sife_bit_length, "bound_x": FH_BOUND_X, "bound_y": 3},
                2,
            ),
            lambda: _bench_sife_helper(
                "SIFE/DamgardDDH",
                SIFEDamgardKeyGenerator,
                SIFEDamgard,
                {"sec_param": 64, "eta": 1, "modulus_length": sife_bit_length, "bound": max(20, sife_bound * 4)},
                3,
            ),
        ])

    def fh_group() -> list[dict]:
        results = [
            _bench_mife_fh_multi_ipe(),
            _bench_mcfe_fh_multi_ipe("MCFE/FHMultiIPE"),
        ]
        if args.profile in {"large", "report"}:
            results.extend(
                [
                    _bench_mcfe_fh_multi_ipe("dMCFE/FHMultiIPE", decentralized=True),
                    _bench_mcfe_fh_multi_ipe("tMCFE/FHMultiIPE", threshold=True),
                ]
            )
        return results

    def quadratic_sgp() -> dict:
        return _bench_quadratic(
            "Quadratic/SGP",
            QuadraticSGP,
            QuadraticSGPKeyGenerator,
            {"sec_param": 64, "n": quadratic_dim, "bound": quadratic_bound, "pairing_group_param": DEFAULT_PAIRING_GROUP},
            [_base_array(0.1 + 0.1 * i) for i in range(quadratic_dim)],
            [_base_array(0.3 + 0.1 * j) for j in range(quadratic_dim)],
            [[1 for _ in range(quadratic_dim)] for _ in range(quadratic_dim)],
        )

    def quadratic_group() -> list[dict]:
        return [
            quadratic_sgp(),
            _bench_quadratic(
                "Quadratic/Quad",
                QuadraticQuad,
                QuadraticQuadKeyGenerator,
                {"sec_param": 64, "n": quadratic_dim, "m": quadratic_dim + 1, "bound": quadratic_bound, "pairing_group_param": DEFAULT_PAIRING_GROUP},
                [_base_array(0.1 + 0.1 * i) for i in range(quadratic_dim)],
                [_base_array(0.3 + 0.1 * j) for j in range(quadratic_dim + 1)],
                [[1 for _ in range(quadratic_dim + 1)] for _ in range(quadratic_dim)],
            ),
        ]

    # (warm-up run, benchmark) per group; only the selected groups run, so the
    # sife group needs no pairing backend
    groups = {
        "sife": (sife_lwe, sife_group),
        "fh": (_bench_mife_fh_multi_ipe, fh_group),
        "quadratic": (quadratic_sgp, quadratic_group),
    }
    selected = list(groups) if args.group == "all" else [args.group]
    for _ in range(args.warmup):
        for name in selected:
            groups[name][0]()
    results = [result for name in selected for result in groups[name][1]()]

    for result in results:
        result["profile"] = args.profile
    emit_results(results, output=args.output, header=BENCHMARK_HEADER)

