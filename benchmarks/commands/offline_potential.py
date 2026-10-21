"""Offline / online potential of the MCFE schemes built on GMP arithmetic.

For one element (``eta = 1``) of the FL workload under the ``standard``
profile, times the scheme's encryption and decryption and their split into

* a part that depends only on the label and the key -- known before the
  round, so it can be computed offline: the client mask ``[u_l]^s`` (DDH,
  CDGPP18) or ``H(l)^s mod N^2`` (DCR, ABMPSW20), the aggregator term
  ``[u_l]^d`` or ``H(l)^{-k}``;
* the rest, online: ``g^x`` times the mask, or ``(1 + x N)`` times it, and
  the aggregator's product of ``c_i^{y_i}``, the division and the dlog.

The split recomputes the scheme's own formulas and checks that it yields
the same ciphertext and result. Usage::

    python benchmarks/run.py offline-potential [--families ddh paillier] [--reps 20]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import tempfile
import time
import warnings
from unittest import mock

import gmpy2 as gp


def _t(fn, reps):
    out = None
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3, out


def ddh(reps: int) -> dict:
    from pyfe4ai.schemes.mcfe import ddh as mod
    from pyfe4ai.ml import FESchemeWrapper

    nids = ["c0", "c1", "c2"]
    w = FESchemeWrapper("mcfe", "ddh", {"eta": 1, "n": 3, "lst_nid": nids, "security_profile": "standard"},
                        precision=2)
    cl = w._get_encryptor("c0")
    agg = w._get_decryptor("a")
    p, g = gp.mpz(w.pp["p"]), gp.mpz(w.pp["g"])
    s = [gp.mpz(v) for v in cl.sk["s"][0]]
    label, x = "elem-label", 57
    full, ct = _t(lambda: cl.encrypt([x], label), reps)
    u = mod._hash_label(label, p)
    t_hash, _ = _t(lambda: mod._hash_label(label, p), reps)
    off, mask = _t(lambda: gp.mul(gp.powmod(u[0], s[0], p), gp.powmod(u[1], s[1], p)) % p, reps)
    on, c = _t(lambda: gp.mul(gp.powmod(g, x, p), mask) % p, reps)
    assert gp.digits(c) == ct["c"][0]
    cts = {n: w._get_encryptor(n).encrypt([x + i], label) for i, n in enumerate(nids)}
    weights = {n: [1] for n in nids}
    dk = w.kg.get_decryption_keys("a", credentials={"fusion_weight": weights})
    dec_full, res = _t(lambda: agg.decrypt(cts, dk, weights, label), reps)
    d = [gp.mpz(v) for v in dk["d"]]
    dec_off, kterm = _t(lambda: gp.mul(gp.powmod(u[0], d[0], p), gp.powmod(u[1], d[1], p)) % p, reps)

    def online_dec():
        prod = gp.mpz(1)
        for n in nids:
            prod = gp.mul(prod, gp.powmod(gp.mpz(cts[n]["c"][0]), 1, p)) % p
        return agg._solve_dlog(gp.divm(prod, kterm, p))

    dec_on, res2 = _t(online_dec, reps)
    assert res == res2 == 3 * x + 3
    t_pow, _ = _t(lambda: gp.powmod(u[0], s[0], p), reps)
    return {"family": "ddh", "group_bits": int(p).bit_length(), "powmod_ms": t_pow, "hash_ms": t_hash,
            "client_full_ms": full, "client_offline_ms": off + t_hash, "client_online_ms": on,
            "agg_full_ms": dec_full, "agg_offline_ms": dec_off + t_hash, "agg_online_ms": dec_on}


def paillier(reps: int) -> dict:
    from pyfe4ai.schemes.mcfe import paillier as mod
    from pyfe4ai.ml import FESchemeWrapper

    nids = ["c0", "c1", "c2"]
    w = FESchemeWrapper("mcfe", "paillier", {"eta": 1, "n": 3, "lst_nid": nids, "security_profile": "standard",
                                             "bound_x": 100, "bound_y": 1}, precision=2)
    cl = w._get_encryptor("c0")
    agg = w._get_decryptor("a")
    n_mod = gp.mpz(w.pp["n_mod"])
    n2 = n_mod * n_mod
    s = gp.mpz(cl.sk["s"][0])
    label, x = "elem-label", 57
    full, ct = _t(lambda: cl.encrypt([x], label), reps)
    t_hash, h = _t(lambda: mod._hash_label(label, n_mod), reps)
    off, mask = _t(lambda: mod.pow_signed(h, s, n2), reps)
    on, c = _t(lambda: (1 + x * n_mod) % n2 * mask % n2, reps)
    assert gp.digits(c) == ct["c"][0]
    cts = {n: w._get_encryptor(n).encrypt([x + i], label) for i, n in enumerate(nids)}
    weights = {n: [1] for n in nids}
    dk = w.kg.get_decryption_keys("a", credentials={"fusion_weight": weights})
    dec_full, res = _t(lambda: agg.decrypt(cts, dk, weights, label), reps)
    dec_off, kterm = _t(lambda: mod.pow_signed(h, -gp.mpz(dk["k"]), n2), reps)

    def online_dec():
        acc = gp.mpz(1)
        for n in nids:
            acc = acc * gp.mpz(cts[n]["c"][0]) % n2
        acc = acc * kterm % n2
        ret = ((acc - 1) % n2) // n_mod
        return int(ret - n_mod if ret > n_mod // 2 else ret)

    dec_on, res2 = _t(online_dec, reps)
    assert res == res2 == 3 * x + 3
    return {"family": "paillier", "n_bits": int(n_mod).bit_length(), "key_bits": int(abs(s)).bit_length(),
            "hash_ms": t_hash, "client_full_ms": full, "client_offline_ms": off + t_hash, "client_online_ms": on,
            "agg_full_ms": dec_full, "agg_offline_ms": dec_off + t_hash, "agg_online_ms": dec_on}


def main(argv=None, prog=None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--families", nargs="+", default=["ddh", "paillier"], choices=["ddh", "paillier"])
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    cwd = os.getcwd()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            for fam in args.families:
                with mock.patch.dict(os.environ, {"PYFE4AI_CACHE_DIR": os.path.join(tmp, "config")}):
                    row = {"ddh": ddh, "paillier": paillier}[fam](args.reps)
                rows.append(row)
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))
        finally:
            os.chdir(cwd)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

