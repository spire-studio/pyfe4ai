"""Replicated key sharing of the LWE threshold layers: dealing and partial
decryptions with a cold share-sum cache (a new server object per call, as an
FL round with a new key), server 1 of E = {1..t} (it adds C(s-1, t-1) pieces).

  ring: tMCFE-Ring-LWE, standard, 3 clients, X = 100, Y = 1, 2 chunk keys
        (10^4-element ndarray); also the 10^4 ndarray partial. Whole key shares
        as JSON on a prime q (key_shares="full", ct_format="json",
        ring_modulus="prime"), comparable with earlier runs; the scheme's
        default split binary key shares on an RNS q: run.py fl-compare
  mife: tMIFE-LWE, toy (standard refuses it), 3 clients, X = Y = 50, lwe_n 1024
  lwe:  tMCFE-LWE, standard, 3 clients, X = 100, Y = 1

Usage: python benchmarks/run.py replicated-sharing KIND S T SHARING [REPS]   (one JSON line;
       the parameter cache goes to PYFE4AI_CACHE_DIR or the user's cache directory)
"""
import json, statistics, sys, time, warnings


def timed(f):
    t0 = time.perf_counter()
    out = f()
    return time.perf_counter() - t0, out


def main(argv=None, prog=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) < 4 or argv[0] in ("-h", "--help"):
        print("usage: {} KIND S T SHARING [REPS]\n\n{}".format(prog or "replicated_sharing", __doc__.strip()))
        raise SystemExit(0 if argv[:1] in (["-h"], ["--help"]) else 2)
    warnings.simplefilter("ignore")
    import numpy as np
    import gmpy2 as gp
    from pyfe4ai.schemes.threshold.sharing import share_for_enrolled

    kind, s, t, sharing = argv[0], int(argv[1]), int(argv[2]), argv[3]
    reps = int(argv[4]) if len(argv) > 4 else 5
    nids, sids = ["c0", "c1", "c2"], ["s%d" % i for i in range(s)]
    base = {"n": 3, "lst_nid": nids, "s": s, "t": t, "lst_sid": sids, "eta": 1, "sharing": sharing}
    if kind == "ring":
        from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWE as S, ThresholdMCFERingLWEKeyGenerator as K
        kg = K(dict(base, bound_x=100, bound_y=1, security_profile="standard", key_shares="full", ct_format="json",
                    ring_modulus="prime"))
    elif kind == "mife":
        from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE as S, ThresholdMIFELWEKeyGenerator as K
        kg = K(dict(base, bound_x=50, bound_y=50, lwe_n=1024))
    else:
        from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWE as S, ThresholdMCFELWEKeyGenerator as K
        kg = K(dict(base, bound_x=100, bound_y=1, security_profile="standard"))
    kg.setup()
    pp = kg.get_public_parameters()
    y = {n: [1] for n in nids}
    enc = {n: S({"id": n, "precision": 2, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}) for n in nids}
    if kind == "ring":
        rng = np.random.default_rng(1)
        nd = {n: enc[n].encrypt_lst_ndarray([rng.uniform(-1, 1, 10000).round(2)], label="r") for n in nids}
        cred = {"fusion_weight": y, "label": "r", "n_chunks": enc["c0"].key_chunks(nd["c0"])}
        cts = {n: enc[n].encrypt([[1, 2, 3]], "r") for n in nids}
        arg = cred
    elif kind == "mife":
        cred = {"fusion_weight": y}
        cts = {n: enc[n].encrypt([7]) for n in nids}
        arg = y
    else:
        cred = {"fusion_weight": y, "label": "r"}
        cts = {n: enc[n].encrypt([7], "r") for n in nids}
        arg = cred
    E = sids[:t]

    deal = []
    for _ in range(reps):
        kg.dict_dk = {}
        d, dk = timed(lambda: kg.get_decryption_keys(sids[0], credentials=cred))
        deal.append(d)
    dks = {sid: kg.get_decryption_keys(sid, credentials=cred) for sid in sids}
    sizes = [len(json.dumps(dks[sid])) for sid in sids]
    q = gp.mpz(pp["q"])
    fields = ("sk_y", "z") if kind == "ring" else None
    sums = [timed(lambda: share_for_enrolled(dks[sids[0]], sids[0], E, t, q, fields=fields,
                                            ring_n=pp.get("ring_n")))[0] for _ in range(reps)]
    cold, warm = [], []
    for _ in range(reps):
        server = S({"id": sids[0], "precision": 2, "keys": {"pp": pp, "dk": dks[sids[0]]}})
        cold.append(timed(lambda: server.share_decrypt(cts, arg, dks[sids[0]], E))[0])
        warm.append(timed(lambda: server.share_decrypt(cts, arg, dks[sids[0]], E))[0])
    row = {"kind": kind, "s": s, "t": t, "sharing": sharing, "q_bits": int(q).bit_length(),
           "dk_bytes_max": max(sizes), "deal_ms": round(1e3 * statistics.median(deal), 1),
           "sum_ms": round(1e3 * statistics.median(sums), 1),
           "partial_cold_ms": round(1e3 * statistics.median(cold), 1),
           "partial_warm_ms": round(1e3 * statistics.median(warm), 1)}
    if kind == "ring":
        nd_cold = []
        for _ in range(3):
            server = S({"id": sids[0], "precision": 2, "keys": {"pp": pp, "dk": dks[sids[0]]}})
            nd_cold.append(timed(lambda: server.compute_lst_ndarray_ct(nd, cred, dks[sids[0]], E))[0])
        row["ndarray_1e4_cold_ms"] = round(1e3 * statistics.median(nd_cold), 1)
    print(json.dumps(row), flush=True)
