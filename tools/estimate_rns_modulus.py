"""128-bit check of the RNS moduli (``ring_modulus="rns"``) at their exact value.

The table ``RING_LWE_128_LOG_Q`` bounds ``log2 q`` (it was computed for
``q = 2^k``); this runs the lattice estimator on the exact composite ``q``
of the FL configurations and on the prime it replaces, with the attacks and
settings of ``tools/estimate_ring_lwe.py``. Two steps, since the estimator
needs SageMath and the moduli need pyfe4ai::

    python tools/estimate_rns_modulus.py derive moduli.json
    python tools/estimate_rns_modulus.py estimate moduli.json --estimator /path/to/lattice-estimator

(the second under Sage's Python, e.g. with the ``sage-lwe`` environment's
``bin`` on ``PATH``). Results: docs/rns_modulus_review.md, section 9.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

#: (clients, bound_x, ring_n) with eta = 1, bound_y = 1, sec_param 128.
CONFIGS = ((3, 100, 4096), (3, 2, 4096), (10, 100, 4096), (3, 100, 8192))
SKIPPED = ("arora-gb", "bkw", "bdd_mitm_hybrid")


def derive(path: str) -> None:
    from pyfe4ai.utils import ring_lwe_utils as rlu

    out = []
    for clients, bound_x, ring_n in CONFIGS:
        args = (128, clients, bound_x, 1, ring_n)
        prime = rlu.bmms21_selective_parameters(*args, security_bound_x=3 * bound_x)
        rns = rlu.bmms21_selective_parameters(*args, security_bound_x=3 * bound_x, ring_modulus="rns")
        out.append({
            "clients": clients, "bound_x": bound_x, "n": ring_n, "sigma": prime["rlwe_width"],
            "q_prime": int(prime["q"]), "q_rns": int(rns["q"]), "factors": [int(f) for f in rns["q_factors"]],
            "table_bound": rlu.ring_lwe_bound(ring_n, prime["rlwe_width"]),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


def estimate(path: str, estimator: str) -> None:
    sys.path.insert(0, estimator)
    from estimator import LWE, ND
    from sage.all import ZZ, oo

    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    for row in rows:
        for key in ("q_rns", "q_prime"):
            error = ND.DiscreteGaussian(float(row["sigma"]))
            params = LWE.Parameters(n=row["n"], q=ZZ(row[key]), Xs=error, Xe=error, m=oo)
            result = LWE.estimate(params, deny_list=SKIPPED, quiet=True, jobs=1)
            row[key + "_log2_cost"] = min(float(math.log2(c["rop"])) for c in result.values())
        print("n {n} clients {clients} bound_x {bound_x}: q {bits} bits, rns {rc:.1f}, prime {pc:.1f} "
              "(table bound {table_bound})".format(bits=ZZ(row["q_rns"]).nbits(), rc=row["q_rns_log2_cost"],
                                                   pc=row["q_prime_log2_cost"], **row), flush=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["derive", "estimate"])
    parser.add_argument("path")
    parser.add_argument("--estimator", help="path of a lattice-estimator checkout (estimate)")
    args = parser.parse_args()
    if args.step == "derive":
        derive(args.path)
    else:
        if not args.estimator:
            parser.error("estimate needs --estimator")
        estimate(args.path, args.estimator)


if __name__ == "__main__":
    main()
