"""Regenerate ``RING_LWE_128_LOG_Q`` of :mod:`pyfe4ai.primitives.lattice.ring_lwe_utils`.

For every error width ``sigma`` of ``SIGMAS`` and every ring dimension ``n``
of ``DIMS`` this finds the largest ``k`` such that LWE in dimension ``n`` with
``q = 2^k``, secret and errors ``D_{Z, sigma}`` and unlimited samples costs at
least ``2^128`` operations for every attack below, by the lattice estimator
(https://github.com/malb/lattice-estimator), default cost model (MATZOV).
Ring-LWE is estimated as plain LWE (no attack exploiting the ring structure
is known), as the Homomorphic Encryption Standard does.

* primal uSVP, BDD and BDD hybrid, dual and dual hybrid;
* skipped: Arora-GB (needs very narrow errors), coded BKW and the
  meet-in-the-middle primal hybrid (which the estimator itself calls overly
  optimistic).

A scheme is held to the row of the largest tabulated width not above the
smallest width among its RLWE secrets and errors: wider Gaussians only make
the attacks more expensive. This replaces the Homomorphic Encryption
Standard table, which this estimator puts at about 125 bits (ternary secret,
sigma = 3.2: 2^125.1 at n = 4096, log2 q = 109).

Needs SageMath. Usage::

    python tools/estimate_ring_lwe.py --estimator /path/to/lattice-estimator
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys

TARGET_BITS = 128
DIMS = (1024, 2048, 4096, 8192, 16384, 32768)
SIGMAS = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
ATTACKS = ("usvp", "bdd", "bdd_hybrid", "dual", "dual_hybrid")
SKIPPED = ("arora-gb", "bkw", "bdd_mitm_hybrid")


def cost_bits(lwe, nd, n: int, log_q: int, sigma: float) -> dict:
    """Return ``{attack: log2(rop)}``; raises if an expected attack gave no cost."""
    from sage.all import ZZ, oo

    error = nd.DiscreteGaussian(float(sigma))
    params = lwe.Parameters(n=n, q=ZZ(2) ** log_q, Xs=error, Xe=error, m=oo)
    result = lwe.estimate(params, deny_list=SKIPPED, quiet=True, jobs=1)
    # an attack that raised is left out of the result (the estimator only logs
    # it), which would silently shrink the minimum below; rop = oo means the
    # attack does not apply, i.e. costs more than any bound
    missing = [a for a in ATTACKS if a not in result]
    if missing:
        raise RuntimeError("no estimate from {} at n={} log2q={}".format(missing, n, log_q))
    return {name: float(math.log2(cost["rop"])) for name, cost in result.items()}


def max_log_q(lwe, nd, n: int, sigma: float) -> tuple:
    # lo secure, hi broken (checked); at n // 64 the dual hybrid fails outright
    # for sigma = 12 (inf - inf), so start a little higher
    lo, hi = n // 44, n // 4
    assert min(cost_bits(lwe, nd, n, lo, sigma).values()) >= TARGET_BITS
    assert min(cost_bits(lwe, nd, n, hi, sigma).values()) < TARGET_BITS
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if min(cost_bits(lwe, nd, n, mid, sigma).values()) >= TARGET_BITS:
            lo = mid
        else:
            hi = mid
    return lo, cost_bits(lwe, nd, n, lo, sigma)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--estimator", required=True, help="path of a lattice-estimator checkout")
    parser.add_argument("--dims", help="comma-separated ring dimensions instead of DIMS")
    parser.add_argument("--sigmas", help="comma-separated widths instead of SIGMAS")
    args = parser.parse_args()
    sys.path.insert(0, args.estimator)
    from estimator import LWE, ND

    commit = subprocess.run(
        ["git", "-C", args.estimator, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print("# lattice-estimator", commit)
    dims = [int(d) for d in args.dims.split(",")] if args.dims else list(DIMS)
    sigmas = [float(s) for s in args.sigmas.split(",")] if args.sigmas else list(SIGMAS)
    table = {}
    for sigma in sigmas:
        table[sigma] = {}
        for n in dims:
            table[sigma][n], costs = max_log_q(LWE, ND, n, sigma)
            print("sigma={} n={} log2q<={}: {}".format(
                sigma, n, table[sigma][n],
                ", ".join("{} 2^{:.1f}".format(k, v) for k, v in sorted(costs.items())),
            ), flush=True)
    print("RING_LWE_128_LOG_Q = {")
    for sigma, row in table.items():
        print("    {}: {{{}}},".format(sigma, ", ".join("{}: {}".format(n, k) for n, k in row.items())))
    print("}")


if __name__ == "__main__":
    main()
