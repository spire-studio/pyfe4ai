"""Regenerate ``WIDE_LWE_128_LOG_RATIO`` of :mod:`pyfe4ai.utils.lwe_utils`.

The ABDP15 LWE IPFE (``sife/lwe``, ``mife/lwe``) publishes ``pk = A S + E``:
LWE in dimension ``n`` with a uniform secret and very wide errors (the
GoFE parameter recipe gives ``sigma`` of 2^30 to 2^45). For every error width
``2^s`` of ``LOG_SIGMAS`` and every ``n`` of the grid this finds the largest
``k`` such that LWE with ``q = 2^(k + s)`` (so ``log2(q / sigma) = k``),
errors ``D_{Z, 2^s}`` and unlimited samples costs at least ``2^128``
operations for every attack below, by the lattice estimator
(https://github.com/malb/lattice-estimator), default cost model (MATZOV):

* primal uSVP and BDD, dual;
* skipped: Arora-GB, coded BKW, the primal hybrids and the dual hybrid, which
  target narrow, small or sparse secrets -- the uniform secret is taken in
  normal form, with the error's width. (The dual hybrid stays more than 20
  bits above the others at ``sigma = 2^16`` and crashes Sage at
  ``sigma >= 2^30``.)

At a fixed ratio ``q / sigma`` the attacks get more expensive as ``sigma``
grows (the secret's ``n`` coordinates grow with it but add no ``q``-volume),
so a scheme is held to the row of the largest tabulated width not above its
own ``sigma``.

Needs SageMath. Usage::

    python tools/estimate_wide_lwe.py --estimator /path/to/lattice-estimator
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys

TARGET_BITS = 128
LOG_SIGMAS = (16, 24, 32, 40)
GRID = range(512, 2048 + 1, 64)
ATTACKS = ("usvp", "bdd", "dual")


def cost_bits(lwe, nd, n: int, log_ratio: int, log_sigma: int) -> dict:
    """Return ``{attack: log2(rop)}`` for ``LWE(n, 2^(log_ratio + log_sigma), 2^log_sigma)``."""
    from sage.all import ZZ, oo

    # a Python float: an RR width drives Sage's symbolic powers into a crash
    error = nd.DiscreteGaussian(float(2.0 ** log_sigma))
    params = lwe.Parameters(n=n, q=ZZ(2) ** (log_ratio + log_sigma), Xs=error, Xe=error, m=oo)
    attacks = {"usvp": lwe.primal_usvp, "bdd": lwe.primal_bdd, "dual": lwe.dual}
    return {name: float(math.log2(attacks[name](params)["rop"])) for name in ATTACKS}


def max_log_ratio(lwe, nd, n: int, log_sigma: int) -> tuple:
    """Largest ``k`` with every attack at ``>= 2^TARGET_BITS``."""
    lo, hi = 4, n // 8  # lo secure, hi broken (checked)
    assert min(cost_bits(lwe, nd, n, lo, log_sigma).values()) >= TARGET_BITS
    assert min(cost_bits(lwe, nd, n, hi, log_sigma).values()) < TARGET_BITS
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if min(cost_bits(lwe, nd, n, mid, log_sigma).values()) >= TARGET_BITS:
            lo = mid
        else:
            hi = mid
    return lo, cost_bits(lwe, nd, n, lo, log_sigma)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--estimator", required=True, help="path of a lattice-estimator checkout")
    parser.add_argument("--dims", help="comma-separated dimensions instead of the whole grid")
    parser.add_argument("--log-sigmas", help="comma-separated log2 widths instead of LOG_SIGMAS")
    args = parser.parse_args()
    sys.path.insert(0, args.estimator)
    from estimator import LWE, ND

    commit = subprocess.run(
        ["git", "-C", args.estimator, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print("# lattice-estimator", commit)
    dims = [int(d) for d in args.dims.split(",")] if args.dims else list(GRID)
    log_sigmas = [int(s) for s in args.log_sigmas.split(",")] if args.log_sigmas else LOG_SIGMAS
    table = {}
    for log_sigma in log_sigmas:
        table[log_sigma] = {}
        for n in dims:
            table[log_sigma][n], costs = max_log_ratio(LWE, ND, n, log_sigma)
            print(
                "sigma=2^{} n={} log2(q/sigma)<={}: {}".format(
                    log_sigma, n, table[log_sigma][n],
                    ", ".join("{} 2^{:.1f}".format(k, v) for k, v in sorted(costs.items())),
                ),
                flush=True,
            )
    print("WIDE_LWE_128_LOG_RATIO = {")
    for log_sigma, row in table.items():
        print("    {}: {{{}}},".format(log_sigma, ", ".join("{}: {}".format(n, k) for n, k in row.items())))
    print("}")


if __name__ == "__main__":
    main()
