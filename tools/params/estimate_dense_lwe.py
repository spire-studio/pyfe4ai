"""Regenerate ``DENSE_LWE_128_LOG_Q`` of :mod:`pyfe4ai.primitives.lattice.lwe_utils`.

For every LWE dimension ``n`` of the grid, finds the largest ``k`` such that
LWE with ``n``, ``q = 2^k``, a uniform secret modulo ``q``, errors
``D_{Z, ceil(2 sqrt(n))}`` and unlimited samples costs at least ``2^128``
operations for every attack below, as estimated by the lattice estimator
(Albrecht, Player, Scott, J. Math. Cryptol. 2015,
https://github.com/malb/lattice-estimator) with its default cost model
(MATZOV) and shape model:

* primal uSVP and BDD, dual and dual hybrid;
* skipped: Arora-GB (needs very narrow errors), coded BKW and the primal
  hybrids, which target small or sparse secrets -- the estimator moves the
  uniform secret to normal form, where it has the error's width
  (``sigma`` of about 130 to 260 here); ``--check-skipped`` re-runs them at
  every bound found.

The error width is the one of the ABMPSW20 LWE MCFE (``sigma_e`` in
:func:`pyfe4ai.primitives.lattice.lwe_utils.abmpsw20_lwe_parameters`), so the table
bounds the LWE problem its security proof reduces to (SCN 2020, Appendix C.2,
game G2: LWE in dimension ``n0`` modulo the scheme's ``q``).

Needs SageMath. Usage::

    python tools/params/estimate_dense_lwe.py --estimator /path/to/lattice-estimator
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys

TARGET_BITS = 128
GRID = range(4096, 16384 + 1, 512)
SKIPPED = ("arora-gb", "bkw", "bdd_hybrid", "bdd_mitm_hybrid")
ATTACKS = ("usvp", "bdd", "dual", "dual_hybrid")


def cost_bits(lwe, nd, n: int, log_q: int, deny=SKIPPED) -> dict:
    """Return ``{attack: log2(rop)}`` for ``LWE(n, 2^log_q, sigma = ceil(2 sqrt(n)))``."""
    from sage.all import ZZ, oo

    error = nd.DiscreteGaussian(math.ceil(2 * math.sqrt(n)))
    # uniform secret in normal form (secret distributed as the error), which is
    # what LWEParameters.normalize() does with unlimited samples; passing
    # nd.UniformMod(q) itself overflows a float for q > 2^512
    params = lwe.Parameters(n=n, q=ZZ(2) ** log_q, Xs=error, Xe=error, m=oo)
    result = lwe.estimate(params, deny_list=deny, quiet=True, jobs=1)
    # an attack that raised is left out of the result (the estimator only logs
    # it), which would silently shrink the minimum taken over the attacks
    missing = [a for a in ATTACKS if a not in deny and a not in result]
    if missing:
        raise RuntimeError("no estimate from {} at n={} log2q={}".format(missing, n, log_q))
    return {name: float(math.log2(cost["rop"])) for name, cost in result.items()}


def max_log_q(lwe, nd, n: int) -> tuple:
    """Largest ``k`` with every attack on ``LWE(n, 2^k)`` at ``>= 2^TARGET_BITS``."""
    lo, hi = 16, n // 16  # lo secure, hi broken (checked)
    assert min(cost_bits(lwe, nd, n, lo).values()) >= TARGET_BITS
    assert min(cost_bits(lwe, nd, n, hi).values()) < TARGET_BITS
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if min(cost_bits(lwe, nd, n, mid).values()) >= TARGET_BITS:
            lo = mid
        else:
            hi = mid
    return lo, cost_bits(lwe, nd, n, lo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--estimator", required=True, help="path of a lattice-estimator checkout")
    parser.add_argument("--check-skipped", action="store_true", help="also run the skipped attacks")
    parser.add_argument("--dims", help="comma-separated dimensions instead of the whole grid")
    args = parser.parse_args()
    sys.path.insert(0, args.estimator)
    from estimator import LWE, ND

    commit = subprocess.run(
        ["git", "-C", args.estimator, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print("# lattice-estimator", commit)
    table = {}
    for n in [int(d) for d in args.dims.split(",")] if args.dims else GRID:
        table[n], costs = max_log_q(LWE, ND, n)
        line = "n={} log2q<={}: {}".format(
            n, table[n], ", ".join("{} 2^{:.1f}".format(k, v) for k, v in sorted(costs.items()))
        )
        if args.check_skipped:
            extra = cost_bits(LWE, ND, n, table[n], deny=("arora-gb",))
            line += " | with skipped: min 2^{:.1f}".format(min(extra.values()))
        print(line, flush=True)
    print("DENSE_LWE_128_LOG_Q = {")
    for n, bits in table.items():
        print("    {}: {},".format(n, bits))
    print("}")


if __name__ == "__main__":
    main()
