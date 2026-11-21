"""Parameter generation and setup time of every registered scheme.

The instantiation benchmarks time ``setup()`` only: parameters are generated
when the key generator is constructed, and cached in ``./config`` after the
first round. This script times both steps for every registry entry, each in a
fresh temporary directory (no cached parameters), with the configuration of
``run.py fl-coverage``; ``bcfe/gc_ristretto255``, which has no FL
configuration, is set up for the ``lr16`` circuit of ``run.py bcfe`` (built
once, outside the timed steps). Under ``--security-profile standard`` the
profile chooses the sizes; refused schemes are reported with the reason. An
entry whose configuration the key generator does not accept is reported as
``unusable`` with the reason, and the other rows are still printed.
Pairing-based entries need a backend for their group (they are reported as
``unverified`` without one).

Usage::

    python benchmarks/run.py setup-costs --security-profile standard --output json
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import time
import warnings

from _common import fresh_cache
from _common import keygen_config
from _common import parse_args
from _common import size_bytes
from _common import stats
from commands.bcfe import CIRCUITS as BCFE_CIRCUITS
from commands.fl_coverage import CONFIGS
from commands.fl_coverage import _needs_pairing
from pyfe4ai import circuits
from pyfe4ai.core.exceptions import FEConfigError
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.core.registry import SCHEMES
from pyfe4ai.core.registry import get_scheme
from pyfe4ai.primitives.pairing.pairing_backend import DEFAULT_PAIRING_GROUP
from pyfe4ai.primitives.pairing.pairing_backend import pairing_available

HEADER = ["scheme", "security_profile", "status", "rounds", "construct_mean_sec",
          "setup_mean_sec", "total_std_sec", "pp_bytes", "params", "note"]

#: The circuit of ``bcfe/gc_ristretto255`` here: a name of ``run.py bcfe``'s ``CIRCUITS``.
BCFE = ("bcfe", "gc_ristretto255")
BCFE_CIRCUIT = "lr16"


@functools.lru_cache(maxsize=None)
def _bcfe_circuit():
    builder, kwargs = BCFE_CIRCUITS[BCFE_CIRCUIT]
    return getattr(circuits, builder)(**kwargs)


def _config(key) -> dict:
    """The key generator configuration of an entry (before the profile's changes)."""
    if key == BCFE:
        return {"circuit": _bcfe_circuit()}
    return dict(CONFIGS[key])


def _params(pp: dict) -> dict:
    """The chosen sizes, from the public parameters: small integers as they
    are, large ones (moduli, group elements) as ``"<bits> bits"``, lists by
    their length, strings (group names) as they are."""
    out = {}
    for key, value in sorted(pp.items()):
        if isinstance(value, str) and value.lstrip("-").isdigit():
            value = int(value)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            out[key] = value if abs(value) < 2 ** 32 else "{} bits".format(abs(value).bit_length())
        elif isinstance(value, float):
            out[key] = round(value, 3)
        elif isinstance(value, (list, tuple)):
            out[key] = "list of {}".format(len(value))
        elif isinstance(value, str) and len(value) <= 32:
            out[key] = value
    return out


def measure(key, rounds: int, profile: str) -> dict:
    row = {"scheme": "/".join(key), "security_profile": profile, "rounds": rounds}
    if _needs_pairing(*key) and not pairing_available():
        return dict(row, status="unverified", note="no backend for pairing group {}".format(DEFAULT_PAIRING_GROUP))
    kg_cls = get_scheme(key).key_generator
    construct, setup, total, pp_bytes = [], [], [], 0
    for _ in range(rounds):
        with fresh_cache():
            try:
                t0 = time.perf_counter()
                kg = kg_cls(keygen_config(_config(key)))
                t1 = time.perf_counter()
                kg.setup()
                t2 = time.perf_counter()
                pp = kg.get_public_parameters()
                pp_bytes = size_bytes(pp)
                params = _params(pp) if isinstance(pp, dict) else {}
            except FEValidationError as exc:
                reason = str(exc).splitlines()[0][:200]
                if profile != "toy" and "parameter set" in str(exc):
                    return dict(row, status="refused", note=reason)
                return dict(row, status="unusable", note=reason)
            except FEConfigError as exc:
                return dict(row, status="unusable", note=str(exc).splitlines()[0][:200])
        construct.append(t1 - t0)
        setup.append(t2 - t1)
        total.append(t2 - t0)
    note = "circuit {}".format(BCFE_CIRCUIT) if key == BCFE else ""
    return dict(
        row, status="ok", construct_mean_sec=stats(construct)[0], setup_mean_sec=stats(setup)[0],
        total_std_sec=stats(total)[2], pp_bytes=pp_bytes, params=params, note=note,
    )


def main(argv=None, prog=None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", choices=["csv", "json"], default="csv")
    parser.add_argument("--schemes", nargs="*", help="family/variant entries (default: all)")
    args = parse_args(parser, argv)
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    keys = [s.key for s in SCHEMES if not args.schemes or s.id in args.schemes]
    rows = [measure(k, args.rounds, args.security_profile) for k in keys]
    if args.output == "json":
        print(json.dumps(rows, indent=2))
        return
    print(",".join(HEADER))
    for r in rows:
        print(",".join(
            "{:.4f}".format(r[h]) if isinstance(r.get(h), float)
            else (json.dumps(r[h]) if isinstance(r.get(h), dict) else str(r.get(h, ""))).replace(",", ";")
            for h in HEADER
        ))

