"""Write the record fixtures of the stable schemes (see :mod:`tests.compat.recipes`).

Run once per release that freezes fixtures, from the repository root::

    python -m tests.compat.gen_fixtures --out tests/compat/fixtures/1.0.0

It writes one ``<family>__<variant>.json`` per scheme, ``markers.json`` (the constants of
:mod:`pyfe4ai.core.versions` and the other version markers, which later versions must keep) and
``manifest.json`` (version, commit, backend, schemes). Fixtures are data once written: the tests read them and
never regenerate them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pyfe4ai
from pyfe4ai.core import stability
from tests.compat import recipes


def marker_snapshot() -> dict:
    """The version markers that records carry: the upper-case constants of :mod:`pyfe4ai.core.versions` (but the
    error text and the pickle state version, which are not part of the promise) and the policy compiler marker."""
    from pyfe4ai.core import versions
    from pyfe4ai.policy import encoding

    skip = {"MIXED_SHARINGS", "TMCFE_RING_LWE_STATE_VERSION"}
    snap = {name: value for name, value in vars(versions).items()
            if name.isupper() and not name.startswith("_") and name not in skip}
    snap["policy.encoding.POLICY_COMPILER"] = encoding.POLICY_COMPILER
    return snap


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--schemes", nargs="*", help="default: every stable scheme")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    schemes = args.schemes or stability.stable_schemes()
    commit = _commit()
    written, failed = [], {}
    for scheme in schemes:
        if not recipes.available(scheme):
            failed[scheme] = "needs a pairing group (the native extension)"
            continue
        start = time.perf_counter()
        try:
            fx = recipes.generate(scheme)
        except Exception as exc:  # noqa: BLE001 -- reported in the manifest
            failed[scheme] = "{}: {}".format(type(exc).__name__, exc)
            print("FAILED", scheme, failed[scheme], file=sys.stderr)
            continue
        fx["commit"] = commit
        fx["tier"] = stability.config_tier(scheme, fx["config"])
        recipes.dump(fx, args.out / (scheme.replace("/", "__") + ".json"))
        written.append(scheme)
        print("{:45s} {:6.2f} s".format(scheme, time.perf_counter() - start))
    recipes.dump(marker_snapshot(), args.out / "markers.json")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = pyfe4ai.backend_info()
    recipes.dump({"fixture_format": recipes.FIXTURE_FORMAT, "pyfe4ai_version": pyfe4ai.__version__, "commit": commit,
                  "backend": info["backend"], "native_build": info.get("native_build"), "schemes": written,
                  "failed": failed}, args.out / "manifest.json")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
