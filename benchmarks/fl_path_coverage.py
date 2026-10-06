"""Enumerate which ML-adapter registry entries support the FL aggregation path.

For every ``(scheme_type, variant)`` in ``pyfe4ai.utils.ml_adapter._SCHEME_REGISTRY``
this script builds a :class:`FESchemeWrapper` with the variant's own test
configuration (``eta`` forced to 1, because the FL helpers are defined for
``eta = 1``), encrypts one small integer gradient per client with
``encrypt_gradient()``, aggregates with ``aggregate_gradients()`` using integer
weight 1 for every client, and compares the result with the plaintext sum.

Status per entry:

* ``usable``     -- the aggregate equals the expected sum;
* ``unusable``   -- an exception or a wrong result (the reason is printed);
* ``unverified`` -- pairing-based entry and ``charm-crypto`` is not installed.

The run happens in a temporary directory because key generators write
``./config/``. Usage::

    python benchmarks/fl_path_coverage.py            # Markdown table
    python benchmarks/fl_path_coverage.py --output json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import traceback

import numpy as np

import pyfe4ai
from pyfe4ai.utils.ml_adapter import (
    _SCHEME_REGISTRY,
    FESchemeWrapper,
    aggregate_gradients,
    encrypt_gradient,
)
from pyfe4ai.utils.pairing_backend import PAIRING_IMPORT_ERROR

NIDS = ["nid_0", "nid_1", "nid_2"]
SIDS = ["sid_0", "sid_1", "sid_2"]
LABEL = "fl-round-1"

_MULTI = {"n": 3, "s": 1, "lst_nid": NIDS}
_THRESHOLD = {"n": 3, "s": 3, "t": 2, "lst_nid": NIDS, "lst_sid": SIDS}

# Configurations taken from each variant's test fixture (tests/<family>/test_*.py).
CONFIGS: dict[tuple[str, str], dict] = {
    ("sife", "ddh"): {"sec_param": 128},
    ("sife", "lwe"): {"sec_param": 64, "lwe_n": 16, "bound_x": 8, "bound_y": 8},
    ("sife", "paillier"): {"sec_param": 64, "bit_length": 64, "bound_x": 10, "bound_y": 10},
    ("sife", "damgard"): {"sec_param": 64, "modulus_length": 64, "bound": 10},
    ("sife", "ring_lwe"): {"sec_param": 64, "ring_n": 16, "bound_x": 2, "bound_y": 2},
    ("sife", "fullysec_lwe"): {"sec_param": 64, "lwe_n": 24, "bound_x": 10, "bound_y": 10},
    ("sife", "ddh_dynamic"): {"sec_param": 128},
    ("sife", "fh_ipe_pairing"): {"sec_param": 64, "bound_x": 8, "bound_y": 8, "pairing_group_param": "SS512"},
    ("sife", "part_fh_ipe_pairing"): {
        "sec_param": 64, "bound": 10, "subspace_dim": 3, "coeff_bound": 1,
        "pairing_group_param": "SS512",
    },
    ("mife", "ddh"): {"sec_param": 256, **_MULTI},
    ("mife", "lwe"): {"sec_param": 64, "lwe_n": 16, "bound_x": 6, "bound_y": 6, **_MULTI},
    ("mife", "paillier"): {"sec_param": 64, "bit_length": 64, "bound_x": 10, "bound_y": 10, **_MULTI},
    ("mife", "damgard"): {"sec_param": 64, "modulus_length": 64, "bound": 10, **_MULTI},
    ("mife", "ring_lwe"): {"sec_param": 64, "ring_n": 16, "bound_x": 2, "bound_y": 2, **_MULTI},
    ("mife", "fullysec_lwe"): {"sec_param": 64, "lwe_n": 24, "bound_x": 8, "bound_y": 8, **_MULTI},
    ("mife", "fh_ipe_pairing"): {
        "sec_param": 64, "bound_x": 6, "bound_y": 6, "pairing_group_param": "SS512", **_MULTI,
    },
    ("mife", "fh_multi_ipe_pairing"): {
        "sec_param": 64, "sec_level": 2, "bound_x": 5, "bound_y": 5,
        "pairing_group_param": "SS512", **_MULTI,
    },
    ("mife", "ddh_threshold"): {"sec_param": 256, **_THRESHOLD},
    ("mife", "lwe_threshold"): {"sec_param": 64, "lwe_n": 16, "bound_x": 6, "bound_y": 6, **_THRESHOLD},
    ("mife", "ddh_hybrid_alpha"): {"sec_param": 256, **_MULTI},
    ("mcfe", "ddh"): {"sec_param": 256, **_MULTI},
    ("mcfe", "lwe"): {
        "sec_param": 64, "lwe_n": 16, "bound_x": 6, "bound_y": 6, "bound_u": 2,
        "label_modulus": 8, **_MULTI,
    },
    ("mcfe", "paillier"): {"sec_param": 64, "bit_length": 64, "bound_x": 10, "bound_y": 10, **_MULTI},
    ("mcfe", "damgard"): {"sec_param": 64, "modulus_length": 64, "bound": 10, **_MULTI},
    ("mcfe", "ring_lwe"): {
        "sec_param": 64, "ring_n": 16, "bound_x": 2, "bound_y": 2, "bound_u": 1,
        "label_modulus": 8, **_MULTI,
    },
    ("mcfe", "fullysec_lwe"): {
        "sec_param": 64, "lwe_n": 24, "bound_x": 8, "bound_y": 8, "bound_u": 2,
        "label_modulus": 8, **_MULTI,
    },
    ("mcfe", "fh_multi_ipe_pairing"): {
        "sec_param": 64, "sec_level": 2, "bound_x": 4, "bound_y": 4, "u_bound": 3,
        "label_modulus": 17, "pairing_group_param": "SS512", **_MULTI,
    },
    ("mcfe", "fh_multi_ipe_pairing_threshold"): {
        "sec_param": 64, "sec_level": 2, "bound_x": 4, "bound_y": 4, "u_bound": 3,
        "label_modulus": 17, "pairing_group_param": "SS512", **_THRESHOLD,
    },
    ("mcfe", "fh_multi_ipe_pairing_decentralized"): {
        "sec_param": 64, "sec_level": 2, "bound_x": 4, "bound_y": 4, "u_bound": 3,
        "label_modulus": 17, "pairing_group_param": "SS512", **_MULTI,
    },
    ("mcfe", "ddh_threshold"): {"sec_param": 128, **_THRESHOLD},
    ("mcfe", "ddh_decentralized"): {"sec_param": 128, **_MULTI},
    ("mcfe", "lwe_threshold"): {
        "sec_param": 64, "lwe_n": 16, "bound_x": 6, "bound_y": 6, "bound_u": 2,
        "label_modulus": 8, **_THRESHOLD,
    },
    ("mcfe", "lwe_decentralized"): {
        "sec_param": 64, "lwe_n": 16, "bound_x": 6, "bound_y": 6, "bound_u": 2,
        "label_modulus": 8, **_MULTI,
    },
    ("mcfe", "ring_lwe_threshold"): {
        "sec_param": 64, "ring_n": 16, "bound_x": 4, "bound_y": 3, "bound_u": 1,
        "label_modulus": 8, **_THRESHOLD,
    },
    ("mcfe", "ring_lwe_decentralized"): {
        "sec_param": 64, "ring_n": 16, "bound_x": 4, "bound_y": 3, "bound_u": 1,
        "label_modulus": 8, **_MULTI,
    },
    ("mcfe", "paillier_decentralized"): {
        "sec_param": 64, "bit_length": 64, "bound_x": 10, "bound_y": 10, **_MULTI,
    },
    ("quadratic", "quad"): {"sec_param": 64, "n": 4, "m": 3, "bound": 5, "pairing_group_param": "SS512"},
    ("quadratic", "sgp"): {"sec_param": 64, "n": 5, "bound": 6, "pairing_group_param": "SS512"},
    ("quadratic", "multi_input_sgp"): {
        "sec_param": 64, "d": 3, "n": 2, "lst_nid": ["client_0", "client_1"], "s": 1,
        "bound": 5, "pairing_group_param": "SS512",
    },
}


def _needs_pairing(scheme_type: str, variant: str) -> bool:
    return scheme_type == "quadratic" or "pairing" in variant


def _gradients(scheme_type: str, config: dict) -> dict[str, np.ndarray]:
    """One small integer gradient per client (|x| <= 2 fits every bound_x)."""
    if scheme_type in ("mife", "mcfe"):
        nids = config["lst_nid"]
    else:
        nids = ["nid_default"]
    return {nid: np.array([1.0, -1.0, float(i % 3)]) for i, nid in enumerate(nids)}


def _describe(exc: Exception) -> str:
    """Exception type and first message line; for an empty message, the
    pyfe4ai function that raised it."""
    message = str(exc).splitlines()[0][:160] if str(exc) else ""
    if not message:
        pkg_dir = os.path.dirname(os.path.abspath(pyfe4ai.__file__))
        frames = [
            f for f in traceback.extract_tb(exc.__traceback__)
            if os.path.abspath(f.filename).startswith(pkg_dir + os.sep)
        ]
        if frames:
            rel = os.path.relpath(frames[-1].filename, os.path.dirname(pkg_dir))
            message = "raised without message in {}:{} {}()".format(
                rel, frames[-1].lineno, frames[-1].name
            )
    return "{}: {}".format(type(exc).__name__, message)


def check_entry(scheme_type: str, variant: str) -> tuple[str, str]:
    if _needs_pairing(scheme_type, variant) and PAIRING_IMPORT_ERROR is not None:
        return "unverified", "charm-crypto not installed"
    config = dict(CONFIGS[(scheme_type, variant)])
    if scheme_type != "quadratic":
        config["eta"] = 1
    label = LABEL if scheme_type == "mcfe" else None
    try:
        wrapper = FESchemeWrapper(scheme_type, variant, config)
        grads = _gradients(scheme_type, config)
        encrypted = {
            nid: encrypt_gradient(g, wrapper, nid=nid, label=label)
            for nid, g in grads.items()
        }
        result = aggregate_gradients(
            encrypted, wrapper, {nid: 1 for nid in grads}, label=label
        )
    except Exception as exc:  # noqa: BLE001 - every failure is a data point
        return "unusable", _describe(exc)
    expected = sum(grads.values())
    try:
        ok = np.allclose(np.asarray(result, dtype=float).ravel(), expected.ravel())
    except (TypeError, ValueError) as exc:
        return "unusable", "result not numeric ({}): {!r}".format(type(exc).__name__, result)
    if not ok:
        return "unusable", "wrong result: got {} expected {}".format(
            np.asarray(result).ravel().tolist(), expected.ravel().tolist()
        )
    return "usable", ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", choices=["md", "json"], default="md")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)

    missing = sorted(set(_SCHEME_REGISTRY) - set(CONFIGS))
    if missing:
        raise SystemExit("no configuration for registry entries: {}".format(missing))

    rows = []
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            for scheme_type, variant in _SCHEME_REGISTRY:
                status, reason = check_entry(scheme_type, variant)
                rows.append(
                    {"scheme": "{}/{}".format(scheme_type, variant), "status": status, "reason": reason}
                )
        finally:
            os.chdir(cwd)

    counts = {s: sum(r["status"] == s for r in rows) for s in ("usable", "unusable", "unverified")}
    if args.output == "json":
        print(json.dumps({"total": len(rows), "counts": counts, "entries": rows}, indent=2))
        return
    print("| # | Scheme | FL path | Reason |")
    print("|---|---|---|---|")
    for i, row in enumerate(rows, 1):
        reason = row["reason"].replace("|", "\\|")
        print("| {} | `{}` | {} | {} |".format(i, row["scheme"], row["status"], reason))
    print()
    print("Total {}: {} usable, {} unusable, {} unverified (charm-crypto {}).".format(
        len(rows), counts["usable"], counts["unusable"], counts["unverified"],
        "missing" if PAIRING_IMPORT_ERROR is not None else "installed",
    ))


if __name__ == "__main__":
    main()
