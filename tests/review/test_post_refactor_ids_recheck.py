"""Ids read back by the ML adapter, and gradients under another client's key (October 2026 code review).

Inner server id of a threshold partial (low, integrity hygiene): ``ThresholdPartial.from_dict`` compares
the ``sid`` of each chunk with the record's ``sid`` by value and then writes the record's ``sid`` into the
chunk. With integer server ids, a chunk whose ``sid`` is ``true`` or ``1.0`` under a record of server ``1``
is therefore accepted and combined, although ``1.0`` and ``True`` are no ids anywhere else after setup
(``core.ids``; the combiner refused such chunks before the chunk ``sid`` was rewritten). The chunk still
belongs to the record's server, so nothing is combined for another server. Check the chunk ``sid`` with
``is_id`` (after ``as_id``) before comparing it.

Gradients under another client's key (low, integrity; also before the id rule): ``aggregate_gradients``
of the lattice MCFE adapters (lwe, ring_lwe, fullysec_lwe) takes ``{nid: EncryptedGradient}`` and
decrypts each gradient as the client its key names, without comparing the key with the gradient's own
``nid``. Two gradients filed under each other's client give a silently wrong aggregate (the DH and
Paillier adapters fail to decrypt them; the threshold adapter refuses them: the client's tags do not
verify). Refuse a gradient whose ``nid`` is not its key, or give the right aggregate.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyfe4ai.core.exceptions import FEError

NIDS = [0, 1, 2]
WEIGHTS = {0: 1, 1: 2, 2: 1}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def _threshold_round():
    from pyfe4ai.ml import FESchemeWrapper, encrypt_gradient, partial_decrypt_gradients, prepare_aggregation

    w = FESchemeWrapper("mcfe", "ring_lwe_threshold", {"eta": 1, "n": 3, "lst_nid": NIDS, "s": 3, "t": 2,
                                                     "lst_sid": [1, 2, 3], "ring_n": 16, "sec_param": 64,
                                                     "bound_x": 50, "bound_y": 2}, precision=1)
    enc = {n: encrypt_gradient(np.array([0.5, -0.5]), w, nid=n, label="a") for n in NIDS}
    prepared = prepare_aggregation(w, WEIGHTS, shape=(2,), label="a", servers=[1, 2])
    wire = {s: json.loads(json.dumps(partial_decrypt_gradients(enc, w, prepared, server=s).to_dict())) for s in (1, 2)}
    return w, prepared, wire


@pytest.mark.parametrize("chunk_sid", [True, 1.0], ids=["true", "1.0"])
def test_a_chunk_sid_that_is_no_id_is_refused(chunk_sid):
    from pyfe4ai.ml import ThresholdPartial, combine_partial_decryptions

    w, prepared, wire = _threshold_round()
    for layer in wire[1]["layers"]:
        for part in layer["parts"]:
            part["sid"] = chunk_sid
    with pytest.raises((ValueError, FEError)):
        received = {s: ThresholdPartial.from_dict(d) for s, d in wire.items()}
        combine_partial_decryptions(received, w, prepared)


@pytest.mark.parametrize("variant", ["ring_lwe", "lwe", "fullysec_lwe"])
def test_gradients_filed_under_another_client_are_refused_or_aggregate_right(variant):
    from pyfe4ai.ml import FESchemeWrapper, aggregate_gradients, encrypt_gradient

    w = FESchemeWrapper("mcfe", variant, {"eta": 1, "n": 3, "lst_nid": NIDS, "ring_n": 16, "lwe_n": 8,
                                          "bound_x": 50, "bound_y": 2, "sec_param": 64}, precision=1)
    grads = {0: np.array([0.5, -0.5]), 1: np.array([0.2, 0.3]), 2: np.array([-0.4, 0.1])}
    enc = {n: encrypt_gradient(grads[n], w, nid=n, label="a") for n in NIDS}
    swapped = {0: enc[1], 1: enc[0], 2: enc[2]}
    try:
        out = aggregate_gradients(swapped, w, WEIGHTS, label="a")
    except (ValueError, FEError):
        return  # refused: consistent
    np.testing.assert_allclose(out, sum(WEIGHTS[n] * grads[n] for n in NIDS), atol=1e-9)
