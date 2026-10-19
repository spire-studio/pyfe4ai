"""Threshold MCFE/Ring-LWE example: federated rounds with split key shares.

With the scheme's defaults (binary ciphertexts, an RNS modulus, split key
shares) each server receives its weight key share once per weight vector and,
per round, only its label key share. The second part runs the same rounds
through the ML adapter, with ciphertexts and partial decryptions sent as JSON.
"""

import json

import numpy as np

from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWE
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWEKeyGenerator
from pyfe4ai.utils.ml_adapter import EncryptedGradient
from pyfe4ai.utils.ml_adapter import FESchemeWrapper
from pyfe4ai.utils.ml_adapter import ThresholdPartial
from pyfe4ai.utils.ml_adapter import combine_partial_decryptions
from pyfe4ai.utils.ml_adapter import encrypt_gradient
from pyfe4ai.utils.ml_adapter import partial_decrypt_gradients
from pyfe4ai.utils.ml_adapter import prepare_aggregation

NIDS = ["nid_0", "nid_1", "nid_2"]
SIDS = ["sid_0", "sid_1", "sid_2"]
CONFIG = {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS, "lst_sid": SIDS,
          "ring_n": 16, "bound_x": 50, "bound_y": 2}


def scheme_classes() -> None:
    kg = ThresholdMCFERingLWEKeyGenerator(CONFIG)
    kg.setup()
    pp = kg.get_public_parameters()
    print("ct_format =", pp["ct_format"], " q factors =", len(pp["q_factors"]), " key shares =", kg.key_shares)
    clients = {nid: ThresholdMCFERingLWE({"id": nid, "precision": 1, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in NIDS}
    weights = {"nid_0": [1], "nid_1": [2], "nid_2": [1]}
    enrolled = ["sid_0", "sid_2"]
    # once per weight vector: each server keeps its weight key share
    servers = {sid: ThresholdMCFERingLWE({"id": sid, "precision": 1, "keys": {"pp": pp}}) for sid in enrolled}
    for sid, server in servers.items():
        server.add_weight_key_share(kg.get_weight_key_share(sid, credentials={"fusion_weight": weights}))
    rng = np.random.default_rng(0)
    for r in range(2):
        label = "round-%d" % r
        grads = {nid: np.round(rng.uniform(-2, 2, (3, 7)), 1) for nid in NIDS}
        cts = {nid: clients[nid].encrypt_lst_ndarray([grads[nid]], label=label) for nid in NIDS}
        # per round: only the label key share (the chunk keys of this label)
        creds = {"fusion_weight": weights, "label": label, "n_chunks": ThresholdMCFERingLWE.key_chunks(cts["nid_0"]),
                 "label_term": False}
        partials = {sid: server.compute_lst_ndarray_ct(cts, creds, kg.get_decryption_keys(sid, credentials=creds),
                                                       enrolled)
                    for sid, server in servers.items()}
        total = ThresholdMCFERingLWE({"id": "combiner", "precision": 1, "keys": {"pp": pp}}).decrypt_lst_ndarray_ct(
            partials)[0]
        expected = sum(weights[nid][0] * grads[nid] for nid in NIDS)
        print(label, "weighted sum correct:", bool(np.allclose(total, expected)))


def ml_adapter() -> None:
    wrapper = FESchemeWrapper("mcfe", "ring_lwe_threshold", CONFIG, precision=1)
    weights = {"nid_0": 1, "nid_1": 2, "nid_2": 1}
    rng = np.random.default_rng(1)
    grads = {nid: np.round(rng.uniform(-2, 2, (3, 7)), 1) for nid in NIDS}
    # the key shares of the round, before the gradients exist
    prepared = prepare_aggregation(wrapper, weights, shape=(3, 7), label="round-a", servers=["sid_1", "sid_2"])
    sent = {nid: json.dumps(encrypt_gradient(grads[nid], wrapper, nid=nid, label="round-a").to_dict())
            for nid in NIDS}
    received = {nid: EncryptedGradient.from_dict(json.loads(data)) for nid, data in sent.items()}
    partials = {sid: json.dumps(partial_decrypt_gradients(received, wrapper, prepared, server=sid).to_dict())
                for sid in prepared.servers}
    total = combine_partial_decryptions(
        {sid: ThresholdPartial.from_dict(json.loads(data)) for sid, data in partials.items()}, wrapper, prepared)
    print("adapter: ciphertext JSON bytes per client =", len(sent["nid_0"]),
          " weighted sum correct:", bool(np.allclose(total, sum(weights[n] * grads[n] for n in NIDS))))


def main() -> None:
    scheme_classes()
    ml_adapter()


if __name__ == "__main__":
    main()
