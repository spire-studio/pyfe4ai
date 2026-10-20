"""Offline / online phases of MCFE-DDH, MCFE-Damgard and MCFE-Paillier (output-identical)."""

import numpy as np
import pytest

from pyfe4ai.schemes.mcfe.damgard_ddh import MCFEDamgard, MCFEDamgardKeyGenerator
from pyfe4ai.schemes.mcfe.ddh import MCFE, MCFEKeyGenerator
from pyfe4ai.schemes.mcfe.paillier import MCFEPaillier, MCFEPaillierKeyGenerator
from pyfe4ai.core.exceptions import FEValidationError
from pyfe4ai.ml import (
    FESchemeWrapper,
    aggregate_gradients,
    encrypt_gradient,
    precompute_gradient,
    prepare_aggregation,
)

NIDS = ["c0", "c1", "c2"]
SCHEMES = {
    "ddh": (MCFEKeyGenerator, MCFE, {"sec_param": 256, "bound_x": 20, "bound_y": 3}),
    "damgard": (MCFEDamgardKeyGenerator, MCFEDamgard, {"sec_param": 256, "bound": 20}),
    "paillier": (MCFEPaillierKeyGenerator, MCFEPaillier, {"bit_length": 256, "bound_x": 20, "bound_y": 3}),
}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _setup(name):
    kg_cls, cls, cfg = SCHEMES[name]
    kg = kg_cls(dict(cfg, n=len(NIDS), eta=1, lst_nid=NIDS))
    kg.setup()
    pp = kg.get_public_parameters()
    clients = {nid: cls({"id": nid, "precision": 0, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}})
               for nid in NIDS}
    return kg, clients, cls({"id": "agg", "precision": 0, "keys": {"pp": pp}})


@pytest.mark.parametrize("name", list(SCHEMES))
def test_precomputed_and_prepared_paths_are_identical(name):
    kg, clients, agg = _setup(name)
    rng = np.random.default_rng(3)
    layers = {nid: [rng.integers(-20, 21, 5), rng.integers(-20, 21, (2, 3))] for nid in NIDS}
    shapes = [(5,), (2, 3)]
    plain = {nid: c.encrypt_lst_ndarray(layers[nid], label="L") for nid, c in clients.items()}
    pre = {nid: c.precompute("L", shapes) for nid, c in clients.items()}
    online = {nid: c.encrypt_lst_ndarray(layers[nid], label="L", precomputed=pre[nid]) for nid, c in clients.items()}
    for nid in NIDS:
        for a, b in zip(plain[nid], online[nid]):
            assert a.tolist() == b.tolist()
    weights = {nid: [int(v)] for nid, v in zip(NIDS, rng.integers(-3, 4, len(NIDS)))}
    dk = kg.get_decryption_keys("agg", credentials={"fusion_weight": weights})
    want = agg.decrypt_lst_ndarray_ct(plain, dk=dk, fusion_weight=weights, label="L")
    prepared = agg.prepare_key(dk, "L", shapes)
    got = agg.decrypt_lst_ndarray_ct(plain, dk=prepared, fusion_weight=weights, label="L")
    expect = [sum(weights[n][0] * layers[n][l] for n in NIDS) for l in range(2)]
    for l in range(2):
        assert np.array_equal(np.asarray(got[l]), np.asarray(want[l])) and np.array_equal(np.asarray(got[l]), expect[l])
    # a prepared key of another label or shape is not used (the plain path decrypts)
    other = agg.prepare_key(dk, "M", shapes)
    assert np.array_equal(np.asarray(agg.decrypt_lst_ndarray_ct(plain, dk=other, fusion_weight=weights, label="L")[0]),
                          expect[0])


@pytest.mark.parametrize("name", list(SCHEMES))
def test_precomputation_is_one_time_and_bound(name):
    _, clients, _ = _setup(name)
    c0, c1 = clients["c0"], clients["c1"]
    x = [np.arange(4)]
    pre = c0.precompute("L", [(4,)])
    with pytest.raises(FEValidationError, match="another label"):
        c0.encrypt_lst_ndarray(x, label="M", precomputed=pre)
    with pytest.raises(FEValidationError, match="another client"):
        c1.encrypt_lst_ndarray(x, label="L", precomputed=pre)
    with pytest.raises(FEValidationError, match="shapes"):
        c0.encrypt_lst_ndarray([np.arange(5)], label="L", precomputed=pre)
    c0.encrypt_lst_ndarray(x, label="L", precomputed=pre)
    with pytest.raises(FEValidationError, match="once"):
        c0.encrypt_lst_ndarray(x, label="L", precomputed=pre)
    # out-of-bound plaintexts: the online phase raises exactly when the plain call does
    big = [np.array([10 ** 6])]
    try:
        c1.encrypt_lst_ndarray(big, label="X")
        plain_raises = False
    except FEValidationError:
        plain_raises = True
    try:
        c1.encrypt_lst_ndarray(big, label="L", precomputed=c1.precompute("L", [(1,)]))
        online_raises = False
    except FEValidationError:
        online_raises = True
    assert online_raises == plain_raises == (name != "ddh")


@pytest.mark.parametrize("variant,cfg", [
    ("ddh", {"sec_param": 256, "bound_x": 100, "bound_y": 1}),
    ("paillier", {"bit_length": 256, "bound_x": 100, "bound_y": 1}),
])
def test_ml_adapter_offline_round(variant, cfg):
    wrapper = FESchemeWrapper("mcfe", variant, dict(cfg, n=3, eta=1, lst_nid=NIDS), precision=2)
    rng = np.random.default_rng(9)
    grads = {nid: rng.uniform(-1, 1, (3, 4)).round(2) for nid in NIDS}
    pre = {nid: precompute_gradient((3, 4), wrapper, nid=nid, label="r1") for nid in NIDS}
    prepared = prepare_aggregation(wrapper, {nid: 1 for nid in NIDS}, shape=(3, 4), label="r1")
    assert "key_terms" in prepared.dk
    enc = {nid: encrypt_gradient(g, wrapper, nid=nid, label="r1", precomputed=pre[nid]) for nid, g in grads.items()}
    total = aggregate_gradients(enc, wrapper, label="r1", prepared=prepared)
    assert np.allclose(total, sum(grads.values()))
