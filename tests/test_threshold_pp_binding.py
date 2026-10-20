"""Threshold key shares are bound to the public parameters they were issued
for (``pp_binding``): a server refuses to answer with a key share under public
parameters that differ in the modulus, the flooding (distribution, width, tail
cut, accounting and its limit) or the label hash, since a second, differently
derived answer to one request reveals key material."""

import os
import warnings

import pytest

from pyfe4ai.schemes.threshold import flooding_accounting as fa
from pyfe4ai.schemes.mcfe import lwe_threshold as dense
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as ring
from pyfe4ai.schemes.mife import lwe_threshold as tmife
from pyfe4ai.schemes.threshold.threshold_utils import pp_binding
from pyfe4ai.core.exceptions import FEKeyError, FESchemeError, FEValidationError

NIDS, SIDS = ["a", "b"], ["s0", "s1", "s2"]
REFUSED = (FEKeyError, FESchemeError, FEValidationError)


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _ring(**extra):
    kg = ring.ThresholdMCFERingLWEKeyGenerator({"sec_param": 64, "n": 2, "lst_nid": NIDS, "eta": 1, "ring_n": 16,
                                                "bound_x": 4, "bound_y": 4, "s": 3, "t": 3, "lst_sid": SIDS, **extra})
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {n: ring.ThresholdMCFERingLWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt([[1]], "L")
           for n in NIDS}
    cred = {"fusion_weight": {n: [1] for n in NIDS}, "label": "L"}
    dk = kg.get_decryption_keys("s0", credentials=cred)  # the label part (split key shares, the default)
    weight = kg.get_weight_key_share("s0", credentials=cred)
    budget = {"partial_budget": fa.PartialDecryptionBudget(pp["max_partial_decryptions"])} if "max_partial_decryptions" in pp else {}

    def serve(p):
        server = ring.ThresholdMCFERingLWE({"id": "s0", **budget, "keys": {"pp": p}})
        server.add_weight_key_share(weight)
        return server.share_decrypt(cts, cred, dk, SIDS)

    return pp, dk, serve


@pytest.mark.parametrize("extra, field, value", [
    ({}, "flood_bits", None),  # the uniform default: another width
    ({"flooding": "gaussian"}, "sigma_f", None),
    ({}, "q", None),
    ({}, "t", 2),
    ({"flooding_accounting": "kl", "max_partial_decryptions": 8}, "max_partial_decryptions", 16),
])
def test_ring_key_share_refused_under_other_parameters(extra, field, value):
    pp, dk, serve = _ring(**extra)
    serve(pp)  # accepted
    if value is None:  # one more than the issued value, keeping the type
        value = pp[field] + 1 if isinstance(pp[field], int) else str(int(pp[field]) + 2)
    with pytest.raises(REFUSED):
        serve(dict(pp, **{field: value}))


def test_ring_kl_tail_cut_and_accounting_are_bound():
    pp, dk, serve = _ring(flooding_accounting="kl", max_partial_decryptions=8)
    serve(pp)
    for bad in (dict(pp, flood_tail_cut=pp["flood_tail_cut"] + 1),
                {k: v for k, v in pp.items() if k not in ("flooding_accounting", "max_partial_decryptions",
                                                           "flood_tail_cut")}):
        with pytest.raises(REFUSED):
            serve(bad)


def test_missing_fields_bind_as_their_defaults():
    """Public parameters without ``flooding`` / ``label_hash`` (v0.3.0's format)
    mean Gaussian flooding and ``"shake256-wide"``: the same binding."""
    pp = {"q": "7", "flooding": "gaussian", "label_hash": "shake256-wide", "flooding_accounting": "per-sample"}
    old = {"q": "7"}
    assert pp_binding(pp, tuple(ring._BINDING_FIELDS), ring._BINDING_DEFAULTS) == pp_binding(
        old, tuple(ring._BINDING_FIELDS), ring._BINDING_DEFAULTS)
    assert pp_binding(dict(pp, flooding="uniform"), ring._BINDING_FIELDS, ring._BINDING_DEFAULTS) != pp_binding(
        old, ring._BINDING_FIELDS, ring._BINDING_DEFAULTS)


def test_tmife_key_share_refused_under_other_parameters():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kg = tmife.ThresholdMIFELWEKeyGenerator({"sec_param": 64, "eta": 1, "n": 2, "lst_nid": NIDS, "lwe_n": 16,
                                                 "bound_x": 4, "bound_y": 4, "s": 3, "t": 3, "lst_sid": SIDS})
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {n: tmife.ThresholdMIFELWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt([1])
           for n in NIDS}
    y = {n: [1] for n in NIDS}
    dk = kg.get_decryption_keys("s0", credentials={"fusion_weight": y})

    def serve(p):
        return tmife.ThresholdMIFELWE({"id": "s0", "keys": {"pp": p}}).share_decrypt(cts, y, dk, SIDS)

    serve(pp)
    for bad in (dict(pp, sigma_f=str(int(pp["sigma_f"]) + 2)), dict(pp, q=str(int(pp["q"]) + 2)),
                dict(pp, threshold_construction=tmife._CONSTRUCTION_KL)):
        with pytest.raises(REFUSED):
            serve(bad)


def test_dense_key_share_records_and_binds_the_label_hash():
    kg = dense.ThresholdMCFELWEKeyGenerator({"sec_param": 64, "n": 2, "lst_nid": NIDS, "eta": 1, "bound_x": 4,
                                             "bound_y": 4, "s": 3, "t": 3, "lst_sid": SIDS})
    kg.setup()
    pp = kg.get_public_parameters()
    fw = {n: [1] for n in NIDS}
    dk = kg.get_decryption_keys("s0", credentials={"fusion_weight": fw})
    assert dk["label_hash"] == pp["label_hash"]
    cts = {n: dense.ThresholdMCFELWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt([1], "r")
           for n in NIDS}
    cred = {"fusion_weight": fw, "label": "r"}
    dense.ThresholdMCFELWE({"id": "s0", "keys": {"pp": pp, "dk": dk}}).share_decrypt(cts, cred, dk, SIDS)
    for bad_pp, bad_dk in ((dict(pp, sigma_f=str(int(pp["sigma_f"]) + 2)), dk),
                           (pp, dict(dk, label_hash="shake256-wide")),
                           (pp, {k: v for k, v in dk.items() if k != "pp_binding"})):
        with pytest.raises(REFUSED):
            dense.ThresholdMCFELWE({"id": "s0", "keys": {"pp": bad_pp, "dk": bad_dk}}).share_decrypt(
                cts, cred, bad_dk, SIDS)


@pytest.mark.parametrize("bypass, leaks", [
    (lambda pp, d, f, df=None: pp_binding(pp, f, df), False),  # the server's own digest seeds the noise
    (lambda pp, d, f, df=None: "constant", True),  # control: one seed for both answers
])
def test_the_flooding_noise_follows_the_public_parameters(monkeypatch, bypass, leaks):
    """Defence in depth: the noise is seeded with the digest of the server's own
    public parameters. Even with both checks bypassed (a key share relabelled
    for another label hash), one request answered under two label hashes gets
    two noises, so ``w_A - w_B`` is not ``lambda <share, H_A(l) - H_B(l)>``; with
    one seed for both (the control) it is."""
    import gmpy2 as gp

    from pyfe4ai.schemes.mcfe.lwe import dot_mod
    from pyfe4ai.schemes.threshold.threshold_utils import lagrange_coefficient_at_zero

    monkeypatch.setattr(dense, "check_pp_binding", bypass)
    kg = dense.ThresholdMCFELWEKeyGenerator({"sec_param": 64, "n": 2, "lst_nid": NIDS, "eta": 1, "bound_x": 4,
                                             "bound_y": 4, "s": 3, "t": 3, "lst_sid": SIDS})
    kg.setup()
    pp_a = kg.get_public_parameters()
    other = "shake256-wide" if pp_a["label_hash"] != "shake256-wide" else "shake128-rejection"
    pp_b = dict(pp_a, label_hash=other)
    fw = {n: [1] for n in NIDS}
    dk = kg.get_decryption_keys("s0", credentials={"fusion_weight": fw})
    cts = {n: dense.ThresholdMCFELWE({"id": n, "keys": {"pp": pp_a, "sk": kg.get_private_keys(n)}}).encrypt([1], "L")
           for n in NIDS}
    cred = {"fusion_weight": fw, "label": "L"}
    srv_a = dense.ThresholdMCFELWE({"id": "s0", "keys": {"pp": pp_a, "dk": dk}})
    w_a = gp.mpz(srv_a.share_decrypt(cts, cred, dk, SIDS)["w"])
    dk_b = dict(dk, label_hash=other)  # the key share claims the other hash
    srv_b = dense.ThresholdMCFELWE({"id": "s0", "keys": {"pp": pp_b, "dk": dk_b}})
    w_b = gp.mpz(srv_b.share_decrypt({n: {"c": ct["c"]} for n, ct in cts.items()}, cred, dk_b, SIDS)["w"])
    q = gp.mpz(pp_a["q"])
    share = [gp.mpz(v) for v in dk["sk_y"]]
    lam = lagrange_coefficient_at_zero(1, [1, 2, 3], q)
    exact = lam * (dot_mod(share, srv_a._hash_label("L"), q) - dot_mod(share, srv_b._hash_label("L"), q)) % q
    assert ((w_a - w_b) % q == exact) is leaks
