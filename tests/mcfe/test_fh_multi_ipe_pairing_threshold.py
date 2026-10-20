import random

import numpy as np
import pytest

from tests.pairing_group import GROUP, skip_unless_available  # noqa: E402

skip_unless_available()

from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold import ThresholdMCFEFHMultiIPE
from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold import ThresholdMCFEFHMultiIPEKeyGenerator


@pytest.fixture
def kg_config():
    return {
        "sec_param": 64,
        "eta": 2,
        "n": 3,
        "s": 3,
        "t": 2,
        "lst_nid": ["nid_0", "nid_1", "nid_2"],
        "lst_sid": ["sid_0", "sid_1", "sid_2"],
        "bound_x": 4,
        "bound_y": 4,
        "pairing_group_param": GROUP,
    }


def test_entire_process(kg_config):
    label = "tmcfe-fhmultiipe"
    kg = ThresholdMCFEFHMultiIPEKeyGenerator(kg_config)
    kg.setup()

    dct_x = {
        nid: [random.randint(-kg_config["bound_x"], kg_config["bound_x"]) for _ in range(kg_config["eta"])]
        for nid in kg_config["lst_nid"]
    }
    dct_y = {
        nid: [random.randint(-kg_config["bound_y"], kg_config["bound_y"]) for _ in range(kg_config["eta"])]
        for nid in kg_config["lst_nid"]
    }
    expected = sum(
        x_i * y_i
        for nid in kg_config["lst_nid"]
        for x_i, y_i in zip(dct_x[nid], dct_y[nid])
    )

    pp = kg.get_public_parameters()
    dct_ct = {}
    for nid in kg_config["lst_nid"]:
        enc = ThresholdMCFEFHMultiIPE(
            {"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}
        )
        dct_ct[nid] = enc.encrypt(dct_x[nid], label)

    credentials = {"fusion_weight": dct_y, "label": label}
    enrolled = random.sample(kg_config["lst_sid"], kg_config["t"])
    dct_prime = {}
    for sid in enrolled:
        dk = kg.get_decryption_keys(sid, credentials=credentials)
        dec = ThresholdMCFEFHMultiIPE({"id": sid, "keys": {"pp": pp, "dk": dk}})
        dct_prime[sid] = dec.share_decrypt(dct_ct, credentials, dk, enrolled, label)

    combiner = ThresholdMCFEFHMultiIPE(
        {"id": kg_config["lst_nid"][0], "keys": {"pp": pp, "sk": kg.get_private_keys(kg_config["lst_nid"][0])}}
    )
    computed = combiner.combine_decrypt(dct_prime)
    assert computed == expected


def test_entire_process_lst_ndarray(kg_config):
    cfg = dict(kg_config)
    cfg["eta"] = 1
    cfg["bound_x"] = 50

    kg = ThresholdMCFEFHMultiIPEKeyGenerator(cfg)
    kg.setup()

    precision = 1
    label = "tmcfe-fh-multi-ipe-ndarray"
    dct_x = {
        nid: [np.random.random_sample((2, 3)), np.random.random_sample((2, 2, 2))]
        for nid in cfg["lst_nid"]
    }
    dct_y = {"nid_0": [1], "nid_1": [2], "nid_2": [-1]}

    expected = []
    for layer_idx in range(len(dct_x[cfg["lst_nid"][0]])):
        expected.append(
            sum(
                (
                    np.rint(dct_x[nid][layer_idx] * pow(10, precision))
                    / pow(10, precision)
                )
                * dct_y[nid][0]
                for nid in cfg["lst_nid"]
            )
        )

    pp = kg.get_public_parameters()
    dct_ct = {}
    for nid in cfg["lst_nid"]:
        enc = ThresholdMCFEFHMultiIPE(
            {
                "id": nid,
                "precision": precision,
                "keys": {"pp": pp, "sk": kg.get_private_keys(nid)},
            }
        )
        dct_ct[nid] = enc.encrypt_lst_ndarray(dct_x[nid], label=label)

    credentials = {"fusion_weight": dct_y, "label": label}
    enrolled = random.sample(cfg["lst_sid"], cfg["t"])
    dct_prime = {}
    for sid in enrolled:
        dk = kg.get_decryption_keys(sid, credentials=credentials)
        dec = ThresholdMCFEFHMultiIPE(
            {"id": sid, "precision": precision, "keys": {"pp": pp, "dk": dk}}
        )
        dct_prime[sid] = dec.compute_lst_ndarray_ct(dct_ct, credentials, dk, enrolled, label)

    combiner = ThresholdMCFEFHMultiIPE(
        {
            "id": cfg["lst_nid"][0],
            "precision": precision,
            "keys": {"pp": pp, "sk": kg.get_private_keys(cfg["lst_nid"][0])},
        }
    )
    computed = combiner.decrypt_lst_ndarray_ct(dct_prime, label)

    for idx in range(len(expected)):
        assert np.allclose(computed[idx], expected[idx], atol=1e-2)


def _setup(cfg):
    kg = ThresholdMCFEFHMultiIPEKeyGenerator(cfg)
    kg.setup()
    pp = kg.get_public_parameters()
    x = {"nid_0": [1, -2], "nid_1": [0, 3], "nid_2": [2, 1]}
    y = {"nid_0": [2, 1], "nid_1": [-1, 2], "nid_2": [1, -2]}
    dct_ct = {
        nid: ThresholdMCFEFHMultiIPE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}).encrypt(
            x[nid], "round-1"
        )
        for nid in cfg["lst_nid"]
    }
    return kg, pp, y, dct_ct


def _partials(kg, pp, y, dct_ct, enrolled, label="round-1"):
    out = {}
    for sid in enrolled:
        dk = kg.get_decryption_keys(sid, credentials={"fusion_weight": y})
        dec = ThresholdMCFEFHMultiIPE({"id": sid, "keys": {"pp": pp, "dk": dk}})
        out[sid] = dec.share_decrypt(dct_ct, {"fusion_weight": y}, dk, enrolled, label)
    return out


def test_every_enrolled_set_decrypts(kg_config):
    kg, pp, y, dct_ct = _setup(kg_config)
    combiner = ThresholdMCFEFHMultiIPE({"id": "c", "keys": {"pp": pp}})
    for enrolled in (["sid_0", "sid_1"], ["sid_1", "sid_2"], ["sid_0", "sid_1", "sid_2"]):
        assert combiner.combine_decrypt(_partials(kg, pp, y, dct_ct, enrolled)) == 6


def test_threshold_and_consistency_checks(kg_config):
    from pyfe4ai.core.exceptions import FESchemeError

    kg, pp, y, dct_ct = _setup(kg_config)
    combiner = ThresholdMCFEFHMultiIPE({"id": "c", "keys": {"pp": pp}})
    parts = _partials(kg, pp, y, dct_ct, ["sid_0", "sid_1"])
    with pytest.raises(FESchemeError):
        combiner.combine_decrypt({"sid_0": parts["sid_0"]})
    dk = kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y})
    dec = ThresholdMCFEFHMultiIPE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    assert "weights_id" not in dk and "key_id" in dk  # no digest of y
    with pytest.raises(FESchemeError):
        dec.share_decrypt(dct_ct, {"fusion_weight": y}, dk, ["sid_0"], "round-1")
    # a server must use its own share (another's has a different Lagrange point)
    other = ThresholdMCFEFHMultiIPE({"id": "sid_1", "keys": {"pp": pp, "dk": dk}})
    with pytest.raises(FESchemeError):
        other.share_decrypt(dct_ct, {}, dk, ["sid_0", "sid_1"], "round-1")
    # servers that decrypted different (authentic) ciphertexts are not combined
    client = ThresholdMCFEFHMultiIPE({"id": "nid_0", "keys": {"pp": pp, "sk": kg.get_private_keys("nid_0")}})
    other = dict(dct_ct, nid_0=client.encrypt([0, 0], "round-1"))
    parts["sid_1"] = _partials(kg, pp, y, other, ["sid_0", "sid_1"])["sid_1"]
    assert combiner.combine_decrypt(parts) is None


def test_single_share_reveals_nothing_on_its_own(kg_config):
    """A partial from fewer than t servers does not decrypt."""
    kg, pp, y, dct_ct = _setup(kg_config)
    parts = _partials(kg, pp, y, dct_ct, ["sid_0", "sid_1"])
    combiner = ThresholdMCFEFHMultiIPE({"id": "c", "keys": {"pp": pp}})
    assert combiner.ctx.solve(combiner.ctx.de(parts["sid_0"]["w"])) is None


def test_partial_decryptions_for_several_enrolled_sets_must_agree(kg_config):
    from pyfe4ai.core.exceptions import FESchemeError

    kg, pp, y, dct_ct = _setup(kg_config)
    combiner = ThresholdMCFEFHMultiIPE({"id": "c", "keys": {"pp": pp}})
    pairs = (["sid_0", "sid_1"], ["sid_0", "sid_2"], ["sid_1", "sid_2"])
    groups = [_partials(kg, pp, y, dct_ct, pair) for pair in pairs]
    assert combiner.combine_decrypt(groups) == 6
    ctx = combiner.ctx
    forged = [
        {s: (dict(p, w=ctx.ser(ctx.de(p["w"]) * ctx.fhipe.gt)) if s == "sid_1" else p) for s, p in g.items()}
        for g in groups
    ]
    with pytest.raises(FESchemeError, match="disagree"):
        combiner.combine_decrypt(forged)


def test_servers_hold_the_key_only_in_the_exponent(kg_config):
    """Regression: key shares were Shamir shares of the key exponents ``e_i``
    in ``Z_p``, so ``t`` colluding servers (one if ``t = 1``) interpolated
    ``e_i`` and solved for ``y``; the FH-MCFE key hides ``y`` only as
    ``[e_i]_2``. Shares are now group elements: ``t`` servers interpolate
    ``[e_i]_2`` and ``[z]_T``, i.e. the FH-MCFE functional key itself."""
    from pyfe4ai.schemes.threshold.threshold_utils import lagrange_coefficient_at_zero, share_points
    from pyfe4ai.primitives.pairing.pairing_backend import GT, pair

    kg, pp, y, dct_ct = _setup(kg_config)
    ctx = ThresholdMCFEFHMultiIPE({"id": "c", "keys": {"pp": pp}}).ctx
    enrolled = ["sid_0", "sid_2"]
    dks = {sid: kg.get_decryption_keys(sid, credentials={"fusion_weight": y}) for sid in enrolled}
    for dk in dks.values():
        for value in [dk["z"]] + [v for entries in dk["k"].values() for v in entries]:
            with pytest.raises(ValueError):
                int(value)
    pts = share_points(kg_config["lst_sid"])
    lam = {s: ctx.zr(lagrange_coefficient_at_zero(pts[s], [pts[e] for e in enrolled], ctx.order))
           for s in enrolled}

    def interpolate(elements):
        out = None
        for s, element in elements.items():
            term = ctx.de(element) ** lam[s]
            out = term if out is None else out * term
        return out

    acc = ctx.group.init(GT, 1)
    for nid, ct in dct_ct.items():
        for j, c in enumerate(ct["ct"]):
            acc *= pair(ctx.de(c), interpolate({s: dks[s]["k"][nid][j] for s in enrolled}))
    acc /= interpolate({s: dks[s]["z"] for s in enrolled})
    assert ctx.solve(acc) == 6


def test_servers_decrypt_only_the_requested_round(kg_config):
    """Regression: FH ciphertexts carry no label in the clear, so servers
    decrypted whatever round the combiner submitted. Ciphertexts now carry
    the label, tagged with the ciphertext for every server."""
    from pyfe4ai.core.exceptions import FESchemeError

    kg, pp, y, dct_ct = _setup(kg_config)
    assert _partials(kg, pp, y, dct_ct, ["sid_0", "sid_1"], "round-1")
    with pytest.raises(FESchemeError, match="requested label"):
        _partials(kg, pp, y, dct_ct, ["sid_0", "sid_1"], "round-2")
    relabelled = {nid: dict(ct, label='["label", "round-2"]') for nid, ct in dct_ct.items()}
    with pytest.raises(FESchemeError, match="not authenticated"):
        _partials(kg, pp, y, relabelled, ["sid_0", "sid_1"], "round-2")
    first = dct_ct["nid_0"]
    for forged in (dict(first, ct=dct_ct["nid_1"]["ct"]), {k: v for k, v in first.items() if k != "tags"}):
        with pytest.raises(FESchemeError, match="not authenticated"):
            _partials(kg, pp, y, dict(dct_ct, nid_0=forged), ["sid_0", "sid_1"])
