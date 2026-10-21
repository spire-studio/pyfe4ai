"""Strict parsing of the threshold DDH key shares and of serialised gradients.

* The values of a threshold DDH key share (``mife/ddh_threshold``: ``d`` per
  client and ``z``; ``mcfe/ddh_threshold``: the two values of ``d``) are
  decimal strings: that is what the key generators write (``gmpy2.digits``,
  as in v0.3.0) and what the formats document. A server read them with
  ``gmpy2.mpz``, which also took ints, ``mpz``, bytes, floats (truncated),
  ``True``, ``" 12"`` or ``"1_2"``; it now refuses any other type or text
  with :class:`~pyfe4ai.core.exceptions.FEKeyError`.
* :meth:`~pyfe4ai.ml.EncryptedGradient.from_dict` checks the field types and
  the layout, as :meth:`~pyfe4ai.ml.ThresholdPartial.from_dict` does: exactly
  the five keys, their types, per-element layers ``{"shape", "data"}`` with
  one ciphertext (a dict) under each index key in the one spelling
  ``to_dict`` writes. It accepted extra keys, any types, other spellings of
  an index (``"( 0,)"``, ``"(+0,)"``, ``"(0,0)"``) and non-ciphertext
  entries.

Everything the library writes still round-trips (every scheme whose
gradients the ML adapter encrypts, through JSON).
"""

from __future__ import annotations

import json

import gmpy2 as gp
import numpy as np
import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai.core.exceptions import FEKeyError
from pyfe4ai.ml import EncryptedGradient, FESchemeWrapper, encrypt_gradient

NIDS, SIDS = ["nid_0", "nid_1"], ["sid_0", "sid_1"]
Y = {n: [1] for n in NIDS}
needs_bls = pytest.mark.skipif(not native_backend.available(), reason="BLS12-381 needs the native extension")


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


# -- threshold DDH key shares


def _tmife_ddh():
    from pyfe4ai.schemes.mife.ddh_threshold import ThresholdMIFEDDH, ThresholdMIFEDDHKeyGenerator

    kg = ThresholdMIFEDDHKeyGenerator({"sec_param": 256, "lst_nid": NIDS, "lst_sid": SIDS, "eta": 1, "n": 2,
                                       "s": 2, "t": 2})
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {n: ThresholdMIFEDDH({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt([2]) for n in NIDS}
    dks = {s: kg.get_decryption_keys(s, credentials={"fusion_weight": Y}) for s in SIDS}

    def round_with(dk0):
        parts = {s: ThresholdMIFEDDH({"id": s, "keys": {"pp": pp, "dk": dk}}).share_decrypt(cts, Y, dk, SIDS)
                 for s, dk in (("sid_0", dk0), ("sid_1", dks["sid_1"]))}
        return ThresholdMIFEDDH({"id": "c", "keys": {"pp": pp}}).combine_decrypt(parts)

    return dks["sid_0"], round_with


def _tmcfe_ddh():
    from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFEDDH, ThresholdMCFEDDHKeyGenerator

    kg = ThresholdMCFEDDHKeyGenerator({"sec_param": 128, "lst_nid": NIDS, "lst_sid": SIDS, "eta": 1, "n": 2,
                                       "s": 2, "t": 2})
    kg.setup()
    pp = kg.get_public_parameters()
    cts = {n: ThresholdMCFEDDH({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt([2], "L")
           for n in NIDS}
    dks = {s: kg.get_decryption_keys(s, credentials={"fusion_weight": Y}) for s in SIDS}

    def round_with(dk0):
        parts = {s: ThresholdMCFEDDH({"id": s, "keys": {"pp": pp, "dk": dk}}).share_decrypt(cts, Y, dk, SIDS, "L")
                 for s, dk in (("sid_0", dk0), ("sid_1", dks["sid_1"]))}
        return ThresholdMCFEDDH({"id": "c", "keys": {"pp": pp}}).combine_decrypt(parts)

    return dks["sid_0"], round_with


#: Values of other types or texts, as functions of the value written.
OTHER_VALUES = {
    "int": int,
    "mpz": gp.mpz,
    "bytes": lambda v: v.encode(),
    "float": lambda v: float(int(v) % 1000),
    "True": lambda v: True,
    "None": lambda v: None,
    "leading space": lambda v: " " + v,
    "underscore": lambda v: v[:3] + "_" + v[3:],
    "sign": lambda v: "+" + v,
}


@pytest.mark.parametrize("other", sorted(OTHER_VALUES))
@pytest.mark.parametrize("field", ["z", "d"])
def test_tmife_ddh_key_share_values_are_decimal_strings(field, other):
    dk, round_with = _tmife_ddh()
    assert round_with(dk) == 4 and round_with(json.loads(json.dumps(dk))) == 4  # as written, and through JSON
    change = OTHER_VALUES[other]
    bad = dict(dk, z=change(dk["z"])) if field == "z" else dict(dk, d=dict(dk["d"], nid_0=change(dk["d"]["nid_0"])))
    with pytest.raises(FEKeyError, match="malformed tMIFE key share: d .per client. and z are decimal strings"):
        round_with(bad)


@pytest.mark.parametrize("other", sorted(OTHER_VALUES))
def test_tmcfe_ddh_key_share_values_are_decimal_strings(other):
    dk, round_with = _tmcfe_ddh()
    assert round_with(dk) == 4 and round_with(json.loads(json.dumps(dk))) == 4
    bad = dict(dk, d=[dk["d"][0], OTHER_VALUES[other](dk["d"][1])])
    with pytest.raises(FEKeyError, match="malformed tMCFE key share: d holds decimal strings"):
        round_with(bad)


# -- EncryptedGradient.from_dict


def _gradient():
    wrapper = FESchemeWrapper("mcfe", "ddh", {"sec_param": 128, "n": 2, "lst_nid": NIDS, "eta": 1}, precision=1)
    return encrypt_gradient(np.array([[1.0, 2.0], [3.0, 4.0]]), wrapper, nid="nid_0", label="r").to_dict()


def _respelled(layer, old, new):
    return dict(layer, data={(new if key == old else key): value for key, value in layer["data"].items()})


MALFORMED = {
    "an extra key": lambda g, lay: dict(g, extra=1),
    "nid not a string": lambda g, lay: dict(g, nid=5),
    "precision a bool": lambda g, lay: dict(g, precision=True),
    "precision a float": lambda g, lay: dict(g, precision=1.0),
    "precision a string": lambda g, lay: dict(g, precision="1"),
    "precision negative": lambda g, lay: dict(g, precision=-1),
    "mode unknown": lambda g, lay: dict(g, mode="hex"),
    "shape a tuple": lambda g, lay: dict(g, shape=(2, 2)),
    "shape a string": lambda g, lay: dict(g, shape="22"),
    "no layers": lambda g, lay: dict(g, ciphertexts=[]),
    "layers a tuple": lambda g, lay: dict(g, ciphertexts=tuple(g["ciphertexts"])),
    "layer with an extra key": lambda g, lay: dict(g, ciphertexts=[dict(lay, extra=1)]),
    "index with a space": lambda g, lay: dict(g, ciphertexts=[_respelled(lay, "(0, 0)", "( 0, 0)")]),
    "index without a space": lambda g, lay: dict(g, ciphertexts=[_respelled(lay, "(0, 0)", "(0,0)")]),
    "index with a sign": lambda g, lay: dict(g, ciphertexts=[_respelled(lay, "(0, 0)", "(+0, 0)")]),
    "entry not a ciphertext": lambda g, lay: dict(g, ciphertexts=[dict(lay, data=dict(lay["data"], **{"(0, 0)": "x"}))]),
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_encrypted_gradient_from_dict_is_strict(case):
    g = _gradient()
    assert EncryptedGradient.from_dict(json.loads(json.dumps(g))).to_dict() == g
    with pytest.raises(ValueError):
        EncryptedGradient.from_dict(MALFORMED[case](g, g["ciphertexts"][0]))


ROUND_TRIP = ["sife/lwe", "sife/paillier", "sife/damgard", "sife/fullysec_ring_lwe", "sife/fullysec_lwe",
              "mcfe/ddh", "mcfe/ddh_ristretto255", "mcfe/lwe", "mcfe/paillier", "mcfe/damgard", "mcfe/ring_lwe",
              "mcfe/fullysec_lwe", "mcfe/ddh_threshold", "mcfe/ddh_decentralized", "mcfe/lwe_threshold",
              "mcfe/lwe_decentralized", "mcfe/ring_lwe_threshold", "mcfe/ring_lwe_decentralized",
              "mcfe/paillier_decentralized"]
ROUND_TRIP += [pytest.param(s, marks=needs_bls) for s in ("mcfe/fh_multi_ipe_pairing", "mcfe/fh_multi_ipe_pairing_threshold",
                                                         "mcfe/fh_multi_ipe_pairing_decentralized")]


@pytest.mark.parametrize("scheme", ROUND_TRIP)
def test_every_gradient_the_library_writes_round_trips(scheme):
    family, variant = scheme.split("/")
    group = any(token in variant for token in ("ddh", "damgard", "paillier"))
    config = {"eta": 1, "n": 2, "lst_nid": NIDS, "ring_n": 16, "sec_param": 128 if group else 64, "bound_x": 50,
              "bound_y": 4, "lwe_n": 8, "s": 2, "t": 2, "lst_sid": SIDS}
    wrapper = FESchemeWrapper(family, variant, config, precision=1)
    for i, gradient in enumerate((np.arange(6.0).reshape(2, 3) / 10, np.array(0.5))):  # a matrix, a scalar
        label = {"label": "r{}".format(i)} if wrapper.has_label else {}
        enc = encrypt_gradient(gradient, wrapper, nid="nid_0", **label)
        sent = json.loads(json.dumps(enc.to_dict()))
        restored = EncryptedGradient.from_dict(sent)
        assert restored.to_dict() == sent and restored.shape == gradient.shape
        with pytest.raises(ValueError):  # and the strict checks hold for this scheme's gradients
            EncryptedGradient.from_dict(dict(sent, extra=1))
