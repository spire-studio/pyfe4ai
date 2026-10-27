"""Partial decryptions that disagree on the ciphertexts they decrypted.

A combiner given partial decryptions of different ciphertexts (a server
that decrypted another round, or a faulty one) returned ``None`` with an
FEWarning from ``combine_decrypt``, while the ndarray path raised
:class:`~pyfe4ai.core.exceptions.FESchemeError` ("element ... does not
decrypt"), as do partial decryptions of several enrolled sets whose results
differ. Both paths of every threshold scheme now raise ``FESchemeError``
with the one message
:data:`~pyfe4ai.schemes.threshold.roles.DISAGREEING_PARTIALS`, without a
warning; ``None`` stays the answer to a value out of range.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai.core.exceptions import FESchemeError, FEWarning

MESSAGE = "^partial decryptions disagree on the ciphertexts they decrypted$"
NIDS, SIDS = ["nid_0", "nid_1", "nid_2"], ["sid_0", "sid_1", "sid_2"]
ENROLLED = SIDS[:2]
Y = {"nid_0": [1], "nid_1": [2], "nid_2": [-1]}  # <x, y> = 2 x for x the same at every client
needs_bls = pytest.mark.skipif(not native_backend.available(), reason="BLS12-381 needs the native extension")


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def _classes(kind):
    if kind == "mife/ddh_threshold":
        from pyfe4ai.schemes.mife import ddh_threshold as m
        return m.ThresholdMIFEDDHKeyGenerator, m.ThresholdMIFEDDH, {"sec_param": 256}
    if kind == "mcfe/ddh_threshold":
        from pyfe4ai.schemes.mcfe import ddh_threshold as m
        return m.ThresholdMCFEDDHKeyGenerator, m.ThresholdMCFEDDH, {"sec_param": 128}
    if kind == "mife/lwe_threshold":
        from pyfe4ai.schemes.mife import lwe_threshold as m
        return m.ThresholdMIFELWEKeyGenerator, m.ThresholdMIFELWE, {"sec_param": 64, "lwe_n": 16, "bound_x": 6,
                                                                   "bound_y": 3}
    if kind == "mcfe/lwe_threshold":
        from pyfe4ai.schemes.mcfe import lwe_threshold as m
        return m.ThresholdMCFELWEKeyGenerator, m.ThresholdMCFELWE, {"sec_param": 64, "lwe_n": 16, "bound_x": 6,
                                                                   "bound_y": 3, "bound_u": 2, "label_modulus": 8}
    if kind == "mcfe/ring_lwe_threshold":
        from pyfe4ai.schemes.mcfe import ring_lwe_threshold as m
        return m.ThresholdMCFERingLWEKeyGenerator, m.ThresholdMCFERingLWE, {"sec_param": 64, "ring_n": 16,
                                                                           "bound_x": 6, "bound_y": 3}
    from pyfe4ai.schemes.mcfe import fh_multi_ipe_pairing_threshold as m
    return (m.ThresholdMCFEFHMultiIPEPairingKeyGenerator, m.ThresholdMCFEFHMultiIPEPairing,
            {"sec_param": 64, "bound_x": 6, "bound_y": 3})


class Scheme:
    """A threshold scheme (precision 0): the enrolled servers' partial
    decryptions of ciphertexts of given values, for one value and for ndarrays,
    and the combiner."""

    def __init__(self, kind):
        self.kind = kind
        kg_cls, self.crypto, extra = _classes(kind)
        self.kg = kg_cls(dict({"lst_nid": NIDS, "lst_sid": SIDS, "eta": 1, "n": 3, "s": 3, "t": 2}, **extra))
        self.kg.setup()
        self.pp = self.kg.get_public_parameters()
        self.mcfe = kind.startswith("mcfe")
        self.creds = {"fusion_weight": Y, "label": "L"} if self.mcfe else {"fusion_weight": Y}
        if kind == "mcfe/ring_lwe_threshold":
            self.creds["key_part"] = "full"
        self.clients = {n: self._crypto(n, sk=self.kg.get_private_keys(n)) for n in NIDS}
        self.combiner = self._crypto("c")

    def _crypto(self, ident, **keys):
        return self.crypto({"id": ident, "precision": 0, "keys": dict({"pp": self.pp}, **keys)})

    def _weights(self):
        """The credentials share_decrypt takes: the weights, or (LWE / Ring-LWE MCFE, FH-MCFE) the request."""
        if self.kind in ("mife/ddh_threshold", "mife/lwe_threshold", "mcfe/ddh_threshold"):
            return Y
        return self.creds

    def partials(self, x, servers=ENROLLED):
        if self.kind == "mcfe/ring_lwe_threshold":
            cts = {n: c.encrypt([[x]], "L") for n, c in self.clients.items()}
        elif self.mcfe:
            cts = {n: c.encrypt([x], "L") for n, c in self.clients.items()}
        else:
            cts = {n: c.encrypt([x]) for n, c in self.clients.items()}
        out = {}
        for s in servers:
            dk = self.kg.get_decryption_keys(s, credentials=self.creds)
            label = ("L",) if self.kind in ("mcfe/ddh_threshold", "mcfe/fh_multi_ipe_pairing_threshold") else ()
            out[s] = self._crypto(s, dk=dk).share_decrypt(cts, self._weights(), dk, ENROLLED, *label)
        return out

    def nd_partials(self, array, servers=ENROLLED):
        label = {"label": "L"} if self.mcfe else {}
        cts = {n: c.encrypt_lst_ndarray([array], **label) for n, c in self.clients.items()}
        creds = dict(self.creds)
        if self.kind == "mcfe/ring_lwe_threshold":
            creds["n_chunks"] = self.crypto.key_chunks(cts["nid_0"])
        out = {}
        for s in servers:
            dk = self.kg.get_decryption_keys(s, credentials=creds)
            extra = ("L",) if self.kind in ("mcfe/ddh_threshold", "mcfe/fh_multi_ipe_pairing_threshold") else ()
            weights = creds if self._weights() is self.creds else Y
            out[s] = self._crypto(s, dk=dk).compute_lst_ndarray_ct(cts, weights, dk, ENROLLED, *extra)
        return out

    def nd_combine(self, partials):
        if self.kind in ("mcfe/ddh_threshold", "mcfe/fh_multi_ipe_pairing_threshold"):
            return self.combiner.decrypt_lst_ndarray_ct(partials, "L")
        return self.combiner.decrypt_lst_ndarray_ct(partials)


KINDS = ["mife/ddh_threshold", "mcfe/ddh_threshold", "mife/lwe_threshold", "mcfe/lwe_threshold",
         "mcfe/ring_lwe_threshold", pytest.param("mcfe/fh_multi_ipe_pairing_threshold", marks=needs_bls)]


def _without_fe_warning(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    assert not [w for w in caught if issubclass(w.category, FEWarning)], [str(w.message) for w in caught]


@pytest.mark.parametrize("kind", KINDS)
def test_both_paths_refuse_partial_decryptions_of_different_ciphertexts(kind):
    scheme = Scheme(kind)
    honest = scheme.partials(1)
    assert np.ravel(scheme.combiner.combine_decrypt(honest)).tolist() == [2]
    other = dict(honest, sid_1=scheme.partials(2, servers=["sid_1"])["sid_1"])  # sid_1 decrypted other ciphertexts

    def scalar():
        with pytest.raises(FESchemeError, match=MESSAGE):  # it was None, with an FEWarning
            scheme.combiner.combine_decrypt(other)
        with pytest.raises(FESchemeError, match=MESSAGE):  # in one of several enrolled sets too
            scheme.combiner.combine_decrypt([honest, other])

    _without_fe_warning(scalar)

    array = np.array([1, 3])
    nd = scheme.nd_partials(array)
    assert np.asarray(scheme.nd_combine(nd)[0], dtype=float).tolist() == [2.0, 6.0]
    nd_other = dict(nd, sid_1=scheme.nd_partials(array + 1, servers=["sid_1"])["sid_1"])

    def ndarray():
        with pytest.raises(FESchemeError, match=MESSAGE):  # the error and message of the scalar path
            scheme.nd_combine(nd_other)

    _without_fe_warning(ndarray)
    if kind == "mcfe/ddh_threshold":  # None stays the answer to a value out of range
        forged = dict(honest, sid_1=dict(honest["sid_1"], u_d="4"))  # another key term: out of the dlog bound
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert scheme.combiner.combine_decrypt(forged) is None
        assert any("out of bound" in str(w.message) for w in caught if issubclass(w.category, FEWarning))


def test_the_ml_adapter_raises_the_same_error():
    """combine_partial_decryptions (mcfe/ring_lwe_threshold) raised "element (0, chunk 0) does not decrypt"."""
    from pyfe4ai.ml import FESchemeWrapper, encrypt_gradient, partial_decrypt_gradients, prepare_aggregation
    from pyfe4ai.ml.threshold import combine_partial_decryptions

    wrapper = FESchemeWrapper("mcfe", "ring_lwe_threshold", {"n": 2, "lst_nid": NIDS[:2], "eta": 1, "ring_n": 16,
                                                             "sec_param": 64, "bound_x": 60, "bound_y": 3,
                                                             "s": 3, "t": 2, "lst_sid": SIDS}, precision=1)
    weights = {"nid_0": 1, "nid_1": 2}
    prepared = prepare_aggregation(wrapper, weights, shape=(2,), label="r1")
    twin = copy.deepcopy(wrapper)  # the same keys, its own record of used labels
    enc = {n: encrypt_gradient(np.array(v), wrapper, nid=n, label="r1")
           for n, v in (("nid_0", [1.0, 2.0]), ("nid_1", [3.0, 4.0]))}
    other = {n: encrypt_gradient(np.array([5.0, 5.0]), twin, nid=n, label="r1") for n in NIDS[:2]}
    partials = {s: partial_decrypt_gradients(enc, wrapper, prepared, server=s) for s in prepared.servers}
    assert np.allclose(combine_partial_decryptions(partials, wrapper, prepared), [7.0, 10.0])
    second = prepared.servers[1]
    partials[second] = partial_decrypt_gradients(other, wrapper, prepared, server=second)
    with pytest.raises(FESchemeError, match=MESSAGE):
        combine_partial_decryptions(partials, wrapper, prepared)

