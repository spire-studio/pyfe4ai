"""Ring-LWE encryption keys whose rows are too few or not ``ring_n`` values.

* SIFE Ring-LWE (``sife/ring_lwe``): a ``pk`` with fewer than ``eta`` rows or
  with a row of another length failed with a bare ``IndexError`` /
  ``ValueError`` / ``TypeError``.
* SIFE FullySec Ring-LWE (``sife/fullysec_ring_lwe``, both ndarray layouts):
  a ``pk`` with fewer rows silently gave ciphertexts with fewer ``ct`` rows
  (refused only by the decryption); a row of another length: ``IndexError``.
* MIFE / MCFE Ring-LWE and the threshold and decentralized MCFE (rows of the
  right count, see test_ring_lwe_short_keys.py): a ``u`` row of another length
  silently gave a ciphertext of another length; a ``pk`` row:
  ``IndexError``.

Every encryption path now refuses such a key with
:class:`~pyfe4ai.core.exceptions.FEKeyError` and a constant message before
anything is drawn, in both backends.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai.core.exceptions import FEKeyError
from pyfe4ai.schemes.mcfe import ring_lwe as mcfe
from pyfe4ai.schemes.mcfe import ring_lwe_decentralized as dmcfe
from pyfe4ai.schemes.mcfe import ring_lwe_threshold as tmcfe
from pyfe4ai.schemes.mife import ring_lwe as mife
from pyfe4ai.schemes.sife.fullysec_ring_lwe import SIFEFullySecRingLWE, SIFEFullySecRingLWEKeyGenerator
from pyfe4ai.schemes.sife.ring_lwe import SIFERingLWE, SIFERingLWEKeyGenerator

BACKENDS = ["python"] + (["native"] if native_backend.available() else [])
NIDS = ["c0", "c1", "c2"]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def rows_message(what):
    return "the encryption key has fewer than eta rows of " + what


def shape_message(what):
    return "the encryption key has a row of " + what + " that is not ring_n values"


#: name -> (change of the field's rows, which message)
MALFORMED = {
    "one row short": (lambda rows: rows[:-1], rows_message),
    "empty": (lambda rows: [], rows_message),
    "row one value short": (lambda rows: [rows[0][:-1]] + rows[1:], shape_message),
    "row one value long": (lambda rows: [rows[0] + ["1"]] + rows[1:], shape_message),
    "row None": (lambda rows: [None] + rows[1:], shape_message),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", sorted(MALFORMED))
@pytest.mark.parametrize("eta", [1, 2])
def test_sife_ring_lwe_refuses_malformed_pk_rows(eta, case, backend):
    with native_backend.backend(backend):
        kg = SIFERingLWEKeyGenerator({"sec_param": 64, "eta": eta, "ring_n": 16, "bound_x": 50, "bound_y": 2})
        kg.setup()
        pp, sk = kg.get_public_parameters(), kg.get_private_keys()
        assert SIFERingLWE({"precision": 0, "keys": {"pp": pp, "sk": sk}}).encrypt([[1, 2, 3]] * eta)
        change, message = MALFORMED[case]
        client = SIFERingLWE({"precision": 0, "keys": {"pp": pp, "sk": dict(sk, pk=change(sk["pk"]))}})
        with pytest.raises(FEKeyError) as info:
            client.encrypt([[1, 2, 3]] * eta)
        assert str(info.value) == message("pk")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", sorted(MALFORMED))
@pytest.mark.parametrize("layout", ["elementwise", "packed"])
@pytest.mark.parametrize("eta", [1, 2])
def test_sife_fullysec_ring_lwe_refuses_malformed_pk_rows(eta, layout, case, backend):
    with native_backend.backend(backend):
        kg = SIFEFullySecRingLWEKeyGenerator({"sec_param": 64, "eta": eta, "ring_n": 16, "bound_x": 5, "bound_y": 2,
                                               "ndarray_layout": layout})
        kg.setup()
        pp, sk = kg.get_public_parameters(), kg.get_private_keys()
        assert SIFEFullySecRingLWE({"precision": 0, "keys": {"pp": pp, "sk": sk}}).encrypt([1] * eta)
        change, message = MALFORMED[case]
        client = SIFEFullySecRingLWE({"precision": 0, "keys": {"pp": pp, "sk": dict(sk, pk=change(sk["pk"]))}})
        calls = [lambda: client.encrypt([1] * eta)]
        if layout == "packed":
            calls.append(lambda: client.encrypt([[1, 2]] * eta))
        if eta == 1:
            calls.append(lambda: client.encrypt_lst_ndarray([np.ones((2, 3))]))
        for call in calls:
            with pytest.raises(FEKeyError) as info:
                call()
            assert str(info.value) == message("pk")


SCHEMES = {
    "mcfe/ring_lwe": (mcfe.MCFERingLWEKeyGenerator, mcfe.MCFERingLWE, {}),
    "mcfe/ring_lwe_decentralized": (dmcfe.DecentralizedMCFERingLWEKeyGenerator, dmcfe.DecentralizedMCFERingLWE, {}),
    "mcfe/ring_lwe_threshold": (tmcfe.ThresholdMCFERingLWEKeyGenerator, tmcfe.ThresholdMCFERingLWE,
                                {"s": 2, "t": 2, "lst_sid": ["s0", "s1"]}),
    "mife/ring_lwe": (mife.MIFERingLWEKeyGenerator, mife.MIFERingLWE, {}),
}
ROW_SHAPES = ["row one value short", "row one value long", "row None"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", ROW_SHAPES)
@pytest.mark.parametrize("field", ["pk", "u"])
@pytest.mark.parametrize("scheme", sorted(SCHEMES))
def test_mife_and_mcfe_refuse_key_rows_of_another_length(scheme, field, case, backend):
    kg_cls, crypto, extra = SCHEMES[scheme]
    with native_backend.backend(backend):
        kg = kg_cls(dict({"sec_param": 64, "eta": 1, "n": 3, "lst_nid": NIDS, "ring_n": 16, "bound_x": 50,
                          "bound_y": 2}, **extra))
        kg.setup()
        pp, sk = kg.get_public_parameters(), kg.get_private_keys("c0")
        change, message = MALFORMED[case]
        client = crypto({"id": "c0", "precision": 0, "keys": {"pp": pp, "sk": dict(sk, **{field: change(sk[field])})}})
        calls = [lambda: client.encrypt([[1, 2, 3]]) if scheme.startswith("mife") else client.encrypt([[1, 2, 3]], "L")]
        if scheme.startswith("mcfe"):
            calls += [lambda: client.encrypt_lst_ndarray([np.ones((2, 3))], label="L"),
                      lambda: client.precompute("L", [(2, 3)])]
        for call in calls:
            with pytest.raises(FEKeyError) as info:
                call()
            assert str(info.value) == message("pk or u")
