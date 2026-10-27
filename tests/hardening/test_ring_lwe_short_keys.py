"""Ring-LWE encryption keys with fewer than ``eta`` rows of the pad ``u`` or of ``pk``.

The MIFE / MCFE Ring-LWE encryption zipped the plaintext rows with the
key's rows: a key with fewer rows of ``u`` gave a single ciphertext without
the missing ``ct0`` rows (refused only at decryption), and the packed
ndarray encryption and its precomputation failed with a bare
``IndexError``; fewer rows of ``pk`` gave an ``IndexError``. Every path now
refuses such a key at encryption with
:class:`~pyfe4ai.core.exceptions.FEKeyError` (a constant message, no key
data), in both backends, for the MCFE schemes and their threshold and
decentralised variants (which inherit the checks) and for the MIFE.
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

BACKENDS = ["python"] + (["native"] if native_backend.available() else [])
NIDS = ["c0", "c1", "c2"]
REFUSED = "the encryption key has fewer than eta rows of pk or u"
SCHEMES = {
    "mcfe/ring_lwe": (mcfe.MCFERingLWEKeyGenerator, mcfe.MCFERingLWE, {}),
    "mcfe/ring_lwe_decentralized": (dmcfe.DecentralizedMCFERingLWEKeyGenerator, dmcfe.DecentralizedMCFERingLWE, {}),
    "mcfe/ring_lwe_threshold": (tmcfe.ThresholdMCFERingLWEKeyGenerator, tmcfe.ThresholdMCFERingLWE,
                                {"s": 2, "t": 2, "lst_sid": ["s0", "s1"]}),
    "mife/ring_lwe": (mife.MIFERingLWEKeyGenerator, mife.MIFERingLWE, {}),
}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def _client(scheme, eta, short, ct_format="json"):
    kg_cls, crypto, extra = SCHEMES[scheme]
    kg = kg_cls(dict({"sec_param": 64, "eta": eta, "n": 3, "lst_nid": NIDS, "ring_n": 16, "bound_x": 50,
                      "bound_y": 2}, **extra))
    kg.setup()
    pp = kg.get_public_parameters()
    sk = kg.get_private_keys("c0")
    if short is not None:
        field, rows = short
        sk = dict(sk, **{field: sk[field][:rows]})
    config = {"id": "c0", "precision": 0, "keys": {"pp": pp, "sk": sk}}
    if ct_format != "json":
        config["ct_format"] = ct_format
    return crypto(config)


SHORT = [("u", 0), ("u", -1), ("pk", -1)]
IDS = ["u-empty", "u-one-row-short", "pk-one-row-short"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("short", SHORT, ids=IDS)
@pytest.mark.parametrize("eta", [1, 2])
@pytest.mark.parametrize("scheme", sorted(SCHEMES))
def test_a_single_encryption_refuses_a_key_with_fewer_rows(scheme, eta, short, backend):
    with native_backend.backend(backend):
        client = _client(scheme, eta, short)
        args = ([[1, 2, 3]] * eta,) if scheme.startswith("mife") else ([[1, 2, 3]] * eta, "L")
        with pytest.raises(FEKeyError) as info:
            client.encrypt(*args)
        assert str(info.value) == REFUSED
        if scheme == "mcfe/ring_lwe":  # the binary ciphertext format too
            with pytest.raises(FEKeyError, match=REFUSED):
                _client(scheme, eta, short, "binary").encrypt(*args)
        whole = _client(scheme, eta, None)  # the same parameters, the whole key: encrypts
        assert whole.encrypt(*args)


MCFE = sorted(name for name in SCHEMES if name.startswith("mcfe"))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("short", SHORT, ids=IDS)
@pytest.mark.parametrize("scheme", MCFE)
def test_ndarray_encryption_and_precomputation_refuse_a_key_with_fewer_rows(scheme, short, backend):
    with native_backend.backend(backend):
        client = _client(scheme, 1, short)
        with pytest.raises(FEKeyError) as info:
            client.encrypt_lst_ndarray([np.ones((2, 3))], label="L")
        assert str(info.value) == REFUSED
        with pytest.raises(FEKeyError) as info:
            client.precompute("L", [(2, 3)])
        assert str(info.value) == REFUSED
        whole = _client(scheme, 1, None)
        pre = whole.precompute("L", [(2, 3)])
        assert whole.encrypt_lst_ndarray([np.ones((2, 3))], label="L", precomputed=pre)
