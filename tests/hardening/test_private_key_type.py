"""An encryption key (``keys["sk"]``) that is not a dict.

The encryption paths that read the fields of the key through ``.get`` -- the
row checks of the Ring-LWE schemes, the label-hash check of the MCFE LWE
schemes, the tag keys of the threshold LWE and Ring-LWE layers -- failed on
such a key with a bare ``AttributeError`` (``'list' object has no attribute
'get'``) or ``TypeError`` (``argument of type 'NoneType' is not
iterable``); SIFE Ring-LWE and SIFE FullySec Ring-LWE raised ``TypeError``
before their row checks. They now refuse it with
:class:`~pyfe4ai.core.exceptions.FEKeyError` and a constant message, in every
encryption path (single values, ndarrays, the precomputation, a
decentralized key share), before anything is drawn.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyfe4ai
from pyfe4ai.core.base import PRIVATE_KEY_NOT_A_DICT
from pyfe4ai.core.exceptions import FEKeyError

NIDS = ["c0", "c1"]
#: name -> what stands for the key
NOT_DICTS = {"a list": [1, 2], "None": None, "a string": "key", "an int": 5, "a tuple": ("pk", "u")}
#: scheme -> the calls that encrypt with the client's key (eta = 1)
CALLS = {
    "sife/ring_lwe": [lambda c: c.encrypt([[1, 2]])],
    "sife/fullysec_ring_lwe": [lambda c: c.encrypt([1]), lambda c: c.encrypt_lst_ndarray([np.ones(3)])],
    "mife/ring_lwe": [lambda c: c.encrypt([[1, 2]])],
    "mife/lwe_threshold": [lambda c: c.encrypt([1]), lambda c: c.encrypt_lst_ndarray([np.ones(3)])],
}
for _scheme in ("mcfe/ring_lwe", "mcfe/ring_lwe_threshold", "mcfe/ring_lwe_decentralized"):
    CALLS[_scheme] = [lambda c: c.encrypt([[1, 2]], "L"), lambda c: c.encrypt_lst_ndarray([np.ones(3)], label="L"),
                      lambda c: c.precompute("L", [(3,)])]
for _scheme in ("mcfe/lwe", "mcfe/fullysec_lwe", "mcfe/lwe_threshold", "mcfe/lwe_decentralized"):
    CALLS[_scheme] = [lambda c: c.encrypt([1], "L"), lambda c: c.encrypt_lst_ndarray([np.ones(3)], label="L")]
CALLS["mcfe/ring_lwe_decentralized"].append(lambda c: c.derive_function_decryption_key_share({n: [1] for n in NIDS},
                                                                                               "L"))


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


@pytest.mark.parametrize("key", sorted(NOT_DICTS))
@pytest.mark.parametrize("scheme", sorted(CALLS))
def test_an_encryption_key_that_is_not_a_dict(scheme, key):
    family = scheme.split("/")[0]
    kg, crypto = pyfe4ai.create(scheme, {"eta": 1, "n": 2, "lst_nid": NIDS, "ring_n": 16, "lwe_n": 16, "bound_x": 6,
                                         "bound_y": 6, "bound_u": 2, "label_modulus": 8, "s": 2, "t": 2,
                                         "lst_sid": ["s0", "s1"], "sec_param": 64})
    kg.setup()
    pp = kg.get_public_parameters()
    sk = kg.get_private_keys() if family == "sife" else kg.get_private_keys("c0")
    for call in CALLS[scheme]:
        call(crypto({"id": "c0", "precision": 0, "keys": {"pp": pp, "sk": sk}}))  # the key as issued
        client = crypto({"id": "c0", "precision": 0, "keys": {"pp": pp, "sk": NOT_DICTS[key]}})
        with pytest.raises(FEKeyError) as info:
            call(client)
        assert str(info.value) == PRIVATE_KEY_NOT_A_DICT
