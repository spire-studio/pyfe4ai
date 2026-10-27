"""The key shares of the decentralised MCFE over Ring-LWE from malformed keys and shares.

``derive_function_decryption_key_share`` reads the client's pad ``u`` and
IPFE secret ``s`` (``eta`` polynomials of ``ring_n`` values each). With
fewer rows the Python backend zipped them away and published a key share
whose combined key decrypted to wrong values inside the bound (e.g.
``[-83, -295, -283]`` for ``[3, 6, 9]``); the native backend failed with a bare
``ValueError``; a row of another length gave ``IndexError`` / ``ValueError``
or, for ``u``, a share wrong in the coefficients beyond the batch.
``combine_function_decryption_key_share`` summed shares whose ``sk_y`` /
``z`` / ``z_chunks`` were not ``ring_n`` values: natively a bare
``ValueError``, in Python the shorter length (a wrong key). Both now refuse
with :class:`~pyfe4ai.core.exceptions.FEKeyError` (constant messages) before
any computation, in both backends.
"""

from __future__ import annotations

import pytest

from pyfe4ai import backend as native_backend
from pyfe4ai.core.exceptions import FEKeyError
from pyfe4ai.schemes.mcfe import ring_lwe_decentralized as dm

BACKENDS = ["python"] + (["native"] if native_backend.available() else [])
NIDS = ["c0", "c1", "c2"]
ROWS = "the encryption key has fewer than eta rows of u or s"
SHAPE = "the encryption key has a row of u or s that is not ring_n values"
SHARE = "malformed key share: sk_y, z and every z_chunks entry are ring_n values"


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


def _setup(eta):
    kg = dm.DecentralizedMCFERingLWEKeyGenerator({"sec_param": 64, "eta": eta, "n": 3, "lst_nid": NIDS, "ring_n": 16,
                                                  "bound_x": 50, "bound_y": 2})
    kg.setup()
    pp = kg.get_public_parameters()
    sks = {n: kg.get_private_keys(n) for n in NIDS}
    clients = {n: dm.DecentralizedMCFERingLWE({"id": n, "precision": 0, "keys": {"pp": pp, "sk": sks[n]}})
               for n in NIDS}
    weights = {n: [1] * eta for n in NIDS}
    cts = {n: clients[n].encrypt([[1, 2, 3]] * eta, "L") for n in NIDS}
    right = [3 * eta * v for v in (1, 2, 3)]
    return pp, sks, clients, weights, cts, right


MALFORMED_KEYS = {
    "u one row short": (lambda sk: dict(sk, u=sk["u"][:-1]), ROWS),
    "u empty": (lambda sk: dict(sk, u=[]), ROWS),
    "s one row short": (lambda sk: dict(sk, s=sk["s"][:-1]), ROWS),
    "u row one value short": (lambda sk: dict(sk, u=[sk["u"][0][:-1]] + sk["u"][1:]), SHAPE),
    "u row one value long": (lambda sk: dict(sk, u=[sk["u"][0] + ["1"]] + sk["u"][1:]), SHAPE),
    "s row one value short": (lambda sk: dict(sk, s=[sk["s"][0][:-1]] + sk["s"][1:]), SHAPE),
    "s row None": (lambda sk: dict(sk, s=[None] + sk["s"][1:]), SHAPE),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", sorted(MALFORMED_KEYS))
@pytest.mark.parametrize("eta", [1, 2])
def test_a_key_share_from_a_malformed_key_is_refused(eta, case, backend):
    with native_backend.backend(backend):
        pp, sks, clients, weights, cts, right = _setup(eta)
        shares = {n: clients[n].derive_function_decryption_key_share(weights, "L") for n in NIDS}
        key = clients["c1"].combine_function_decryption_key_share(shares)
        assert clients["c1"].decrypt(cts, key, weights, "L") == right  # the whole keys
        change, message = MALFORMED_KEYS[case]
        bad = dm.DecentralizedMCFERingLWE({"id": "c0", "precision": 0, "keys": {"pp": pp, "sk": change(sks["c0"])}})
        with pytest.raises(FEKeyError) as info:
            bad.derive_function_decryption_key_share(weights, "L")
        assert str(info.value) == message


MALFORMED_SHARES = {
    "z one value short": lambda s: dict(s, z=s["z"][:-1]),
    "z one value long": lambda s: dict(s, z=s["z"] + ["1"]),
    "z first value dropped": lambda s: dict(s, z=s["z"][1:]),
    "sk_y one value short": lambda s: dict(s, sk_y=s["sk_y"][:-1]),
    "z_chunks entry short": lambda s: dict(s, z_chunks=[s["z_chunks"][0][:-1]]),
    "z not a list": lambda s: dict(s, z="0"),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", sorted(MALFORMED_SHARES))
def test_combining_a_malformed_key_share_is_refused(case, backend):
    with native_backend.backend(backend):
        pp, sks, clients, weights, cts, right = _setup(1)
        shares = {n: clients[n].derive_function_decryption_key_share(weights, "L", n_chunks=1) for n in NIDS}
        assert clients["c1"].decrypt(cts, clients["c1"].combine_function_decryption_key_share(shares), weights,
                                     "L") == right
        with pytest.raises(FEKeyError) as info:
            clients["c1"].combine_function_decryption_key_share(dict(shares, c0=MALFORMED_SHARES[case](shares["c0"])))
        assert str(info.value) == SHARE
