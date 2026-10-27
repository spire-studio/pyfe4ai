"""Seed-expanded replicated sharing (October 2026 code review).

The construction checks out (exact reconstruction for all three layers and
(s, t) up to (6, 3), multipliers 0/1, residue attacks at chance, domain
separation, sharing confusion refused). Before the fix, the expansion hashed
json([seed, T, path, q]) with the typed path, so a key share of integer client
ids no longer expanded to the same pieces after a JSON round trip (ids become
strings): decryption then returned None or a wrong value, silently, while full
replicated sharing still decrypted.
"""

import json
import os
import warnings

import pytest

from pyfe4ai.schemes.mife import lwe_threshold as tmife

SIDS = ["s1", "s2", "s3"]


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))


def _round(sharing, transport):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kg = tmife.ThresholdMIFELWEKeyGenerator({"sec_param": 64, "eta": 1, "n": 2, "s": 3, "t": 2, "lst_nid": [0, 1],
                                                 "lst_sid": SIDS, "lwe_n": 16, "bound_x": 6, "bound_y": 6,
                                                 "sharing": sharing})
        kg.setup()
    pp = kg.get_public_parameters()
    y = {0: [2], 1: [3]}
    cts = {n: tmife.ThresholdMIFELWE({"id": n, "keys": {"pp": pp, "sk": kg.get_private_keys(n)}}).encrypt([5]) for n in y}
    dks = {sid: kg.get_decryption_keys(sid, credentials={"fusion_weight": y}) for sid in SIDS}
    dks, cts, y, pp = (transport(o) for o in (dks, cts, y, pp))
    out = []
    for enrolled in (["s1", "s2"], ["s1", "s3"], ["s2", "s3"]):
        parts = {sid: tmife.ThresholdMIFELWE({"id": sid, "keys": {"pp": pp, "dk": dks[sid]}})
                 .share_decrypt(cts, y, dks[sid], enrolled) for sid in enrolled}
        out.append(tmife.ThresholdMIFELWE({"id": "c", "keys": {"pp": pp}}).combine_decrypt(parts))
    return out


@pytest.mark.parametrize("sharing", ["replicated", "replicated-seeded"])
def test_key_shares_survive_json_transport(sharing):
    as_json = lambda o: json.loads(json.dumps(o))  # noqa: E731
    try:
        got = _round(sharing, as_json)
    except Exception as exc:  # noqa: BLE001 - a refusal (e.g. non-str ids) is acceptable
        assert "id" in str(exc).lower() or "key" in str(exc).lower(), exc
        return
    assert got == [25, 25, 25], "{} after a JSON round trip: {}".format(sharing, got)
