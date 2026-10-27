"""FEWarning texts of unknown server ids (October 2026 code review of the threshold module split).

Low (availability, log hygiene; introduced by the split): the seven ``"invalid sid: {}".format(sid)``
FEWarnings carry the caller's sid in their text. Under the default warning filters Python keeps one
``__warningregistry__`` entry per distinct text in the calling module and prints each one to stderr,
so a key-distribution service that passes request sids to ``get_decryption_keys`` /
``get_weight_key_share`` grows without bound (20000 unknown sids: 20000 entries, +8.8 MB, 40000
stderr lines) and writes caller-controlled text to stderr. Before the split the sid was only logged
(the ``pyfe4ai`` logger has a NullHandler). A constant warning text (the sid in the log record
only) keeps one entry per site.
"""

from __future__ import annotations

import warnings

import pytest

from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFEDDHKeyGenerator
from pyfe4ai.schemes.mcfe.lwe_threshold import ThresholdMCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.ring_lwe_threshold import ThresholdMCFERingLWEKeyGenerator
from pyfe4ai.schemes.mife.ddh_threshold import ThresholdMIFEDDHKeyGenerator
from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWEKeyGenerator

NIDS, SIDS = ["nid_0", "nid_1", "nid_2"], ["sid_0", "sid_1", "sid_2"]
LWE = {"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS, "lst_sid": SIDS, "lwe_n": 16,
       "bound_x": 6, "bound_y": 6}
DDH = {"sec_param": 128, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS, "lst_sid": SIDS}
CREDS = {"fusion_weight": {nid: [1] for nid in NIDS}, "label": "L"}


def _pairing():
    from tests.pairing_group import GROUP
    from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing_threshold import ThresholdMCFEFHMultiIPEPairingKeyGenerator

    kg = ThresholdMCFEFHMultiIPEPairingKeyGenerator({"sec_param": 64, "eta": 1, "n": 3, "s": 3, "t": 2, "lst_nid": NIDS,
                                              "lst_sid": SIDS, "bound_x": 4, "bound_y": 4,
                                              "pairing_group_param": GROUP})
    return kg.get_decryption_keys


SITES = {
    "tMIFE-LWE": lambda: ThresholdMIFELWEKeyGenerator(dict(LWE)).get_decryption_keys,
    "tMCFE-LWE": lambda: ThresholdMCFELWEKeyGenerator(dict(LWE, bound_u=2, label_modulus=8)).get_decryption_keys,
    "tMCFE-Ring-LWE": lambda: ThresholdMCFERingLWEKeyGenerator(
        dict(n=3, lst_nid=NIDS, s=3, t=2, lst_sid=SIDS, eta=1, ring_n=32, sec_param=64, bound_x=100, bound_y=3,
             stat_param=64)).get_decryption_keys,
    "tMCFE-Ring-LWE weight part": lambda: ThresholdMCFERingLWEKeyGenerator(
        dict(n=3, lst_nid=NIDS, s=3, t=2, lst_sid=SIDS, eta=1, ring_n=32, sec_param=64, bound_x=100, bound_y=3,
             stat_param=64)).get_weight_key_share,
    "tMIFE-DDH": lambda: ThresholdMIFEDDHKeyGenerator(dict(DDH)).get_decryption_keys,
    "tMCFE-DDH": lambda: ThresholdMCFEDDHKeyGenerator(dict(DDH)).get_decryption_keys,
    "tMCFE-FH-pairing": _pairing,
}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path / "config"))


@pytest.mark.parametrize("site", sorted(SITES))
def test_unknown_sids_do_not_grow_the_warning_registry(site):
    try:
        get_keys = SITES[site]()
    except ImportError as exc:  # the pairing backend
        pytest.skip(str(exc))
    with warnings.catch_warnings():
        warnings.simplefilter("default")  # an application's default filters
        for i in range(50):
            assert get_keys("unknown-sid-%d" % i, credentials=CREDS) is None
        registry = globals().get("__warningregistry__", {})
        entries = [key for key in registry if isinstance(key, tuple) and "unknown-sid-" in str(key[0])]
    assert not entries, (
        "{}: {} warning texts with the caller's sid kept for 50 unknown sids, e.g. {!r} ({}): the text "
        "should not carry it".format(site, len(entries), entries[0][0], entries[0][1].__name__))
