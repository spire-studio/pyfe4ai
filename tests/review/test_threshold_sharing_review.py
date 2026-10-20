"""Replicated sharing by default for t < s (October 2026 code review).

The fix itself checks out (the simulation lemma holds exactly in code; residue
attacks against replicated shares are at chance). Before the follow-up fixes:

* a boolean threshold (``t=True``) passed validate_threshold as 1 but escaped
  the t < s policy of resolve_sharing: Shamir with t = 1 < s, no warning, no
  refusal, and nu mod 2 leaked from a single honest partial (no corrupted
  server needed);
* key shares and public parameters of v0.3.0 (Shamir, no ``sharing`` field)
  with t < s were still served without any warning, so an upgraded deployment
  kept the leak silently. Servers now refuse them by default, with a
  per-server opt-out ``legacy_shamir="warn"``.
"""

import warnings

import pytest

from pyfe4ai.schemes.mife.lwe_threshold import ThresholdMIFELWE, ThresholdMIFELWEKeyGenerator
from pyfe4ai.core.exceptions import FESchemeError, FESecurityWarning, FEValidationError

NIDS = ["nid_0", "nid_1"]
SIDS = ["sid_0", "sid_1", "sid_2"]
CFG = {"sec_param": 64, "n": 2, "lst_nid": NIDS, "s": 3, "lst_sid": SIDS, "eta": 1,
       "lwe_n": 16, "bound_x": 4, "bound_y": 5}


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_boolean_threshold_is_refused():
    with pytest.raises(FEValidationError):
        ThresholdMIFELWEKeyGenerator(dict(CFG, t=True))


def _v030_round():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kg = ThresholdMIFELWEKeyGenerator(dict(CFG, t=2, sharing="shamir"))  # toy: warned at key generation
        kg.setup()
    new = ("sharing", "threshold_construction", "security_profile")
    pp = {k: v for k, v in kg.get_public_parameters().items() if k not in new}  # v0.3.0 format
    y = {nid: [1] for nid in NIDS}
    dk = {k: v for k, v in kg.get_decryption_keys("sid_0", credentials={"fusion_weight": y}).items() if k != "sharing"}
    encs = {nid: ThresholdMIFELWE({"id": nid, "keys": {"pp": pp, "sk": kg.get_private_keys(nid)}}) for nid in NIDS}
    return pp, dk, y, {nid: encs[nid].encrypt([1]) for nid in NIDS}


def test_v030_shamir_shares_with_t_below_s_are_refused_by_default():
    """Refused by default, with an explicit per-server opt-out."""
    pp, dk, y, cts = _v030_round()
    server = ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}})
    with pytest.raises(FESchemeError):
        server.share_decrypt(cts, y, dk, ["sid_0", "sid_2"])


def test_v030_shamir_shares_served_only_with_an_explicit_warning_opt_out():
    pp, dk, y, cts = _v030_round()
    server = ThresholdMIFELWE({"id": "sid_0", "keys": {"pp": pp, "dk": dk}, "legacy_shamir": "warn"})
    with pytest.warns(FESecurityWarning):
        server.share_decrypt(cts, y, dk, ["sid_0", "sid_2"])
