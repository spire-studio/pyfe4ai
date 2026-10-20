"""Client and server lists must agree with n and s and hold distinct ids."""

import os
import pytest

from pyfe4ai.schemes.mcfe.lwe import MCFELWEKeyGenerator
from pyfe4ai.schemes.mcfe.ddh_threshold import ThresholdMCFEKeyGenerator
from pyfe4ai.schemes.threshold.threshold_utils import validate_threshold
from pyfe4ai.core.exceptions import FEValidationError


def test_n_must_match_the_client_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", os.path.join(tmp_path, "config"))
    nids = ["nid_{}".format(i) for i in range(10)]
    with pytest.raises(FEValidationError, match="lst_nid"):
        MCFELWEKeyGenerator({"n": 2, "eta": 1, "lst_nid": nids, "lwe_n": 16})
    assert MCFELWEKeyGenerator({"n": 10, "eta": 1, "lst_nid": nids, "lwe_n": 16}).n == 10


def test_server_ids_must_be_distinct():
    with pytest.raises(FEValidationError, match="duplicate"):
        validate_threshold(2, ["sid_0", "sid_0", "sid_1"])
    validate_threshold(2, ["sid_0", "sid_1", "sid_2"])
