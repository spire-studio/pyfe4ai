"""Re-check of the fixes of review R1 (5316b73 to b6ae92b): one failing test per new finding.

Every test here fails on b6ae92b for the reason its finding states. The names carry the ids of the re-check section
of the review report: R1-N01 and R1-N02 are Low, R1-N03 and R1-N04 are Info. The nine findings of the review itself
are fixed; ``test_v05_m1_review.py`` covers them.

R1-N02 pins the rule that the review recommended and the library adopted: ``standard`` refuses an option of another
scheme.
"""

import copy
import ctypes
import pickle
from pathlib import Path

import pytest

import pyfe4ai
from pyfe4ai.core import config as configs
from pyfe4ai.core.exceptions import FEConfigError, FEValidationError
from pyfe4ai.primitives.pairing import bls12_381 as bls
from pyfe4ai.schemes.sife.ddh import SIFEDDHKeyGenerator

SERVERS = ["sid_{}".format(i) for i in range(5)]


@pytest.fixture(autouse=True)
def own_parameter_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path))


# ---------------------------------------------------------------------------
# Low


def test_r1_n01_a_subclass_keeps_its_own_options():
    """R1-N01: ``scheme_of`` gives a subclass the scheme of its base, so the option check holds the subclass to the
    options of that scheme, and a key that the subclass itself reads is refused. The subclass cannot declare it; on
    5316b73 this built. The escape for key generators outside the registry does not reach it."""

    class Audited(SIFEDDHKeyGenerator):
        def __init__(self, config, **kwargs):
            self.audit_log = config.get("audit_log", "none")
            super().__init__(config, **kwargs)

    try:
        assert Audited({"eta": 2, "audit_log": "yes"}).audit_log == "yes"
    except FEConfigError as exc:
        pytest.fail("the option of the subclass is refused: {}".format(exc))


def test_r1_n02_the_standard_profile_does_not_lower_a_threshold_it_cannot_floor():
    """R1-N02: a name that another scheme declares only warns, and ``t`` is the one setting found that the profile
    does not put a floor under. ``m=4`` (m-of-n, the usual words for it) on a threshold scheme of five servers leaves
    the default ``t = 2``: two servers decrypt where the application asked for four, under ``standard`` too."""
    try:
        created = pyfe4ai.create("mcfe/ddh_threshold", n=2, eta=2, lst_nid=["nid_0", "nid_1"], lst_sid=SERVERS, s=5, m=4,
                                 security_profile="standard")
    except (FEConfigError, FEValidationError):
        return                                          # refused: fixed
    assert created.key_generator.t == 4, "m=4 was ignored with a warning: t = {}".format(created.key_generator.t)


# ---------------------------------------------------------------------------
# Info


class _ReduceFails:
    def __init__(self, error):
        self.error = error

    def __reduce_ex__(self, protocol):
        raise self.error


@pytest.mark.skipif(not bls.available(), reason="BLS12-381 backend not built")
@pytest.mark.parametrize("extra", [ctypes.pointer(ctypes.c_int(1)), _ReduceFails(pickle.PicklingError("no")),
                                   _ReduceFails(copy.Error("no"))], ids=["ctypes", "pickling-error", "copy-error"])
def test_r1_n03_parameters_that_cannot_be_copied_stay_a_validation_error(extra):
    """R1-N03: the copy of R1-F03 now comes before the checks and turns ``TypeError`` and ``RecursionError`` into
    ``FEValidationError``, but ``deepcopy`` raises others. On 5316b73 these parameters drew ``FEValidationError``."""
    from pyfe4ai.schemes.sife.cp_ab_pairing import SIFECPABPairing, SIFECPABPairingKeyGenerator

    kg = SIFECPABPairingKeyGenerator({"universe": ["l:a"]})
    kg.setup()
    with pytest.raises(FEValidationError):
        SIFECPABPairing({"keys": {"pp": dict(kg.get_public_parameters(), extra=extra)}})


def test_r1_n04_the_examples_pass_no_key_that_draws_the_warning():
    """R1-N04: three examples still pass ``bound_u`` and ``label_modulus``, so what the project hands to a new user
    now prints two warnings about keys of versions <= 0.2.0."""
    root = Path(__file__).resolve().parents[2] / "examples"
    keys = configs.LEGACY_OPTIONS - set(configs.MIFEFHMultiIPEPairingConfig.OPTIONS)    # sec_level is live there
    stale = {path.name: sorted(key for key in keys if '"{}"'.format(key) in path.read_text("utf-8"))
             for path in root.glob("*.py")}
    assert not {name: keys for name, keys in stale.items() if keys}
