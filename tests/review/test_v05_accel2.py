"""Review of the second acceleration round (c3b7d87~1..28ab306): one failing test per finding.

Every test here fails on 28ab306 for the reason its finding states. The names carry the ids of the report: A2-01 and
A2-02 are Low, A2-03 and A2-04 are Info. The rest of the review found nothing and left no test here.
"""

import warnings
from pathlib import Path

import pytest

from pyfe4ai.core import config as configs
from pyfe4ai.core.exceptions import FEConfigError, FEWarning

ROOT = Path(__file__).resolve().parents[2]


def test_a2_01_standard_refuses_sec_level_on_a_scheme_that_ignores_it():
    """A2-01: ``sec_level`` is the ``k`` of k-Lin on ``mife/fh_multi_ipe_pairing``. In ``LEGACY_OPTIONS``, which applies
    to every scheme, it passes ``standard`` with a warning on the 42 schemes that ignore it, as ``m=4`` did before
    R1-N02. No config of versions <= 0.2.0 holds ``standard``: ``security_profile`` came in 0.3.0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FEWarning)
        with pytest.raises(FEConfigError, match="sec_level"):
            configs.check_options({"sec_level": 3, "security_profile": "standard"}, ("mcfe", "fh_multi_ipe_pairing"))


def test_a2_02_the_documents_say_what_the_time_of_the_online_phase_shows():
    """A2-02: one measurement of the online phase reads the share of negative coordinates to about 2%, as ``x % order``
    makes a negative plaintext a 255-bit exponent and no ``s`` adds noise. The time itself is measured by
    ``p16f_online_timing.py``, not here. The plan lists ``docs/security.rst`` for this round and 28ab306 does not touch
    it; the module says only that the table reads depend on ``x``."""
    from pyfe4ai.schemes.sife import cp_ab_pairing

    assert "precompute" in (ROOT / "docs/security.rst").read_text("utf-8")
    online = cp_ab_pairing.__doc__.split("Offline / online:")[1].split("\n\n")[0]
    assert "negative" in online or "sign" in online


def test_a2_03_the_docstring_of_encrypt_lists_every_refusal():
    """A2-03: ``encrypt`` also refuses a precomputation made in another process."""
    from pyfe4ai.schemes.sife.cp_ab_pairing import SIFECPABPairing

    assert "process" in SIFECPABPairing.encrypt.__doc__


def test_a2_04_the_head_of_msm_rs_says_how_one_power_in_g2_is_taken():
    """A2-04: the head of ``msm.rs`` still calls one exponentiation in G1 / G2 the one-term case of the GLV method, and
    the comment in ``g2_pow`` gives a carry past bit 63 as the reason for two limbs. The wNAF can end at bit 64, but
    U + 15 < 2^64 keeps its working value in one limb."""
    source = (ROOT / "rust/native/src/msm.rs").read_text("utf-8")
    assert "One\n//! exponentiation is the one-term case" not in source
    assert "can carry past bit 63" not in source
