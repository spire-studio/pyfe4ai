"""Confirmation of the fixes of R1-N01 to R1-N04 (b6ae92b to 9478f20): one failing test per new finding.

Every test here fails on 9478f20 for the reason its finding states. Both findings come from the rule that ``standard``
refuses an option of another scheme. The names carry the ids of the report: R1-N05 is Low, R1-N06 is Info.
"""

import importlib.util
from pathlib import Path

from pyfe4ai.core import config as configs
from pyfe4ai.core.exceptions import FEConfigError

ROOT = Path(__file__).resolve().parents[2]


def _fl_compare():
    spec = importlib.util.spec_from_file_location("fl_compare_under_review", ROOT / "benchmarks/commands/fl_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r1_n05_fl_compare_runs_every_family_under_its_default_profile():
    """R1-N05: ``fl-compare`` defaults to ``standard`` and gives every family but one ``bound_x`` and ``bound_y``. The
    DDH schemes have neither, which used to be ignored and is now ``FEConfigError``, so the command stops before it
    measures. The suite does not run the benchmarks; on 5316b73 both families ran."""
    fl_compare = _fl_compare()
    refused = {}
    for family, (scheme, base, _) in fl_compare.FAMILIES.items():
        config = dict(base, n=3, lst_nid=["c0", "c1", "c2"], security_profile="standard",
                      **fl_compare._bounds(family, 100))
        if family.endswith("_threshold"):
            config.update(s=3, t=2, lst_sid=["s0", "s1", "s2"])
        try:
            configs.check_options(config, scheme)
        except FEConfigError as exc:
            refused[family] = str(exc).split(";")[0]
    assert not refused


def test_r1_n06_a_key_the_docstring_calls_ignored_is_not_refused():
    """R1-N06: the docstring of ``mcfe/fh_multi_ipe_pairing`` lists ``sec_level`` with ``u_bound`` and ``label_modulus``
    as keys of versions <= 0.2.0, ignored with a warning. ``LEGACY_OPTIONS`` holds only the other two, so under
    ``standard`` ``sec_level`` counts as an option of another scheme (``mife/fh_multi_ipe_pairing``) and is refused.
    Either side may change: the test holds once the two agree."""
    from pyfe4ai.schemes.mcfe.fh_multi_ipe_pairing import MCFEFHMultiIPEPairingKeyGenerator

    if "sec_level" in MCFEFHMultiIPEPairingKeyGenerator.__init__.__doc__:
        configs.check_options({"sec_level": 2, "security_profile": "standard"}, ("mcfe", "fh_multi_ipe_pairing"))
