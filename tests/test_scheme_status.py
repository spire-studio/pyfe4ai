"""The scheme status table, its runtime warning and its README rendering."""

import re
import warnings
from pathlib import Path

import pytest

from pyfe4ai import FESecurityWarning, get_scheme_status
from pyfe4ai.core import status as st
from pyfe4ai.ml.wrapper import _SCHEME_REGISTRY

README = Path(__file__).resolve().parents[1] / "README.md"


def test_every_registered_scheme_has_a_status():
    assert set(st.SCHEME_STATUS) == set(_SCHEME_REGISTRY)
    for key, (module, _, _) in _SCHEME_REGISTRY.items():
        entry = st.SCHEME_STATUS[key]
        assert entry.module == module
        assert entry.status in (st.REVIEWED, st.EXPERIMENTAL, st.UNREVIEWED, st.KNOWN_INSECURE)
        assert entry.fidelity in (st.FAITHFUL, st.ADAPTED, st.AD_HOC, st.UNKNOWN)
        assert bool(entry.issue) == (entry.status == st.KNOWN_INSECURE)
        if entry.status == st.REVIEWED:
            assert entry.fidelity in (st.FAITHFUL, st.ADAPTED)
        if entry.status == st.EXPERIMENTAL:
            assert entry.note
        if entry.fidelity == st.ADAPTED:
            assert entry.note


@pytest.fixture
def insecure_mife_lwe(monkeypatch):
    """Pretend mife/lwe is known-insecure (independent of the real table)."""
    import dataclasses

    key = ("mife", "lwe")
    fake = dataclasses.replace(
        st.SCHEME_STATUS[key], status=st.KNOWN_INSECURE, issue="test issue"
    )
    monkeypatch.setitem(st.SCHEME_STATUS, key, fake)


def test_known_insecure_key_generator_warns(insecure_mife_lwe):
    from pyfe4ai.schemes.mife.lwe import MIFELWEKeyGenerator

    with pytest.warns(FESecurityWarning, match="mife/lwe is known to be insecure: test issue"):
        MIFELWEKeyGenerator({"n": 2, "lwe_n": 8, "bound_x": 2, "bound_y": 2})


def test_known_insecure_crypto_object_warns(insecure_mife_lwe):
    from pyfe4ai.schemes.mife.lwe import MIFELWE

    with pytest.warns(FESecurityWarning, match="mife/lwe"):
        MIFELWE({"keys": {"pp": {"A": [["1"]]}}})


def test_reviewed_key_generator_does_not_warn():
    from pyfe4ai.schemes.mcfe.lwe import MCFELWEKeyGenerator

    with warnings.catch_warnings():
        warnings.simplefilter("error", FESecurityWarning)
        MCFELWEKeyGenerator({"n": 2, "lwe_n": 8, "bound_x": 2, "bound_y": 2})


def test_wrapper_exposes_status():
    from pyfe4ai.ml import FESchemeWrapper

    wrapper = FESchemeWrapper(
        "mcfe", "paillier", {"n": 2, "bit_length": 64, "bound_x": 10, "bound_y": 10}
    )
    assert wrapper.status is get_scheme_status("mcfe", "paillier")
    assert wrapper.status.status == st.REVIEWED


def test_readme_status_table_matches():
    """README "Scheme Status" rows must agree with SCHEME_STATUS."""
    text = README.read_text(encoding="utf-8")
    section = text.split("## Scheme Status", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        m = re.match(r"\|\s*`(\w+)/(\w+)`\s*\|\s*([\w-]+)\s*\|\s*([\w-]+)\s*\|", line)
        if m:
            rows[(m.group(1), m.group(2))] = (m.group(3), m.group(4))
    expected = {k: (v.status, v.fidelity) for k, v in st.SCHEME_STATUS.items()}
    assert rows == expected


def test_readme_status_table_is_the_rendered_table():
    """The whole README table (construction and notes too) is
    st.markdown_table(); regenerate it rather than editing it by hand."""
    text = README.read_text(encoding="utf-8")
    section = text.split("## Scheme Status", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("|")]
    assert "\n".join(rows) == st.markdown_table()
