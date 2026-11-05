"""The scheme status table, its runtime warning and its README rendering."""

import re
import warnings
from collections import Counter
from pathlib import Path

import pytest

import pyfe4ai
from pyfe4ai import FESecurityWarning, FEValidationError, get_scheme_status
from pyfe4ai.core import registry
from pyfe4ai.core import status as st

README = Path(__file__).resolve().parents[2] / "README.md"


def test_every_registered_scheme_has_a_status():
    assert list(st.SCHEME_STATUS) == [scheme.key for scheme in registry.SCHEMES]
    for scheme in registry.SCHEMES:
        entry = st.SCHEME_STATUS[scheme.key]
        assert entry.module == scheme.module
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


def test_the_standard_profile_refuses_a_known_insecure_scheme(insecure_mife_lwe, monkeypatch):
    """``toy`` warns (see above) and goes on; ``standard`` is for security, so it refuses, whatever the scheme, through
    the ``refused`` of its registry entry like the other refusals (``FEValidationError``)."""
    from pyfe4ai.schemes.mife.lwe import MIFELWEKeyGenerator

    real = registry.get_scheme("mife/lwe")
    known_insecure = registry._scheme("mife/lwe", real.kind, st.KNOWN_INSECURE, real.status.fidelity,
                                      real.status.reference, note=real.status.note, issue="test issue",
                                      capabilities=real.capabilities)
    monkeypatch.setitem(registry._BY_KEY, ("mife", "lwe"), known_insecure)
    config = {"n": 2, "bound_x": 2, "bound_y": 2}
    message = "no 'standard' parameter set for mife/lwe: known to be insecure: test issue"
    with pytest.raises(FEValidationError, match=message):
        MIFELWEKeyGenerator(dict(config, security_profile="standard"))
    with pytest.raises(FEValidationError, match=message):
        pyfe4ai.create("mife/lwe", dict(config, security_profile="standard"))
    with pytest.warns(FESecurityWarning, match="known to be insecure: test issue"):
        MIFELWEKeyGenerator(dict(config, lwe_n=8))


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


def test_the_attribute_based_schemes_have_the_status_of_their_reviews():
    """The three schemes of 0.5 were checked against their papers, with the fidelity recorded. The multi-client
    scheme is faithful to Fig. 3 and known-insecure: the construction fails with one corrupted client (its issue,
    2026-11); the per-client scheme stays known-insecure (its construction fails, see its issue)."""
    expected = {
        "sife/cp_ab_pairing": (st.REVIEWED, st.ADAPTED),
        "sife/kp_ab_pairing": (st.REVIEWED, st.FAITHFUL),
        "mcfe/kp_ab_pairing": (st.KNOWN_INSECURE, st.FAITHFUL),
        "mcfe/kp_ab_pairing_per_client": (st.KNOWN_INSECURE, st.ADAPTED),
    }
    assert {i: (get_scheme_status(i).status, get_scheme_status(i).fidelity) for i in expected} == expected


def test_readme_counts_are_the_registry_counts():
    """The sentence of the README that counts the schemes by status."""
    text = README.read_text(encoding="utf-8")
    found = re.search(r"Currently (\d+) reviewed, (\d+) experimental, (\d+) unreviewed and (\d+) known-insecure out\s+"
                      r"of (\d+) registered schemes", text)
    assert found is not None
    counts = Counter(entry.status for entry in st.SCHEME_STATUS.values())
    assert [int(g) for g in found.groups()] == [counts[s] for s in registry.STATUSES] + [len(st.SCHEME_STATUS)]


def test_no_text_calls_a_reviewed_attribute_based_scheme_unreviewed():
    """A sentence of the README, SECURITY.md, the documentation, the project page or the module of one of the three
    attribute-based schemes of 0.5 that names it does not call it unreviewed (the per-client scheme, which several
    of them name beside, is known-insecure)."""
    root = README.parent
    named = ("sife/cp_ab_pairing", "sife/kp_ab_pairing", "mcfe/kp_ab_pairing")
    paths = [README, root / "SECURITY.md", root / "site" / "index.html", *sorted((root / "docs").glob("*.rst")),
             *(root / "pyfe4ai" / "schemes" / (name + ".py") for name in named)]
    for path in paths:
        if not path.exists():       # a source distribution carries neither the page nor the sources of the docs
            continue
        # the matrix of the page is generated from the registry and tested against it (test_site_matrix.py)
        text = re.sub(r"<!--matrix:start-->.*?<!--matrix:end-->", "", path.read_text(encoding="utf-8"), flags=re.S)
        text = re.sub(r"\s+", " ", text)
        for sentence in re.split(r"(?<=[.;])\s|</dd>|</li>|</div>|</p>", text):
            if "unreviewed" in sentence.lower() or "not yet reviewed" in sentence:
                bare = sentence.replace("mcfe/kp_ab_pairing_per_client", "")
                assert not any(name in bare for name in named), (path.name, sentence)
