"""R4 increment 4 (integrate/1.0 081d4a1): ``benchmarks/run.py setup-costs`` on the 1.0 registry.

N-1: the command walks every registry entry with the configuration of ``fl-coverage``. For
``bcfe/gc_ristretto255`` that configuration is ``{}`` (its FL path is unusable), and the key generator
raises ``FEConfigError`` (it needs a circuit). ``setup-costs`` catches only ``FEValidationError``, so the
whole command aborts before it prints any row, under ``toy`` and under ``standard``. The entry should be
measured with a library circuit, or reported (``refused`` / ``unusable``) with the reason, like the others.
"""

import importlib

import pytest

from tests.api.test_registry import ROOT


@pytest.fixture
def setup_costs(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))
    common = importlib.import_module("_common")
    module = importlib.import_module("commands.setup_costs")
    yield common, module
    common.set_security_profile("toy")


@pytest.mark.parametrize("profile", ["toy", "standard"])
def test_setup_costs_reports_bcfe_instead_of_aborting(setup_costs, profile):
    common, module = setup_costs
    common.set_security_profile(profile)
    row = module.measure(("bcfe", "gc_ristretto255"), 1, profile)
    assert row["scheme"] == "bcfe/gc_ristretto255" and row["status"] in {"ok", "refused", "unusable", "unverified"}


def test_setup_costs_prints_a_row_for_every_entry_given(setup_costs, capsys):
    _, module = setup_costs
    module.main(["--rounds", "1", "--schemes", "sife/ddh", "bcfe/gc_ristretto255"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("scheme,") and [line.split(",")[0] for line in lines[1:]] == [
        "sife/ddh", "bcfe/gc_ristretto255"]
