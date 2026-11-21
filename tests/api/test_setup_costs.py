"""``benchmarks/run.py setup-costs``: an entry whose configuration the key generator does not accept is reported as
``unusable`` with the reason, and the command still prints the other rows."""

import importlib

import pytest

from tests.api.test_registry import ROOT


@pytest.fixture
def setup_costs(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))
    common = importlib.import_module("_common")
    module = importlib.import_module("commands.setup_costs")
    yield module
    common.set_security_profile("toy")


def test_a_config_error_is_an_unusable_row_and_the_command_goes_on(setup_costs, monkeypatch, capsys):
    real = setup_costs._config
    monkeypatch.setattr(setup_costs, "_config", lambda key: {} if key == setup_costs.BCFE else real(key))
    row = setup_costs.measure(setup_costs.BCFE, 1, "toy")
    assert row["status"] == "unusable" and "needs a circuit" in row["note"]
    setup_costs.main(["--rounds", "1", "--schemes", "sife/ddh", "bcfe/gc_ristretto255"])
    lines = capsys.readouterr().out.strip().splitlines()  # in registry order
    assert [line.split(",")[:3] for line in lines[1:]] == [
        ["sife/ddh", "toy", "ok"], ["bcfe/gc_ristretto255", "toy", "unusable"]]


def test_every_entry_sets_up_under_toy(setup_costs):
    """unusable is a reported row, not a failure of the command: this test makes a broken setup fail. Entries that
    need a pairing backend read unverified where there is none (the Python-only lanes)."""
    from commands.fl_coverage import _needs_pairing
    from pyfe4ai.core.registry import SCHEMES
    from pyfe4ai.primitives.pairing.pairing_backend import pairing_available

    importlib.import_module("_common").set_security_profile("toy")
    pairing = pairing_available()
    wrong = []
    for scheme in SCHEMES:
        row = setup_costs.measure(scheme.key, 1, "toy")
        want = "ok" if pairing or not _needs_pairing(*scheme.key) else "unverified"
        if row["status"] != want:
            wrong.append("{}: {} ({})".format(row["scheme"], row["status"], row.get("note", "")))
    assert not wrong, "entries that did not set up under toy:\n" + "\n".join(wrong)


def test_bcfe_is_set_up_with_the_library_circuit(setup_costs):
    row = setup_costs.measure(setup_costs.BCFE, 1, "toy")
    assert row["status"] == "ok" and row["note"] == "circuit {}".format(setup_costs.BCFE_CIRCUIT)
    assert row["params"]["scheme"] == "bcfe/gc_ristretto255"
