"""``benchmarks/run.py setup-costs``: an entry whose configuration the key generator does not accept is reported as
``unusable`` with the reason; the command still prints the other rows, then names every unusable entry on stderr and
exits with status 1."""

import importlib
import json
import os
import subprocess
import sys
import textwrap

import pytest

from tests.api.test_registry import ROOT


@pytest.fixture
def setup_costs(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))
    common = importlib.import_module("_common")
    module = importlib.import_module("commands.setup_costs")
    yield module
    common.set_security_profile("toy")  # main() also disables logging: tests/conftest.py restores it


def test_a_config_error_is_an_unusable_row_and_the_command_exits_non_zero(setup_costs, monkeypatch, capsys):
    real = setup_costs._config
    monkeypatch.setattr(setup_costs, "_config", lambda key: {} if key == setup_costs.BCFE else real(key))
    row = setup_costs.measure(setup_costs.BCFE, 1, "toy")
    assert row["status"] == "unusable" and "needs a circuit" in row["note"]
    with pytest.raises(SystemExit) as exit_:
        setup_costs.main(["--rounds", "1", "--schemes", "sife/ddh", "bcfe/gc_ristretto255"])
    assert exit_.value.code == 1
    out, err = capsys.readouterr()
    lines = out.strip().splitlines()  # every row, in registry order
    assert [line.split(",")[:3] for line in lines[1:]] == [
        ["sife/ddh", "toy", "ok"], ["bcfe/gc_ristretto255", "toy", "unusable"]]
    assert err.strip().splitlines() == ["setup-costs: bcfe/gc_ristretto255: unusable ({})".format(row["note"])]


def test_every_entry_sets_up_under_toy(setup_costs, capsys):
    """unusable makes the command exit non-zero; this test also makes a broken setup fail with its reason. Entries
    that need a pairing backend read unverified where there is none (the Python-only lanes)."""
    from commands.fl_coverage import _needs_pairing
    from pyfe4ai.primitives.pairing.pairing_backend import pairing_available

    try:
        setup_costs.main(["--rounds", "1", "--output", "json"])
        code = 0
    except SystemExit as exc:
        code = exc.code
    out, err = capsys.readouterr()
    pairing = pairing_available()
    wrong = []
    for row in json.loads(out):
        want = "ok" if pairing or not _needs_pairing(*row["scheme"].split("/")) else "unverified"
        if row["status"] != want:
            wrong.append("{}: {} ({})".format(row["scheme"], row["status"], row.get("note", "")))
    assert not wrong, "entries that did not set up under toy:\n" + "\n".join(wrong)
    assert code == 0 and err == "", (code, err)


def test_the_configuration_is_made_outside_the_timed_steps(setup_costs, monkeypatch):
    """The bcfe circuit is built when the configuration is made (once, on a cold cache): before t0. A fake clock
    ticks 1 ms per reading, and making the configuration advances it by 100 s, which must reach no timing."""
    clock = [0.0]

    def tick():
        clock[0] += 0.001
        return clock[0]

    real = setup_costs._config

    def slow_config(key):
        clock[0] += 100.0
        return real(key)

    setup_costs._bcfe_circuit.cache_clear()
    monkeypatch.setattr(setup_costs.time, "perf_counter", tick)
    monkeypatch.setattr(setup_costs, "_config", slow_config)
    row = setup_costs.measure(setup_costs.BCFE, 2, "toy")
    assert row["status"] == "ok"
    assert row["construct_mean_sec"] < 1 and row["setup_mean_sec"] < 1 and row["total_std_sec"] < 1, row


def test_bcfe_is_set_up_with_the_library_circuit(setup_costs):
    row = setup_costs.measure(setup_costs.BCFE, 1, "toy")
    assert row["status"] == "ok" and row["note"] == "circuit {}, no slots yet".format(setup_costs.BCFE_CIRCUIT)
    assert row["params"]["scheme"] == "bcfe/gc_ristretto255"


def test_logging_is_on_again_in_the_next_test_after_main(tmp_path):
    """main() disables logging for the process; tests/conftest.py restores the level after every test. Checked in a
    pytest run of its own (the suite's conftest loaded as a plugin), where the two tests run in this order whatever
    order or worker the suite itself uses."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_order.py").write_text(textwrap.dedent("""
        import logging
        import sys

        sys.path.insert(0, {benchmarks!r})
        from commands import setup_costs


        def test_1_main_disables_logging():
            setup_costs.main(["--rounds", "1", "--schemes", "sife/ddh"])
            assert logging.root.manager.disable == logging.CRITICAL


        def test_2_logging_is_on_again():
            assert logging.root.manager.disable == logging.NOTSET
    """).format(benchmarks=str(ROOT / "benchmarks")))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])))
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "tests.conftest",
         "-c", str(tmp_path / "pytest.ini"), str(tmp_path / "test_order.py")],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0 and "2 passed" in run.stdout, run.stdout + run.stderr
