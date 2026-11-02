"""R2, the crypto review of 0.5 M2 (e7324f2..ed82de4): one failing test per finding.

Every test here fails on ed82de4 for the reason its finding states. The names carry the ids of the report: R2-F01 and
R2-F02 are Low, R2-F03 to R2-F05 Info. The measurements behind them are the probes ``p20a`` to ``p20h`` of the review
records, not here.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(path: str) -> str:
    return " ".join((ROOT / path).read_text("utf-8").split())


def test_r2_f01_a_relabelled_replay_is_not_said_to_need_every_client():
    """R2-F01: the label of a ciphertext is a field of the record, outside everything that is encrypted. Whoever holds
    the ciphertexts of a round, an aggregator with no key at all among them, can write another label into each and
    have them aggregated, silently, as that round (p20g). The documents make it an act of "all clients"."""
    for path in ("pyfe4ai/ml/__init__.py", "docs/api/ml.rst", "pyfe4ai/schemes/mcfe/kp_ab_pairing.py"):
        text = _text(path)
        assert "replay by all clients" not in text, path
        assert "if all clients copy the label" not in text, path


def test_r2_f01_the_claims_are_not_said_to_need_the_master_key():
    """R2-F01: the encryption key of a client checks the label claim of its own ciphertexts (t_i over the points of the
    claimed tag to the powers s_i, u_i is a small power of g1 only under the right tag, p20h); the ML adapter holds
    every client's key. "Only the master key could check them" is not so."""
    for path in ("docs/api/ml.rst", "docs/security.rst"):
        assert "only the master key could check" not in _text(path), path


def test_r2_f02_the_labels_capability_does_not_say_keys_of_these_schemes_are_bound_to_a_label():
    """R2-F02: a functional key of mcfe/kp_ab_pairing opens every tag (Fig. 3 has no tag in Extract): the key the
    adapter issued for round 2 opens round 1, and with the weights of both rounds isolates one client's gradient of
    round 1 (p20g, 4), while the adapter refuses to aggregate round 1 with those weights. The capability that the
    registry gives every MCFE scheme says the opposite."""
    from pyfe4ai.core.registry import CAPABILITIES, LABELS, SCHEMES

    label_free = [s.id for s in SCHEMES if s.id.startswith("mcfe/kp_ab_pairing") and LABELS in s.capabilities]
    assert not (label_free and "functional keys are bound to a label" in CAPABILITIES[LABELS]), label_free


def test_r2_f02_the_adapter_says_a_key_of_another_round_opens_this_one():
    """R2-F02: "each label is aggregated with one weight vector only. These checks stop the obvious ways of isolating
    one client" needs, for mcfe/kp_ab_pairing, that a key issued for any round also opens this one; the final review
    of M2 left that sentence to the fix wave (its "declined to judge" 1) and it was not written."""
    text = _text("pyfe4ai/ml/__init__.py") + " " + _text("docs/api/ml.rst")
    assert re.search(r"(key|keys) [^.]{0,80}(every|any|other) (label|round)", text), \
        "no sentence says that a key of one round opens the other rounds"


def test_r2_f03_the_status_table_does_not_call_the_selective_version_proved():
    """R2-F03: the paper proves the boxed (adaptive) version of Fig. 1 and only says the selective version "can be
    proved using similar techniques". docs/security.rst says the scheme has no proof of its own; the note of the
    registry, which the README and the status table print, says "selective" alone, as cp's note does for a proved
    SEL-SIM."""
    from pyfe4ai.core.registry import SCHEMES

    note = next(s for s in SCHEMES if s.id == "sife/kp_ab_pairing").status.note
    assert "proof" in note or "proved" in note, note


def test_r2_f04_the_profile_table_does_not_list_a_refused_scheme_as_accepted():
    """R2-F04: the table of the standard profile in docs/configuration.rst lists mcfe/kp_ab_pairing_per_client in the
    pairing row, "BLS12-381 only, with a FESecurityWarning", although standard refuses it (the list below the table
    says so)."""
    text = (ROOT / "docs/configuration.rst").read_text("utf-8")
    row = text.split("   * - pairing", 1)[1].split("   * -", 2)
    assert "kp_ab_pairing_per_client" not in row[0] + row[1].split("\n", 1)[0], "listed in the pairing row"


def test_r2_f05_one_scalar_per_client_is_not_put_on_remark_10():
    """R2-F05: Remark 10 is about repetitions (no two ciphertexts under one (i, tag, ac-ct_i)) and about every honest
    client having a challenge ciphertext; that each client encrypts one scalar is Fig. 3 (x_i in Z_q)."""
    for path in ("pyfe4ai/schemes/mcfe/kp_ab_pairing.py",):
        text = _text(path)
        assert "scalar (``eta = 1``), as Remark 10 has it" not in text, path
        assert "one scalar under a tag (NPP22 Remark 10)" not in text, path
