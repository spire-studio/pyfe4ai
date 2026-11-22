"""Re-check of the fixes of A2-01 to A2-04 (28ab306..a7014a4): one failing test per new finding.

Every test here fails on a7014a4 for the reason its finding states. The names carry the ids of the report: A2-N01 is Low
and A2-N02 is Info. The four fixes themselves hold; the time behind both findings is measured by ``p17c_sign_timing.py``
and ``p17d_online_or_encrypt.py``, not here.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(path: str) -> str:
    return " ".join((ROOT / path).read_text("utf-8").split())


def test_a2_n01_the_documents_do_not_call_one_path_the_weaker():
    """A2-N01: one timed call of the online phase shows the sizes and the zeros better than one of ``encrypt`` (11 to 17
    and 9 to 12 sigma against about 1 and 3 to 4), but the signs worse (0.2 to 0.4 sigma against 2.4 to 4.4). So neither
    path is the weaker one, and a reader who keeps ``encrypt`` to protect the signs gets the opposite."""
    assert "the online phase is the weaker of the two" not in _text("docs/security.rst")


def test_a2_n02_no_document_says_the_time_shows_nothing_of_the_signs():
    """A2-N02: signs in a random order cost 0.012 ms more than signs in one block at eta = 1000, an effect of the branch
    on ``x_i < 0`` that 10000 rounds resolve and one call does not (0.3 sigma). The time hardly shows the signs."""
    for path in ("docs/security.rst", "CHANGELOG.md", "pyfe4ai/schemes/sife/cp_ab_pairing.py"):
        assert "does not show the signs" not in _text(path)


def test_a2_n02_precompute_does_not_say_it_cannot_be_copied():
    """A2-N02: A2-03 took this claim out of the changelog, as ``core/one_time.py`` says that copying the attributes by
    hand is not prevented; the same words stay in the Returns of ``precompute``."""
    from pyfe4ai.schemes.sife.cp_ab_pairing import SIFECPABPairing

    assert "cannot be copied" not in " ".join(SIFECPABPairing.precompute.__doc__.split())
