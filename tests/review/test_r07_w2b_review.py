"""R-0.7 round 1: failing tests of the review of the identity directory (feat/0.7 5b51558..97dfd02).

Each test fails on 97dfd02 for the reason its finding states; the ids are those of the review.
"""

import threading
from pathlib import Path

import pytest

from pyfe4ai.ac import identity_directory as idd
from pyfe4ai.ac.identity_directory import IdentityDirectory, IdentityKeyCentre, check_identity
from pyfe4ai.core.exceptions import FESchemeError, FEValidationError
from pyfe4ai.schemes.sife.fullysec_ring_lwe import SIFEFullySecRingLWEKeyGenerator

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {"eta": 3, "bound_x": 4, "bound_y": 4}


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path))


@pytest.mark.parametrize("code_point", ["͸", "࢐", "࢑", "\U000E0080"])
def test_r07_m1_an_identity_accepted_on_one_python_is_accepted_on_every_python(code_point):
    """R07-M1 (Medium), the adding side: ``check_identity`` accepts unassigned code points (category Cn). Python 3.10
    (Unicode 13.0) has U+0890 and U+0891 unassigned, Python 3.13 (Unicode 15.1) has them as format characters (Cf),
    which the rule refuses. A key centre on 3.10 adds ``"x\\u0890"``; an encryptor on 3.13 then refuses the whole
    directory, since ``IdentityDirectory._checked`` checks every entry (``probe_cross_version.py``). A key centre should
    not add a code point its own Unicode database does not know. This alone does not make the rule the same on every
    Python: a character assigned in 15.1 is Cn on 3.10, so the reading side needs the second test as well."""
    with pytest.raises(FEValidationError):
        check_identity("x" + code_point)


class _Unicode151:
    """``unicodedata`` as Python 3.13 (Unicode 15.1) answers for U+0890, under any Python."""

    unidata_version = "15.1.0"

    def __getattr__(self, name):
        import unicodedata
        return getattr(unicodedata, name)

    @staticmethod
    def category(c):
        import unicodedata
        return "Cf" if c == "࢐" else unicodedata.category(c)


def test_r07_m1_a_directory_is_read_alike_whatever_the_unicode_version_of_the_reader(monkeypatch):
    """R07-M1 (Medium), the reading side: whether a published directory loads must not depend on the Unicode database
    of the reader's Python. The entries are bound by their digests (and, with ``expected_digest``, by the trusted
    digest), and ``encrypt`` looks an identity up by exact string, so the reader needs only a ``str`` of 1 to 256
    characters; the Unicode rule belongs to ``add_identity``. Here a directory with ``"x\\u0890"`` is made under
    Unicode 13.0 semantics (3.10) and read under 15.1 semantics (emulated)."""
    if idd.unicodedata.category("࢐") != "Cn":
        pytest.skip("this Python already knows U+0890; the test emulates the newer version on an older one")
    kc = IdentityKeyCentre(CONFIG)
    # once the adding side refuses unassigned code points (the first test), no key centre on this Python adds
    # "x\u0890"; one whose Unicode database classed it as an allowed character would (a newer one, for a newly
    # assigned letter), which the rule replaced here stands for
    with monkeypatch.context() as m:
        m.setattr(idd, "check_identity", lambda identity: identity)
        for who in ("alice", "x࢐"):
            kc.add_identity(who)
    record = kc.directory()
    monkeypatch.setattr(idd, "unicodedata", _Unicode151())
    IdentityDirectory(record, expected_digest=record["digest"])


def test_r07_l1_two_threads_cannot_add_one_identity_twice(monkeypatch):
    """R07-L1 (Low): ``add_identity`` checks ``identity in self._instances`` and inserts after ``setup()``. Two threads
    that add one identity at once both pass the check; the second overwrites the first, and with interleaved writes
    ``_instances`` and ``_entries`` come from different instances: keys are issued from one instance under the entry
    (``pk``) of the other, so nothing they decrypt is right. The second call must be refused (``FESchemeError``)."""
    kc = IdentityKeyCentre(CONFIG)
    barrier = threading.Barrier(2, timeout=3)
    real = SIFEFullySecRingLWEKeyGenerator.setup

    def setup(self):
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        real(self)

    monkeypatch.setattr(SIFEFullySecRingLWEKeyGenerator, "setup", setup)
    outcomes = []

    def add():
        try:
            kc.add_identity("alice")
            outcomes.append("added")
        except FESchemeError:
            outcomes.append("refused")

    pool = [threading.Thread(target=add) for _ in range(2)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    assert sorted(outcomes) == ["added", "refused"], outcomes


def _identity_section() -> str:
    text = (ROOT / "docs" / "security.rst").read_text("utf-8")
    return " ".join(text.split("Identity directory (experimental)", 1)[1].split())


def test_r07_l2_the_documentation_says_the_identity_fields_are_not_authenticated():
    """R07-L2 (Low): the ``identity`` and ``entry`` fields of a ciphertext or key are claims. Rewriting both fields of
    alice's ciphertext to bob's passes every check, and bob's key then decrypts it to a wrong value without an error
    (23 for a true -3 in ``probe_w2b.py``; the toy parameters make a random value look valid). The documentation says
    decryption refuses unless the fields agree, which a reader takes as a binding against tampering."""
    section = _identity_section()
    assert "not authenticated" in section or "claims" in section


def test_r07_l3_the_documentation_names_the_status_of_the_packed_layout():
    """R07-L3 (Low): the directory says the scheme underneath is reviewed and unchanged, but a key centre configured
    with ``ndarray_layout="packed"`` runs the packed layout of ``sife/fullysec_ring_lwe``, which is experimental."""
    section = _identity_section()
    module = " ".join((idd.__doc__ or "").split())
    assert "packed" in section and "packed" in module


def test_r07_relabelling_reaches_the_scheme(tmp_path):
    """Evidence for R07-L2 (passes today; it shows the behaviour, not a fix): a relabelled ciphertext is decrypted."""
    kc = IdentityKeyCentre(CONFIG)
    for who in ("alice", "bob"):
        kc.add_identity(who)
    d = IdentityDirectory(kc.directory())
    ct = d.encrypt("alice", [1, -2, 3])
    relabelled = dict(ct, identity="bob", entry=d.entry_digest("bob"))
    dk = kc.get_decryption_keys("bob", [2, 1, -1])
    d.decrypt(relabelled, dk, [2, 1, -1])       # no FEKeyError: the fields agree with the directory
