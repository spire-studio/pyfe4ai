"""Hand-written status words in the README and the security page agree with the registry.

The status table of the README and the counts of the documentation are generated, but some prose is written by
hand: the markers of the README's "Implemented Schemes" list (``*(unreviewed)*``, ``*(pairing, known-insecure)*``)
and the sentences of ``docs/security.rst`` that say which schemes are reviewed or known-insecure. When
``mcfe/kp_ab_pairing`` became known-insecure (5b51558) three such sentences and one marker kept the old status
(R-1.0 part A, M-1 and M-2). These tests read the statuses from the registry, so a status change fails here until
the prose follows.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pyfe4ai.core import registry

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
SECURITY = (ROOT / "docs" / "security.rst").read_text(encoding="utf-8")
INDEX = (ROOT / "docs" / "index.rst").read_text(encoding="utf-8")

STATUS = {"{}/{}".format(s.family, s.variant): s.status.status for s in registry.SCHEMES}
AB = sorted("{}/{}".format(s.family, s.variant) for s in registry.SCHEMES if registry.ACCESS_CONTROL in s.capabilities)
MARKED = (registry.UNREVIEWED, registry.KNOWN_INSECURE)  # the statuses the README marks by hand
#: The statuses a row of one scheme states: the two above and *experimental*, which a row of one experimental scheme
#: carries too (the bundled rows of the experimental threshold layers say it in a footnote instead).
ROW_STATUSES = MARKED + (registry.EXPERIMENTAL,)

#: Rows of the README's "Implemented Schemes" list that stand for one scheme each: (family, instantiation) -> id.
#: A row that bundles several schemes ("DDH, Damgård-DDH, LWE") carries no marker and is not listed.
ROWS = {
    ("SIFE", "DDH over ristretto255"): "sife/ddh_ristretto255",
    ("SIFE", "CP-AB-IPFE, attribute-based"): "sife/cp_ab_pairing",
    ("SIFE", "KP-AB-IPFE, attribute-based"): "sife/kp_ab_pairing",
    ("MIFE", "DDH over ristretto255"): "mife/ddh_ristretto255",
    ("MCFE", "DDH over ristretto255"): "mcfe/ddh_ristretto255",
    ("MCFE", "KP-AB, attribute-based"): "mcfe/kp_ab_pairing",
    ("MCFE", "KP-AB with an attribute set per client"): "mcfe/kp_ab_pairing_per_client",
    ("BCFE", "Garbled circuits over ristretto255, one slot per issued function"): "bcfe/gc_ristretto255",
}

#: Schemes named inside a bundled row with a marker of their own, which names the part that has the status
#: ("dMCFE-DDH *(experimental carry mask)*"): (family, item) -> id.
ITEMS = {
    ("Decentralized", "dMCFE-DDH"): "mcfe/ddh_decentralized",
}

NUMBERS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}


def flat(text: str) -> str:
    return " ".join(text.split())


def implemented_rows() -> dict:
    """(family, instantiation without its marker) -> the set of status markers of the row."""
    section = README.split("## Implemented Schemes", 1)[1].split("\n## ", 1)[0]
    rows, family = {}, None
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or set(cells[0]) <= set("-") and cells[0]:
            continue
        found = re.match(r"\*\*(\w[\w-]*)\*\*", cells[0])
        if found:
            family = found.group(1)
        if family is None or cells[1] in ("Instantiations", "") or set(cells[1]) <= set("-"):
            continue
        marker = re.search(r"\*\(([^)]*)\)\*", cells[1])
        name = re.sub(r"\s*\*\([^)]*\)\*", "", cells[1]).strip()
        words = {w.strip() for w in marker.group(1).split(",")} if marker else set()
        rows[(family, name)] = words & set(ROW_STATUSES)
    return rows


def marked_items() -> list:
    """(family, row without its markers, item, words of the item's marker) for every item of the list that carries
    a marker of its own (the items of a row are separated by ", ")."""
    section = README.split("## Implemented Schemes", 1)[1].split("\n## ", 1)[0]
    found, family = [], None
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        head = re.match(r"\*\*(\w[\w-]*)\*\*", cells[0])
        if head:
            family = head.group(1)
        if family is None:
            continue
        row = re.sub(r"\s*\*\([^)]*\)\*", "", cells[1]).strip()
        for item in cells[1].split(", "):
            marker = re.fullmatch(r"(.+?)\s*\*\(([^)]*)\)\*", item.strip())
            if marker:
                words = [w.strip() for w in marker.group(2).split(",")]
                found.append((family, row, marker.group(1), words))
    return found


def implemented_items() -> dict:
    """(family, item of a row) -> the first word of the item's own marker, for the items that carry one."""
    return {(family, item): words[0].split()[0] for family, _, item, words in marked_items()}


def test_every_marked_status_has_its_row():
    marked = {i for i, st in STATUS.items() if st in MARKED}
    assert marked <= set(ROWS.values()), sorted(marked - set(ROWS.values()))


@pytest.mark.parametrize("row", sorted(ROWS), ids=lambda r: ROWS[r])
def test_the_readme_marks_each_row_with_the_status_of_its_scheme(row):
    rows = implemented_rows()
    assert row in rows, "no README row {!r} under {}".format(row[1], row[0])
    want = {STATUS[ROWS[row]]} & set(ROW_STATUSES)
    assert rows[row] == want, "{}: README says {}, the registry {}".format(ROWS[row], rows[row] or "nothing",
                                                                       STATUS[ROWS[row]])


def test_no_other_row_carries_a_status_marker():
    stray = {row: words for row, words in implemented_rows().items() if words and row not in ROWS}
    assert not stray, stray


@pytest.mark.parametrize("item", sorted(ITEMS), ids=lambda r: ITEMS[r])
def test_a_scheme_marked_inside_a_bundled_row_has_that_status(item):
    items = implemented_items()
    assert item in items, "no README item {!r} with a marker under {}".format(item[1], item[0])
    assert items[item] == STATUS[ITEMS[item]], "{}: README says {}, the registry {}".format(
        ITEMS[item], items[item], STATUS[ITEMS[item]])


def test_no_bundled_row_hides_a_status_marker():
    """A status word in the marker of an item of a row that stands for several schemes belongs to an ITEMS entry."""
    for family, row, item, words in marked_items():
        if (family, row) in ROWS:
            continue
        if {w.split()[0] for w in words} & set(ROW_STATUSES):
            assert (family, item) in ITEMS, "{} / {}: marker {} not in ITEMS".format(family, item, words)


def test_every_experimental_scheme_is_marked_in_its_row_or_a_footnote():
    """A row or an item of its own (ROWS, ITEMS), or one of the threshold layers that footnote 1 names."""
    footnote = flat(README.split("> ¹", 1)[1].split("\n\n", 1)[0])
    covered = set(ROWS.values()) | set(ITEMS.values())
    for scheme in sorted(i for i, st in STATUS.items() if st == registry.EXPERIMENTAL and i not in covered):
        assert scheme.endswith("_threshold"), scheme
        assert "(*experimental*)" in footnote
    assert all(STATUS[i] == registry.EXPERIMENTAL for i in ITEMS.values())


def test_the_status_paragraph_names_exactly_the_known_insecure_schemes():
    # from the sentence that names them, so the experimental list before it (which names ids too) is left out
    paragraph = flat(SECURITY).split("The known-insecure ones are", 1)[1].split("Query a scheme at runtime", 1)[0]
    named = set(re.findall(r"``([a-z0-9_]+/[a-z0-9_]+)``", paragraph))
    assert named == {i for i, st in STATUS.items() if st == registry.KNOWN_INSECURE}, sorted(named)


def ids_before(text: str, phrase: str) -> list:
    """For each occurrence of *phrase*, the set of ``family/variant`` ids that the list just before it names."""
    found = []
    for m in re.finditer(r"((?:``[a-z0-9_]+/[a-z0-9_]+``(?:, | and |, and )?)+) " + re.escape(phrase), text):
        found.append(set(re.findall(r"``([a-z0-9_]+/[a-z0-9_]+)``", m.group(1))))
    return found


def test_the_attribute_based_section_states_the_registry_statuses():
    text = flat(SECURITY)
    count = NUMBERS[len(AB)]
    assert "{} schemes release a result only to a key whose policy and attributes fit".format(count) in text
    reviewed = {i for i in AB if STATUS[i] == registry.REVIEWED}
    insecure = {i for i in AB if STATUS[i] == registry.KNOWN_INSECURE}
    said_reviewed = ids_before(text, "were checked against their papers (status *reviewed*")
    said_insecure = [ids for ids in ids_before(text, "are known-insecure") if ids <= set(AB)]
    assert said_reviewed and all(ids == reviewed for ids in said_reviewed), said_reviewed
    assert said_insecure and all(ids == insecure for ids in said_insecure), said_insecure


@pytest.mark.parametrize("name,text", [("README.md", README), ("docs/security.rst", SECURITY),
                                       ("docs/index.rst", INDEX)], ids=["README.md", "security.rst", "index.rst"])
def test_no_sentence_speaks_of_a_single_known_insecure_scheme(name, text):
    n = sum(st == registry.KNOWN_INSECURE for st in STATUS.values())
    if n == 1:
        pytest.skip("one known-insecure scheme: the singular is right")
    stale = [p for p in ("The known-insecure one is", "the last one is known-insecure", "The first three were checked")
             if p in flat(text)]
    assert not stale, "{}: {}".format(name, stale)
