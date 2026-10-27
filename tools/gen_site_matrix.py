"""Render the scheme matrix of the project page (``site/index.html``) from the registry.

The page shows the registered schemes as families (rows) by instantiations
(columns), one mark per scheme with its review status; a cell lists its
schemes and whether ``standard`` refuses them. The table sits between the
markers ``<!--matrix:start-->`` and ``<!--matrix:end-->`` of the page and is
generated, not edited by hand: ``tests/api/test_site_matrix.py`` fails when
it differs from :func:`render`.

Usage::

    python tools/gen_site_matrix.py            # print the table
    python tools/gen_site_matrix.py --write    # replace it in site/index.html
    python tools/gen_site_matrix.py --check    # exit 1 if the page differs
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "site" / "index.html"
START, END = "<!--matrix:start-->", "<!--matrix:end-->"

#: The rows: (key, label, subtitle).
ROWS = (
    ("sife", "SIFE", "single-input"),
    ("mife", "MIFE", "multi-input"),
    ("mcfe", "MCFE", "multi-client, labelled"),
    ("dmcfe", "dMCFE", "decentralized"),
    ("tmife", "tMIFE", "threshold"),
    ("tmcfe", "tMCFE", "threshold"),
    ("quadratic", "Quadratic", "SGP · Quad · MIQFE"),
)
#: The columns: (key, label); the instantiations, by the registry's kinds.
COLS = (("ddh", "DDH"), ("damgard", "Damgård"), ("paillier", "Paillier"), ("lwe", "LWE"),
        ("ring_lwe", "Ring-LWE"), ("pairing", "Pairing"))
#: The CSS class of a status mark.
MARK = {"reviewed": "rev", "experimental": "exp", "unreviewed": "unr", "known-insecure": "ins"}


def _row(scheme) -> str:
    from pyfe4ai.core import registry

    if scheme.family == "mcfe" and scheme.has(registry.THRESHOLD):
        return "tmcfe"
    if scheme.family == "mife" and scheme.has(registry.THRESHOLD):
        return "tmife"
    if scheme.family == "mcfe" and scheme.has(registry.DECENTRALIZED):
        return "dmcfe"
    return scheme.family


def _col(scheme) -> str:
    if scheme.kind == "dl":
        return "damgard" if scheme.variant.startswith("damgard") else "ddh"
    return {"ec": "ddh", "dcr": "paillier", "lwe": "lwe", "lwe_abdp15": "lwe", "lwe_matrix": "lwe",
            "ring_lwe": "ring_lwe", "pairing": "pairing"}[scheme.kind]


def cells() -> dict:
    """``{(row, column): [scheme, ...]}`` in registry order; every scheme in one cell."""
    from pyfe4ai.core import registry

    out: dict = {}
    for scheme in registry.SCHEMES:
        key = (_row(scheme), _col(scheme))
        if key[0] not in {r[0] for r in ROWS} or key[1] not in {c[0] for c in COLS}:
            raise ValueError("no cell of the site matrix for {}".format(scheme.id))
        out.setdefault(key, []).append(scheme)
    return out


def _cell(rname: str, cname: str, schemes: list) -> str:
    if not schemes:
        return '<td class="none"><span aria-hidden="true">—</span><span class="sr-only">none</span></td>'
    data = ";".join("{}|{}|{}".format(s.id, s.status.status, "refused" if s.refused else "") for s in schemes)
    marks = "".join('<i class="dot {}"></i>'.format(MARK[s.status.status]) for s in schemes)
    spoken = ", ".join("{} {}{}".format(s.id, s.status.status, ", refused under standard" if s.refused else "")
                       for s in schemes)
    n = len(schemes)
    return ('<td><button type="button" class="cell" data-cell="{} × {}" data-schemes="{}">'
            '<span class="dots" aria-hidden="true">{}</span><span class="cnt" aria-hidden="true">{}</span>'
            '<span class="sr-only">{} scheme{}: {}</span></button></td>').format(
        rname, cname, html.escape(data), marks, n, n, "" if n == 1 else "s", html.escape(spoken))


def render() -> str:
    """The matrix table (HTML), from the registry."""
    grid = cells()
    lines = [
        '<table class="matrix">',
        "  <caption class=\"sr-only\">Registered schemes by family (rows) and instantiation (columns), "
        "with each scheme's review status</caption>",
        '  <thead><tr><th scope="col">Family</th>'
        + "".join('<th scope="col">{}</th>'.format(html.escape(label)) for _, label in COLS) + "</tr></thead>",
        "  <tbody>",
    ]
    for rkey, rname, rsub in ROWS:
        tds = "".join(_cell(rname, cname, grid.get((rkey, ckey), [])) for ckey, cname in COLS)
        lines.append('    <tr><th scope="row">{} <small>{}</small></th>{}</tr>'.format(rname, html.escape(rsub), tds))
    lines += ["  </tbody>", "</table>"]
    return "\n".join(lines)


def block() -> str:
    """The marked block of the page."""
    return "{}\n{}\n{}".format(START, render(), END)


def page_block(text: str) -> str:
    """The marked block of a page's text."""
    i, j = text.find(START), text.find(END)
    if i < 0 or j < i:
        raise ValueError("the page has no {} ... {} block".format(START, END))
    return text[i:j + len(END)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="replace the table in site/index.html")
    mode.add_argument("--check", action="store_true", help="exit 1 if site/index.html has another table")
    args = parser.parse_args(argv)
    if not (args.write or args.check):
        print(render())
        return 0
    text = PAGE.read_text(encoding="utf-8")
    current = page_block(text)
    if args.check:
        if current != block():
            print("site/index.html: the scheme matrix differs from the registry; "
                  "run python tools/gen_site_matrix.py --write", file=sys.stderr)
            return 1
        return 0
    if current != block():
        PAGE.write_text(text.replace(current, block()), encoding="utf-8")
        print("site/index.html: scheme matrix updated")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
