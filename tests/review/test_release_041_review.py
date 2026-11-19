"""Review of release/0.4.1 (782a30c), R041-1: the site's description still names 0.4.0 as the version.

site/index.html line 12 (og:description) ends "Version 0.4.0." while the pill, the citations and
pyfe4ai.__version__ say 0.4.1."""

import pathlib
import re

import pyfe4ai

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_r041_1_every_version_the_site_names_is_the_package_version():
    html = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    named = re.findall(r"Version (\d+\.\d+\.\d+)", html)
    assert named, "the site names no version"
    assert set(named) == {pyfe4ai.__version__}, named
