"""Parameter records written under ``standard`` hold the string ``"auto"`` where the config left the dimension to the
library (``ring_n_config``, ``lwe_n_config``). The fixtures of 1.0.0 were written under ``toy``, where these fields
hold an int or null, so this test keeps the third encoding (R2: the field and its JSON form) and its reuse (R5: a
key generator with the same configuration reads the record back without regenerating it).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from pyfe4ai.core.exceptions import FEWarning
from pyfe4ai.schemes.sife.fullysec_ring_lwe import SIFEFullySecRingLWEKeyGenerator
from pyfe4ai.schemes.sife.lwe import SIFELWEKeyGenerator

CASES = {
    "ring_n_config": (SIFEFullySecRingLWEKeyGenerator, {"eta": 1, "bound_x": 4, "bound_y": 4}),
    "lwe_n_config": (SIFELWEKeyGenerator, {"eta": 1, "bound_x": 4, "bound_y": 4}),
}


@pytest.mark.parametrize("field", list(CASES))
def test_a_dimension_left_to_the_library_is_written_as_auto_and_reused(field, tmp_path, monkeypatch):
    monkeypatch.setenv("PYFE4AI_CACHE_DIR", str(tmp_path))
    cls, config = CASES[field]
    config = {**config, "security_profile": "standard"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cls(config)
    records = [p for p in Path(tmp_path).rglob("param.json")]
    assert len(records) == 1, records
    text = records[0].read_text(encoding="utf-8")
    assert json.loads(text)[field] == "auto"
    assert '"{}": "auto"'.format(field) in text.replace('":"', '": "')  # the JSON form: a plain string
    with warnings.catch_warnings():
        warnings.simplefilter("error", FEWarning)  # a regenerated record warns
        cls(config)
    assert records[0].read_text(encoding="utf-8") == text
