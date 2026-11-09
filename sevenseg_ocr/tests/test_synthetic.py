from pathlib import Path

import numpy as np
import pytest

from sevenseg.pipeline import read_image, read_meter
from sevenseg.recognize import PATTERNS, TABLE, SEGS
from sevenseg.synth import render


def test_pattern_table_unique_and_complete():
    assert {c for c in TABLE.values()} == set("0123456789")
    for ch, segs in PATTERNS.items():
        assert TABLE[tuple(int(s in segs) for s in SEGS)] == ch


@pytest.mark.parametrize("text", ["57901", "92312.5", "245.9", "04050", "6308.0", "1.2345"])
@pytest.mark.parametrize("kw", [{}, {"italic": 0.2}, {"tilt": 0.6}])
def test_reads_clean_synthetic(text, kw):
    rng = np.random.default_rng(0)
    assert read_image(render(text, rng, **kw))["raw_text"] == text


def test_reads_meter_02_sample():
    sample = Path(__file__).resolve().parents[1] / "samples" / "meter_02.png"
    assert read_meter(str(sample))["detected_reading"] == 4859.12
