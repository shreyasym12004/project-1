"""End-to-end: image path -> JSON-able dict."""
from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import numpy as np

from .digits import apply_shear, estimate_shear, find_digits
from .preprocess import binarize_strokes, enhance_gray, load_image
from .recognize import recognise
from .roi import find_display, locate_by_strokes

DISPLAY_H = 220   # rectified display is normalised to this height


def _read_display(roi_bgr: np.ndarray):
    s = DISPLAY_H / roi_bgr.shape[0]
    roi = cv2.resize(roi_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    gray = enhance_gray(roi)
    binary = binarize_strokes(gray)
    boxes, clean = find_digits(binary)
    if len(boxes) >= 2:                               # de-slant italic fonts, then redo
        top = min(b.y for b in boxes); bot = max(b.y + b.h for b in boxes)
        shear = estimate_shear(clean[top:bot])
        if abs(shear) > 0.04:
            gray = apply_shear(gray, shear)
            binary = binarize_strokes(gray)
            boxes, clean = find_digits(binary)
    return clean, boxes, roi


def _score(text: str, confs) -> float:
    if not confs:
        return 0.0
    ok = 0.0 if "?" in text else 1.0
    return (0.3 + 0.7 * ok) * float(np.mean(confs)) * min(1.0, len(confs) / 4)


def read_image(img: np.ndarray, debug: Optional[dict] = None) -> dict:
    gray_full = enhance_gray(img)
    attempts = []
    for locate in (find_display, locate_by_strokes):    # 1) bezel/contour  2) stroke-based
        roi = locate(img, gray_full)
        clean, boxes, disp = _read_display(roi.image)
        text, confs = recognise(clean, boxes)
        attempts.append((_score(text, confs) * (0.6 + 0.4 * roi.score), len(boxes), text, confs, roi, clean, boxes, disp))
        if len(boxes) >= 3 and roi.score >= 0.5:       # a plausible display with digits: trust it
            break
    sc, _, text, confs, roi, clean, boxes, disp = attempts[-1] if len(attempts[-1][6]) >= 3 and len(attempts) == 1 \
        else max(attempts, key=lambda a: a[0])
    if debug is not None:
        debug.update(roi=roi, display=disp, binary=clean, boxes=boxes, text=text)
    value = None
    if text and "?" not in text:
        try:
            value = float(text) if "." in text else int(text)
        except ValueError:
            value = None
    conf = float(np.mean(confs)) * min(1.0, 0.6 + 0.4 * len(boxes) / 4) if confs else 0.0
    return {"detected_reading": value, "confidence_score": round(conf, 3), "raw_text": text}


def read_meter(path: str, debug: Optional[dict] = None) -> dict:
    t0 = time.perf_counter()
    img = load_image(path)
    res = read_image(img, debug)
    return {
        "filename": os.path.basename(path),
        "detected_reading": res["detected_reading"],
        "confidence_score": res["confidence_score"],
        "processing_time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
