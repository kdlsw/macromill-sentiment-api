from __future__ import annotations

import html
import re
from dataclasses import asdict

from macromill_sentiment.config import PreprocessConfig


_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: str, cfg: PreprocessConfig) -> str:
    s = text
    if cfg.strip_html:
        s = html.unescape(s)
        s = _TAG_RE.sub(" ", s)
    if cfg.lowercase:
        s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def describe_preprocess(cfg: PreprocessConfig) -> dict:
    return asdict(cfg)
