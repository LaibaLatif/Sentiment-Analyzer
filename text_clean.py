"""Fast text cleaning for the NLP pipeline (compiled regexes)."""
from __future__ import annotations

import re

import pandas as pd

# Compile once — reused on every row / request.
_RE_HTTP = re.compile(r"http\S+")
_RE_WWW = re.compile(r"www\S+")
_RE_NON_WORD = re.compile(r"[^\w\s]")
_RE_DIGITS = re.compile(r"\d+")
_RE_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = _RE_HTTP.sub(" ", text)
    text = _RE_WWW.sub(" ", text)
    text = _RE_NON_WORD.sub(" ", text)
    text = _RE_DIGITS.sub(" ", text)
    text = _RE_SPACE.sub(" ", text).strip()
    return text


def prepare_dataframe(df: pd.DataFrame, text_col: str = "statement") -> pd.DataFrame:
    """Drop empty rows and add a `clean_text` column."""
    out = df.copy()
    out = out.dropna(subset=[text_col])
    out[text_col] = out[text_col].astype(str)
    out = out[out[text_col].str.strip() != ""]
    out["clean_text"] = out[text_col].map(clean_text)
    out = out[out["clean_text"].str.len() > 0]
    return out.reset_index(drop=True)
