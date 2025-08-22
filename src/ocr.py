"""
Lightweight OCR helpers (pytesseract + Pillow only).
"""

from __future__ import annotations
from typing import Iterable, List
import re

import pytesseract
from PIL import Image

# --- OCR Text Extraction ---
def extract_cv_lines(image_path: str) -> list[str]:
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image) or ""

    lines = []
    for line in text.split("\n"):
        cleaned = line.strip()
        if cleaned:   # skip empty lines
            lines.append(cleaned)

    return lines

# --- Skills Section Detection ---
def extract_skills_section(lines: Iterable[str]) -> list[str]:
    lines = list(lines)
    skill_start = -1
    for i, line in enumerate(lines):
        ll = line.lower()
        if "skill" in ll and len(line.split()) <= 3:
            skill_start = i
            break

    next_section_keywords = [
        "education", "experience", "projects",
        "summary", "certifications", "language"
    ]
    skill_end = len(lines)

    if skill_start != -1:
        for j in range(skill_start + 1, len(lines)):
            l2 = lines[j].lower()
            if any(k in l2 for k in next_section_keywords) or "skilled in" in l2:
                skill_end = j
                break

        section = lines[skill_start + 1:skill_end]
        return section
    else:
        return []



# --- Tokenization (NO space splitting) ---
_BULLET_LEADING = re.compile(r"^[\+\-\*\u2022•·—–]+\s*")
_SPLIT_DELIMS = re.compile(r"[,\|;/]+")

def _split_line_preserve_phrases(line: str) -> list[str]:
    """
    Split on commas/; | / (NOT spaces), then strip leading bullets.
    """
    parts = _SPLIT_DELIMS.split(line) if _SPLIT_DELIMS.search(line) else [line]
    clean = []
    for p in parts:
        t = _BULLET_LEADING.sub("", p).strip()
        if t and len(t) > 1 and t not in {"+", "-", "•", "—", "–", "*"}:
            clean.append(t)
    return clean

def parse_skills(skills_section: Iterable[str]) -> List[str]:
    tokens: List[str] = []
    for line in skills_section:
        tokens.extend(_split_line_preserve_phrases(line))
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def extract_skills_from_cv(image_path: str) -> List[str]:
    lines = extract_cv_lines(image_path)
    if not lines:
        return []
    section = extract_skills_section(lines)
    # if section not found, fall back to all lines
    base = section or lines
    return parse_skills(base)
