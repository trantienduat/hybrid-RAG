#!/usr/bin/env python3
"""Validate the English thesis against the approved baseline format contract."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
THESIS_PATH = ROOT / "docs/thesis_report_en.tex"


def _ordered(text: str, markers: list[str]) -> bool:
    cursor = 0
    for marker in markers:
        position = text.find(marker, cursor)
        if position < 0:
            return False
        cursor = position + len(marker)
    return True


def main() -> None:
    text = THESIS_PATH.read_text(encoding="utf-8")
    errors: list[str] = []

    required_snippets = {
        "Letter paper": r"\documentclass[12pt,letterpaper,oneside]{report}",
        "Times New Roman body font": r"\setmainfont{Times New Roman}",
        "Arial heading font": r"\setsansfont{Arial}",
        "1.25 inch left margin": "left=1.25in",
        "0.75 inch right margin": "right=0.75in",
        "20.7 point body leading": r"\setstretch{1.43}",
        "Table of Contents title": r"\renewcommand{\contentsname}{Table of Contents}",
        "self-contained abbreviation generation": r"\makenoidxglossaries",
        "hidden front-matter numbering": r"\pagenumbering{gobble}",
        "main-matter page reset": r"\pagenumbering{arabic}",
        "top-right page style": r"\fancyhead[R]{\thepage}",
        "continuous figure numbering": r"\counterwithout{figure}{chapter}",
        "continuous table numbering": r"\counterwithout{table}{chapter}",
        "caption period separator": "labelsep=period",
        "References title": r"\renewcommand{\bibname}{References}",
        "decimal reference labels": r"\renewcommand{\@biblabel}[1]{#1.}",
        "black hyperlinks": "allcolors=black",
        "Acknowledgments": "Acknowledgments",
        "Copyright Acknowledgements": "Copyright Acknowledgements",
    }
    for description, snippet in required_snippets.items():
        if snippet not in text:
            errors.append(f"missing {description}: {snippet}")

    if text.count(r"\begin{titlepage}") != 2:
        errors.append("front matter must contain exactly two title pages")

    front_matter_order = [
        r"\begin{titlepage}",
        "% SUPERVISOR COVER",
        "% TITLE AND ABSTRACT",
        "% ACKNOWLEDGMENTS",
        r"\tableofcontents",
        r"\thesislistoftables",
        r"\thesislistoffigures",
        r"\printnoidxglossary",
        r"\pagenumbering{arabic}",
        r"\chapter{Introduction}",
    ]
    if not _ordered(text, front_matter_order):
        errors.append("front-matter blocks are missing or in the wrong order")

    if not re.search(
        r"\\titleformat\{\\chapter\}\[hang\].*?"
        r"\\fontsize\{18\}\{21\.6\}\\selectfont",
        text,
        re.DOTALL,
    ):
        errors.append("numbered chapter headings must be single-line Arial 18 pt")

    if not re.search(
        r"\\titleformat\{\\section\}\[hang\].*?"
        r"\\fontsize\{16\}\{19\.2\}\\selectfont",
        text,
        re.DOTALL,
    ):
        errors.append("section headings must use Arial 16 pt")

    post_defense_evidence = [
        "RRF was selected instead of adding graph and vector scores directly",
        "no valid schema-v3 Hit Rate@5 result is used as confirmatory evidence",
        "The gold dataset is a manually maintained, checked-in artifact",
        "sub-second retrieval does not imply interactive end-to-end response time",
        "Literature Refresh",
    ]
    for evidence in post_defense_evidence:
        if evidence not in text:
            errors.append(f"post-defense content was lost: {evidence}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("PASS: English thesis matches the approved baseline format contract")


if __name__ == "__main__":
    main()
