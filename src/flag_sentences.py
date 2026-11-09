"""Flag labeled "sentences" that are speech headers (title, byline, venue) rather than text.

Some speech rows in the labeled data are the web-page header scraped in front of the
speech, e.g. "May 13, 2021 The Economic Outlook ... Governor Christopher J. Waller At
The Global Interdependence Center's ... (via webcast) Share ...". Their label describes
metadata, not tone, so they should be excluded from training and evaluation.

Writes data/processed/sentence_flags.csv (one row per flagged sentence, keyed by
`row_id`, the row position in labeled_sentences.csv) for manual review. It does not
modify labeled_sentences.csv.

Run from the repo root:
    python src/flag_sentences.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
STARTS_WITH_DATE = rf"^\s*(?:{MONTHS})\s+\d{{1,2}},\s+\d{{4}}"
SPEAKER = r"\b(?:Chair(?:man|woman)?|Vice Chair|Governor|President)\s+[A-Z][a-z]+"
WEB_MARKER = r"\b(?:Share|via webcast|Watch Live|Return to top)\b"


def flag(sentences):
    starts_with_date = sentences.str.contains(STARTS_WITH_DATE, regex=True)
    has_speaker = sentences.str.contains(SPEAKER, regex=True)
    has_marker = sentences.str.contains(WEB_MARKER, regex=True)

    reason = pd.Series("", index=sentences.index)
    # A leading date plus a speaker or web marker is the page header; a sentence that
    # merely starts with a date (e.g. "March 15, 2020, was ...") should not match
    reason[starts_with_date & (has_speaker | has_marker)] = "date + speaker/web marker"
    reason[(reason == "") & has_speaker & has_marker] = "speaker + web marker"
    return reason


def main():
    df = pd.read_csv(PROCESSED_DIR / "labeled_sentences.csv")
    reason = flag(df["sentence"])
    flagged = df[reason != ""].assign(reason=reason[reason != ""])
    flagged.index.name = "row_id"
    out = PROCESSED_DIR / "sentence_flags.csv"
    flagged[
        ["index", "orig_index", "split", "label_name", "doc_id", "doc_type", "reason", "sentence"]
    ].to_csv(out, encoding="utf-8")
    print(f"flagged {len(flagged)} of {len(df)} sentences -> {out.relative_to(ROOT)}")
    print(flagged["reason"].value_counts().to_string())
    print(flagged["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
