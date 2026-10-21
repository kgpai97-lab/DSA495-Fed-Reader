"""Recover the real source document, type and date for each labeled sentence.

The `year` column in gtfintechlab/fomc_communication is misaligned with the
sentences (see results/dataset_summary.md), and there is no document or meeting
field. The authors' repo (gtfintechlab/fomc-hawkish-dovish) contains the
filtered source documents the sentences were sampled from, with the document
date in each file name. This script matches every labeled sentence back to
those documents.

Matching uses normalized text (lowercase, alphanumerics only). A sentence is
matched exactly first; otherwise it is looked up as a substring, because some
labeled rows are fragments of longer sentences split at contrast words.
Boilerplate sentences can occur in many documents; for these we keep every
candidate and assign the earliest one, so a time split never puts a sentence
after a date when it had already appeared.

Requires data/raw/fomc_communication_{train,test}.csv from
src/load_fomc_dataset.py. Clones the authors' repo into data/raw/ if missing.

Run from the repo root:
    python src/recover_metadata.py
"""

import re
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

REPO_URL = "https://github.com/gtfintechlab/fomc-hawkish-dovish"
REPO_COMMIT = "98646987452b326507479cf641571b33814bb73f"
REPO_DIR = RAW_DIR / "fomc-hawkish-dovish"
FILTERED = REPO_DIR / "data" / "filtered_data"
MINUTES_INDEX = REPO_DIR / "data" / "master_files" / "master_mm_final.xlsx"

# The three document types the labeled data was sampled from
SOURCES = {
    "minutes": FILTERED / "meeting_minutes",
    "press_conference": FILTERED / "press_conference",
    "speech": FILTERED / "speech" / "select",
}

LABEL_NAMES = {0: "dovish", 1: "hawkish", 2: "neutral"}


def ensure_repo():
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
    head = subprocess.run(
        ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != REPO_COMMIT:
        print(f"warning: {REPO_DIR.name} is at {head[:7]}, expected {REPO_COMMIT[:7]}")


def norm(text):
    # The source files contain some mis-decoded UTF-8 (e.g. "â€™" for an apostrophe);
    # dropping all non-alphanumerics makes both sides comparable
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def load_documents():
    rows = []
    for doc_type, folder in SOURCES.items():
        for f in sorted(folder.glob("*.csv")):
            # File names vary (19960130_filtered, FOMCpresconf20110427_select_filtered,
            # bernanke20060224a_filtered, 199709122_filtered); the first 8 digits are the date
            date = pd.to_datetime(re.search(r"\d{8}", f.name).group(0), format="%Y%m%d")
            doc_id = f"{doc_type}:{f.stem.removesuffix('_filtered')}"
            for sentence in pd.read_csv(f, encoding="utf-8")["sentence"].dropna():
                rows.append((doc_id, doc_type, date, norm(sentence)))
    return pd.DataFrame(rows, columns=["doc_id", "doc_type", "date", "norm"])


def minutes_release_dates():
    # Minutes come out weeks after the meeting (after the next meeting before Dec 2004).
    # The index Url ends in the same 8-digit date as the minutes file name.
    index = pd.read_excel(MINUTES_INDEX)
    key = index["Url"].str.extract(r"(\d{8})\.htm$", expand=False)
    return pd.Series(pd.to_datetime(index["ReleaseDate"]).values, index="minutes:" + key)


def load_labeled():
    df = pd.concat(
        [
            pd.read_csv(RAW_DIR / f"fomc_communication_{s}.csv").assign(split=s)
            for s in ("train", "test")
        ],
        ignore_index=True,
    )
    df["norm"] = df["sentence"].map(norm)
    return df


def match(labeled, docs):
    meta = docs.drop_duplicates("doc_id").set_index("doc_id")[["doc_type", "date"]]
    exact = docs.groupby("norm")["doc_id"].apply(set).to_dict()
    text_by_doc = docs.groupby("doc_id")["norm"].apply(" | ".join).to_dict()

    candidates, methods = [], []
    for n in labeled["norm"]:
        if n in exact:
            found, how = exact[n], "exact"
        else:
            found = {d for d, text in text_by_doc.items() if len(n) > 20 and n in text}
            how = "substring" if found else "none"
        candidates.append(sorted(found, key=lambda d: (meta.loc[d, "date"], d)))
        methods.append(how)

    out = labeled.drop(columns="norm").rename(columns={"year": "year_given"})
    out["label_name"] = out["label"].map(LABEL_NAMES)
    out["match"] = methods
    out["n_candidate_docs"] = [len(c) for c in candidates]
    out["candidate_docs"] = [";".join(c) for c in candidates]
    out["doc_id"] = [c[0] if c else None for c in candidates]
    out["doc_type"] = out["doc_id"].map(meta["doc_type"])
    out["date"] = out["doc_id"].map(meta["date"])
    out["year"] = out["date"].dt.year.astype("Int64")
    # When the text became public: speeches and press conferences are public on `date`
    out["release_date"] = out["doc_id"].map(minutes_release_dates()).fillna(out["date"])
    return out


def summarize(df):
    matched = df[df["match"] != "none"]
    unique = df[df["n_candidate_docs"] == 1]
    lines = ["# Recovered metadata for the labeled sentences", ""]
    lines.append(
        f"Generated by `src/recover_metadata.py`, matching against "
        f"[{REPO_URL.removeprefix('https://github.com/')}]({REPO_URL}) "
        f"at commit `{REPO_COMMIT[:7]}`."
    )
    lines.append("")
    lines.append("## Match rate")
    lines.append(df["match"].value_counts().rename("sentences").to_frame().to_markdown())
    lines.append("")
    lines.append(
        f"- Matched to exactly one document: {len(unique)}; to several "
        f"(boilerplate, assigned to the earliest): {(df['n_candidate_docs'] > 1).sum()}"
    )
    lines.append(f"- Distinct source documents: {matched['doc_id'].nunique()}")
    lines.append("")
    lines.append("## Given `year` vs recovered year (uniquely matched sentences)")
    lines.append(f"- Equal: {(unique['year_given'] == unique['year']).mean():.1%}")
    lines.append(
        f"- Within one year: {((unique['year_given'] - unique['year']).abs() <= 1).mean():.1%}"
    )
    lines.append(
        f"- Chance level for equal, with {df['year_given'].nunique()} years: about "
        f"{1 / df['year_given'].nunique():.1%}"
    )
    lines.append("")
    mins = df[df["doc_type"] == "minutes"].drop_duplicates("doc_id")
    lag = (mins["release_date"] - mins["date"]).dt.days
    lines.append("## Minutes release dates")
    lines.append(
        f"- `release_date` is from the authors' minutes index for minutes, and equals "
        f"`date` for speeches and press conferences."
    )
    lines.append(
        f"- Days from `date` to release, minutes: median {lag.median():.0f} "
        f"(before 2005: {lag[mins['date'] < '2005'].median():.0f}; "
        f"2005 on: {lag[mins['date'] >= '2005'].median():.0f}); range {lag.min()} to {lag.max()}"
    )
    lines.append("")
    lines.append("## Document type by split")
    lines.append(pd.crosstab(df["doc_type"], df["split"], margins=True).to_markdown())
    lines.append("")
    lines.append("## Label balance by document type")
    lines.append(
        (pd.crosstab(df["doc_type"], df["label_name"], normalize="index") * 100)
        .round(1)
        .to_markdown()
    )
    lines.append("")
    lines.append("## Recovered year by split")
    lines.append(pd.crosstab(df["year"], df["split"], margins=True).to_markdown())
    lines.append("")
    in_both = set(df.loc[df["split"] == "train", "doc_id"]) & set(
        df.loc[df["split"] == "test", "doc_id"]
    )
    test = df[df["split"] == "test"]
    lines.append("## Document overlap in the official split")
    lines.append(
        f"- Documents with sentences in both train and test: {len(in_both)} of "
        f"{df['doc_id'].nunique()}"
    )
    lines.append(
        f"- Test sentences whose document also appears in train: "
        f"{test['doc_id'].isin(in_both).sum()} of {len(test)}"
    )
    lines.append("")
    return "\n".join(lines)


def main():
    ensure_repo()
    docs = load_documents()
    print(f"indexed {len(docs)} source sentences from {docs['doc_id'].nunique()} documents")
    df = match(load_labeled(), docs)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "labeled_sentences.csv"
    df.to_csv(out, index=False)
    print(f"saved {out.relative_to(ROOT)} ({len(df)} rows)")

    summary = summarize(df)
    out = RESULTS_DIR / "metadata_recovery.md"
    out.write_text(summary, encoding="utf-8")
    print(f"saved {out.relative_to(ROOT)}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
