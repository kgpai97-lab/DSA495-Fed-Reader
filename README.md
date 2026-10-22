# Reading the Fed: Classifying Tone in FOMC Communications 

> **DSA 495 Topic Brief**

## Team Members

- Gaurav Pai
- Jacob Jones
- Angelus Jones

## One-Sentence Task

**Given sentences from Federal Open Market Committee (FOMC) meeting minutes, we will classify each as hawkish, dovish, or neutral, aggregate the labels into a meeting-level tone score, and test whether that score helps anticipate the next rate decision and relates to Treasury yield reactions when the minutes are released.**

## Illustrative Input and Desired Output

- **Example input:** "Indeed, in the interest of fostering a continuation of sustainable growth of the economy, it would be desirable to tighten before any sign of actual higher inflation were to become evident." (from the [Feb 4-5, 1997 minutes](https://www.federalreserve.gov/fomc/minutes/19970204.htm))
- **Desired output:** the label **hawkish**, with a probability for each of the three classes. At the document level, the system returns a tone score per meeting (share hawkish minus share dovish) dated by the day the minutes were released, a timeline of that score against the fed funds target rate, and a hike/hold/cut prediction for the next decision.

## Background and Starting Point

The Fed expresses its stance in hedged language, so general sentiment tools often misread it: "increase" can signal tightening or easing depending on context. Our team wants a measurable signal that connects to macro regimes and risk. We start from the "Trillion Dollar Words" paper (Shah, Paturi, and Chava, ACL 2023), which introduced a labeled hawkish-dovish dataset: https://arxiv.org/abs/2305.07972

## Intended Output and Success

The project will produce a reproducible Python pipeline that scrapes and cleans FOMC minutes, a classifier comparison with error analysis, and a meeting-level tone series. Users would be students, analysts, and researchers who want a transparent tone measure for central bank text.

**Success statement:** our best classifier clearly beats a simple baseline on held-out data, including later time periods than it was trained on, we can explain where it fails, and the tone series gives an answer on whether it adds information about policy decisions or yield moves.

## Preferred Data

### Corpus

We use two corpora:

- **A. Labeled sentences (training and evaluation):** the "Trillion Dollar Words" dataset. One item is one sentence from FOMC minutes, press conferences, or speeches, labeled hawkish, dovish, or neutral (0 = dovish, 1 = hawkish, 2 = neutral, per the dataset card). Fields: `sentence`, `year`, `label`. Size: 2,480 sentences (1,984 train, 496 test) from 473 documents, 1996 to 2022: 1,131 from minutes, 1,026 from speeches, and 323 from press conferences. About 49% neutral, 26% dovish, 24% hawkish.
- **B. FOMC minutes (scored by the trained classifier):** one item is one set of meeting minutes, split into sentences, with meeting date, release date, URL, and text. About 8 meetings a year, so roughly 250 sets of minutes from 1996 to 2026, each 100+ sentences.
- **Outcome data:** fed funds target rate and Treasury yields from FRED.

### Source and Access

**A. Labeled sentences**
- **Dataset or corpus name:** Trillion Dollar Words (`gtfintechlab/fomc_communication`)
- **Original source organization or creator:** Financial Services Innovation Lab, Georgia Tech (Shah, Paturi, and Chava). The underlying text is published by the Federal Reserve.
- **Original collection process:** the authors collected FOMC minutes, press conference transcripts, and speeches from federalreserve.gov (1996 to 2022), split them into sentences, sampled sentences, and annotated them by hand (see the paper for their guidelines). Hugging Face only redistributes the dataset; the authors' [GitHub repo](https://github.com/gtfintechlab/fomc-hawkish-dovish) holds their source files.
- **Direct link:** https://huggingface.co/datasets/gtfintechlab/fomc_communication
- **Access method:** download with the Hugging Face `datasets` library (`src/load_fomc_dataset.py`).
- **Access, licensing, privacy, or redistribution considerations:** CC BY-NC 4.0, fine for non-commercial coursework. No private data; the text is public government communication.

**B. FOMC minutes**
- **Dataset or corpus name:** Minutes of the Federal Open Market Committee
- **Original source organization or creator:** Board of Governors of the Federal Reserve System
- **Original collection process:** the FOMC publishes minutes of each meeting, about three weeks after the meeting since December 2004 and after the next meeting before that. For 1996 to 2022, the dataset authors already collected and sentence-split all 214 sets of minutes, with Fed URLs and release dates; we will spot-check a sample against federalreserve.gov before relying on them.
- **Direct link:** [FOMC meeting calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm); minutes URLs follow `/monetarypolicy/fomcminutesYYYYMMDD.htm`.
- **Access method:** web collection of 2023 to 2026 minutes with `requests` and `BeautifulSoup` (static HTML); download of the authors' files for 1996 to 2022.
- **Access, licensing, privacy, or redistribution considerations:** Fed publications are public, but we will read the site's Website Policies before sharing raw text, use an identifying User-Agent, rate-limit requests, and cache everything. Raw downloaded text is kept out of the repository.
- **Backup corpus (optional):** if the Hugging Face dataset is unusable, Stanford's FOMC minutes corpus (1967 to 2008) provides unlabeled text.

### Examples

| Input sentence | Gold label |
|---|---|
| "Indeed, in the interest of fostering a continuation of sustainable growth of the economy, it would be desirable to tighten before any sign of actual higher inflation were to become evident." ([Feb 4-5, 1997 minutes](https://www.federalreserve.gov/fomc/minutes/19970204.htm)) | hawkish |
| "Despite a recent uptick in consumer prices, year-over-year consumer inflation remained at a very low level." ([Aug 12, 2003 minutes](https://www.federalreserve.gov/fomc/minutes/20030812.htm)) | dovish |
| "In Latin America, indicators suggested that economic activity in Mexico and Brazil expanded further in the first quarter." ([Apr 27-28, 2010 minutes](https://www.federalreserve.gov/monetarypolicy/fomcminutes20100428.htm)) | neutral |
| "Participants judged that it would be appropriate to move the stance of monetary policy toward a neutral posture expeditiously." ([Mar 15-16, 2022 minutes](https://www.federalreserve.gov/monetarypolicy/fomcminutes20220316.htm)) | neutral (a hard case: in March 2022 this signaled rate hikes) |

All four sentences and labels are taken from the labeled dataset; sources were traced through the dataset authors' files. The [March 2023 minutes](https://www.federalreserve.gov/monetarypolicy/fomcminutes20230322.htm) are an example of a recent, unlabeled document we would score.

### Fit with the Task

Corpus A supplies the labeled sentences needed to train and test the classifier, and corpus B supplies the minutes it scores to build the tone series, including 2023 to 2026, which A does not cover. Important gaps:

- The dataset's `year` field is misaligned with the sentences (it matches the true document year for only 6% of sentences, close to chance), and there is no document or date field. We matched every sentence back to the authors' source files and recovered its document, document type, date, and release date (`src/recover_metadata.py`, `results/metadata_recovery.md`).
- In the official train/test split, every test sentence comes from a document that also has sentences in train, so it overstates accuracy.
- Only 46% of the labeled sentences come from minutes, and there are no labels after 2022.
- We do not score FOMC statements: they are short, mostly repeat the previous statement, announce the decision we would predict, and are absent from the labeled data.

## Feasibility and Immediate Next Steps

- **Minimum viable project:** load the labeled data, compare TF-IDF + logistic regression with a fine-tuned transformer, and report macro-F1, a confusion matrix, and error analysis. The tone series, rate-decision prediction, and yield analysis are extensions.
- **Data-access check:** Gaurav Pai has downloaded the Hugging Face dataset, summarized it (`results/dataset_summary.md`), and recovered each sentence's source document, date, and minutes release date (`results/metadata_recovery.md`). We confirmed the Fed calendar loads as static HTML with predictable links.
- **Largest current risk:** label and metadata quality. Our time-based splits rely on dates recovered from the authors' files, which we have not yet checked against federalreserve.gov, and some labeled "sentences" are speech titles and bylines. A second risk is domain shift: the labeled data mixes minutes with speeches and press conferences and ends in 2022.
- **Backup plan:** report results on the official test split alongside our document-grouped and time-based splits, and train a minutes-only variant to compare. If the recovered dates fail the spot check, fall back to document-grouped splits only.
- **Next two actions:**
  1. Spot-check recovered documents against federalreserve.gov, then run the TF-IDF baseline on document-grouped and time-based splits — [Team member TBD — 30 Oct
  2. Write and test the 2023 to 2026 minutes scraper — Team member TBD — 30 Oct

## Possible Text-Analysis Method and Evaluation (Optional)

- **Methods:** TF-IDF + logistic regression baseline; zero-shot FinBERT (trained for general financial sentiment, so we expect it to do poorly); fine-tuned FinBERT and RoBERTa; a meeting-level tone score with logistic regression for the next decision, using only minutes released before that decision (the previous meeting's minutes from 2005 on, two meetings back before then), with walk-forward validation against a "repeat the last move" baseline; event-window analysis of yield changes on minutes release days against tone change.
- **Classifier evaluation:** macro-F1, per-class precision and recall, and confusion matrices on the official test split, document-grouped splits, and time-based splits on recovered dates (for example, train through 2015, test on 2016 to 2022), reported overall and for minutes sentences only, plus error analysis by failure type and confidence intervals.
- **Recent minutes (2023 to 2026):** with no gold labels, we will check face validity: whether tone separates minutes before hikes, holds, and cuts, agreement with the published FOMC-RoBERTa model, and the sentences where our models disagree most. These are sanity checks, not proof of accuracy.
- **Stretch:** an official-level hawkishness tracker and a simple hidden Markov model on the tone series.

## Questions for the Instructor

- Is the scope (classification core, with the tone series and market analysis as extensions) appropriate, or should we narrow it to the classifier comparison?
- Is using an existing labeled dataset acceptable for the "raw text to analysis-ready data" requirement, given that we also scrape and clean our own documents?
- Are time-based splits plus face-validity checks sufficient validation, given that we will not hand-label recent text?
- Do you have expectations for the market-link component, given the small number of meetings and the limits of daily yield data?

## AI-Use Statement

We used Claude (Anthropic), through Claude Code, while preparing this brief. 

Claude was used to revise this brief to match the course template and our decision to score minutes only. Advice we incorporated: dropping FOMC statements from the scored text, dating tone by minutes release date, and not relying on the leaky official train/test split.

