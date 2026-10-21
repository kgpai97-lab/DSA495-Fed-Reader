# Reading the Fed: Classifying Hawkish and Dovish Tone in FOMC Communications and Testing Its Link to Rate Decisions

## 1. Team Members
- Gaurav Pai
- Jacob Jones
- Angelus Jones

## 2. One-Sentence Task
Given sentences from Federal Open Market Committee (FOMC) meeting minutes, we will classify each as hawkish, dovish, or neutral, aggregate the labels into a meeting-level tone score, and test whether that score helps anticipate the next rate decision and relates to Treasury yield reactions when the minutes are released.

## 3. Illustrative Input and Desired Output
| Input sentence | Desired output |
|---|---|
| "Indeed, in the interest of fostering a continuation of sustainable growth of the economy, it would be desirable to tighten before any sign of actual higher inflation were to become evident." ([Feb 4-5, 1997 minutes](https://www.federalreserve.gov/fomc/minutes/19970204.htm)) | hawkish |
| "Despite a recent uptick in consumer prices, year-over-year consumer inflation remained at a very low level." ([Aug 12, 2003 minutes](https://www.federalreserve.gov/fomc/minutes/20030812.htm)) | dovish |
| "In Latin America, indicators suggested that economic activity in Mexico and Brazil expanded further in the first quarter." ([Apr 27-28, 2010 minutes](https://www.federalreserve.gov/monetarypolicy/fomcminutes20100428.htm)) | neutral |
| "Participants judged that it would be appropriate to move the stance of monetary policy toward a neutral posture expeditiously." ([Mar 15-16, 2022 minutes](https://www.federalreserve.gov/monetarypolicy/fomcminutes20220316.htm)) | neutral (a hard case: the gold label is neutral, but in March 2022 this signaled rate hikes) |

All four sentences and labels are taken from the labeled dataset (section 6A); sources were traced through the dataset authors' files. Each output includes a probability for the three classes. At the document level, the output is a tone score per meeting (share hawkish minus share dovish), dated by the day the minutes were released, a timeline of that score against the fed funds target rate, and a hike/hold/cut prediction for the next decision.

## 4. Background and Starting Point
The Fed expresses its stance in hedged language, so general sentiment tools often misread it: "increase" can signal tightening or easing depending on context. Our team wants a measurable signal that connects to macro regimes and risk. We start from the "Trillion Dollar Words" paper (Shah, Paturi, and Chava, ACL 2023), which introduced a labeled hawkish-dovish dataset: https://arxiv.org/abs/2305.07972

## 5. Intended Output and Success
**Output:** a reproducible Python pipeline that scrapes and cleans FOMC minutes, a classifier comparison with error analysis, a meeting-level tone series. Users would be students, analysts, and researchers who want a transparent tone measure for central bank text.

**Success:** our best classifier clearly beats a simple baseline on held-out data, including later time periods than it was trained on, we can explain where it fails, and the tone series gives an answer on whether it adds information about policy decisions or yield moves.

## 6. Preferred Data
**A. Labeled sentences (training and evaluation): "Trillion Dollar Words"**
- **Source:** Georgia Tech Financial Services Innovation Lab; sentences from FOMC minutes, press conferences, and speeches, annotated hawkish, dovish, or neutral.
- **Access:** Hugging Face [`gtfintechlab/fomc_communication`](https://huggingface.co/datasets/gtfintechlab/fomc_communication), via the `datasets` library.
- **Unit and fields:** one sentence; `sentence`, `year`, `label` (0 = dovish, 1 = hawkish, 2 = neutral, per the dataset card).
- **Size and period:** 2,480 sentences (1,984 train, 496 test) from 473 documents, 1996 to 2022: 1,131 from minutes, 1,026 from speeches, and 323 from press conferences. About 49% neutral, 26% dovish, 24% hawkish.
- **License:** CC BY-NC 4.0, fine for coursework; the text is public government communication.

**B. FOMC minutes (scored by the trained classifier)**
- **1996 to 2022:** the dataset authors' repo includes sentence-split minutes for all 214 meetings, with Fed URLs and release dates. We will spot-check a sample against federalreserve.gov before relying on them.
- **2023 to 2026:** scraped from the Federal Reserve [meeting calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm); minutes URLs follow `/monetarypolicy/fomcminutesYYYYMMDD.htm`. The text is in static HTML, so we expect to scrape with `requests` and `BeautifulSoup`.
- **Unit and fields:** one document per meeting, split into sentences; meeting date, release date, URL, text.
- **Size:** about 8 meetings a year, so roughly 250 sets of minutes from 1996 to 2026.
- **Timing:** minutes are released about three weeks after each meeting since December 2004; before that they were released after the next meeting. Each meeting's tone is dated by its release date.
- **Concerns:** Fed publications are public, but we will read the site's Website Policies before sharing raw text, use an identifying User-Agent, rate-limit requests, and cache everything.

**Outcome data:** fed funds target rate and Treasury yields from FRED.

**Examples:** the minutes linked in section 3, plus the [March 2023 minutes](https://www.federalreserve.gov/monetarypolicy/fomcminutes20230322.htm).

**How it supports the task:** A trains and tests the classifier; B provides the minutes it scores to build the tone series, including 2023 to 2026, which A does not cover.

**Backup:** if the Hugging Face dataset is unusable, Stanford's FOMC minutes corpus (1967 to 2008) provides unlabeled text.

## 7. Feasibility and Immediate Next Steps
- **Minimum viable project:** load the labeled data, compare TF-IDF + logistic regression with a fine-tuned transformer, and report macro-F1, a confusion matrix, and error analysis. The tone series, rate-decision prediction, and yield analysis are extensions.
- **Data access status:** we have loaded the Hugging Face dataset, summarized it (`results/dataset_summary.md`), and recovered each sentence's source document and date (`results/metadata_recovery.md`). We confirmed the Fed calendar loads as static HTML with predictable links.
- **Largest risk:** label and metadata quality. The dataset's `year` field is unusable, so our time-based splits rely on dates we recovered from the authors' files, which we have not yet checked against federalreserve.gov. The official train/test split shares documents between train and test, so it overstates accuracy. Some labeled "sentences" are speech titles and bylines.
- **Second risk:** domain shift. Only 46% of the labeled sentences come from minutes (the rest are speeches and press conferences), and they end in 2022, so 2023 to 2026 minutes have no labels.
- **Backup plan:** report results on the official test split alongside our document-grouped and time-based splits, and train a minutes-only variant to compare. If the recovered dates fail the spot check, fall back to document-grouped splits only.
- **Next two actions:**
  1. [Team member], by [date]: spot-check recovered documents against federalreserve.gov, then run the TF-IDF baseline on document-grouped and time-based splits.
  2. [Team member], by [date]: write and test the 2023 to 2026 minutes scraper 

## 8. Possible Method and Evaluation (Optional)
- **Methods:** TF-IDF + logistic regression baseline; zero-shot FinBERT (trained for general financial sentiment, so we expect it to do poorly); fine-tuned FinBERT and RoBERTa; a meeting-level tone score with logistic regression for the next decision, using only minutes released before that decision (the previous meeting's minutes from 2005 on, two meetings back before then), with walk-forward validation against a "repeat the last move" baseline; event-window analysis of yield changes on minutes release days against tone change.
- **Classifier evaluation:** macro-F1, per-class precision and recall, and confusion matrices on the official test split, document-grouped splits, and time-based splits on recovered dates (for example, train through 2015, test on 2016 to 2022), reported overall and for minutes sentences only, plus error analysis by failure type and confidence intervals.
- **Recent minutes (2023 to 2026):** with no gold labels, we will check face validity: whether tone separates minutes before hikes, holds, and cuts, agreement with the published FOMC-RoBERTa model, and the sentences where our models disagree most. These are sanity checks, not proof of accuracy.
- **Stretch:** an official-level hawkishness tracker and a simple hidden Markov model on the tone series.

## 9. Questions for the Instructor
1. Is the scope (classification core, with the tone series and market analysis as extensions) appropriate, or should we narrow it to the classifier comparison?
2. Is using an existing labeled dataset acceptable for the "raw text to analysis-ready data" requirement, given that we also scrape and clean our own documents?
3. Are time-based splits plus face-validity checks sufficient validation, given that we will not hand-label recent text?
4. Do you have expectations for the market-link component, given the small number of meetings and the limits of daily yield data?
