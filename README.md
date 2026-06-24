# Alice Comprehension Neural Prediction

This repository contains the analysis scaffold for testing whether subject-level
neural measures from the Alice EEG dataset predict post-story comprehension
performance.

## Research Question

Can neural measures extracted while participants listen to chapter one of
*Alice's Adventures in Wonderland* predict how well each participant answers the
post-listening story comprehension questions?

In modeling terms:

```text
X = subject-level EEG/neural features
y = subject-level comprehension score
```

## Local Data

Raw data are not stored in this repository. The current local paths are:

- BIDS EEG data: `/Users/yanyuwoo/Data/bids`
- Full/raw Alice dataset: `/Users/yanyuwoo/Data/r`
- Comprehension scores: `/Users/yanyuwoo/Data/r/comprehension-scores.txt`
- Comprehension questions: `/Users/yanyuwoo/Data/r/comprehension-questions.doc`

## Repository Layout

```text
analysis/                  Exploratory notebooks or one-off analysis scripts
config/                    Local path templates and analysis configuration
data/derived/              Small derived tables created from raw data
docs/                      Notes, SOPs, and analysis decisions
figures/                   Generated figures
results/                   Model outputs and summary tables
scripts/                   Command-line scripts
src/alice_comprehension/   Reusable Python code
```

## First Step

Create a clean subject-level comprehension table:

```bash
python3 scripts/parse_comprehension_scores.py
```

This writes:

```text
data/derived/comprehension_scores_clean.csv
```

## Analysis Plan

1. Parse and QC comprehension scores.
2. Extract subject-level neural features from TRF/mTRF outputs.
3. Merge behavioral and neural features into one subject-level table.
4. Fit simple predictive models for comprehension performance.
5. Run sensitivity analyses for high-noise subjects, `/4` scores, and S39.

