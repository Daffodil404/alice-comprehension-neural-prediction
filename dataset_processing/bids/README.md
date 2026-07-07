# BIDS Dataset Processing

This folder contains notebooks and scripts used to create, inspect, and update
the Alice comprehension BIDS dataset.

These files are for dataset preparation, not feature extraction or TRF model
estimation.

## Files

- `00_convert_alice_to_bids_original.ipynb`
  - Migrated from `/Users/yanyuwoo/Desktop/convert_alice_to_bids.ipynb`.
  - Converts the original BrainVision files in `/Users/yanyuwoo/Data/r` into
    the BIDS dataset under `/Users/yanyuwoo/Data/bids`.
  - Outputs were cleared after migration so the notebook can be reviewed and
    re-run step by step.

## Planned Next Notebook

The next notebook should implement the OpenNeuro/BIDS event-table cleanup:

- remove redundant `raw_trial_type` columns when they duplicate `trial_type`
- add `stimulus_id` metadata to `events.json`
- verify whether `stimulus_id` refers to WAV segment identity rather than raw
  trigger `value`
- replace trigger-pulse durations with WAV segment durations in `events.tsv`
- document that BrainVision `.vmrk` markers are not rewritten unless explicitly
  regenerated
