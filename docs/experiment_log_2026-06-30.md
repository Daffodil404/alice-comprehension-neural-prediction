# Experiment Log: 2026-06-30

## Acoustic Tracking Feature 1A: Speech Envelope

### Completed

- Created separate notebook structure for stimulus-level predictors and
  subject-level features.
- Built stimulus-level predictor notebooks:
  - `analysis/features/predictors/01_speech_envelope.ipynb`
  - `analysis/features/predictors/02_gammatone_envelope_8.ipynb`
- Built subject-level feature notebook:
  - `analysis/features/features/01_acoustic_tracking_speech_envelope.ipynb`
- Generated speech-envelope acoustic tracking features for 49 subjects.
- Unified BIDS event marker format with:
  - `trial_type = Stimulus/N`
  - `stimulus_id = N`
  - `raw_trial_type` preserving the original marker.
- Added BIDS event unification notebook:
  - `analysis/bids/01_unify_bids_events_trial_type.ipynb`
- Added leakage diagnostic script:
  - `scripts/check_audio_aux_channel_leakage.py`

### Issues Found

- Original BIDS `events.tsv` files used inconsistent stimulus marker formats:
  - `Stimulus/1`
  - `Stimulus/S  1`
  - `Stimulus/S 12`
- The first acoustic tracking run included only 20 subjects because the parser
  failed on `Stimulus/S  N` marker strings.
- After BIDS event unification, all 49 subjects could be included in the
  speech-envelope feature table.
- `tracking_r_best_channel` was suspiciously high for many subjects
  (`~0.90-0.93`), which is implausible for neural acoustic tracking.
- Leakage check showed that `AUD` was present in many subjects and MNE treated
  it as an EEG channel:
  - `Audio-like channel survives raw.pick("eeg"): 47`
  - `Best channel is audio-like: 43`
  - `Suspected leakage flag: 47`

### Fixes Applied

- Updated BIDS events so standardized stimulus IDs are available directly from
  `stimulus_id`.
- Updated the speech-envelope feature notebook so `load_events()` prefers the
  normalized `stimulus_id` column and only parses `trial_type` as a fallback.
- Updated `load_subject_eeg()` to mark audio/auxiliary-like channels as `misc`
  before selecting EEG channels.
- Audio/auxiliary-like channel names currently flagged include:
  - `AUD`
  - `Audio`
  - `Aux`
  - `Aux5`
  - `OX`
  - names containing `audio`, `aux`, `stim`, `trigger`, or `trig`
- Added feature/QC output fields:
  - `excluded_non_neural_channels`
  - `tracking_r_best_channel_name`

### Remaining Checks

- Rerun `analysis/features/features/01_acoustic_tracking_speech_envelope.ipynb`
  after the `AUD` exclusion change.
- Re-run `scripts/check_audio_aux_channel_leakage.py` after regenerating
  `fold_channel_scores.csv`.
- Confirm that `AUD` no longer appears after EEG channel selection.
- Confirm that `tracking_r_best_channel_name` is no longer usually `AUD`.
- Confirm that `tracking_r_best_channel` drops from the suspicious `~0.90`
  range to a plausible EEG range.
- Recompute comprehension association only after leakage is fixed.
- Treat current pre-fix comprehension correlations as invalid for scientific
  interpretation.
- Next feature to build after this fix:
  - `gammatone_envelope_8` acoustic tracking.

