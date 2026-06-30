# Feature Construction Notebooks

This directory separates stimulus-level predictors from subject-level features.

## `predictors/`

Notebooks here build stimulus-level predictor time series from audio files.

Examples:

- `speech_envelope`: one predictor file per audio stimulus
- `gammatone_envelope_8`: one predictor file per audio stimulus

These are not subject-specific.

## `features/`

Notebooks here combine predictors with subject EEG to build subject-level feature
tables.

Examples:

- `acoustic_tracking_speech_envelope`: one row per subject

