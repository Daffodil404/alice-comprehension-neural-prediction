# TRF-Tools Pipeline

This directory contains the formal TRF-Tools pipeline scaffold for the Alice
comprehension acoustic-tracking analysis.

The goal of this branch is to replace the custom hand-written ridge-regression
feature extraction with the TRF-Tools BIDS pipeline.

## Environment

Use the local `trf` environment:

```bash
mamba activate trf
```

The current environment has:

- `trftools`
- `eelbrain`
- `mne`

## First Model

The first formal pipeline model is:

```text
gammatone-8
```

This model name is registered in `AliceComprehensionTRF.models` and uses
existing predictor files under:

```text
/Users/yanyuwoo/Data/bids/derivatives/predictors/
```

with names like:

```text
1~gammatone-8.pickle
2~gammatone-8.pickle
...
```

Segment durations are read dynamically from BIDS `events.tsv` files:

```text
/Users/yanyuwoo/Data/bids/sub-*/eeg/*_events.tsv
```

The pipeline uses the `duration` column, keyed by `stimulus_id`, to define each
story-listening epoch. WAV durations are still read in the setup check only as a
QC comparison.

## Important Precondition

Before running TRFs, fix the `AUD` channel metadata:

```bash
python scripts/set_aud_channel_type_misc.py --apply
```

This changes BIDS `channels.tsv` metadata only. It does not modify `.eeg`
binary files.

## Planned Flow

1. Fix `AUD` channel type from `EEG` to `MISC`.
2. Check TRF pipeline inputs.
3. Run the first `gammatone-8` TRF model.
4. Extract subject-level tracking scores.
5. Merge tracking scores with comprehension scores.
