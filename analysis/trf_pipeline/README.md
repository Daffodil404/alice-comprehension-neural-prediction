# Eelbrain-Main TRF Pipeline

This directory contains the formal Eelbrain pipeline scaffold for the Alice
comprehension acoustic-tracking analysis.

The pipeline uses Eelbrain's built-in TRF API from the latest main branch.

## Environment

Use the Eelbrain-main environment:

```bash
mamba env create -f environment-trf-eelbrain-main.yml
mamba activate trf-eelbrain-main
```

This environment intentionally does not install `trftools`. It uses Eelbrain's
native TRF API:

- `eelbrain`
- `mne`

## Experiment Definition

Use this experiment definition:

```text
analysis/trf_pipeline/alice_eelbrain_main_experiment.py
```

## First Model

The first formal pipeline model is:

```text
gammatone-8
```

This model name is registered in `AliceComprehensionEelbrainMain.models` and uses
existing predictor files under:

```text
/Users/yanyuwoo/Data/derivatives/predictors/
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

The BrainVision files do not provide electrode digitization points, and the EEG
channels are numbered rather than named with a standard montage. The pipeline
therefore assigns finite placeholder EEG positions and disables sensor
adjacency. This is only to let Eelbrain create sensor-level NDVars; it is not
used for spatial inference.

## Planned Flow

1. Fix `AUD` channel type from `EEG` to `MISC`.
2. Check TRF pipeline inputs.
3. Validate the Eelbrain-main experiment definition.
4. Run the first `gammatone-8` TRF model with Eelbrain main.
5. Extract subject-level tracking scores.
6. Merge tracking scores with comprehension scores.
