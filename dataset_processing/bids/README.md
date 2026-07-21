# BIDS Dataset Processing

This folder contains notebooks for creating, checking, and updating the Alice
comprehension BIDS dataset.

The organization is by function:

```text
dataset_processing/bids/
├── conversion/
├── qc/
└── apply/
```

## Conversion

- `conversion/convert_alice_to_bids_original.ipynb`
  - Migrated from `/Users/yanyuwoo/Desktop/convert_alice_to_bids.ipynb`.
  - Converts original BrainVision files in `/Users/yanyuwoo/Data/r` into the
    BIDS dataset under `/Users/yanyuwoo/Data/bids`.
  - Outputs were cleared after migration.

## QC

- `qc/openneuro_events_qc.ipynb`
  - Read-only event-table QC for OpenNeuro/BIDS feedback.
  - Checks `raw_trial_type`, `stimulus_id`, `value`, and WAV-based `duration`.
  - Does not write to the BIDS dataset.

- `qc/stimulus_id_audio_alignment_qc.ipynb`
  - Read-only audio alignment QC.
  - Compares the recorded audio/AUD channel after each event onset against WAV
    stimulus envelopes.
  - Used to verify `stimulus_id` before editing BIDS events.

## Apply

- `apply/openneuro_events_cleanup.ipynb`
  - Processing/write-back notebook.
  - Defaults to dry-run mode.
  - Run only after the event-table QC and audio-alignment QC are reviewed.
  - Backs up edited files before writing to `/Users/yanyuwoo/Data/bids`.

## Suggested Order

1. Review conversion provenance if needed.
2. Run `qc/openneuro_events_qc.ipynb`.
3. Run `qc/stimulus_id_audio_alignment_qc.ipynb`.
4. Only after QC is accepted, run `apply/openneuro_events_cleanup.ipynb`.
