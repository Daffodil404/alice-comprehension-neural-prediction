#!/usr/bin/env python3
"""Check inputs for the Alice TRF-Tools pipeline.

This script does not estimate TRFs. It verifies the metadata and predictor
inputs required before running the first formal pipeline model.
"""

from __future__ import annotations

import csv
from pathlib import Path

from alice_trf_experiment import (
    BIDS_ROOT,
    BIDS_SEGMENT_DURATION,
    EVENT_TO_SEGMENT,
    WAV_SEGMENT_DURATION,
    alice,
)


PREDICTOR_DIR = BIDS_ROOT / "derivatives" / "predictors"


def check_aud_channel_types() -> tuple[int, list[str]]:
    bad_subjects: list[str] = []
    total_aud = 0

    for path in sorted(BIDS_ROOT.glob("sub-*/eeg/*_channels.tsv")):
        subject = path.parts[-3]
        with path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["name"] == "AUD":
                    total_aud += 1
                    if row["type"] != "MISC":
                        bad_subjects.append(subject)

    return total_aud, bad_subjects


def check_predictors() -> list[Path]:
    missing: list[Path] = []
    for segment in range(1, 13):
        path = PREDICTOR_DIR / f"{segment}~gammatone-8.pickle"
        if not path.exists():
            missing.append(path)
    return missing


def check_durations() -> None:
    print(f"BIDS duration segments: {len(BIDS_SEGMENT_DURATION)}")
    print(f"WAV duration segments: {len(WAV_SEGMENT_DURATION)}")
    for segment, bids_duration in BIDS_SEGMENT_DURATION.items():
        wav_duration = WAV_SEGMENT_DURATION.get(segment)
        if wav_duration is None:
            print(f"  stimulus_id={segment}: missing WAV duration")
            continue
        delta = bids_duration - wav_duration
        print(
            f"  stimulus_id={segment}: "
            f"BIDS={bids_duration:.6f}s WAV={wav_duration:.6f}s delta={delta:.6f}s"
        )


def check_events(subject: str = "01") -> None:
    events = alice.load_events(subject)
    print(f"Loaded events for subject {subject}: {events.n_cases} rows")
    print(events.head())
    print("segment values:", sorted(set(events["segment"])))


def check_raw_channel_types(subject: str = "01") -> None:
    alice.set(subject=subject, raw="0.5-20")
    raw = alice.load_raw(preload=False)
    channel_types = raw.get_channel_types()
    n_eeg = channel_types.count("eeg")
    n_misc = channel_types.count("misc")
    print(f"Raw channel types for subject {subject}: eeg={n_eeg}, misc={n_misc}")
    if "AUD" in raw.ch_names:
        print(f"AUD type in pipeline raw: {raw.get_channel_types(picks=['AUD'])[0]}")


def main() -> None:
    print(f"BIDS root: {BIDS_ROOT}")
    print(f"Predictor dir: {PREDICTOR_DIR}")
    subjects = alice.get_field_values("subject")
    print(f"Pipeline subjects: {len(subjects)} ({subjects[0]}..{subjects[-1]})")
    print(f"Event marker mappings: {len(EVENT_TO_SEGMENT)}")
    check_durations()

    total_aud, bad_subjects = check_aud_channel_types()
    print(f"AUD rows in channels.tsv: {total_aud}")
    if bad_subjects:
        print("AUD rows still marked as non-MISC:")
        print(", ".join(bad_subjects))
    else:
        print("AUD metadata OK: all AUD channels are MISC")

    missing_predictors = check_predictors()
    if missing_predictors:
        print("Missing gammatone-8 predictors:")
        for path in missing_predictors:
            print(f"  {path}")
    else:
        print("Predictors OK: all 12 gammatone-8 files exist")

    check_events("01")
    check_raw_channel_types("01")


if __name__ == "__main__":
    main()
