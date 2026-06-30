#!/usr/bin/env python3
"""Check whether audio/auxiliary channels leaked into acoustic tracking targets.

This script does not modify BIDS files and does not rerun acoustic tracking. It
only inspects BIDS channel metadata plus the existing fold/channel tracking
scores.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mne
import pandas as pd


DEFAULT_BIDS_ROOT = Path("/Users/yanyuwoo/Data/bids")
DEFAULT_ANALYSIS_ROOT = Path("/Users/yanyuwoo/Data/Alice Comprehension")
DEFAULT_FOLD_SCORES = (
    DEFAULT_ANALYSIS_ROOT
    / "intermediate"
    / "acoustic_tracking_speech_envelope"
    / "fold_channel_scores.csv"
)
DEFAULT_OUTPUT = DEFAULT_ANALYSIS_ROOT / "qc" / "audio_aux_channel_leakage_check.csv"

AUDIO_LIKE_EXACT = {
    "aud",
    "audio",
    "aux",
    "aux5",
    "ox",
}
AUDIO_LIKE_PATTERN = re.compile(r"(aud|audio|aux|stim|trigger|trig)", re.IGNORECASE)


def is_audio_like_channel(name: str) -> bool:
    normalized = name.strip().lower()
    return normalized in AUDIO_LIKE_EXACT or AUDIO_LIKE_PATTERN.search(normalized) is not None


def read_raw_header(vhdr_path: Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")


def subject_sort_key(subject: str) -> int:
    return int(subject.split("-")[1])


def inspect_subject(bids_root: Path, subject: str) -> dict[str, object]:
    vhdr_path = bids_root / subject / "eeg" / f"{subject}_task-alice_eeg.vhdr"
    if not vhdr_path.exists():
        return {
            "subject": subject,
            "vhdr_path": str(vhdr_path),
            "raw_read_ok": False,
            "raw_read_error": f"missing file: {vhdr_path}",
        }

    try:
        raw = read_raw_header(vhdr_path)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {
            "subject": subject,
            "vhdr_path": str(vhdr_path),
            "raw_read_ok": False,
            "raw_read_error": str(exc),
        }

    ch_names = list(raw.ch_names)
    ch_types = raw.get_channel_types()
    audio_like = [ch for ch in ch_names if is_audio_like_channel(ch)]
    audio_like_types = [
        f"{ch}:{ch_types[ch_names.index(ch)]}"
        for ch in audio_like
        if ch in ch_names
    ]

    picked = raw.copy().pick("eeg")
    picked_names = list(picked.ch_names)
    audio_like_after_pick = [ch for ch in picked_names if is_audio_like_channel(ch)]

    return {
        "subject": subject,
        "vhdr_path": str(vhdr_path),
        "raw_read_ok": True,
        "raw_read_error": "",
        "n_raw_channels": len(ch_names),
        "n_picked_eeg_channels": len(picked_names),
        "audio_like_channels_raw": ";".join(audio_like),
        "audio_like_channel_types_raw": ";".join(audio_like_types),
        "audio_like_channels_after_pick_eeg": ";".join(audio_like_after_pick),
        "audio_like_after_pick_eeg_flag": bool(audio_like_after_pick),
    }


def summarize_best_channels(fold_scores_path: Path) -> pd.DataFrame:
    scores = pd.read_csv(fold_scores_path)
    required = {"subject", "channel", "tracking_r"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"{fold_scores_path} missing columns: {sorted(missing)}")

    channel_scores = (
        scores.groupby(["subject", "channel"], as_index=False)["tracking_r"]
        .mean()
        .rename(columns={"tracking_r": "channel_mean_tracking_r"})
    )
    best_idx = channel_scores.groupby("subject")["channel_mean_tracking_r"].idxmax()
    best = channel_scores.loc[best_idx].copy()
    best = best.rename(
        columns={
            "channel": "best_channel_name",
            "channel_mean_tracking_r": "best_channel_r",
        }
    )
    best["best_channel_audio_like_flag"] = best["best_channel_name"].map(is_audio_like_channel)
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument("--fold-scores", type=Path, default=DEFAULT_FOLD_SCORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mne.set_log_level("WARNING")

    subjects = sorted(
        [p.name for p in args.bids_root.glob("sub-*") if p.is_dir()],
        key=subject_sort_key,
    )
    metadata = pd.DataFrame([inspect_subject(args.bids_root, subject) for subject in subjects])

    best = summarize_best_channels(args.fold_scores)
    report = metadata.merge(best, on="subject", how="left")
    report["suspected_leakage_flag"] = (
        report["audio_like_after_pick_eeg_flag"].fillna(False)
        | report["best_channel_audio_like_flag"].fillna(False)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)

    print(f"Wrote: {args.output}")
    print(f"Subjects inspected: {len(report)}")
    print(f"Audio-like channel survives raw.pick('eeg'): {int(report['audio_like_after_pick_eeg_flag'].sum())}")
    print(f"Best channel is audio-like: {int(report['best_channel_audio_like_flag'].sum())}")
    print(f"Suspected leakage flag: {int(report['suspected_leakage_flag'].sum())}")

    suspicious = report[
        report["audio_like_after_pick_eeg_flag"] | report["best_channel_audio_like_flag"].fillna(False)
    ][
        [
            "subject",
            "audio_like_channels_raw",
            "audio_like_channel_types_raw",
            "audio_like_channels_after_pick_eeg",
            "best_channel_name",
            "best_channel_r",
            "best_channel_audio_like_flag",
        ]
    ]
    if not suspicious.empty:
        print("\nSuspicious subjects:")
        print(suspicious.to_string(index=False))


if __name__ == "__main__":
    main()
