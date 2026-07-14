"""Inspect BIDS stimulus-id/trigger alignment for one Alice subject.

This is a BIDS-native companion to the original Alice
``import_dataset/check-triggers.py`` script. The original script uses the Alice
Eelbrain pipeline to load corrected events. This script instead reads this
project's BIDS files directly:

- ``sub-*/eeg/*_events.tsv`` for ``stimulus_id``, ``value`` and sample onset
- ``sub-*/eeg/*_eeg.vhdr`` for the recorded AUD/Aux5 channel
- ``derivatives/predictors/*~gammatone-1.pickle`` for the expected stimulus
  envelope

The script is read-only. It creates an Eelbrain plot for visual inspection and
prints the event table used for the comparison. The plot is saved to the QC
directory by default, because command-line plotting might not open a GUI window.

Run from the alice-comprehension-neural-prediction repository root:

    python dataset_processing/bids/qc/scripts/check_random_subject_triggers.py --show
    python dataset_processing/bids/qc/scripts/check_random_subject_triggers.py --subject S1 --show
    python dataset_processing/bids/qc/scripts/check_random_subject_triggers.py --subject sub-30 --show
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eelbrain import NDVar, UTS, load, plot


BIDS_ROOT = Path("/Users/yanyuwoo/Data/bids")
PREDICTOR_DIR = BIDS_ROOT / "derivatives" / "predictors"
DEFAULT_OUTPUT_DIR = Path(
    "/Users/yanyuwoo/Data/Alice Comprehension/qc/stimulus_id_audio_alignment/plots"
)
AUDIO_CHANNEL_CANDIDATES = ("AUD", "Aux5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check BIDS stimulus_id/value alignment for one Alice subject."
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=BIDS_ROOT,
        help="Path to the BIDS dataset.",
    )
    parser.add_argument(
        "--subject",
        help="Subject to inspect, e.g. S1, S01, sub-01, or sub-30. If omitted, choose a random subject.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used when --subject is omitted.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of raw audio-channel samples to compare after each event onset.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Keep the Eelbrain plot open after saving. This depends on the active graphics backend.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved plot images.",
    )
    return parser.parse_args()


def normalize_subject(subject: str) -> str:
    """Normalize S1/S01/sub-01 style inputs to BIDS style sub-01."""

    match = re.fullmatch(r"(?:sub-?|S)?0*(\d{1,2})", subject, flags=re.IGNORECASE)
    if match is None:
        return subject
    return f"sub-{int(match.group(1)):02d}"


def available_subjects(bids_root: Path) -> list[str]:
    return sorted(
        [path.name for path in bids_root.glob("sub-*") if path.is_dir()],
        key=lambda item: int(item.split("-")[1]),
    )


def choose_subject(bids_root: Path, requested_subject: str | None, seed: int) -> str:
    subjects = available_subjects(bids_root)
    if requested_subject:
        subject = normalize_subject(requested_subject)
        if subject not in subjects:
            raise ValueError(
                f"{requested_subject!r} normalized to {subject!r}, but it is not in {bids_root}."
            )
        return subject
    rng = random.Random(seed)
    return rng.choice(subjects)


def subject_paths(bids_root: Path, subject: str) -> tuple[Path, Path]:
    eeg_dir = bids_root / subject / "eeg"
    events_path = eeg_dir / f"{subject}_task-alice_events.tsv"
    vhdr_path = eeg_dir / f"{subject}_task-alice_eeg.vhdr"
    if not events_path.exists():
        raise FileNotFoundError(events_path)
    if not vhdr_path.exists():
        raise FileNotFoundError(vhdr_path)
    return events_path, vhdr_path


def find_audio_channel(raw) -> tuple[str, int]:
    for name in AUDIO_CHANNEL_CANDIDATES:
        if name in raw.ch_names:
            return name, raw.ch_names.index(name)
    raise RuntimeError(f"No AUD/Aux5 channel found. Channels: {raw.ch_names}")


def load_gammatone_predictors(predictor_dir: Path) -> dict[str, NDVar]:
    predictors = {}
    for stimulus_id in range(1, 13):
        path = predictor_dir / f"{stimulus_id}~gammatone-1.pickle"
        if not path.exists():
            raise FileNotFoundError(path)
        predictor = load.unpickle(path)
        predictor.name = "expected WAV-derived gammatone-1 predictor"
        predictor /= predictor.std()
        predictors[str(stimulus_id)] = predictor
    return predictors


def normalize_for_plot(values) -> np.ndarray:
    """Match the original visual-QC normalization while keeping arrays plottable."""

    array = np.asarray(values, dtype=float).copy()
    array -= np.nanmin(array)
    std = np.nanstd(array)
    if std > 0 and np.isfinite(std):
        array /= std
    return array


def crop_ndvar_samples(ndvar: NDVar, n_samples: int) -> NDVar:
    """Return the first ``n_samples`` from a uniformly sampled NDVar."""

    stop_time = min(n_samples, len(ndvar.x)) * 0.002
    return ndvar.sub(time=(0, stop_time))


def save_transparent_overlay(plot_rows: list[dict], output_path: Path) -> None:
    """Save a transparent overlay plot with explicit matplotlib alpha settings."""

    n_rows = len(plot_rows)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(10, max(2, 1.5 * n_rows)),
        sharex=False,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    for ax, row in zip(axes, plot_rows):
        recorded = row["recorded"]
        predictor = row["predictor"]
        n = min(len(recorded), len(predictor))
        time = np.arange(n) * 0.002

        ax.plot(
            time,
            recorded[:n],
            color="tab:blue",
            alpha=0.55,
            lw=1.2,
            label="recorded AUD/Aux5 channel",
        )
        ax.plot(
            time,
            predictor[:n],
            color="tab:orange",
            alpha=0.55,
            lw=1.2,
            label="expected WAV-derived gammatone-1 predictor",
        )
        ax.set_title(
            f"event {row['event_index']} | stimulus_id={row['stimulus_id']} | value={row['value']}",
            fontsize=9,
        )
        ax.set_ylabel("norm.")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time from event onset (s)")
    axes[0].legend(loc="upper right")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    bids_root = args.bids_root
    predictor_dir = bids_root / "derivatives" / "predictors"

    subject = choose_subject(bids_root, args.subject, args.seed)
    events_path, vhdr_path = subject_paths(bids_root, subject)

    print(f"Subject: {subject}")
    print(f"Events: {events_path}")
    print(f"Raw EEG: {vhdr_path}")

    events = pd.read_csv(events_path, sep="\t")
    required = {"sample", "stimulus_id", "value"}
    missing = required - set(events.columns)
    if missing:
        raise RuntimeError(f"{events_path} is missing required columns: {sorted(missing)}")

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
    audio_name, audio_index = find_audio_channel(raw)
    print(f"Audio channel: {audio_name}")

    print("\nEvent table from BIDS")
    print("stimulus_id = WAV/predictor identity")
    print("value = raw trigger code")
    print(events[["trial_type", "stimulus_id", "value", "sample"]])

    gammatone = load_gammatone_predictors(predictor_dir)
    xs = []
    plot_rows = []

    for event_index, row in events.iterrows():
        stimulus_id = str(int(row["stimulus_id"]))
        value = row["value"]
        i_start = int(row["sample"])
        predictor = crop_ndvar_samples(gammatone[stimulus_id], args.samples)

        recorded = NDVar(
            raw._data[audio_index, i_start : i_start + args.samples],
            UTS(0, 0.002, args.samples),
            name=f"recorded {audio_name} channel",
        )
        recorded -= recorded.min()
        recorded /= recorded.std()
        xs.append([recorded, predictor])
        plot_rows.append(
            {
                "event_index": event_index,
                "stimulus_id": stimulus_id,
                "value": value,
                "recorded": normalize_for_plot(recorded.x),
                "predictor": normalize_for_plot(predictor.x),
            }
        )

        print(
            f"event_index={event_index}, stimulus_id={stimulus_id}, "
            f"value={value}, sample={i_start}"
        )

    p = plot.UTS(
        xs,
        axh=2,
        w=10,
        title=f"{subject}: recorded AUD/Aux5 channel vs expected WAV-derived gammatone-1 predictor",
        axtitle=events["value"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{subject}_stimulus_id_audio_alignment_eelbrain.png"
    overlay_output_path = args.output_dir / f"{subject}_stimulus_id_audio_alignment_overlay.png"
    p.save(output_path, dpi=150, bbox_inches="tight")
    save_transparent_overlay(plot_rows, overlay_output_path)

    print("\nCreated Eelbrain UTS plot.")
    print(f"Saved Eelbrain plot: {output_path}")
    print(f"Saved transparent overlay plot: {overlay_output_path}")
    print("Each row compares the recorded AUD/Aux5 channel against the expected WAV-derived gammatone-1 predictor.")
    print("If shapes align, the BIDS stimulus_id is supported for that row.")
    print("If a row looks shifted or wrong, inspect that event before trusting stimulus_id.")

    if not args.show:
        p.close()
        print("Plot closed after saving. Re-run with --show to also keep it open if your backend supports it.")


if __name__ == "__main__":
    main()
