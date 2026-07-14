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
    python dataset_processing/bids/qc/scripts/check_random_subject_triggers.py --random-count 10 --include-subjects sub-09 sub-21 sub-30
    python dataset_processing/bids/qc/scripts/check_random_subject_triggers.py --random-count 10 --include-subjects sub-09 sub-21 sub-30 --include-mismatch-subjects
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
DEFAULT_SUBJECT_QC_PATH = Path(
    "/Users/yanyuwoo/Data/Alice Comprehension/qc/stimulus_id_audio_alignment/"
    "stimulus_id_audio_alignment_subjects.csv"
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
        "--random-count",
        type=int,
        help="Randomly select this many subjects. Can be combined with --include-subjects.",
    )
    parser.add_argument(
        "--include-subjects",
        nargs="*",
        default=[],
        help="Additional subjects to always inspect, e.g. sub-09 sub-21.",
    )
    parser.add_argument(
        "--include-mismatch-subjects",
        action="store_true",
        help="Also inspect all subjects marked has_mismatch in the automatic audio-alignment QC report.",
    )
    parser.add_argument(
        "--subject-qc-path",
        type=Path,
        default=DEFAULT_SUBJECT_QC_PATH,
        help="Path to stimulus_id_audio_alignment_subjects.csv for --include-mismatch-subjects.",
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


def read_mismatch_subjects(subject_qc_path: Path) -> list[str]:
    if not subject_qc_path.exists():
        raise FileNotFoundError(subject_qc_path)
    qc = pd.read_csv(subject_qc_path)
    if "subject" not in qc.columns or "status" not in qc.columns:
        raise RuntimeError(f"{subject_qc_path} must contain subject and status columns")
    return qc.loc[qc["status"].eq("has_mismatch"), "subject"].astype(str).tolist()


def choose_subjects(args: argparse.Namespace) -> list[str]:
    subjects = available_subjects(args.bids_root)
    selected: list[str] = []

    if args.subject:
        selected.append(normalize_subject(args.subject))
    else:
        if args.random_count:
            rng = random.Random(args.seed)
            selected.extend(rng.sample(subjects, min(args.random_count, len(subjects))))
        elif not args.include_subjects and not args.include_mismatch_subjects:
            selected.append(choose_subject(args.bids_root, None, args.seed))

    selected.extend(normalize_subject(subject) for subject in args.include_subjects)

    if args.include_mismatch_subjects:
        selected.extend(read_mismatch_subjects(args.subject_qc_path))

    deduplicated = []
    for subject in selected:
        if subject not in subjects:
            raise ValueError(f"{subject!r} is not in {args.bids_root}")
        if subject not in deduplicated:
            deduplicated.append(subject)
    return deduplicated


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


def inspect_subject(
    bids_root: Path,
    subject: str,
    gammatone: dict[str, NDVar],
    samples: int,
    output_dir: Path,
    show: bool,
) -> dict:
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

    xs = []
    plot_rows = []

    for event_index, row in events.iterrows():
        stimulus_id = str(int(row["stimulus_id"]))
        value = row["value"]
        i_start = int(row["sample"])
        predictor = crop_ndvar_samples(gammatone[stimulus_id], samples)

        recorded = NDVar(
            raw._data[audio_index, i_start : i_start + samples],
            UTS(0, 0.002, samples),
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

    subject_output_dir = output_dir / subject
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    events_table_path = subject_output_dir / f"{subject}_events_used_for_visual_qc.csv"
    events[["trial_type", "stimulus_id", "value", "sample"]].to_csv(events_table_path, index=False)

    p = plot.UTS(
        xs,
        axh=2,
        w=10,
        title=f"{subject}: recorded AUD/Aux5 channel vs expected WAV-derived gammatone-1 predictor",
        axtitle=events["value"],
    )
    output_path = subject_output_dir / f"{subject}_stimulus_id_audio_alignment_eelbrain.png"
    overlay_output_path = subject_output_dir / f"{subject}_stimulus_id_audio_alignment_overlay.png"
    p.save(output_path, dpi=150, bbox_inches="tight")
    save_transparent_overlay(plot_rows, overlay_output_path)

    print("\nCreated Eelbrain UTS plot.")
    print(f"Saved event table: {events_table_path}")
    print(f"Saved Eelbrain plot: {output_path}")
    print(f"Saved transparent overlay plot: {overlay_output_path}")
    print("Each row compares the recorded AUD/Aux5 channel against the expected WAV-derived gammatone-1 predictor.")
    print("If shapes align, the BIDS stimulus_id is supported for that row.")
    print("If a row looks shifted or wrong, inspect that event before trusting stimulus_id.")

    if not show:
        p.close()
        print("Plot closed after saving. Re-run with --show to also keep it open if your backend supports it.")

    return {
        "subject": subject,
        "status": "ok",
        "audio_channel": audio_name,
        "n_events": len(events),
        "event_table": str(events_table_path),
        "eelbrain_plot": str(output_path),
        "overlay_plot": str(overlay_output_path),
    }


def main() -> None:
    args = parse_args()
    bids_root = args.bids_root
    predictor_dir = bids_root / "derivatives" / "predictors"
    subjects = choose_subjects(args)

    print("Subjects selected for visual QC:")
    print(", ".join(subjects))

    gammatone = load_gammatone_predictors(predictor_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for subject in subjects:
        try:
            summary_rows.append(
                inspect_subject(
                    bids_root=bids_root,
                    subject=subject,
                    gammatone=gammatone,
                    samples=args.samples,
                    output_dir=args.output_dir,
                    show=args.show,
                )
            )
        except Exception as exc:
            print(f"{subject} failed: {exc!r}")
            subject_output_dir = args.output_dir / subject
            subject_output_dir.mkdir(parents=True, exist_ok=True)
            summary_rows.append({"subject": subject, "status": "failed", "error": repr(exc)})

    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "visual_qc_subject_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved visual QC summary: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
