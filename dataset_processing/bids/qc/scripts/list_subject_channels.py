"""List BrainVision/BIDS channels for one Alice subject.

This is a read-only helper for inspecting which channels are available in a
subject's BIDS EEG recording. It compares the channel names/types reported by
MNE from the BrainVision header with the metadata in the BIDS channels.tsv file.

Run from anywhere:

    python dataset_processing/bids/qc/scripts/list_subject_channels.py --subject sub-09
    python dataset_processing/bids/qc/scripts/list_subject_channels.py --subject S09
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mne
import pandas as pd


BIDS_ROOT = Path("/Users/yanyuwoo/Data/bids")
AUDIO_LIKE_EXACT = {"aud", "audio", "aux", "aux5", "ox"}
AUDIO_LIKE_PATTERN = re.compile(r"(aud|audio|aux|stim|trigger|trig)", re.IGNORECASE)
EOG_LIKE_PATTERN = re.compile(r"eog", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List all channels for one Alice BIDS subject."
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=BIDS_ROOT,
        help="Path to the BIDS dataset.",
    )
    parser.add_argument(
        "--subject",
        default="sub-09",
        help="Subject to inspect, e.g. sub-09, S09, or 09.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path for saving the channel table.",
    )
    return parser.parse_args()


def normalize_subject(subject: str) -> str:
    match = re.fullmatch(r"(?:sub-?|S)?0*(\d{1,2})", subject, flags=re.IGNORECASE)
    if match is None:
        return subject
    return f"sub-{int(match.group(1)):02d}"


def is_audio_like(channel_name: str) -> bool:
    lower = channel_name.strip().lower()
    return lower in AUDIO_LIKE_EXACT or AUDIO_LIKE_PATTERN.search(lower) is not None


def is_eog_like(channel_name: str) -> bool:
    return EOG_LIKE_PATTERN.search(channel_name.strip()) is not None


def subject_paths(bids_root: Path, subject: str) -> tuple[Path, Path]:
    eeg_dir = bids_root / subject / "eeg"
    vhdr_path = eeg_dir / f"{subject}_task-alice_eeg.vhdr"
    channels_path = eeg_dir / f"{subject}_task-alice_channels.tsv"
    if not vhdr_path.exists():
        raise FileNotFoundError(vhdr_path)
    if not channels_path.exists():
        raise FileNotFoundError(channels_path)
    return vhdr_path, channels_path


def load_bids_channel_metadata(channels_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(channels_path, sep="\t")
    if "name" not in metadata.columns:
        raise ValueError(f"{channels_path} does not contain a 'name' column")
    return metadata.set_index("name", drop=False)


def build_channel_table(vhdr_path: Path, channels_path: Path) -> pd.DataFrame:
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
    bids_metadata = load_bids_channel_metadata(channels_path)

    rows = []
    mne_types = raw.get_channel_types()
    for index, (name, mne_type) in enumerate(zip(raw.ch_names, mne_types), start=1):
        metadata = bids_metadata.loc[name] if name in bids_metadata.index else None
        bids_type = metadata.get("type", pd.NA) if metadata is not None else pd.NA
        status = metadata.get("status", pd.NA) if metadata is not None else pd.NA
        units = metadata.get("units", pd.NA) if metadata is not None else pd.NA
        rows.append(
            {
                "index_1based": index,
                "name": name,
                "mne_type": mne_type,
                "bids_type": bids_type,
                "status": status,
                "units": units,
                "audio_like": is_audio_like(name),
                "eog_like": is_eog_like(name),
            }
        )

    return pd.DataFrame(rows)


def print_summary(subject: str, table: pd.DataFrame) -> None:
    print(f"Subject: {subject}")
    print(f"Total channels: {len(table)}")
    print("\nMNE channel types:")
    print(table["mne_type"].value_counts(dropna=False).to_string())
    print("\nBIDS channel types:")
    print(table["bids_type"].value_counts(dropna=False).to_string())

    audio_like = table.loc[table["audio_like"], ["index_1based", "name", "mne_type", "bids_type"]]
    eog_like = table.loc[table["eog_like"], ["index_1based", "name", "mne_type", "bids_type"]]
    print("\nAudio-like channels:")
    print(audio_like.to_string(index=False) if not audio_like.empty else "None")
    print("\nEOG-like channels:")
    print(eog_like.to_string(index=False) if not eog_like.empty else "None")

    print("\nAll channels:")
    columns = ["index_1based", "name", "mne_type", "bids_type", "status", "units", "audio_like", "eog_like"]
    print(table[columns].to_string(index=False))


def main() -> None:
    args = parse_args()
    subject = normalize_subject(args.subject)
    vhdr_path, channels_path = subject_paths(args.bids_root, subject)

    print(f"BrainVision header: {vhdr_path}")
    print(f"BIDS channels.tsv: {channels_path}\n")

    table = build_channel_table(vhdr_path, channels_path)
    print_summary(subject, table)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
