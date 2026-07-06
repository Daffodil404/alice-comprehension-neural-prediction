#!/usr/bin/env python3
"""Set Alice BIDS audio auxiliary channels to MISC.

This script edits BIDS metadata only. It does not modify BrainVision binary
``.eeg`` files and it does not add or remove channels.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_BIDS_ROOT = Path("/Users/yanyuwoo/Data/bids")
DEFAULT_ANALYSIS_ROOT = Path("/Users/yanyuwoo/Data/Alice Comprehension")
DEFAULT_REPORT = DEFAULT_ANALYSIS_ROOT / "qc" / "aud_channel_type_misc_report.csv"
DEFAULT_BACKUP_ROOT = DEFAULT_ANALYSIS_ROOT / "intermediate" / "aud_channel_type_backups"

AUDIO_NAMES = {"AUD", "Audio", "AUDIO", "Aux", "AUX", "Aux5", "OX", "Ox", "ox"}


@dataclass
class Update:
    subject: str
    path: Path
    channel_name: str
    old_type: str
    new_type: str
    old_description: str
    new_description: str
    changed: bool
    written: bool
    backup_path: Path | None


def read_tsv(path: Path) -> tuple[str, list[str], list[list[str]]]:
    text = path.read_text(encoding="utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    rows = list(csv.reader(text.splitlines(), delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty TSV: {path}")
    return newline, rows[0], rows[1:]


def write_tsv(path: Path, newline: str, header: list[str], rows: list[list[str]]) -> None:
    rendered = ["\t".join(header), *("\t".join(row) for row in rows)]
    path.write_text(newline.join(rendered) + newline, encoding="utf-8-sig")


def backup_file(path: Path, backup_root: Path, bids_root: Path, timestamp: str) -> Path:
    backup_path = backup_root / timestamp / path.relative_to(bids_root)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def update_channels_tsv(
    path: Path,
    bids_root: Path,
    apply: bool,
    backup_root: Path,
    timestamp: str,
) -> list[Update]:
    subject = path.parts[-3]
    newline, header, rows = read_tsv(path)
    name_i = header.index("name")
    type_i = header.index("type")
    desc_i = header.index("description") if "description" in header else None

    updates: list[Update] = []
    changed_file = False
    backup_path: Path | None = None

    for row in rows:
        channel_name = row[name_i]
        if channel_name not in AUDIO_NAMES:
            continue

        old_type = row[type_i]
        old_description = row[desc_i] if desc_i is not None else ""
        new_type = "MISC"
        new_description = "Audio auxiliary channel"
        changed = old_type != new_type or (desc_i is not None and old_description != new_description)

        if changed:
            changed_file = True
            if apply:
                row[type_i] = new_type
                if desc_i is not None:
                    row[desc_i] = new_description

        updates.append(
            Update(
                subject=subject,
                path=path,
                channel_name=channel_name,
                old_type=old_type,
                new_type=new_type,
                old_description=old_description,
                new_description=new_description,
                changed=changed,
                written=False,
                backup_path=None,
            )
        )

    if changed_file and apply:
        backup_path = backup_file(path, backup_root, bids_root, timestamp)
        write_tsv(path, newline, header, rows)
        updates = [
            Update(
                subject=u.subject,
                path=u.path,
                channel_name=u.channel_name,
                old_type=u.old_type,
                new_type=u.new_type,
                old_description=u.old_description,
                new_description=u.new_description,
                changed=u.changed,
                written=u.changed,
                backup_path=backup_path if u.changed else None,
            )
            for u in updates
        ]

    return updates


def write_report(updates: list[Update], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "path",
                "channel_name",
                "old_type",
                "new_type",
                "old_description",
                "new_description",
                "changed",
                "written",
                "backup_path",
            ],
        )
        writer.writeheader()
        for update in updates:
            writer.writerow(
                {
                    "subject": update.subject,
                    "path": str(update.path),
                    "channel_name": update.channel_name,
                    "old_type": update.old_type,
                    "new_type": update.new_type,
                    "old_description": update.old_description,
                    "new_description": update.new_description,
                    "changed": update.changed,
                    "written": update.written,
                    "backup_path": str(update.backup_path) if update.backup_path else "",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--apply", action="store_true", help="Write metadata changes to channels.tsv files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    updates: list[Update] = []

    for path in sorted(args.bids_root.glob("sub-*/eeg/*_channels.tsv")):
        updates.extend(update_channels_tsv(path, args.bids_root, args.apply, args.backup_root, timestamp))

    write_report(updates, args.report)

    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Report: {args.report}")
    print(f"Audio-like channel rows found: {len(updates)}")
    print(f"Rows needing change: {sum(update.changed for update in updates)}")
    print(f"Rows written: {sum(update.written for update in updates)}")


if __name__ == "__main__":
    main()
