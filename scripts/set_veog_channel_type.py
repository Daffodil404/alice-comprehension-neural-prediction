#!/usr/bin/env python3
"""Set Alice BIDS VEOG channels to the BIDS ``VEOG`` channel type.

The script edits ``*_channels.tsv`` metadata only. It does not modify the
BrainVision ``.eeg``, ``.vhdr``, or ``.vmrk`` files and does not add or remove
channels. By default it performs a dry run; pass ``--apply`` to write changes.
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
DEFAULT_REPORT = DEFAULT_ANALYSIS_ROOT / "qc" / "veog_channel_type_report.csv"
DEFAULT_BACKUP_ROOT = DEFAULT_ANALYSIS_ROOT / "intermediate" / "veog_channel_type_backups"


@dataclass
class Finding:
    subject: str
    path: Path
    old_type: str
    new_type: str
    action: str
    backup_path: Path | None = None


def read_tsv(path: Path) -> tuple[str, list[str], list[list[str]]]:
    text = path.read_text(encoding="utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    rows = list(csv.reader(text.splitlines(), delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty TSV: {path}")
    header, data = rows[0], rows[1:]
    missing = {"name", "type"}.difference(header)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return newline, header, data


def write_tsv(path: Path, newline: str, header: list[str], rows: list[list[str]]) -> None:
    rendered = ["\t".join(header), *("\t".join(row) for row in rows)]
    path.write_text(newline.join(rendered) + newline, encoding="utf-8-sig")


def backup_file(path: Path, backup_root: Path, bids_root: Path, timestamp: str) -> Path:
    backup_path = backup_root / timestamp / path.relative_to(bids_root)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)
    return backup_path


def inspect_file(
    path: Path,
    bids_root: Path,
    apply: bool,
    backup_root: Path,
    timestamp: str,
) -> Finding:
    subject = path.parts[-3]
    newline, header, rows = read_tsv(path)
    name_i = header.index("name")
    type_i = header.index("type")
    matches = [row for row in rows if row[name_i] == "VEOG"]

    if not matches:
        return Finding(subject, path, "", "VEOG", "missing_veog")
    if len(matches) > 1:
        raise ValueError(f"Multiple VEOG rows in {path}")

    row = matches[0]
    old_type = row[type_i]
    if old_type == "VEOG":
        return Finding(subject, path, old_type, "VEOG", "already_veog")
    if old_type != "EEG":
        return Finding(subject, path, old_type, "VEOG", "unexpected_type")
    if not apply:
        return Finding(subject, path, old_type, "VEOG", "would_change")

    backup_path = backup_file(path, backup_root, bids_root, timestamp)
    row[type_i] = "VEOG"
    write_tsv(path, newline, header, rows)
    return Finding(subject, path, old_type, "VEOG", "changed", backup_path)


def write_report(findings: list[Finding], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "path", "old_type", "new_type", "action", "backup_path"],
        )
        writer.writeheader()
        for finding in findings:
            writer.writerow(
                {
                    "subject": finding.subject,
                    "path": str(finding.path),
                    "old_type": finding.old_type,
                    "new_type": finding.new_type,
                    "action": finding.action,
                    "backup_path": str(finding.backup_path) if finding.backup_path else "",
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
    paths = sorted(args.bids_root.glob("sub-*/eeg/*_channels.tsv"))
    if not paths:
        raise FileNotFoundError(f"No channels.tsv files found under {args.bids_root}")

    findings = [
        inspect_file(path, args.bids_root, args.apply, args.backup_root, timestamp)
        for path in paths
    ]
    write_report(findings, args.report)

    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.action] = counts.get(finding.action, 0) + 1

    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"BIDS root: {args.bids_root}")
    print(f"channels.tsv files checked: {len(paths)}")
    for action, count in sorted(counts.items()):
        print(f"{action}: {count}")
    print(f"Report: {args.report}")
    if args.apply and any(finding.action == "changed" for finding in findings):
        print(f"Backup: {args.backup_root / timestamp}")


if __name__ == "__main__":
    main()
