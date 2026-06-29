#!/usr/bin/env python3
"""Check or standardize the Alice BIDS auxiliary/audio channel name.

This script fixes metadata names only. It does not edit the binary EEG data and
does not add channels for subjects that have fewer channels.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_BIDS_ROOT = Path("/Users/yanyuwoo/Data/bids")
DEFAULT_TARGET = "AUD"
DEFAULT_ALIASES = ("Aux5", "OX", "Ox", "ox")
CHANNEL_LINE_RE = re.compile(r"^(Ch(?P<number>\d+)=)(?P<name>[^,]*)(?P<rest>,.*)$")


@dataclass
class ChannelFinding:
    participant_id: str
    file_type: str
    path: Path
    channel_index: int | None
    current_name: str | None
    action: str


def read_channels_tsv(path: Path) -> tuple[str, list[str], list[list[str]]]:
    text = path.read_text(encoding="utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    reader = csv.reader(lines, delimiter="\t")
    rows = list(reader)
    if not rows:
        raise ValueError(f"Empty channels.tsv: {path}")
    return newline, rows[0], rows[1:]


def write_channels_tsv(path: Path, newline: str, header: list[str], rows: list[list[str]]) -> None:
    rendered = ["\t".join(header), *("\t".join(row) for row in rows)]
    path.write_text(newline.join(rendered) + newline, encoding="utf-8-sig")


def inspect_channels_tsv(
    path: Path,
    target: str,
    aliases: set[str],
    apply: bool,
    backup_dir: Path | None,
) -> ChannelFinding:
    participant_id = path.parts[-3]
    newline, header, rows = read_channels_tsv(path)
    name_idx = header.index("name")

    for index, row in enumerate(rows, start=1):
        name = row[name_idx]
        if name == target:
            return ChannelFinding(participant_id, "channels.tsv", path, index, name, "ok")
        if name in aliases:
            action = f"rename_to_{target}"
            if apply:
                backup_file(path, backup_dir)
                row[name_idx] = target
                write_channels_tsv(path, newline, header, rows)
                action = f"renamed_to_{target}"
            return ChannelFinding(participant_id, "channels.tsv", path, index, name, action)

    return ChannelFinding(participant_id, "channels.tsv", path, None, None, "no_target_or_alias")


def inspect_vhdr(
    path: Path,
    target: str,
    aliases: set[str],
    apply: bool,
    backup_dir: Path | None,
) -> ChannelFinding:
    participant_id = path.parts[-3]
    text = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()

    for i, line in enumerate(lines):
        match = CHANNEL_LINE_RE.match(line)
        if not match:
            continue

        channel_index = int(match.group("number"))
        name = match.group("name")
        if name == target:
            return ChannelFinding(participant_id, "vhdr", path, channel_index, name, "ok")
        if name in aliases:
            action = f"rename_to_{target}"
            if apply:
                backup_file(path, backup_dir)
                lines[i] = f"{match.group(1)}{target}{match.group('rest')}"
                path.write_text(newline.join(lines) + newline, encoding="utf-8")
                action = f"renamed_to_{target}"
            return ChannelFinding(participant_id, "vhdr", path, channel_index, name, action)

    return ChannelFinding(participant_id, "vhdr", path, None, None, "no_target_or_alias")


def backup_file(path: Path, backup_dir: Path | None) -> None:
    if backup_dir is None:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / path.relative_to(path.parents[3])
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def write_report(findings: list[ChannelFinding], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "participant_id",
                "file_type",
                "path",
                "channel_index",
                "current_name",
                "action",
            ],
        )
        writer.writeheader()
        for finding in findings:
            writer.writerow(
                {
                    "participant_id": finding.participant_id,
                    "file_type": finding.file_type,
                    "path": str(finding.path),
                    "channel_index": finding.channel_index,
                    "current_name": finding.current_name,
                    "action": finding.action,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--aliases", nargs="+", default=list(DEFAULT_ALIASES))
    parser.add_argument("--apply", action="store_true", help="Modify files in place.")
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aliases = set(args.aliases)
    findings: list[ChannelFinding] = []

    for channels_path in sorted(args.bids_root.glob("sub-*/eeg/*_channels.tsv")):
        findings.append(
            inspect_channels_tsv(
                channels_path,
                args.target,
                aliases,
                args.apply,
                args.backup_dir,
            )
        )

    for vhdr_path in sorted(args.bids_root.glob("sub-*/eeg/*_eeg.vhdr")):
        findings.append(
            inspect_vhdr(
                vhdr_path,
                args.target,
                aliases,
                args.apply,
                args.backup_dir,
            )
        )

    if args.report:
        write_report(findings, args.report)

    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.action] = counts.get(finding.action, 0) + 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"BIDS root: {args.bids_root}")
    print(f"Target name: {args.target}")
    print(f"Aliases: {', '.join(sorted(aliases))}")
    for action, count in sorted(counts.items()):
        print(f"{action}: {count}")

    changed = [f for f in findings if f.action.startswith("rename") or f.action.startswith("renamed")]
    if changed:
        print("Affected files:")
        for finding in changed:
            print(
                f"  {finding.participant_id} {finding.file_type} "
                f"channel={finding.channel_index} {finding.current_name} -> {args.target}"
            )


if __name__ == "__main__":
    main()

