from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComprehensionScore:
    subject_raw: str
    participant_id: str
    correct: int | None
    total: int | None
    score_prop: float | None
    low_score_flag: bool
    high_noise_flag: bool
    original_use_flag: bool
    four_question_subject: bool
    s39_flag: bool
    all_scored_subjects: bool
    exclude_high_noise: bool
    exclude_low_perf: bool
    exclude_4_question_subjects: bool
    exclude_S39: bool
    original_use_only: bool
    notes: str


SCORE_RE = re.compile(r"^(?P<correct>\d+)/(?P<total>\d+)$")
SUBJECT_RE = re.compile(r"^S(?P<num>\d{2})$")


def subject_to_participant_id(subject: str) -> str:
    match = SUBJECT_RE.match(subject)
    if not match:
        raise ValueError(f"Unexpected subject id: {subject!r}")
    return f"sub-{match.group('num')}"


def parse_score_file(path: Path) -> list[ComprehensionScore]:
    rows: list[ComprehensionScore] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if line_number == 1 or not line.strip():
            continue

        parts = line.split()
        subject = parts[0]
        score_match = SCORE_RE.match(parts[1]) if len(parts) > 1 else None

        if score_match:
            correct = int(score_match.group("correct"))
            total = int(score_match.group("total"))
            notes = " ".join(parts[2:])
            score_prop = correct / total
        else:
            correct = None
            total = None
            score_prop = None
            notes = " ".join(parts[1:])

        low_score_flag = "low score" in notes
        high_noise_flag = "noise" in notes
        original_use_flag = notes == "use"
        all_scored_subjects = score_prop is not None
        four_question_subject = total == 4
        s39_flag = subject == "S39"

        rows.append(
            ComprehensionScore(
                subject_raw=subject,
                participant_id=subject_to_participant_id(subject),
                correct=correct,
                total=total,
                score_prop=score_prop,
                low_score_flag=low_score_flag,
                high_noise_flag=high_noise_flag,
                original_use_flag=original_use_flag,
                four_question_subject=four_question_subject,
                s39_flag=s39_flag,
                all_scored_subjects=all_scored_subjects,
                exclude_high_noise=all_scored_subjects and not high_noise_flag,
                exclude_low_perf=all_scored_subjects and not low_score_flag,
                exclude_4_question_subjects=all_scored_subjects and not four_question_subject,
                exclude_S39=all_scored_subjects and not s39_flag,
                original_use_only=all_scored_subjects and original_use_flag,
                notes=notes,
            )
        )
    return rows


def write_scores_csv(rows: list[ComprehensionScore], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_raw",
                "participant_id",
                "correct",
                "total",
                "score_prop",
                "low_score_flag",
                "high_noise_flag",
                "original_use_flag",
                "four_question_subject",
                "s39_flag",
                "all_scored_subjects",
                "exclude_high_noise",
                "exclude_low_perf",
                "exclude_4_question_subjects",
                "exclude_S39",
                "original_use_only",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
