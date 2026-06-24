#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alice_comprehension.comprehension import parse_score_file, write_scores_csv
from alice_comprehension.paths import (
    DEFAULT_COMPREHENSION_SCORES,
    DERIVED_DATA_DIR,
)


def main() -> None:
    rows = parse_score_file(DEFAULT_COMPREHENSION_SCORES)
    output_path = DERIVED_DATA_DIR / "comprehension_scores_clean.csv"
    write_scores_csv(rows, output_path)

    scored = [row for row in rows if row.score_prop is not None]
    missing = [row.subject_raw for row in rows if row.score_prop is None]
    high_noise = [row.subject_raw for row in rows if row.high_noise_flag]
    low_score = [row.subject_raw for row in rows if row.low_score_flag]
    four_question = [row.subject_raw for row in rows if row.four_question_subject]

    print(f"Wrote {output_path}")
    print(f"Subjects: {len(rows)}")
    print(f"Subjects with scores: {len(scored)}")
    print(f"Missing score: {', '.join(missing) if missing else 'none'}")
    print(f"Four-question subjects: {', '.join(four_question)}")
    print(f"High-noise subjects: {', '.join(high_noise)}")
    print(f"Low-score subjects: {', '.join(low_score)}")


if __name__ == "__main__":
    main()
