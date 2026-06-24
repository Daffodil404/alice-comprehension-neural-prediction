# SOP: Alice Comprehension Neural Prediction

## Goal

Test whether subject-level EEG-derived neural measures predict post-listening
story comprehension performance.

## Primary Outcome

- `score_prop = correct / total`
- Exclude S21 from score-based models unless the missing score is recovered.
- Track `/4` subjects separately: S14, S15, S31.

## Initial Data Products

1. `data/derived/comprehension_scores_clean.csv`
2. A future neural feature table with one row per subject.
3. A merged modeling table with behavioral and neural columns.

## Behavioral Table Columns

- `subject_raw`: original subject ID, for example `S01`.
- `participant_id`: BIDS participant ID, for example `sub-01`.
- `correct`: number of correct comprehension answers.
- `total`: number of attempted/scored questions.
- `score_prop`: primary outcome, computed as `correct / total`.
- `low_score_flag`: source notes indicate low comprehension score.
- `high_noise_flag`: source notes indicate high EEG noise.
- `original_use_flag`: source notes mark the subject as `use`.
- `four_question_subject`: subject was scored out of 4 instead of 8.
- `s39_flag`: documented special subject S39.

Sensitivity columns indicate whether the subject is included in that analysis
subset:

- `all_scored_subjects`
- `exclude_high_noise`
- `exclude_low_perf`
- `exclude_4_question_subjects`
- `exclude_S39`
- `original_use_only`

## Modeling Principle

Keep models simple because the sample size is small. Start with one or a few
predefined neural features, then compare against a mean-prediction baseline.

## Sensitivity Analyses

- All scored subjects.
- Exclude high-noise subjects.
- Exclude subjects with `/4` comprehension totals.
- Check whether conclusions depend on S39.
