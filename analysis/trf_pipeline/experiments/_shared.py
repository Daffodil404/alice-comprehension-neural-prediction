"""Shared runner for single-predictor Alice TRF experiments."""

from __future__ import annotations

import os
import re
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENTS_DIR = Path(__file__).resolve().parent
TRF_PIPELINE_DIR = EXPERIMENTS_DIR.parent
REPO_ROOT = TRF_PIPELINE_DIR.parents[1]
RESULTS_ROOT = TRF_PIPELINE_DIR / "results" / "by_predictor"
SCORES_PATH = REPO_ROOT / "data" / "derived" / "comprehension_scores_clean.csv"

# These settings define the controlled part of every predictor experiment.
# A predictor entry script should not override them.
TRF_SETTINGS = {
    "estimator": "boosting",
    "raw": "0.5-20",
    "epoch": "story-segments",
    "reference": "mastoids",
    "inv": "",
    "tstart": -0.100,
    "tstop": 1.000,
    "samplingrate": 50,
    "filter_x": "continuous",
}

# The mastoid reference reconstructs implicit acquisition-reference channel 29,
# yielding channels 1-61 as EEG targets (VEOG remains excluded).
EXPECTED_RESULT_CHANNELS = 61
AUDIO_ALIGNMENT_FAILED = {"09", "16"}


def _subject_number(subject: object) -> str:
    """Normalize an Eelbrain or BIDS subject label to two digits."""

    value = str(subject)
    if value.startswith("sub-"):
        value = value[4:]
    if not value.isdigit():
        raise ValueError(f"Invalid subject label: {subject!r}")
    return f"{int(value):02d}"


def _array_shape(value: object) -> str:
    """Represent an Eelbrain NDVar/array shape without serializing its data."""

    array = getattr(value, "x", value)
    return str(np.asarray(array).shape)


def _sensor_names(r: object) -> list[str]:
    """Extract sensor names from a sensor-dimensional Eelbrain NDVar."""

    sensor = r.get_dim("sensor")
    values = getattr(sensor, "values", getattr(sensor, "names", ()))
    return [str(value) for value in values]


def _mean_value(value: object) -> float:
    """Calculate a scalar mean from an NDVar, array, or scalar."""

    array = getattr(value, "x", value)
    return float(np.asarray(array, dtype=float).mean())


def _summarize_result(result: object) -> dict[str, object]:
    """Extract the predictor-independent subject metrics used downstream."""

    r_values = np.asarray(result.r.x, dtype=float).reshape(-1)
    sensors = _sensor_names(result.r)
    if len(sensors) != len(r_values):
        raise ValueError(
            f"Found {len(sensors)} sensor names for {len(r_values)} r values"
        )

    best_index = int(np.nanargmax(r_values))
    minimum_index = int(np.nanargmin(r_values))
    h_values = np.asarray(result.h.x, dtype=float)
    finite_r = bool(np.isfinite(r_values).all())
    finite_h = bool(np.isfinite(h_values).all())

    summary = {
        "tracking_r_mean": float(np.mean(r_values)),
        "tracking_r_median": float(np.median(r_values)),
        "tracking_r_max": float(np.max(r_values)),
        "tracking_r_min": float(np.min(r_values)),
        "tracking_r_sd": float(np.std(r_values)),
        "positive_channel_count": int((r_values > 0).sum()),
        "positive_channel_fraction": float((r_values > 0).mean()),
        "best_channel": sensors[best_index],
        "minimum_channel": sensors[minimum_index],
        "n_result_channels": len(r_values),
        "contains_veog": "VEOG" in sensors,
        "h_shape": _array_shape(result.h),
        "finite_r": finite_r,
        "finite_h": finite_h,
        "finite_result": finite_r and finite_h,
        "result_summary": str(result),
    }

    proportion_explained = getattr(result, "proportion_explained", None)
    summary["proportion_explained_mean"] = (
        _mean_value(proportion_explained)
        if proportion_explained is not None
        else np.nan
    )
    return summary


def _fixed_columns(experiment_id: str, predictor: str) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "predictor": predictor,
        "estimator": TRF_SETTINGS["estimator"],
        "raw": TRF_SETTINGS["raw"],
        "epoch": TRF_SETTINGS["epoch"],
        "reference": TRF_SETTINGS["reference"],
        "inv": TRF_SETTINGS["inv"],
        "tstart": TRF_SETTINGS["tstart"],
        "tstop": TRF_SETTINGS["tstop"],
        "samplingrate": TRF_SETTINGS["samplingrate"],
        "filter_x": TRF_SETTINGS["filter_x"],
    }


def _load_kwargs(subject: str) -> dict[str, object]:
    return {
        "subject": subject,
        "estimator": TRF_SETTINGS["estimator"],
        "raw": TRF_SETTINGS["raw"],
        "epoch": TRF_SETTINGS["epoch"],
        "reference": TRF_SETTINGS["reference"],
        "inv": TRF_SETTINGS["inv"],
        "samplingrate": TRF_SETTINGS["samplingrate"],
        "filter_x": TRF_SETTINGS["filter_x"],
    }


def _add_qc_columns(row: dict[str, object], subject: str) -> None:
    reasons: list[str] = []
    if row["status"] != "ok":
        reasons.append("trf_error")
    else:
        if not row["finite_result"]:
            reasons.append("non_finite_result")
        if row["n_result_channels"] != EXPECTED_RESULT_CHANNELS:
            reasons.append(
                f"expected_{EXPECTED_RESULT_CHANNELS}_channels_found_"
                f"{row['n_result_channels']}"
            )
        if row["contains_veog"]:
            reasons.append("result_contains_VEOG")

    if subject in AUDIO_ALIGNMENT_FAILED:
        row["audio_alignment_qc"] = "fail"
        reasons.append("audio_alignment_failed")
    else:
        row["audio_alignment_qc"] = "pass"

    row["include_trf_primary"] = not reasons
    row["trf_exclusion_reason"] = ";".join(reasons)


def _merge_behavior(results: pd.DataFrame) -> pd.DataFrame:
    """Attach behavioral values without using them to fit or select a TRF."""

    if not SCORES_PATH.exists():
        return results

    scores = pd.read_csv(SCORES_PATH)
    behavior_columns = [
        "participant_id",
        "correct",
        "total",
        "score_prop",
        "low_score_flag",
        "high_noise_flag",
        "four_question_subject",
        "s39_flag",
        "exclude_high_noise",
        "notes",
    ]
    results = results.merge(
        scores[behavior_columns],
        left_on="subject",
        right_on="participant_id",
        how="left",
        validate="one_to_one",
    ).drop(columns="participant_id")
    results["include_prediction_primary"] = (
        results["include_trf_primary"] & results["score_prop"].notna()
    )
    results["include_sensitivity"] = (
        results["include_prediction_primary"]
        & results["exclude_high_noise"].fillna(False)
    )
    return results


def _atomic_write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.csv")
    table.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _write_results(rows: list[dict[str, object]], output_path: Path) -> None:
    results = _merge_behavior(pd.DataFrame(rows))
    _atomic_write_csv(results, output_path)

    failures_path = output_path.with_name(output_path.stem + "_failures.csv")
    failures = results.loc[results["status"] != "ok"]
    _atomic_write_csv(failures, failures_path)


def run_experiment(
    *,
    experiment_id: str,
    predictor: str,
    subjects: list[str] | None = None,
) -> Path:
    """Fit/load one predictor for every requested subject and save one row each."""

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", experiment_id):
        raise ValueError(
            "experiment_id must contain only lowercase letters, digits, '_' or '-'"
        )

    # Keep plotting/font caches out of the repository and writable in restricted
    # environments. These defaults do not override a user's configured paths.
    temporary_root = Path(tempfile.gettempdir()) / "alice-comprehension-cache"
    temporary_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(temporary_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(temporary_root / "xdg"))

    from eelbrain import load_pipeline

    pipeline = load_pipeline(str(TRF_PIPELINE_DIR))
    available_subjects = [
        _subject_number(value) for value in pipeline.get_field_values("subject")
    ]

    if subjects:
        selected_subjects = [_subject_number(value) for value in subjects]
        unknown = sorted(set(selected_subjects) - set(available_subjects))
        if unknown:
            raise ValueError(f"Subjects not found in the pipeline: {unknown}")
    else:
        selected_subjects = available_subjects

    output_dir = RESULTS_ROOT / experiment_id
    if subjects:
        label = "-".join(selected_subjects)
        output_path = output_dir / f"subject_results_subset-{label}.csv"
    else:
        output_path = output_dir / "subject_results.csv"

    rows: list[dict[str, object]] = []
    fixed_columns = _fixed_columns(experiment_id, predictor)
    print(
        f"Running {experiment_id}: predictor={predictor!r}, "
        f"subjects={len(selected_subjects)}, reference={TRF_SETTINGS['reference']!r}"
    )

    for index, subject in enumerate(selected_subjects, start=1):
        row: dict[str, object] = {
            "run_time": datetime.now().isoformat(timespec="seconds"),
            "subject": f"sub-{subject}",
            **fixed_columns,
            "status": "error",
            "error_type": "",
            "error": "",
            "traceback": "",
        }
        print(f"[{index:02d}/{len(selected_subjects):02d}] sub-{subject}")

        try:
            kwargs = _load_kwargs(subject)
            result = pipeline.load_trf(
                predictor,
                TRF_SETTINGS["tstart"],
                TRF_SETTINGS["tstop"],
                **kwargs,
            )
            cache_path = pipeline.load_trf(
                predictor,
                TRF_SETTINGS["tstart"],
                TRF_SETTINGS["tstop"],
                path_only=True,
                **kwargs,
            )
            row.update(_summarize_result(result))
            row["cache_path"] = str(cache_path)
            row["status"] = "ok"
        except Exception as error:  # keep later subjects running after one failure
            row["error_type"] = type(error).__name__
            row["error"] = str(error)
            row["traceback"] = traceback.format_exc()

        _add_qc_columns(row, subject)
        rows.append(row)
        _write_results(rows, output_path)

    print(f"Saved {len(rows)} subject rows to {output_path}")
    return output_path
