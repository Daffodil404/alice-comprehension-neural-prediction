"""Eelbrain-main experiment definition for Alice comprehension TRFs.

This is the forward migration target for the acoustic-tracking pipeline. It
uses Eelbrain's built-in TRF pipeline API instead of TRF-Tools.
"""

from __future__ import annotations

import csv
import math
import re
import wave
from collections.abc import Mapping
from typing import Any
from pathlib import Path

import mne
from eelbrain.pipeline import LabelVar, Pipeline, PrimaryEpoch, RawFilter, RawSource
from eelbrain._experiment.trf import Boosting, NUTSPredictor, UTSPredictor
from eelbrain._experiment.trf.model import Term


BIDS_ROOT = Path("/Users/yanyuwoo/Data/bids")
STIMULI_DIR = BIDS_ROOT / "stimuli"
PREDICTOR_ROOT = Path("/Users/yanyuwoo/Data/derivatives/predictors")
DURATION_TOLERANCE_SEC = 0.001


def _load_segment_durations_from_wav(stimuli_dir: Path = STIMULI_DIR) -> dict[str, float]:
    """Read segment durations from stimulus WAV files for QC comparison."""

    durations: dict[str, float] = {}
    for wav_path in sorted(stimuli_dir.glob("*.wav"), key=lambda path: int(path.stem)):
        with wave.open(str(wav_path), "rb") as wav:
            durations[wav_path.stem] = wav.getnframes() / wav.getframerate()
    if not durations:
        raise FileNotFoundError(f"No stimulus WAV files found in {stimuli_dir}")
    return durations


def _load_segment_durations_from_events(bids_root: Path = BIDS_ROOT) -> dict[str, float]:
    """Read segment durations from BIDS events.tsv files."""

    durations_by_segment: dict[str, list[float]] = {}
    event_paths = sorted(bids_root.glob("sub-*/eeg/*_events.tsv"))
    if not event_paths:
        raise FileNotFoundError(f"No BIDS events.tsv files found under {bids_root}")

    for path in event_paths:
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            missing_columns = {"duration", "stimulus_id"} - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

            for row in reader:
                stimulus_id = row["stimulus_id"].strip()
                duration = row["duration"].strip()
                if not stimulus_id or not duration:
                    continue
                durations_by_segment.setdefault(stimulus_id, []).append(float(duration))

    if not durations_by_segment:
        raise ValueError(f"No stimulus durations found in BIDS events.tsv files under {bids_root}")

    durations: dict[str, float] = {}
    for segment, values in durations_by_segment.items():
        reference = values[0]
        if any(abs(value - reference) > DURATION_TOLERANCE_SEC for value in values):
            raise ValueError(
                f"Inconsistent duration values for stimulus_id={segment}: "
                f"min={min(values):.6f}, max={max(values):.6f}"
            )
        durations[segment] = reference

    return dict(sorted(durations.items(), key=lambda item: int(item[0])))


BIDS_SEGMENT_DURATION = _load_segment_durations_from_events()
WAV_SEGMENT_DURATION = _load_segment_durations_from_wav()
SEGMENT_DURATION = BIDS_SEGMENT_DURATION


def _event_to_segment_map() -> dict[str, str]:
    """Map BrainVision marker labels to predictor stimulus keys."""

    mapping: dict[str, str] = {}
    for key in sorted(SEGMENT_DURATION, key=int):
        segment = int(key)
        mapping[f"Stimulus/{segment}"] = key
        mapping[f"Stimulus/S {segment:2d}"] = key
        mapping[f"Stimulus/S {segment}"] = key
        mapping[f"Stimulus/S{segment}"] = key
    return mapping


EVENT_TO_SEGMENT = _event_to_segment_map()

FALLBACK_MONTAGE_CHANNELS = tuple(str(i) for i in range(1, 62) if i != 29) + ("VEOG",)

AUDIO_LIKE_EXACT = {"aud", "audio", "aux", "aux5", "ox"}
AUDIO_LIKE_PATTERN = re.compile(r"(aud|audio|aux|stim|trigger|trig)", re.IGNORECASE)
EOG_LIKE_EXACT = {"eog", "veog", "heog"}
EOG_LIKE_PATTERN = re.compile(r"eog", re.IGNORECASE)


def is_audio_like_channel(channel_name: str) -> bool:
    lower = channel_name.strip().lower()
    return lower in AUDIO_LIKE_EXACT or AUDIO_LIKE_PATTERN.search(lower) is not None


def is_eog_like_channel(channel_name: str) -> bool:
    lower = channel_name.strip().lower()
    return lower in EOG_LIKE_EXACT or EOG_LIKE_PATTERN.search(lower) is not None


def _has_valid_position(raw: mne.io.BaseRaw, channel_name: str) -> bool:
    loc = raw.info["chs"][raw.ch_names.index(channel_name)]["loc"][:3]
    return all(math.isfinite(float(value)) for value in loc)


def _set_fallback_eeg_positions(raw: mne.io.BaseRaw) -> None:
    """Set finite placeholder EEG positions when BIDS has no montage."""

    eeg_channels = [
        ch for ch, kind in zip(raw.ch_names, raw.get_channel_types())
        if kind == "eeg"
    ]
    if not eeg_channels or all(_has_valid_position(raw, ch) for ch in eeg_channels):
        return

    radius = 0.095
    ch_pos = {}
    for index, ch in enumerate(eeg_channels):
        angle = 2 * math.pi * index / len(eeg_channels)
        ch_pos[ch] = (radius * math.cos(angle), radius * math.sin(angle), 0.0)

    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    raw.set_montage(montage, on_missing="ignore")


def _make_fallback_montage() -> mne.channels.DigMontage:
    """Create placeholder positions for Alice's numbered EEG channels."""

    radius = 0.095
    ch_pos = {}
    for index, ch in enumerate(FALLBACK_MONTAGE_CHANNELS):
        angle = 2 * math.pi * index / len(FALLBACK_MONTAGE_CHANNELS)
        ch_pos[ch] = (radius * math.cos(angle), radius * math.sin(angle), 0.0)
    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")


class AudioAwareRawSource(RawSource):
    """RawSource that marks audio/auxiliary channels as non-EEG targets."""

    def __init__(self, **kwargs):
        kwargs.setdefault("montage", _make_fallback_montage())
        super().__init__(**kwargs)

    def _load(self, path, preload):
        raw = super()._load(path, preload)
        audio_like = [ch for ch in raw.ch_names if is_audio_like_channel(ch)]
        if audio_like:
            raw.set_channel_types({ch: "misc" for ch in audio_like}, on_unit_change="ignore")
        eog_like = [ch for ch in raw.ch_names if is_eog_like_channel(ch)]
        if eog_like:
            raw.set_channel_types({ch: "eog" for ch in eog_like}, on_unit_change="ignore")
        _set_fallback_eeg_positions(raw)
        return raw


class AlicePredictorMixin:
    """Load Alice predictors from the existing external predictor directory."""

    def _path(self, term: Term, state: Mapping[str, Any], root: Path) -> Path:
        return PREDICTOR_ROOT / f"{self._file_stem(term)}.pickle"


class AliceUTSPredictor(AlicePredictorMixin, UTSPredictor):
    pass


class AliceNUTSPredictor(AlicePredictorMixin, NUTSPredictor):
    pass


TRF_OPTIONS = {
    "samplingrate": 50,
    "data": "eeg",
    "tstart": -0.100,
    "tstop": 1.000,
    "filter_x": "continuous",
}


class AliceComprehensionEelbrainMain(Pipeline):
    """Eelbrain-main pipeline for the BIDS Alice comprehension dataset."""

    data_dir = "."
    subject_re = r"sub-\d\d"
    sessions = ["alice"]
    default_data = "eeg"

    raw = {
        "raw": AudioAwareRawSource(adjacency="none"),
        "0.5-20": RawFilter("raw", 0.5, 20, cache=False),
    }

    variables = {
        "segment": LabelVar("trial_type", EVENT_TO_SEGMENT, default=""),
        "duration": LabelVar(
            "segment",
            {segment: duration + 1 for segment, duration in SEGMENT_DURATION.items()},
        ),
    }

    epochs = {
        "chapter-1": PrimaryEpoch(
            "alice",
            tmin=0,
            tmax="duration",
            samplingrate=50,
            sel="segment != ''",
        ),
    }

    stim_var = "segment"

    predictors = {
        "gammatone": AliceUTSPredictor(resample="bin"),
        "word": AliceNUTSPredictor(),
    }

    estimators = {
        "boosting": Boosting(
            basis=0.050,
            error="l1",
            partitions=-5,
            selective_stopping=1,
        ),
    }

    models = {
        "gammatone-8": "gammatone-8",
        "acoustic-onset-8": "gammatone-on-8",
        "auditory-gammatone": "gammatone-8 + gammatone-on-8",
        "word-onset": "word",
        "word-logfreq": "word-LogFreq",
        "word-ngram": "word-NGRAM",
        "word-rnn": "word-RNN",
        "word-cfg": "word-CFG",
        "lexical-onset": "word-lexical",
        "nonlexical-onset": "word-nlexical",
        "gammatone-plus-word": "gammatone-8 + word",
    }


alice = AliceComprehensionEelbrainMain(BIDS_ROOT)
