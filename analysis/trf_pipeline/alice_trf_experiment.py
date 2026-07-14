"""TRF-Tools experiment definition for Alice comprehension.

This module defines the formal TRF-Tools pipeline used to estimate acoustic
tracking features. It intentionally replaces the earlier custom ridge-regression
implementation.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import re
import wave

import mne
from eelbrain.pipeline import LabelVar, PrimaryEpoch, RawFilter, RawSource
from trftools.pipeline import FilePredictor, TRFExperiment


BIDS_ROOT = Path("/Users/yanyuwoo/Data/bids")
STIMULI_DIR = BIDS_ROOT / "stimuli"
DURATION_TOLERANCE_SEC = 0.001


def _load_segment_durations_from_wav(stimuli_dir: Path = STIMULI_DIR) -> dict[str, float]:
    """Read segment durations from stimulus WAV files for QC comparison.

    Segment keys match the predictor file convention:
    ``1.wav`` -> ``1~gammatone-8.pickle``.
    """

    durations: dict[str, float] = {}
    for wav_path in sorted(stimuli_dir.glob("*.wav"), key=lambda path: int(path.stem)):
        with wave.open(str(wav_path), "rb") as wav:
            durations[wav_path.stem] = wav.getnframes() / wav.getframerate()
    if not durations:
        raise FileNotFoundError(f"No stimulus WAV files found in {stimuli_dir}")
    return durations


def _load_segment_durations_from_events(bids_root: Path = BIDS_ROOT) -> dict[str, float]:
    """Read segment durations from BIDS events.tsv files.

    The BIDS events table is the source of truth for epoch duration. Durations
    are keyed by ``stimulus_id`` so they match predictor file stems.
    """

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

# Keep this public name for the TRF-Tools experiment and notebooks.
SEGMENT_DURATION = BIDS_SEGMENT_DURATION


def _event_to_segment_map() -> dict[str, str]:
    """Map BrainVision marker labels to predictor stimulus keys.

    TRF-Tools/Eelbrain reads BrainVision marker labels from the raw files. Those
    labels can still contain the old marker strings even when BIDS events.tsv has
    already been standardized. Predictor filenames use the simple keys
    ``1~gammatone-8.pickle`` ... ``12~gammatone-8.pickle``.
    """

    mapping: dict[str, str] = {}
    for key in sorted(SEGMENT_DURATION, key=int):
        segment = int(key)
        mapping[f"Stimulus/{segment}"] = key
        mapping[f"Stimulus/S {segment:2d}"] = key
        mapping[f"Stimulus/S {segment}"] = key
        mapping[f"Stimulus/S{segment}"] = key
    return mapping


EVENT_TO_SEGMENT = _event_to_segment_map()

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
    """Set finite placeholder EEG positions when BIDS has no montage.

    Eelbrain needs finite sensor locations to build an NDVar. These positions
    are not used for spatial inference because adjacency is explicitly "none".
    """

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


class AudioAwareRawSource(RawSource):
    """RawSource that marks audio/auxiliary channels as non-EEG targets."""

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

PARAMETERS = {
    "raw": "0.5-20",
    "samplingrate": 50,
    "data": "eeg",
    "tstart": -0.100,
    "tstop": 1.000,
    "filter_x": "continuous",
    "error": "l1",
    "basis": 0.050,
    "partitions": -5,
    "selective_stopping": 1,
}


class AliceComprehensionTRF(TRFExperiment):
    """TRF-Tools pipeline for the BIDS Alice comprehension dataset."""

    data_dir = "."
    subject_re = r"sub-\d\d"
    sessions = ["alice"]

    raw = {
        "raw": AudioAwareRawSource(adjacency="none"),
        "0.5-20": RawFilter("raw", 0.5, 20, cache=False),
    }

    variables = {
        "segment": LabelVar("event", EVENT_TO_SEGMENT, default=""),
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

    # Predictor files are named {segment}~gammatone-8.pickle in derivatives.
    stim_var = "segment"

    predictors = {
        "gammatone": FilePredictor(resample="bin"),
        "word": FilePredictor(columns=True),
    }

    models = {
        "gammatone-8": "gammatone-8",
        "auditory-gammatone": "gammatone-8 + gammatone-on-8",
    }


alice = AliceComprehensionTRF(BIDS_ROOT)
