"""TRF-Tools experiment definition for Alice comprehension.

This module defines the formal TRF-Tools pipeline used to estimate acoustic
tracking features. It intentionally replaces the earlier custom ridge-regression
implementation.
"""

from __future__ import annotations

from pathlib import Path

import re
import wave

from eelbrain.pipeline import LabelVar, PrimaryEpoch, RawFilter, RawSource
from trftools.pipeline import FilePredictor, TRFExperiment


BIDS_ROOT = Path("/Users/yanyuwoo/Data/bids")
STIMULI_DIR = BIDS_ROOT / "stimuli"


def _load_segment_durations(stimuli_dir: Path = STIMULI_DIR) -> dict[str, float]:
    """Read segment durations from stimulus WAV files.

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


SEGMENT_DURATION = _load_segment_durations()


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


def is_audio_like_channel(channel_name: str) -> bool:
    lower = channel_name.strip().lower()
    return lower in AUDIO_LIKE_EXACT or AUDIO_LIKE_PATTERN.search(lower) is not None


class AudioAwareRawSource(RawSource):
    """RawSource that marks audio/auxiliary channels as non-EEG targets."""

    def _load(self, path, preload):
        raw = super()._load(path, preload)
        audio_like = [ch for ch in raw.ch_names if is_audio_like_channel(ch)]
        if audio_like:
            raw.set_channel_types({ch: "misc" for ch in audio_like}, on_unit_change="ignore")
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
        "raw": AudioAwareRawSource(adjacency="auto"),
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
        "auditory-gammatone": "gammatone-8 + gammatone-on-8",
    }


alice = AliceComprehensionTRF(BIDS_ROOT)
