"""Eelbrain pipeline for the Alice comprehension TRF analysis."""

from pathlib import Path

import mne
from eelbrain.pipeline import (
    Boosting,
    LabelVar,
    NUTSPredictor,
    Pipeline,
    PrimaryEpoch,
    RawFilter,
    RawSource,
    Reference,
    UTSPredictor,
)


BIDS_ROOT = Path("/Users/yanyuwoo/Data/bids")
MONTAGE_PATH = Path("/Users/yanyuwoo/Data/r/easycapM10-acti61_elec.sfp")
root = BIDS_ROOT


class AliceComprehension(Pipeline):
    """Alice comprehension analysis pipeline."""

    raw = {
        "raw": RawSource(
            montage=mne.channels.read_custom_montage(MONTAGE_PATH),
            adjacency="none",
        ),
        "0.5-20": RawFilter("raw", 0.5, 20, cache=False),
    }

    references = {
        "mastoids": Reference(["25", "29"], add="29"),
    }

    variables = {
        "stimulus": LabelVar(
            "stimulus_id",
            {stimulus_id: str(stimulus_id) for stimulus_id in range(1, 13)},
        ),
    }

    epochs = {
        "story-segments": PrimaryEpoch(
            tmin=0,
            tmax="duration + 1",
            samplingrate=50,
        ),
    }

    stim_var = "stimulus"

    predictors = {
        "gammatone": UTSPredictor(resample="bin"),
        "word": NUTSPredictor(),
    }

    estimators = {
        "boosting": Boosting(
            basis=0.050,
            error="l1",
            partitions=-5,
            selective_stopping=1,
        ),
    }
