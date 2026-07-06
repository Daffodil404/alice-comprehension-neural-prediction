"""Batch jobs for the Alice comprehension TRF-Tools pipeline.

Run from this directory with:

    trf-tools-make-jobs jobs.py

Start with the single-predictor model before expanding to model comparisons.
"""

from alice_trf_experiment import PARAMETERS, alice


JOBS = [
    alice.trf_job("gammatone-8", **PARAMETERS),
]

