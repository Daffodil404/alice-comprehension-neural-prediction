"""Eelbrain-main TRF job helpers for Alice comprehension.

This file replaces the old TRF-Tools ``trf-tools-make-jobs`` entrypoint. It
keeps job construction explicit and local to Eelbrain's built-in TRF API.
"""

from __future__ import annotations

from alice_eelbrain_main_experiment import TRF_OPTIONS, alice


MODEL = "gammatone-8"
SUBJECTS = tuple(alice.get_field_values("subject"))


def trf_path(subject: str):
    """Return the cache path for a subject/model without fitting it."""

    return alice.load_trf(
        MODEL,
        subject=subject,
        raw="0.5-20",
        epoch="story-segments",
        inv="",
        path_only=True,
        **TRF_OPTIONS,
    )


def make_trf_job(subject: str):
    """Load the data-carrying Eelbrain TRF job for one subject."""

    return alice.load_trf_job(
        MODEL,
        subject=subject,
        raw="0.5-20",
        epoch="story-segments",
        inv="",
        **TRF_OPTIONS,
    )


def fit_subject(subject: str):
    """Fit or load the gammatone-8 TRF for one subject."""

    return alice.load_trf(
        MODEL,
        subject=subject,
        raw="0.5-20",
        epoch="story-segments",
        inv="",
        **TRF_OPTIONS,
    )
