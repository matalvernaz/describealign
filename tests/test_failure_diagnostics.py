"""
Property tests for diagnose_alignment_failure.

Energy arrays are at the algorithm's internal 420-samples/sec rate. We
synthesise envelopes that match common failure shapes and check the
heuristic produces a useful summary.
"""
import numpy as np
import pytest

import describealaign as da


SR = 420  # internal energy-array rate (samples per second)


def _envelope(duration_minutes, mean=1.0, quiet_fraction=0.3):
    """Generate a fake energy envelope matching realistic AD/video shape."""
    n = int(duration_minutes * 60 * SR)
    rng = np.random.default_rng(seed=42)
    arr = rng.uniform(0.6, 1.5, size=n) * mean
    # Stamp the requested fraction of samples below the QUIET threshold.
    quiet_count = int(n * quiet_fraction)
    arr[:quiet_count] = 0.2
    rng.shuffle(arr)
    return arr.astype(np.float32)


def test_short_audio_against_long_video_is_classified():
    video = _envelope(45, mean=1.0)
    audio = _envelope(5, mean=1.0)
    diag = da.diagnose_alignment_failure(video, audio)
    assert "wrong episode" in diag["summary"] or "truncated" in diag["summary"]
    assert diag["duration_ratio"] < 0.5


def test_long_audio_against_short_video_is_classified():
    video = _envelope(20, mean=1.0)
    audio = _envelope(60, mean=1.0)  # 3× — full-season AD vs single episode
    diag = da.diagnose_alignment_failure(video, audio)
    assert "full-season" in diag["summary"] or "season" in diag["summary"].lower()
    assert diag["duration_ratio"] > 2.0


def test_silent_audio_is_classified():
    video = _envelope(45, mean=1.0, quiet_fraction=0.3)
    audio = _envelope(45, mean=1.0, quiet_fraction=0.95)
    diag = da.diagnose_alignment_failure(video, audio)
    assert "silence" in diag["summary"].lower()
    assert diag["audio_quiet_fraction"] > 0.9


def test_energy_mismatch_is_classified():
    """AD much quieter than video, but both with audible content.

    The "loud" half of an _envelope(mean=M) lives at 0.6M..1.5M, so we have
    to keep audio mean >= QUIET/0.6 ≈ 0.83 to keep its "loud" samples above
    the quiet threshold. Otherwise the silence-classifier fires first."""
    video = _envelope(45, mean=50.0, quiet_fraction=0.3)
    audio = _envelope(45, mean=1.0, quiet_fraction=0.3)
    diag = da.diagnose_alignment_failure(video, audio)
    assert "quieter" in diag["summary"].lower() or "energy" in diag["summary"].lower()


def test_reasonable_inputs_get_neutral_summary():
    """When everything looks sane, the diagnostic should say so rather
    than confidently misclassifying."""
    video = _envelope(45, mean=1.0, quiet_fraction=0.3)
    audio = _envelope(45, mean=1.0, quiet_fraction=0.3)
    diag = da.diagnose_alignment_failure(video, audio)
    assert "reasonable" in diag["summary"].lower() \
        or "no obvious cause" in diag["summary"].lower() \
        or "not be matching content" in diag["summary"].lower()


def test_diagnostic_dict_has_stable_keys():
    video = _envelope(20)
    audio = _envelope(20)
    diag = da.diagnose_alignment_failure(video, audio)
    for key in ("summary", "duration_ratio", "video_duration_minutes",
                "audio_duration_minutes", "energy_ratio",
                "video_quiet_fraction", "audio_quiet_fraction"):
        assert key in diag, f"diagnostic missing key {key!r}"


def test_alignment_mismatch_error_carries_diagnostic():
    """The custom exception type lets callers fish out the diagnostic
    without parsing the message string."""
    diag = {"summary": "test", "duration_ratio": 0.5}
    err = da.AlignmentMismatchError("boom", diagnostic=diag)
    assert err.diagnostic == diag
    assert str(err) == "boom"
