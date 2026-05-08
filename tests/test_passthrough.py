"""
Property tests for is_passthrough_alignment.

The check must:
  - Return True for a near-identity alignment (slope ~1.0 everywhere).
  - Return False the moment any segment shows real drift.
  - Return False on PAL/NTSC magnitudes (>4% drift).
  - Tolerate sample-level rounding noise on otherwise-1.0 segments.
"""
import numpy as np
import pytest

import describealaign as da


def _flat_alignment(num_segments, slope, segment_seconds=200.0):
    """Generate audio_desc_times, video_times for a straight alignment at
    the given slope, with no seam discontinuities."""
    video_times = [0.0]
    audio_times = [0.0]
    for _ in range(num_segments):
        video_times.append(video_times[-1] + segment_seconds)
        audio_times.append(audio_times[-1] + segment_seconds * slope)
    return np.array(audio_times), np.array(video_times)


def test_pure_identity_is_passthrough():
    audio, video = _flat_alignment(num_segments=10, slope=1.0)
    assert da.is_passthrough_alignment(audio, video, median_slope=1.0)


def test_pal_to_ntsc_is_not_passthrough():
    """4.27% drift — needs real correction, not passthrough."""
    audio, video = _flat_alignment(num_segments=10, slope=23.976 / 25.0)
    assert not da.is_passthrough_alignment(audio, video, median_slope=23.976 / 25.0)


def test_ntsc_to_pal_is_not_passthrough():
    audio, video = _flat_alignment(num_segments=10, slope=25.0 / 23.976)
    assert not da.is_passthrough_alignment(audio, video, median_slope=25.0 / 23.976)


def test_within_tolerance_is_passthrough():
    """Slope of 1.0005 (0.05% drift) — sub-tolerance, treat as identity."""
    audio, video = _flat_alignment(num_segments=10, slope=1.0005)
    assert da.is_passthrough_alignment(audio, video, median_slope=1.0005)


def test_just_outside_tolerance_is_not_passthrough():
    """Slope of 1.005 (0.5% drift) — beyond default 0.1% tolerance."""
    audio, video = _flat_alignment(num_segments=10, slope=1.005)
    assert not da.is_passthrough_alignment(audio, video, median_slope=1.005)


def test_one_drifty_segment_disqualifies():
    """If 9 segments are slope-1.0 but one is at 1.05, that's a seam or
    real drift somewhere; passthrough is unsafe."""
    audio, video = _flat_alignment(num_segments=10, slope=1.0)
    # Force segment 5 to drift.
    audio[5] += 1.0  # adds 1s of drift starting from segment 5
    assert not da.is_passthrough_alignment(audio, video, median_slope=1.0)


def test_seam_disqualifies():
    """A short, extreme-rate segment (typical commercial-break seam)
    must disqualify even if median is ~1.0."""
    # 9 stable segments + 1 short seam at 200% slope (audio jumps 0.001s
    # relative to video's 0.04s).
    audio, video = _flat_alignment(num_segments=9, slope=1.0)
    # Append a short seam.
    video = np.append(video, video[-1] + 0.04)
    audio = np.append(audio, audio[-1] + 0.001)
    assert not da.is_passthrough_alignment(audio, video, median_slope=1.0)


def test_zero_or_negative_durations_skipped():
    """Adjacent identical timestamps shouldn't crash or mis-fire."""
    audio = np.array([0.0, 100.0, 100.0, 200.0])
    video = np.array([0.0, 100.0, 100.0, 200.0])
    # Should pass (the duplicate point is skipped, not treated as a slope).
    assert da.is_passthrough_alignment(audio, video, median_slope=1.0)
