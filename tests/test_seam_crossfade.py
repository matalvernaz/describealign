"""
Property tests for the seam crossfade in describealaign.replace_aligned_segments.

The crossfade is supposed to:
  - Smooth the amplitude transition at every AD↔original boundary.
  - Preserve sample-level timing (a sample at time T stays at time T).
  - Be a no-op far from any boundary.
  - Apply equal-power (sin² + cos² = 1) so the perceived loudness is
    constant through the fade.
"""
import numpy as np
import pytest

import describealaign as da

SR = da.AUDIO_SAMPLE_RATE


def _three_segment_setup():
    """Three 5-second segments. Middle is skipped (huge slope)."""
    seg_dur = 5.0
    seg_samples = int(seg_dur * SR)
    n_segs = 3
    total_samples = seg_samples * n_segs

    # Constant-amplitude streams make per-sample expectations easy.
    video_arr = np.full((2, total_samples), 0.2, dtype=np.float32)
    audio_desc_arr = np.full((2, total_samples), 0.8, dtype=np.float32)

    video_times = np.array([0.0, 5.0, 10.0, 15.0])
    audio_desc_times = np.array([0.0, 5.0, 5.001, 10.001])
    return video_arr, audio_desc_arr, audio_desc_times, video_times


def test_seam_amplitude_is_smooth_at_fade_out_boundary():
    """At the AD→original boundary, the fade-out midpoint should sit at the
    equal-power midpoint, not jump straight from 0.8 to 0.2."""
    video_arr, audio_desc_arr, ad_times, vid_times = _three_segment_setup()
    da.replace_aligned_segments(
        video_arr, audio_desc_arr, ad_times, vid_times, no_pitch_correction=True
    )
    mono = video_arr.mean(axis=0)

    seam = int(5.0 * SR)
    fade = int(da.SEAM_CROSSFADE_SECONDS * SR)
    pre = mono[seam - fade - 100]
    mid = mono[seam - fade // 2]
    post = mono[seam + 100]

    assert pre == pytest.approx(0.8, abs=0.01), "AD region body should be at AD level"
    assert post == pytest.approx(0.2, abs=0.01), "skipped region should retain original"
    # Equal-power crossfade gains are sin(θ) and cos(θ) with θ ramping
    # 0→π/2 across the fade. Midpoint of two constants A and B is
    # A·sin(π/4) + B·cos(π/4) = (A+B)/√2. For A=0.8, B=0.2 that's
    # 1.0/√2 ≈ 0.707.
    expected = (0.8 + 0.2) / np.sqrt(2)
    assert mid == pytest.approx(expected, abs=0.02), \
        f"midpoint {mid} should be ≈ {expected:.3f} (equal-power)"


def test_seam_amplitude_is_smooth_at_fade_in_boundary():
    """At the original→AD boundary, the fade-in should ramp the AD up."""
    video_arr, audio_desc_arr, ad_times, vid_times = _three_segment_setup()
    da.replace_aligned_segments(
        video_arr, audio_desc_arr, ad_times, vid_times, no_pitch_correction=True
    )
    mono = video_arr.mean(axis=0)

    seam = int(10.0 * SR)
    fade = int(da.SEAM_CROSSFADE_SECONDS * SR)
    pre = mono[seam - 100]
    mid = mono[seam + fade // 2]
    post = mono[seam + fade + 100]

    assert pre == pytest.approx(0.2, abs=0.01)
    assert post == pytest.approx(0.8, abs=0.01)
    expected = (0.8 + 0.2) / np.sqrt(2)
    assert mid == pytest.approx(expected, abs=0.02), \
        f"midpoint {mid} should be ≈ {expected:.3f} (equal-power)"


def test_fade_out_matches_equal_power_curve():
    """The fade-out window should follow the analytical equal-power curve
    (A·cos(θ) + B·sin(θ), θ ramping 0→π/2 across the fade) sample-for-sample,
    not jump in steps. Note: for two constants A>B>0 this curve is *not*
    strictly monotonic — it briefly rises before falling — but real audio
    fluctuates around zero so this only matters for test signals."""
    video_arr, audio_desc_arr, ad_times, vid_times = _three_segment_setup()
    da.replace_aligned_segments(
        video_arr, audio_desc_arr, ad_times, vid_times, no_pitch_correction=True
    )
    mono = video_arr.mean(axis=0)
    seam = int(5.0 * SR)
    fade = int(da.SEAM_CROSSFADE_SECONDS * SR)
    window = mono[seam - fade : seam]
    theta = (0.5 * np.pi) * np.linspace(0., 1., fade, dtype=np.float64)
    expected = 0.8 * np.cos(theta) + 0.2 * np.sin(theta)
    np.testing.assert_allclose(window, expected, atol=1e-3)


def test_no_op_far_from_any_seam():
    """Samples deep inside an AD region — outside any fade window —
    should be exactly the AD content the algorithm produced before crossfade
    (i.e. unchanged by the crossfade pass)."""
    video_arr, audio_desc_arr, ad_times, vid_times = _three_segment_setup()
    sr = SR

    # Probe at video time 2.5 s — middle of AD region 0, well clear of the
    # fade window at the start (0 → 0.2 s) and end (4.8 → 5.0 s) of that
    # region.
    da.replace_aligned_segments(
        video_arr, audio_desc_arr, ad_times, vid_times, no_pitch_correction=True
    )
    mono = video_arr.mean(axis=0)

    deep_inside = mono[int(2.5 * sr)]
    assert deep_inside == pytest.approx(0.8, abs=0.001), \
        "AD content far from any fade should be unchanged"


def test_continuous_narration_does_not_silence_adjust():
    """Regression: when the AD audio is loud throughout the trailing
    SEAM_LOOKBACK_SECONDS (i.e. the narrator is still talking), the
    silence-detect must NOT shorten the AD region. Earlier behaviour
    picked the lowest-energy 11 ms window unconditionally and overwrote
    up to ~2 s of narration with original program audio."""
    video_arr, audio_desc_arr, ad_times, vid_times = _three_segment_setup()
    da.replace_aligned_segments(
        video_arr, audio_desc_arr, ad_times, vid_times, no_pitch_correction=True
    )
    mono = video_arr.mean(axis=0)

    # 1.0 s before the AD→original seam should still be solid AD content.
    # Under the old buggy code this sample would have been overwritten
    # with original-program audio (0.2) by the silence-adjusted tail.
    seam = int(5.0 * SR)
    sample_one_sec_before = mono[seam - SR]
    assert sample_one_sec_before == pytest.approx(0.8, abs=0.001), \
        "1.0 s before seam should still be AD content, not original"


def test_silence_in_ad_tail_lands_seam_at_silence():
    """When the AD audio truly goes silent in the last second of a region,
    the silence-detect should advance the seam to that pause so the
    crossfade lands between words rather than mid-phrase."""
    seg_samples = int(5.0 * SR)
    total_samples = seg_samples * 3
    video_arr = np.full((2, total_samples), 0.2, dtype=np.float32)
    audio_desc_arr = np.full((2, total_samples), 0.8, dtype=np.float32)
    # Make the last 0.5 s of AD region 0 truly silent.
    silence_start = seg_samples - int(0.5 * SR)
    audio_desc_arr[:, silence_start:seg_samples] = 0.0

    video_times = np.array([0.0, 5.0, 10.0, 15.0])
    audio_desc_times = np.array([0.0, 5.0, 5.001, 10.001])

    da.replace_aligned_segments(
        video_arr, audio_desc_arr, audio_desc_times, video_times,
        no_pitch_correction=True
    )
    mono = video_arr.mean(axis=0)

    # The advanced seam should land somewhere inside the silent tail —
    # i.e. samples near the *end* of the AD region should have been
    # restored to original (0.2) rather than left as the AD's silence (0).
    seam_nominal = seg_samples
    assert mono[seam_nominal - 100] == pytest.approx(0.2, abs=0.05), \
        "samples just before nominal seam should be restored original (silence detected)"
