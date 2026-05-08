"""
Property tests for the parsed-audio cache.

These exercise the cache machinery in isolation (key generation, eviction,
mtime-touch) without hitting ffmpeg. The full round-trip through
parse_audio_from_file is covered indirectly by an integration smoke test.
"""
import os
import tempfile
import time

import numpy as np
import pytest

import describealaign as da


def test_cache_key_changes_with_mtime(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x" * 100)
    k1 = da._audio_cache_key(str(f), 2)

    # Touch with a different mtime; key should change.
    new_time = (f.stat().st_atime, f.stat().st_mtime + 10)
    os.utime(f, new_time)
    k2 = da._audio_cache_key(str(f), 2)

    assert k1 is not None and k2 is not None
    assert k1 != k2, "different mtime must produce different cache key"


def test_cache_key_changes_with_size(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x" * 100)
    k1 = da._audio_cache_key(str(f), 2)
    f.write_bytes(b"x" * 200)
    k2 = da._audio_cache_key(str(f), 2)
    assert k1 != k2


def test_cache_key_changes_with_channel_count(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x" * 100)
    k1 = da._audio_cache_key(str(f), 1)
    k2 = da._audio_cache_key(str(f), 2)
    assert k1 != k2


def test_cache_key_returns_none_for_missing_file(tmp_path):
    assert da._audio_cache_key(str(tmp_path / "nope.mp3"), 2) is None


def test_eviction_drops_oldest_entries_first(tmp_path, monkeypatch):
    """When the cache exceeds its cap, oldest-by-mtime should evict first."""
    monkeypatch.setattr(da, "AUDIO_CACHE_SIZE_BYTES", 1000)

    def write_with_mtime(name, size, mtime):
        path = tmp_path / name
        path.write_bytes(b"x" * size)
        os.utime(path, (mtime, mtime))

    write_with_mtime("oldest.npy", 400, 1000.0)
    write_with_mtime("middle.npy", 400, 2000.0)
    write_with_mtime("newest.npy", 400, 3000.0)

    # Total = 1200, cap = 1000, incoming = 300. Need to free at least 500.
    da._evict_until_fits(str(tmp_path), 300)

    survivors = sorted(p.name for p in tmp_path.iterdir())
    assert "oldest.npy" not in survivors, "oldest must be evicted first"
    assert "newest.npy" in survivors, "newest must survive"


def test_eviction_no_op_when_under_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(da, "AUDIO_CACHE_SIZE_BYTES", 10000)
    (tmp_path / "a.npy").write_bytes(b"x" * 100)
    (tmp_path / "b.npy").write_bytes(b"x" * 100)
    da._evict_until_fits(str(tmp_path), 100)
    assert {p.name for p in tmp_path.iterdir()} == {"a.npy", "b.npy"}
