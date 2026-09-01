from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cv_car_counter import media


def _have(cmd: str) -> bool:
    return subprocess.run(
        ["which", cmd], capture_output=True, text=True
    ).returncode == 0


def _ffmpeg_available() -> bool:
    """True if ffmpeg/ffprobe are on PATH OR via the static_ffmpeg fallback."""
    if _have("ffmpeg") and _have("ffprobe"):
        return True
    try:
        import static_ffmpeg  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _ffmpeg_available(),
    reason="ffmpeg/ffprobe not installed (and static-ffmpeg not available)",
)


def test_probe_missing_file_raises():
    with pytest.raises(media.MediaError):
        media.probe("/nonexistent/video.mp4")


def test_validate_canonical_frame_count_mismatch(tmp_path):
    # create a tiny 1-second 5fps video, then assert validation expects 300 frames
    src = tmp_path / "src.mp4"
    out = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=64x64:rate=5",
            "-frames:v", "5", str(src),
        ],
        check=True,
    )
    # Pretend the canonical clip should be 20s @ 15fps = 300 frames. It only has 5.
    with pytest.raises(media.MediaError, match="frame count"):
        media.validate_canonical_clip(src, duration_seconds=20.0, analysis_fps=15)


def test_make_canonical_clip_roundtrip(tmp_path):
    src = tmp_path / "src.mp4"
    out = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30",
            "-frames:v", "60", str(src),
        ],
        check=True,
    )
    media.make_canonical_clip(src, 0.0, 1.0, out, analysis_fps=10)
    assert out.exists()
    probe = media.probe(out)
    assert probe.width == 64
    assert probe.height == 64


def test_sha256_of_file(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    h = media.sha256_of_file(p)
    assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
