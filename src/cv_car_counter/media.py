"""Media handling: probe, transcode the canonical 20-second CFR clip, validate it.

Stream-copying arbitrary cut points snaps to keyframes, so we transcode to a
constant-frame-rate clip with timestamps reset to zero and an exact frame count.
Analytics must use presentation timestamps, never wall-clock speed or an
unverified `CAP_PROP_FPS`.

ffmpeg/ffprobe are resolved in this order: (1) on the system PATH, (2) the
bundled ``static_ffmpeg`` package if installed. This keeps the canonical-clip
contract while letting the tool run on machines without a system ffmpeg.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class MediaError(RuntimeError):
    pass


def _resolve_bin(name: str) -> str:
    """Return a usable path to ``name`` (ffmpeg or ffprobe)."""
    found = shutil.which(name)
    if found:
        return found
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
    except ImportError:
        static_ffmpeg = None  # type: ignore[assignment]
    found = shutil.which(name)
    if found:
        return found
    raise MediaError(
        f"{name!r} not found on PATH. Install ffmpeg or run "
        "`pip install static-ffmpeg` to get a bundled build."
    )


@dataclass(frozen=True)
class ProbeResult:
    duration_seconds: float
    width: int
    height: int
    fps_num: int
    fps_den: int

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den if self.fps_den else float(self.fps_num)


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise MediaError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}"
        )
    return proc.stdout


def probe(path: str | Path) -> ProbeResult:
    path = str(path)
    out = _run(
        [
            _resolve_bin("ffprobe"), "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration:format=duration",
            "-of", "json", path,
        ]
    )
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fr = stream.get("r_frame_rate", "0/1")
    num_s, _, den_s = fr.partition("/")
    fps_num, fps_den = int(num_s or 0), int(den_s or 1)
    duration = float(
        stream.get("duration")
        or (data.get("format") or {}).get("duration")
        or 0.0
    )
    if not (width and height and fps_num and duration):
        raise MediaError(f"could not probe video stream from {path!r}")
    return ProbeResult(duration_seconds=duration, width=width, height=height,
                       fps_num=fps_num, fps_den=fps_den)


def make_canonical_clip(
    source: str | Path,
    start_seconds: float,
    duration_seconds: float,
    out_path: str | Path,
    analysis_fps: int = 15,
) -> Path:
    """Transcode an exact constant-frame-rate clip with reset timestamps.

    Uses accurate seeking (`-ss` after `-i`) and re-encodes so cuts are not
    snapped to keyframes. Output is CFR at `analysis_fps` with PTS reset to 0.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _resolve_bin("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-ss", f"{start_seconds:.6f}",
        "-t", f"{duration_seconds:.6f}",
        "-frames:v", str(int(round(duration_seconds * analysis_fps))),
        "-vf", f"fps={analysis_fps},format=yuv420p",
        "-r", str(analysis_fps),
        "-an",
        "-reset_timestamps", "1",
        # Disable B-frames so packet order == presentation order; this keeps
        # PTS strictly increasing in file order and makes the canonical-clip
        # validation below hold without sorting.
        "-bf", "0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(out_path),
    ]
    _run(cmd)
    validate_canonical_clip(out_path, duration_seconds, analysis_fps)
    return out_path


def validate_canonical_clip(
    path: str | Path, duration_seconds: float, analysis_fps: int
) -> None:
    """Enforce the canonical-clip invariants from the plan."""
    path = str(path)
    out = _run(
        [
            _resolve_bin("ffprobe"), "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "packet=pts_time:stream=nb_frames,r_frame_rate:format=duration",
            "-of", "json", path,
        ]
    )
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    nb_frames = int(stream.get("nb_frames") or 0)
    expected = int(round(duration_seconds * analysis_fps))
    if nb_frames != expected:
        raise MediaError(
            f"canonical clip frame count mismatch: got {nb_frames}, expected {expected}"
        )
    packets = data.get("packets") or []
    if packets:
        pts = sorted(float(p["pts_time"]) for p in packets if "pts_time" in p)
        first_pts = pts[0]
        if abs(first_pts) > 1e-3:
            raise MediaError(f"first PTS not zero: {first_pts}")
        for a, b in zip(pts, pts[1:]):
            if not b > a:
                raise MediaError(f"timestamps not strictly increasing at {a}->{b}")


def sha256_of_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
