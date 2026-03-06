# 视频分段与音频抽取（依赖 ffmpeg）

import re
import subprocess
from pathlib import Path

DEFAULT_SEGMENT_DURATION_SEC = 600  # 10 分钟（与 LEMON 论文一致）


def get_video_duration_sec(video_path: str | Path, ffmpeg: str = "ffmpeg") -> float:
    """获取视频时长（秒）。失败返回 0。"""
    video_path = Path(video_path)
    try:
        out = subprocess.run(
            [ffmpeg, "-i", str(video_path), "-t", "0.1", "-f", "null", "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        dur_match = re.search(r"Duration: (\d+):(\d+):(\d+)\.?\d*", out.stderr or "")
        if dur_match:
            h, m, s = int(dur_match.group(1)), int(dur_match.group(2)), float(dur_match.group(3))
            return h * 3600 + m * 60 + s
    except Exception:
        pass
    return 0.0


def segment_video_by_duration(
    video_path: str | Path,
    output_dir: str | Path,
    duration_sec: int = DEFAULT_SEGMENT_DURATION_SEC,
    ffmpeg: str = "ffmpeg",
) -> list[dict]:
    """
    将视频按固定时长切分为多个片段。
    返回 [{"path": str, "start_sec": int, "end_sec": int, "index": int}, ...]。
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_sec = get_video_duration_sec(video_path, ffmpeg)
    if total_sec <= 0:
        return []
    segments = []
    start = 0
    idx = 0
    while start < total_sec:
        end = min(start + duration_sec, total_sec)
        out_path = output_dir / f"{video_path.stem}_seg{idx:04d}.mp4"
        try:
            subprocess.run(
                [
                    ffmpeg, "-y", "-i", str(video_path),
                    "-ss", str(start), "-t", str(end - start),
                    "-c", "copy", str(out_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            break
        segments.append({"path": str(out_path), "start_sec": start, "end_sec": end, "index": idx})
        start = end
        idx += 1
    return segments


def extract_audio(
    video_path: str | Path,
    output_path: str | Path | None = None,
    sample_rate: int = 16000,
    ffmpeg: str = "ffmpeg",
) -> str | None:
    """从视频抽取音频，16kHz 单声道。返回输出路径或 None。"""
    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(video_path),
                "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate),
                str(output_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=True,
        )
        return str(output_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
