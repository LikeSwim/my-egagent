# 视频分段与音频抽取（依赖 ffmpeg）

import re
import subprocess
import sys
from pathlib import Path

DEFAULT_SEGMENT_DURATION_SEC = 600  # 10 分钟（与 LEMON 论文一致）


def get_video_duration_sec(video_path: str | Path, ffmpeg: str = "ffmpeg") -> float:
    """获取视频时长（秒）。先尝试 ffprobe，再回退到解析 ffmpeg 输出。失败返回 0。"""
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        return 0.0

    # 1. 优先用 ffprobe（输出稳定，不依赖 locale）
    if ffmpeg == "ffmpeg" or ffmpeg == "ffprobe":
        ffprobe = "ffprobe"
    else:
        p = Path(ffmpeg).parent
        ffprobe = str(p / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe"))
    try:
        out = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if out.returncode == 0 and out.stdout:
            for line in out.stdout.strip().splitlines():
                s = line.strip().strip("\ufeff")
                if not s:
                    continue
                try:
                    v = float(s.replace(",", "."))
                    if v >= 0:
                        return v
                except ValueError:
                    pass
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass

    # 2. 回退：解析 ffmpeg -i 的 stderr（兼容 Duration: 00:01:23.45 或 00:01:23,45）
    try:
        out = subprocess.run(
            [ffmpeg, "-i", str(video_path), "-t", "0.1", "-f", "null", "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        combined = (out.stderr or "") + (out.stdout or "")
        # 匹配 Duration: HH:MM:SS.ms 或 HH:MM:SS,ms（放宽空格）
        dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)[.,]?\d*", combined, re.IGNORECASE)
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
    verbose: bool = False,
) -> list[dict]:
    """
    将视频按固定时长切分为多个片段。
    返回 [{"path": str, "start_sec": int, "end_sec": int, "index": int}, ...]。
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_sec = get_video_duration_sec(video_path, ffmpeg)
    if verbose:
        print(f"  视频时长: {total_sec:.1f} 秒" if total_sec > 0 else "  视频时长: 无法解析（请检查 ffprobe/ffmpeg 及视频文件）")
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
