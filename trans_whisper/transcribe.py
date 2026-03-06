# Whisper 转写：音频 → 字幕文本 / 带时间戳的段落（用于 SRT）

from pathlib import Path


def _sec_to_srt_time(sec: float) -> str:
    """将秒数转为 SRT 时间轴格式 HH:MM:SS,mmm"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    """
    将 Whisper 风格的段落列表转为标准 SRT 字符串。
    :param segments: [{"start": float, "end": float, "text": str}, ...]，时间单位为秒
    :return: 标准 SRT 内容，可直接写入 .srt 文件
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        start_s = seg.get("start", 0.0)
        end_s = seg.get("end", 0.0)
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{_sec_to_srt_time(start_s)} --> {_sec_to_srt_time(end_s)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n" if lines else ""


def transcribe_audio(
    audio_path: str | Path,
    model: str = "base",
    language: str | None = "en",
    device: str | None = None,
) -> str:
    """
    使用 OpenAI Whisper 将音频转写为文本。
    :param audio_path: 音频文件路径（wav/mp3 等）
    :param model: Whisper 模型名，如 base / small / medium / large-v3
    :param language: 语言代码，如 en/zh，None 表示自动检测
    :param device: cuda/cpu，None 表示自动
    :return: 转写文本，失败返回空字符串
    """
    segs = transcribe_audio_to_segments(audio_path, model=model, language=language, device=device)
    if not segs:
        return ""
    return " ".join(s.get("text", "").strip() for s in segs).strip()


def transcribe_audio_to_segments(
    audio_path: str | Path,
    model: str = "base",
    language: str | None = "en",
    device: str | None = None,
) -> list[dict]:
    """
    使用 Whisper 将音频转写为带时间戳的段落，可用于生成 SRT。
    :param audio_path: 音频文件路径（wav/mp3 等）
    :param model: Whisper 模型名
    :param language: 语言代码，None 表示自动检测
    :param device: cuda/cpu，None 表示自动
    :return: [{"start": float, "end": float, "text": str}, ...]，失败返回 []
    """
    try:
        import whisper as _whisper
    except ImportError:
        raise ImportError("请安装 openai-whisper: pip install openai-whisper") from None

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        return []

    model_w = _whisper.load_model(model, device=device)
    kwargs = {"language": language} if language else {}
    result = model_w.transcribe(str(audio_path), **kwargs)
    raw = result.get("segments") or []
    return [
        {"start": float(s.get("start", 0)), "end": float(s.get("end", 0)), "text": (s.get("text") or "").strip()}
        for s in raw
    ]
