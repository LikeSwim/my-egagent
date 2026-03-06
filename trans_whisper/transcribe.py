# Whisper 转写：音频 → 字幕文本 / 带时间戳的段落（用于 SRT）

from pathlib import Path


def _sec_to_srt_time(sec: float) -> str:
    """将秒数转为 SRT 时间轴格式 HH:MM:SS,mmm"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[dict], speaker_prefix: str = "Speaker") -> str:
    """
    将 Whisper 风格的段落列表转为标准 SRT 字符串。
    :param segments: [{"start": float, "end": float, "text": str, "speaker": str?}, ...]，时间单位为秒；含 speaker 时输出说话人分离
    :param speaker_prefix: 说话人标签前缀，如 "Speaker" 或 "说话人"
    :return: 标准 SRT 内容，可直接写入 .srt 文件
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        start_s = seg.get("start", 0.0)
        end_s = seg.get("end", 0.0)
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        if speaker is not None and str(speaker).strip():
            # 将 SPEAKER_00 转为 "Speaker 0" / "说话人 0"
            s = str(speaker).strip()
            if s.upper().startswith("SPEAKER_"):
                try:
                    n = int(s.split("_")[-1])
                    text = f"{speaker_prefix} {n}: {text}"
                except ValueError:
                    text = f"{speaker_prefix} {s}: {text}"
            else:
                text = f"{speaker_prefix} {s}: {text}"
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


def transcribe_audio_to_segments_diarized(
    audio_path: str | Path,
    model: str = "large-v3",
    language: str | None = "zh",
    device: str | None = None,
    hf_token: str | None = None,
    speaker_prefix: str = "说话人",
) -> list[dict] | None:
    """
    使用 WhisperX 做转写 + 说话人分离，返回带 speaker 的段落（用于生成带说话人标注的 SRT）。
    需要安装: pip install whisperx
    使用 pyannote 做 diarization 时需在 https://huggingface.co/pyannote/speaker-diarization-3.1 同意条款并传入 hf_token。
    :param audio_path: 音频文件路径（建议 16kHz 单声道 wav）
    :param model: Whisper 模型名，如 base / large-v3
    :param language: 语言代码，如 zh / en，None 为自动检测
    :param device: cuda/cpu，None 为自动
    :param hf_token: HuggingFace token（pyannote 模型需在网页同意条款后使用）
    :param speaker_prefix: 返回段落里 speaker 会原样保留，仅影响后续 segments_to_srt 的前缀
    :return: [{"start", "end", "text", "speaker"}, ...]，失败或未安装 whisperx 返回 None
    """
    try:
        import whisperx
    except ImportError:
        return None

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        return None

    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    try:
        audio = whisperx.load_audio(str(audio_path))
        if audio is None or len(audio) == 0:
            return None
        model_wx = whisperx.load_model(model, device, compute_type=compute_type)
        result = model_wx.transcribe(audio, batch_size=16)
        if not result or "segments" not in result or not result["segments"]:
            return None

        lang = result.get("language") or language or "zh"
        try:
            model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
            aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)
            result = {"segments": aligned} if isinstance(aligned, list) else (aligned if isinstance(aligned, dict) else result)
        except Exception:
            pass

        def _seg_list():
            if isinstance(result, list):
                return result
            return (result.get("segments") or []) if isinstance(result, dict) else []

        if not hf_token:
            return [
                {"start": float(s.get("start", 0)), "end": float(s.get("end", 0)), "text": (s.get("text") or "").strip()}
                for s in _seg_list()
            ]

        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=20)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        segments = _seg_list()
        out = []
        for s in segments:
            start_s = float(s.get("start", 0))
            end_s = float(s.get("end", 0))
            text = (s.get("text") or "").strip()
            if not text:
                continue
            speaker = s.get("speaker")
            out.append({"start": start_s, "end": end_s, "text": text, "speaker": speaker})
        return out if out else None
    except Exception:
        return None
