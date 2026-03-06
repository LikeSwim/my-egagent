# 讲座视频 → 分段 → 抽音频 → Whisper 得字幕（完整流水线），并生成视频配套 SRT

from pathlib import Path

from trans_whisper.segment import segment_video_by_duration, extract_audio
from trans_whisper.transcribe import transcribe_audio, transcribe_audio_to_segments, segments_to_srt


def video_to_transcripts(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    segment_duration_sec: int = 600,
    whisper_model: str = "base",
    language: str | None = "en",
    ffmpeg: str = "ffmpeg",
    keep_segments: bool = True,
    keep_audio: bool = False,
) -> list[dict]:
    """
    输入讲座视频，输出每段的字幕及合并字幕文件。

    :param video_path: 讲座视频路径
    :param output_dir: 输出目录（下建 segments/、audio/、transcripts/ 及 all_transcript.txt）
    :param segment_duration_sec: 每段时长（秒），默认 600（10 分钟）
    :param whisper_model: Whisper 模型，如 base / small / medium / large-v3
    :param language: 语言代码，如 en / zh，None 为自动检测
    :param ffmpeg: ffmpeg 可执行路径
    :param keep_segments: 是否保留分段视频文件
    :param keep_audio: 是否保留抽取的 wav 文件
    :return: [{"segment_index": int, "path": str, "start_sec": int, "end_sec": int, "transcript": str, "transcript_file": str}, ...]
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_segments = output_dir / "segments"
    out_audio = output_dir / "audio"
    out_audio.mkdir(parents=True, exist_ok=True)
    out_transcripts = output_dir / "transcripts"
    out_transcripts.mkdir(parents=True, exist_ok=True)

    # 0. 整段视频抽音 → Whisper 带时间戳转写 → 生成视频配套 SRT（与 create_kg_custom_video 等兼容）
    full_audio = out_audio / f"{video_path.stem}_full.wav"
    srt_path = output_dir / f"{video_path.stem}.srt"
    if extract_audio(video_path, output_path=full_audio, ffmpeg=ffmpeg):
        segs = transcribe_audio_to_segments(full_audio, model=whisper_model, language=language)
        if segs:
            srt_path.write_text(segments_to_srt(segs), encoding="utf-8")
        if not keep_audio:
            try:
                full_audio.unlink(missing_ok=True)
            except Exception:
                pass

    # 1. 分段
    segments = segment_video_by_duration(
        video_path, out_segments, duration_sec=segment_duration_sec, ffmpeg=ffmpeg
    )
    if not segments:
        # 分段失败（如无法解析时长）时，若已生成 SRT 仍视为部分成功
        if srt_path.exists():
            return [{"segment_index": 0, "path": "", "start_sec": 0, "end_sec": 0, "transcript": "", "transcript_file": "", "srt_path": str(srt_path), "srt_only": True}]
        return []

    results = []
    all_lines = []

    for seg in segments:
        idx = seg["index"]
        seg_path = seg["path"]
        start_sec, end_sec = seg["start_sec"], seg["end_sec"]

        # 2. 抽音频
        wav_path = out_audio / f"{video_path.stem}_seg{idx:04d}.wav"
        out_audio.mkdir(parents=True, exist_ok=True)
        audio_file = extract_audio(seg_path, output_path=wav_path, ffmpeg=ffmpeg)
        if not audio_file:
            results.append({
                "segment_index": idx,
                "path": seg_path,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "transcript": "",
                "transcript_file": "",
            })
            continue

        # 3. Whisper 转写
        transcript = transcribe_audio(audio_file, model=whisper_model, language=language)
        txt_path = out_transcripts / f"seg{idx:04d}.txt"
        txt_path.write_text(transcript, encoding="utf-8")

        results.append({
            "segment_index": idx,
            "path": seg_path,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "transcript": transcript,
            "transcript_file": str(txt_path),
        })
        all_lines.append(f"[Segment {idx} | {start_sec}s - {end_sec}s]\n{transcript}")

        if not keep_audio:
            try:
                Path(audio_file).unlink(missing_ok=True)
            except Exception:
                pass
        if not keep_segments:
            try:
                Path(seg_path).unlink(missing_ok=True)
            except Exception:
                pass

    # 4. 合并字幕
    merged_path = output_dir / "all_transcript.txt"
    merged_path.write_text("\n\n".join(all_lines), encoding="utf-8")

    return results
