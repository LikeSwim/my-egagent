#!/usr/bin/env python3
# 命令行入口：讲座视频 → 分段 → 抽音频 → Whisper 得字幕

import argparse
from pathlib import Path

from trans_whisper.pipeline import video_to_transcripts


def main():
    parser = argparse.ArgumentParser(
        description="讲座视频 → 分段 → 抽音频 → Whisper 得字幕",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=str, help="讲座视频路径（mp4 等）")
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认在视频同目录下建 <视频名>_whisper",
    )
    parser.add_argument(
        "--segment-duration",
        type=int,
        default=600,
        help="每段时长（秒），默认 600（10 分钟）",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper 模型",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="语言代码，如 en/zh，设为 none 表示自动检测",
    )
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="ffmpeg 可执行路径",
    )
    parser.add_argument(
        "--no-keep-segments",
        action="store_true",
        help="转写完成后删除分段视频以节省空间",
    )
    parser.add_argument(
        "--no-keep-audio",
        action="store_true",
        help="转写完成后删除 wav 以节省空间",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="启用说话人分离（需安装 whisperx，且建议传入 --hf-token 以使用 pyannote 模型）",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token，用于 pyannote 说话人分离（在 hf.co/pyannote/speaker-diarization-3.1 同意条款后获取）",
    )
    parser.add_argument(
        "--speaker-prefix",
        type=str,
        default="说话人",
        help="SRT 中说话人标签前缀，如「说话人」或「Speaker」",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"错误：视频文件不存在 {video_path}")
        return 1

    output_dir = args.output_dir or str(video_path.parent / f"{video_path.stem}_whisper")
    language = None if args.language.lower() == "none" else args.language

    print(f"输入视频: {video_path}")
    print(f"输出目录: {output_dir}")
    print(f"分段时长: {args.segment_duration}s | Whisper: {args.whisper_model} | 语言: {language or '自动'}" + (" | 说话人分离: 开" if args.diarize else ""))
    print("开始处理...")

    results = video_to_transcripts(
        video_path,
        output_dir,
        segment_duration_sec=args.segment_duration,
        whisper_model=args.whisper_model,
        language=language,
        ffmpeg=args.ffmpeg,
        keep_segments=not args.no_keep_segments,
        keep_audio=not args.no_keep_audio,
        diarize=args.diarize,
        hf_token=args.hf_token,
        speaker_prefix=args.speaker_prefix,
    )

    if not results:
        print("未生成任何分段或字幕，请检查视频与 ffmpeg（需能正确解析时长）。")
        return 1

    srt_path = Path(output_dir) / f"{video_path.stem}.srt"
    srt_only = results and results[0].get("srt_only") is True
    if srt_only:
        print(f"完成。已生成视频配套 SRT（分段未执行，可能因无法解析视频时长）:")
        print(f"  {srt_path}")
        return 0
    print(f"完成。共 {len(results)} 段，字幕目录: {output_dir}/transcripts/")
    print(f"合并字幕: {output_dir}/all_transcript.txt")
    if srt_path.exists():
        print(f"视频配套 SRT: {srt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
