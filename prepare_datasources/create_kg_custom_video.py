# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc/4.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
从自有单视频（如一堂课 MP4 + 字幕 SRT）创建知识图谱（实体图）并写入 SQLite。
适用于：一个 MP4 文件对应一节课/一个讲座等场景。

用法示例：
  python create_kg_custom_video.py --video path/to/lecture.mp4 --video-id lecture_01 --transcript path/to/lecture.srt --output-dir ./my_lectures
"""

import argparse
import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import EGAGENT_ROOT
from utils import load_srt_hhmmss, load_srt_only_text


def extract_1fps(video_path: Path, out_dir: Path) -> None:
    """从 MP4 按 1 帧/秒抽取图像到 out_dir，命名为 000001.jpg, 000002.jpg, ..."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", "fps=1",
        str(out_dir / "%06d.jpg"),
        "-hide_banner", "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    print(f"已抽取 1fps 帧到: {out_dir}")


def build_db_from_entity_graph_json(json_path: Path, db_path: Path, video_id: str) -> int:
    """
    从实体图 JSON 构建 SQLite（与 create_db_entity_graph 中 videomme 的 schema 一致）。
    返回插入行数。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("relationships"), list):
        rels = data["relationships"]
    elif isinstance(data, list):
        rels = data
    else:
        raise ValueError(f"不支持的 JSON 结构: {json_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS entity_graph_table (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT,
        start_t TEXT,
        end_t TEXT,
        transcript TEXT,
        source_id TEXT,
        source_type TEXT,
        target_id TEXT,
        target_type TEXT,
        rel_type TEXT
    )
    """)

    insert_sql = """
    INSERT INTO entity_graph_table
      (video_id, start_t, end_t, transcript, source_id, source_type, target_id, target_type, rel_type)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    total = 0
    for rel in rels:
        source_id = rel.get("source_id")
        source_type = rel.get("source_type")
        target_id = rel.get("target_id")
        target_type = rel.get("target_type")
        rel_type = rel.get("rel_type")
        intervals = rel.get("intervals") or []
        for interval in intervals:
            start_s = interval.get("start_t")
            end_s = interval.get("end_t")
            transcript = interval.get("transcript") or interval.get("text") or None
            cur.execute(insert_sql, (
                video_id, start_s, end_s, transcript,
                source_id, source_type, target_id, target_type, rel_type,
            ))
            total += 1

    cur.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON entity_graph_table(video_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_start ON entity_graph_table(start_t)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_end ON entity_graph_table(end_t)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_video_id_start ON entity_graph_table(video_id, start_t)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON entity_graph_table(source_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_target ON entity_graph_table(target_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_source_type ON entity_graph_table(source_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_target_type ON entity_graph_table(target_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON entity_graph_table(rel_type)")
    conn.commit()
    conn.close()
    return total


async def run_entity_graph_for_custom(
    video_id: str,
    transcript_path: Path,
    output_json: Path,
    mllm: str = "gpt-4.1",
) -> None:
    """对单视频的转写文本抽取实体图并打时间戳，保存为 JSON。"""
    from create_entity_graph import (
        generate_graph_for_hour,
        parse_relationships,
        get_rel_timestamper_llm,
        attach_transcripts_to_videomme_graph,
    )

    if not transcript_path.exists():
        raise FileNotFoundError(f"转写文件不存在: {transcript_path}。请提供 SRT 或先用 ASR（如 Whisper）生成。")

    subtitles_only_text = load_srt_only_text(str(transcript_path))
    subtitles_with_timestamps = load_srt_hhmmss(str(transcript_path))

    if not subtitles_only_text.strip():
        raise ValueError("转写内容为空，无法抽取实体图。请检查 SRT 文件。")

    rel_outfile = output_json.parent / "relationships" / f"{video_id}_relationships.json"
    rel_outfile.parent.mkdir(parents=True, exist_ok=True)

    # 1) 从转写文本抽取实体与关系
    if not rel_outfile.exists():
        print("正在从转写文本抽取实体与关系...")
        graph_documents = await generate_graph_for_hour(subtitles_only_text)
        relationships_parsed = parse_relationships(graph_documents)
        with open(rel_outfile, "w", encoding="utf-8") as f:
            json.dump(relationships_parsed, f, indent=4, ensure_ascii=False)
    else:
        with open(rel_outfile, "r", encoding="utf-8") as f:
            relationships_parsed = json.load(f)

    # 仅转写、无视觉 caption 时使用 diarized_transcripts_only 的 prompt
    rel_timestamper = get_rel_timestamper_llm(model=mllm, config="diarized_transcripts_only")
    rel_lookup = {r["rel_id"]: r for r in relationships_parsed}

    # 2) 为关系打时间戳并附着转写
    rel_with_timestamps = rel_timestamper.invoke({
        "relationships": relationships_parsed,
        "transcripts": subtitles_with_timestamps,
    })
    rel_timestamped_dicts = [item.model_dump() for item in rel_with_timestamps.relationships]
    for entry in rel_timestamped_dicts:
        rel_id = entry["relationship_id"]
        if rel_id in rel_lookup:
            r = rel_lookup[rel_id]
            entry["source_id"] = r["source_id"]
            entry["source_type"] = r["source_type"]
            entry["target_id"] = r["target_id"]
            entry["target_type"] = r["target_type"]
            entry["rel_type"] = r["rel_type"]
    updated_graph = attach_transcripts_to_videomme_graph(rel_timestamped_dicts, subtitles_with_timestamps)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(updated_graph, f, indent=4, ensure_ascii=False)
    print(f"已保存实体图 JSON: {output_json}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="从自有单视频（MP4 + 字幕 SRT）创建知识图谱并写入 SQLite。"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="MP4 视频路径，例如 path/to/lecture.mp4",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="视频 ID，用于目录与 DB 命名（默认使用视频文件名不含后缀）",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        required=True,
        help="SRT 字幕/转写文件路径（需与视频时间轴对应）。若无，可先用 Whisper 等 ASR 生成。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出根目录（默认：项目下的 custom_lectures）。将在此目录下生成 video_1fps、timestamp_episodes、dbs。",
    )
    parser.add_argument(
        "--mllm",
        type=str,
        default="gpt-4.1",
        help="用于关系时间戳的多模态 LLM（默认 gpt-4.1，可改为 openai_compatible 等）",
    )
    parser.add_argument(
        "--skip-1fps",
        action="store_true",
        help="若已抽取过 1fps 帧，可跳过帧抽取步骤",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="仅生成实体图 JSON，不生成 SQLite",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")
    transcript_path = Path(args.transcript).resolve()

    video_id = args.video_id or video_path.stem
    if args.output_dir:
        output_root = Path(args.output_dir).resolve()
    else:
        output_root = EGAGENT_ROOT / "custom_lectures"

    config = "custom_transcript_only"
    frames_dir = output_root / "video_1fps" / video_id
    json_dir = output_root / "timestamp_episodes" / config / "custom"
    output_json = json_dir / f"{video_id}.json"
    db_dir = output_root / "dbs" / "entity_graph" / "custom" / config
    db_path = db_dir / f"custom_{video_id}.db"

    # 1) 抽取 1fps 帧（供后续视觉检索等使用，本脚本主要产出知识图谱）
    if not args.skip_1fps:
        extract_1fps(video_path, frames_dir)
    else:
        print("跳过 1fps 抽取（--skip-1fps）")

    # 2) 从转写抽取实体图并保存 JSON
    asyncio.run(run_entity_graph_for_custom(
        video_id=video_id,
        transcript_path=transcript_path,
        output_json=output_json,
        mllm=args.mllm,
    ))

    # 3) 写入 SQLite
    if not args.skip_db:
        n = build_db_from_entity_graph_json(output_json, db_path, video_id)
        print(f"已写入 SQLite: {db_path}（共 {n} 条记录）")

    print("完成。知识图谱 JSON 与 SQLite 路径见上。")


if __name__ == "__main__":
    main()
