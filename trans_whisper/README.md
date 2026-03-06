# 讲座视频 → 分段 → 抽音频 → Whisper 得字幕，并生成视频配套 SRT

将一条讲座视频按固定时长切段、每段抽取音频并用 OpenAI Whisper 转写为字幕；同时对**整段视频**做一次带时间戳的转写，生成**标准 SRT 文件**（与 EGAgent 自有视频知识图谱流程兼容）。

## 依赖

- **ffmpeg**：分段与抽音频（需已安装并加入 PATH）
- **openai-whisper**：`pip install openai-whisper`（会安装 PyTorch）

## 用法

### 命令行（推荐）

在项目根目录执行：

```bash
# 默认：10 分钟一段，Whisper base 模型，输出到 <视频名>_whisper/
python -m trans_whisper.run path/to/lecture.mp4

# 指定输出目录
python -m trans_whisper.run path/to/lecture.mp4 -o output/lecture_01

# 5 分钟一段、使用 large-v3、中文
python -m trans_whisper.run lecture.mp4 -o out --segment-duration 300 --whisper-model large-v3 --language zh

# 转写后删除分段视频和 wav 以节省空间
python -m trans_whisper.run lecture.mp4 -o out --no-keep-segments --no-keep-audio
```

### 输出结构

```
<output_dir>/
├── <视频名>.srt       # 视频配套 SRT（整段转写、标准时间轴，可直接用于 create_kg_custom_video）
├── segments/         # 分段视频（可选保留）
│   ├── lecture_seg0000.mp4
│   └── ...
├── audio/            # 每段抽取的 wav（可选保留）
│   ├── lecture_seg0000.wav
│   └── ...
├── transcripts/      # 每段字幕
│   ├── seg0000.txt
│   ├── seg0001.txt
│   └── ...
└── all_transcript.txt # 合并后的全文字幕
```

### 在代码中调用

```python
from trans_whisper.pipeline import video_to_transcripts

results = video_to_transcripts(
    "path/to/lecture.mp4",
    "path/to/output_dir",
    segment_duration_sec=600,
    whisper_model="base",
    language="en",
)
for r in results:
    print(r["segment_index"], r["start_sec"], r["transcript"][:100])
```

## 与 LEMON 衔接

得到字幕后，可用 LEMON 的 build 从字幕生成 QA：

```bash
python run_lemon_full.py build --transcript-file <output_dir>/all_transcript.txt -o data/lemon_qa.jsonl
```

或对每一段分别生成 QA（循环调用 build，每段使用 `transcripts/seg0000.txt` 等）。

## 与 EGAgent 自有视频知识图谱衔接

生成的 **`<视频名>.srt`** 为标准 SRT（时间从 00:00:00 起），可直接作为 `create_kg_custom_video.py` 的 `--transcript` 输入：

```bash
python -m trans_whisper.run path/to/lecture.mp4 -o ./my_lectures
python prepare_datasources/create_kg_custom_video.py \
  --video path/to/lecture.mp4 --video-id lecture_01 \
  --transcript ./my_lectures/lecture.srt --output-dir ./my_lectures
```
