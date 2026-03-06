## 下载与准备数据集
下载 EgoLife 和 Video-MME 数据集，并在 [`paths.py`](../paths.py) 中更新 EGOLIFE_ROOT 和 VIDEO_MME_ROOT 路径。
```
cd /path/to/datasets/
git lfs install
git clone https://huggingface.co/datasets/lmms-lab/EgoLife
git clone https://huggingface.co/datasets/lmms-lab/Video-MME

# 解压 Video-MME 中的压缩文件（约 20 分钟）
bash prepare_videomme.sh
```

## 按 1 FPS 采样视频
首先，从原始长视频中以 1 帧/秒采样图像帧，用于生成字幕和实体图。
```
python sample_videos_1fps.py --dataset egolife
python sample_videos_1fps.py --dataset videomme # 约 35 分钟
```

## 融合音频转写与字幕以用于实体图抽取

<p align="center">
<img src="./../figs/eg_creation.png" width="768"/>
</p>

我们使用多模态 LLM（默认 GPT‑4.1）将原始视觉字幕与带说话人信息的音频转写进行融合。EgoLife 上默认的字幕模型为 GPT-4.1，对每段 30 秒、按 1 FPS 采样的原始视频片段进行描述。Video-MME (Long) 上默认使用 LLaVA-Video-7B（对每个视频按 1 FPS 采样的 64 张图像窗口进行描述）。对于 Video-MME，我们按 batch-start 索引对 Long 子集中的 300 个视频进行分批，以便在多 GPU 上并行处理。

### 参数说明

- **`--dataset`**：`egolife` 或 `videomme`（默认：`egolife`）
- **`--mllm`**：用于融合/摘要的多模态 LLM（默认：`gpt-4.1`）
- **`--day`**：EgoLife 的日期（1–7）。仅当 `--dataset egolife` 时有效（默认：`1`）
- **`--batch-start`**：Video-MME (Long) 中待处理长视频批次的起始索引（从 0 开始）（默认：`0`）
- **`--batch-size`**：Video-MME (Long) 中从 `--batch-start` 起处理的视频数量（默认：`50`）

### 示例

**在 EgoLife 第 1 天融合转写与字幕：**
```bash
python summarize_and_fuse_captions.py \
  --dataset egolife \
  --mllm gpt-4.1 \
  --day 1
```

**在 Video-MME (Long) 上融合转写与字幕：**

默认我们按 6 批、每批 50 个视频并行运行，即 `batch-start` = {0, 50, 100, 150, 200, 250}。例如，处理 Video-MME (Long) 中第 100–150 个视频的第三批：
```bash
python summarize_and_fuse_captions.py \
  --dataset videomme \
  --mllm gpt-4.1 \
  --batch-start 100 \
  --batch-size 50
```

## 从融合后的转写与字幕创建实体图
随后将融合后的字幕与转写转换为带时间戳的实体图，并保存为 JSON 文件。

### 参数说明

- **`--dataset`**：`egolife` 或 `videomme`（默认：`egolife`）
- **`--day`**：EgoLife 的日期索引（1–7）。仅当 `--dataset egolife` 时有效（默认：`6`）
- **`--batch-start`**：Video-MME (Long) 中待处理长视频批次的起始索引（从 0 开始）（默认：`0`）
- **`--batch-size`**：Video-MME (Long) 中从 `--batch-start` 起处理的视频数量（默认：`50`）

### 示例

**在 EgoLife 第 6 天创建实体图：**
```bash
python create_entity_graph.py \
  --dataset egolife \
  --day 6
```

**在 Video-MME (Long) 上创建实体图：**
与上述转写-字幕融合类似，由于 Video-MME 中各视频相互独立，我们通过分批对 300 个视频分别创建实体图以实现并行。例如，处理 Video-MME (Long) 中第 250–300 个视频的第五批：
```bash
python create_entity_graph.py \
  --dataset videomme \
  --batch-start 250 \
  --batch-size 50
```

## 准备实体图数据库（SQLite）
在通过 `create_entity_graph.py` 生成实体图 JSON 后，在此步骤中构建 EGAgent 推理时查询的 SQLite 表。论文中 EgoLife 使用 GPT-4.1 字幕，Video-MME (Long) 使用 LLaVA-Video-7B 字幕。

### 参数说明

- **`--dataset`**：`egolife` 或 `videomme`（默认：`egolife`）
- **`--config`**：`TIMESTAMP_EPISODES_ROOT` 下的配置目录名，例如 `fused_dt_and_gpt-4.1_captions`、`fused_dt_and_llava-video-7bcaptions`

### 示例

**EgoLife：融合的带说话人转写 + GPT-4.1 字幕**
```bash
python create_db_entity_graph.py --dataset egolife --config fused_dt_and_gpt-4.1_captions
```

**Video-MME (Long)：融合转写 + LLaVA-Video-7B 字幕**
```bash
python create_db_entity_graph.py --dataset videomme --config fused_dt_and_llava-video-7bcaptions
```

**输出说明：**
1. EgoLife → 单个 DB 文件：`DB_ROOT/entity_graph/egolife_entity_graph_{config}.db`
2. Video-MME (Long) → 300 个视频各对应一个 DB，路径为 `DB_ROOT/entity_graph/videomme/{config}/videomme_{video_id}.db`

## 准备视觉数据库（SQLite）
使用强视觉编码器（我们使用 SigLIP 2）对 1 fps 采样的帧进行嵌入，并将嵌入写入表中供 EGAgent 推理时查询。在 EgoLife 上，该过程约需 3 小时，需要足够的 CPU 内存用于从磁盘加载图像（约 40 GB）以及 GPU 内存用于嵌入批推理（2 × 24 GB = 48 GB，以 RTX 4090 为例）。

请根据你的计算资源选择数据集、嵌入批大小以及加载图像时的大致 CPU 内存上限。

- **`--dataset`**：`egolife` 或 `videomme`
- **`--batch-size`**：EgoLife 帧编码的嵌入批大小（默认：`256`）
- **`--max-ram-gb`**：分块加载 EgoLife 帧时使用的大致 CPU 内存上限（GB）（默认：`40.0`）

### EgoLife
我们按天（1 到 7）分别对帧进行嵌入并创建嵌入表，之后再将各天的表合并为一张表。
```bash
python create_db_visual_frames.py \
  --dataset egolife \
  --batch-size 256 \
  --max-ram-gb 40
```

### Video-MME (Long)
在 Video-MME 上，我们加载 Long 子集中全部 300 个视频的预计算嵌入（.npy）。请在 `paths.py` 中设置 VMME_EMBS_PATH。
```bash
python create_db_visual_frames.py --dataset videomme
```

---

## 自有数据（单 MP4，如一堂课）创建知识图谱
若你只有自有视频（一个 MP4 = 一节课），可不用 EgoLife/Video-MME 目录结构，直接使用脚本 **`create_kg_custom_video.py`**：只需提供 **MP4 + 该视频的 SRT 字幕/转写**，即可生成实体图 JSON 与 SQLite 知识图谱。

**示例：**
```bash
python create_kg_custom_video.py \
  --video   /path/to/lecture.mp4 \
  --video-id lecture_01 \
  --transcript /path/to/lecture.srt \
  --output-dir ./my_lectures
```

无 SRT 时可先用 [Whisper](https://github.com/openai/whisper) 等 ASR 从音频生成。  
**EgoLife / Video-MME 的 SRT 来源**：二者均为数据集自带（EgoLife 在 `EgoLifeCap/Transcript/`，Video-MME 在 `subtitle/`）；自有视频需自行生成 SRT，格式与二者一致（标准 SRT、时间从 00:00:00 起）。  
更多说明见 [docs/INSTALL_CN.md](../docs/INSTALL_CN.md) 中的「六、自有数据」与「七、SRT 字幕/转写从哪里来」。
