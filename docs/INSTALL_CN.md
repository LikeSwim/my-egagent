# EGAgent 安装与配置说明（中文）

## 一、如何安装

### 1. 使用 Conda 创建环境并安装依赖

```bash
conda env create -f environment.yml
conda activate egagent
```

### 2. 配置路径与密钥

在运行任何脚本前，请先编辑项目根目录下的 **`paths.py`**，填写以下路径与 API 密钥文件位置：

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `EGOLIFE_ROOT` | EgoLife 数据集根目录 | `"path/to/EgoLife"` |
| `VIDEO_MME_ROOT` | VideoMME 数据集根目录 | `"path/to/VideoMME"` |
| `MODEL_ROOT` | 多模态嵌入模型所在目录（如 SigLIP） | 默认 `EGAGENT_ROOT` |
| `OPENAI_API_KEY_PATH` | 存放 OpenAI API Key 的文本文件路径 | `"path/to/openai-api-key.txt"` |
| `GOOGLE_GENAI_KEY_PATH` | 存放 Google GenAI Key 的文本文件路径（若用 Gemini） | `"path/to/google-genai-key.txt"` |
| `VMME_EMBS_PATH` | Video-MME 预计算嵌入的 .npy 文件路径 | `"path/to/videomme_embeddings"` |

### 3. 下载多模态嵌入模型（视觉检索用）

项目默认使用 SigLIP 2，需单独下载：

```bash
git lfs install
git clone https://huggingface.co/google/siglip2-giant-opt-patch16-384
```

将克隆后的目录放在 `MODEL_ROOT` 所指向的路径下（或修改 `paths.py` 中的 `MODEL_ROOT` 指向该目录）。

---

## 二、需要提供的资源

- **运行环境**：Python 3.10、Conda；需 CUDA 以使用 PyTorch 与嵌入模型。
- **数据集**（二选一或都备）：
  - [EgoLife](https://huggingface.co/datasets/lmms-lab/EgoLife)
  - [Video-MME (Long)](https://huggingface.co/datasets/lmms-lab/Video-MME)
- **API 密钥**（按所用模型准备）：
  - 使用 OpenAI 系列（如 gpt-4.1、gpt-4o）：在 `OPENAI_API_KEY_PATH` 指向的文件中写入 OpenAI API Key。
  - 使用 Gemini：在 `GOOGLE_GENAI_KEY_PATH` 指向的文件中写入 Google GenAI Key。
- **多模态嵌入模型**：如 SigLIP 2（见上一节）。
- **预计算数据**（按流程需要）：
  - 运行 `prepare_datasources` 会生成视觉检索 DB、实体图等；Video-MME 需准备 `VMME_EMBS_PATH` 下的嵌入。

---

## 三、如何把模型改成第三方支持 OpenAI 格式的 API

项目已支持通过 **OpenAI 兼容接口** 使用第三方模型（任何提供 `/v1/chat/completions` 等 OpenAI 风格接口的服务）。

### 1. 在 `paths.py` 中配置第三方 API

在 `paths.py` 中新增或修改以下变量（若文件中没有，可手动添加）：

```python
# 第三方 OpenAI 兼容 API（可选）
THIRD_PARTY_OPENAI_BASE_URL = "https://your-api.com/v1"   # 第三方 API 的 base URL
THIRD_PARTY_OPENAI_MODEL = "your-model-name"              # 第三方模型名称，如 gpt-4o、qwen-vl 等
THIRD_PARTY_OPENAI_KEY = "your-api-key"                   # 第三方 API 的 Key（直接填写；勿提交到 git，可用环境变量或本地占位）
```

- `THIRD_PARTY_OPENAI_BASE_URL`：必填，例如 `https://api.xxx.com/v1`（注意保留末尾 `/v1`）。
- `THIRD_PARTY_OPENAI_MODEL`：该第三方服务里的模型名。
- `THIRD_PARTY_OPENAI_KEY`：第三方 API 的密钥字符串，直接填写；生产环境建议用环境变量注入，避免写死在代码里。

### 2. 在推理时使用第三方模型

- **EGAgent 推理**：编辑 **`egagent/langgraph_agent.py`**，将顶部的 `agent_backbone` 改为：

  ```python
  agent_backbone = 'openai_compatible'   # 使用 paths.py 中配置的第三方 OpenAI 兼容 API
  ```

- **prepare_datasources**（如实体图、摘要等）：在调用 `get_llm_worker` 或等价逻辑时，将 `model` 参数改为 `'openai_compatible'`，即会使用上述同一套配置。

这样，所有通过 `get_llm_worker` / `get_vision_llm` 等走 OpenAI 兼容路径的调用，都会使用你在 `paths.py` 里配置的 **base_url + model + api_key**，从而接入第三方支持 OpenAI 格式的 API。

**仅用第三方 API、不配置 OpenAI/Google 时**：项目已改为“按需读取” API key，只要不调用官方 OpenAI 或 Gemini，就不会去读 `OPENAI_API_KEY_PATH` / `GOOGLE_GENAI_KEY_PATH`。你只需在 `paths.py` 里正确填写 `THIRD_PARTY_OPENAI_BASE_URL`、`THIRD_PARTY_OPENAI_MODEL`、`THIRD_PARTY_OPENAI_KEY`（第三方 API 的密钥），并在运行脚本时指定使用第三方模型（见下）。  
**自有视频知识图谱**：运行 `create_kg_custom_video.py` 时加上 `--mllm openai_compatible`，即会用上述第三方配置。

---

## 四、运行示例

- **EgoLifeQA**（500 道选择题）：
  ```bash
  python egagent/run_egagent_on_egolife.py --tscript-search llm
  ```
- **VideoMME-long**（900 道选择题）：
  ```bash
  python egagent/run_egagent_on_videomme.py --tscript-search llm
  ```

更多数据准备与 baseline 说明见 [prepare_datasources/README.md](../prepare_datasources/README.md)、[egagent/README.md](../egagent/README.md) 和 [baselines/README.md](../baselines/README.md)。

---

## 五、如何抽取视频知识图谱，知识图谱保存在哪里

### 1. 抽取流程概览

视频的**实体图（Entity Graph）**即项目中的“知识图谱”，由以下三步得到：

1. **1 FPS 采样** → 用 `sample_videos_1fps.py` 从长视频中按 1 帧/秒采样图像。
2. **字幕与转写融合** → 用 `summarize_and_fuse_captions.py` 将视觉 caption 与音频转写（含说话人）用多模态 LLM 融合。
3. **实体图抽取** → 用 `create_entity_graph.py` 从融合后的文本中抽取实体与关系，并用 LLM 为关系打时间戳，得到带时间戳的实体图 JSON。
4. **写入 SQLite** → 用 `create_db_entity_graph.py` 将上述 JSON 导入 SQLite，供 EGAgent 推理时查询。

### 2. 具体命令（按顺序）

**EgoLife 示例（以 day 6 为例）：**

```bash
# 若尚未做：采样 + 融合（先做 day 1–7 的融合）
python prepare_datasources/sample_videos_1fps.py --dataset egolife
python prepare_datasources/summarize_and_fuse_captions.py --dataset egolife --mllm gpt-4.1 --day 6

# 从融合结果抽取实体图（生成 JSON）
python prepare_datasources/create_entity_graph.py --dataset egolife --day 6

# 将 JSON 转为 SQLite（生成最终知识图谱 DB）
python prepare_datasources/create_db_entity_graph.py --dataset egolife --config fused_dt_and_gpt-4.1_captions
```

**Video-MME 示例（按 batch 并行）：**

```bash
python prepare_datasources/sample_videos_1fps.py --dataset videomme
python prepare_datasources/summarize_and_fuse_captions.py --dataset videomme --mllm gpt-4.1 --batch-start 0 --batch-size 50
python prepare_datasources/create_entity_graph.py --dataset videomme --batch-start 0 --batch-size 50
python prepare_datasources/create_db_entity_graph.py --dataset videomme --config fused_dt_and_llava-video-7bcaptions
```

### 3. 知识图谱保存位置

| 阶段 | 路径（均在项目根目录下） | 说明 |
|------|---------------------------|------|
| **中间 JSON（实体图 + 时间戳）** | `timestamp_episodes/{config}/egolife/` 或 `timestamp_episodes/{config}/videomme/` | EgoLife：`day{d}_hour{h}.json`；Video-MME：`{video_id}.json`。关系中间结果在各自 `relationships/` 子目录。 |
| **最终 SQLite（供推理查询）** | `dbs/entity_graph/` | EgoLife：单文件 `dbs/entity_graph/egolife_entity_graph_{config}.db`。Video-MME：每个视频一个库 `dbs/entity_graph/videomme/{config}/videomme_{video_id}.db`。 |

上述路径由 `paths.py` 中的 `TIMESTAMP_EPISODES_ROOT`（默认 `timestamp_episodes`）和 `DB_ROOT`（默认 `dbs`）决定；**实际被 EGAgent 使用的是 `DB_ROOT` 下的 SQLite 文件**，表名为 `entity_graph_table`。

---

## 六、自有数据（如一堂课 MP4）如何创建知识图谱

若你只有**自有视频**（例如一个 MP4 = 一堂课），不依赖 EgoLife/Video-MME 的目录结构，可以用脚本 **`prepare_datasources/create_kg_custom_video.py`** 从**单视频 + 字幕/转写**直接生成知识图谱。

### 1. 需要准备的内容

- **一个 MP4 视频**：例如一节课的录像。
- **该视频对应的 SRT 字幕/转写**：与视频时间轴一致（若没有，可先用 [Whisper](https://github.com/openai/whisper) 等 ASR 从音频生成 SRT，再与本脚本配合使用）。

### 2. 一条命令完成：1fps 抽帧 + 实体图抽取 + SQLite

在项目根目录下执行（或在 `prepare_datasources` 下执行时把脚本路径改为 `create_kg_custom_video.py`）：

```bash
python prepare_datasources/create_kg_custom_video.py \
  --video   /path/to/your/lecture.mp4 \
  --video-id lecture_01 \
  --transcript /path/to/lecture.srt \
  --output-dir ./my_lectures
```

- `--video`：MP4 路径。
- `--video-id`：用于生成子目录和 DB 文件名（如 `lecture_01`），不填则用视频文件名（不含后缀）。
- `--transcript`：SRT 路径（**必填**）；内容将用于抽取实体与关系并为关系打时间戳。
- `--output-dir`：输出根目录；不填则默认为项目下的 `custom_lectures`。
- `--mllm`：打时间戳用的 LLM（默认 `gpt-4.1`，可改为 `openai_compatible` 等，需在 `paths.py` 中配置好第三方 API）。
- `--skip-1fps`：若已抽过 1fps 帧可加此参数跳过抽帧。
- `--skip-db`：只生成实体图 JSON，不生成 SQLite。

脚本会依次完成：从 MP4 按 1 帧/秒抽帧 → 从 SRT 转写抽取实体图并打时间戳 → 将结果写入 SQLite（与项目内实体图 DB 同 schema）。

### 3. 自有数据的输出位置（默认 `--output-dir ./my_lectures`）

| 内容 | 路径 |
|------|------|
| 1fps 帧图像 | `my_lectures/video_1fps/lecture_01/`（000001.jpg, 000002.jpg, ...） |
| 实体图 JSON | `my_lectures/timestamp_episodes/custom_transcript_only/custom/lecture_01.json` |
| 知识图谱 SQLite | `my_lectures/dbs/entity_graph/custom/custom_transcript_only/custom_lecture_01.db` |

表名为 `entity_graph_table`，与 EgoLife/Video-MME 的实体图 DB 一致，便于后续对接 EGAgent 的实体图检索（需在推理配置中指定该 DB 路径）。

---

## 七、SRT 字幕/转写从哪里来：EgoLife、Video-MME 与自有视频

### 1. EgoLife 的 SRT 字幕/转写

- **来源**：**随 EgoLife 数据集一起提供**，无需自行生成。下载 [EgoLife](https://huggingface.co/datasets/lmms-lab/EgoLife) 并解压后即可在指定目录下找到。
- **路径**（在 `paths.py` 中 `EGOLIFE_ROOT` 指向数据集根目录时）：
  - `{EGOLIFE_ROOT}/EgoLifeCap/Transcript/A1_JAKE/DAY1/` … `DAY7/`
  - 每个 DAY 目录下有多段 SRT，文件名形如 `DAY4_11000000.srt`（数字为时段起始时间，单位百分秒 HHMMSScc）。
- **格式与内容**：
  - 标准 SRT（序号 + 时间轴 `HH:MM:SS,mmm --> HH:MM:SS,mmm` + 文本行）。
  - 文本行为**带说话人标注的英文转写**（diarized），例如 `Jake: ...`、`Tasha: ...`。项目会按需过滤中文、去除说话人前缀等。
- **数据集说明**：转写由 EgoLife 官方 pipeline 生成（ASR + 说话人分离等），详见 [EgoLife 官网](https://egolife-ai.github.io/) 与论文 [EgoLife (arXiv:2503.03803)](https://arxiv.org/abs/2503.03803)。

### 2. Video-MME 的 SRT 字幕/转写

- **来源**：**随 Video-MME 数据集一起提供**，无需自行生成。下载 [Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME) 并按要求解压/放置后即可使用。
- **路径**（`paths.py` 中 `VMME_ASR_DIR = VIDEO_MME_ROOT + "subtitle"`）：
  - `{VIDEO_MME_ROOT}/subtitle/{video_id}.srt`
  - 每个 long 视频一个 SRT，`video_id` 与数据集中的视频 ID 一致。
- **格式与内容**：
  - 标准 SRT，时间轴为**视频内相对时间**（从 00:00:00 起），与视频逐句对齐；多为英文或与视频语言一致的字幕/转写。

### 3. 自有视频的 SRT 如何获得（参考 EgoLife / Video-MME）

自有视频（如一堂课 MP4）**没有现成数据集**，需要自己生成与视频时间轴一致的 SRT，再交给 `create_kg_custom_video.py` 使用。做法上可参考 EgoLife/Video-MME：**标准 SRT + 与视频对齐的时间轴**。

#### 推荐方式：用 ASR 从音频生成 SRT

1. **从 MP4 提取音频**（若工具不直接读 MP4）：
   ```bash
   ffmpeg -i your_lecture.mp4 -vn -acodec copy audio.wav
   ```
2. **用语音识别（ASR）生成带时间戳的转写**，并导出为 **标准 SRT**：
   - **[Whisper](https://github.com/openai/whisper)**（OpenAI）：支持多语言，可直接输出 SRT。
     ```bash
     whisper audio.wav --output_format srt --language zh  # 中文课用 zh，英文课可省略或 en
     ```
   - 其他支持「带时间戳转写 → SRT」的工具（如 [whisperX](https://github.com/m-bain/whisperX)、[FunASR](https://github.com/alibaba-damo-academy/FunASR)、各云厂商 ASR 的 SRT 导出）也可，只要输出为标准 SRT 即可。

#### 自有 SRT 需满足的格式（与 EgoLife/Video-MME 一致）

- **标准 SRT**：每条为「序号 + 空行 + 时间轴行 + 一行或多行文本 + 空行」。
- **时间轴**：`HH:MM:SS,mmm --> HH:MM:SS,mmm`（与项目内 `load_srt` / `load_srt_hhmmss` 使用的格式一致）。
- **时间语义**：从 00:00:00 起的**视频内相对时间**，与你的 MP4 播放进度一致。
- **可选**：若有多人（如老师/学生），可在文本行前加说话人，如 `Teacher: 今天讲…`；当前自有流程会保留整行文本参与实体图抽取，不做强制要求。

#### 简要对比

| 项目 | EgoLife | Video-MME | 自有视频 |
|------|--------|-----------|----------|
| SRT 来源 | 数据集自带 | 数据集自带 | 自行用 ASR 等生成 |
| 路径 | `EgoLifeCap/Transcript/A1_JAKE/DAYd/` | `subtitle/{video_id}.srt` | 任意，通过 `--transcript` 传入 |
| 格式 | 标准 SRT，多段文件，带说话人 | 标准 SRT，单文件/视频 | 标准 SRT，单文件/视频 |
| 时间 | 每段文件名含偏移，内容为时段内时间 | 相对时间 00:00:00 起 | 建议相对时间 00:00:00 起 |

按上述方式准备好自有视频的 SRT 后，直接使用「六、自有数据如何创建知识图谱」中的 `create_kg_custom_video.py` 命令即可。
