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
THIRD_PARTY_OPENAI_BASE_URL = "https://your-api.com/v1"   # 第三方 API 的 base URL，例如 https://api.openai.com/v1
THIRD_PARTY_OPENAI_MODEL = "your-model-name"              # 第三方模型名称，如 gpt-4o、qwen-vl 等
THIRD_PARTY_OPENAI_KEY_PATH = "path/to/your-api-key.txt"  # 存 API Key 的文本文件；若与 OpenAI 相同可填 OPENAI_API_KEY_PATH
```

- `THIRD_PARTY_OPENAI_BASE_URL`：必填，例如 `https://api.xxx.com/v1`（注意保留末尾 `/v1`）。
- `THIRD_PARTY_OPENAI_MODEL`：该第三方服务里的模型名。
- `THIRD_PARTY_OPENAI_KEY_PATH`：存 API Key 的文本文件路径；若与 OpenAI 用同一 Key 文件，可设为 `OPENAI_API_KEY_PATH`。

### 2. 在推理时使用第三方模型

- **EGAgent 推理**：编辑 **`egagent/langgraph_agent.py`**，将顶部的 `agent_backbone` 改为：

  ```python
  agent_backbone = 'openai_compatible'   # 使用 paths.py 中配置的第三方 OpenAI 兼容 API
  ```

- **prepare_datasources**（如实体图、摘要等）：在调用 `get_llm_worker` 或等价逻辑时，将 `model` 参数改为 `'openai_compatible'`，即会使用上述同一套配置。

这样，所有通过 `get_llm_worker` / `get_vision_llm` 等走 OpenAI 兼容路径的调用，都会使用你在 `paths.py` 里配置的 **base_url + model + api_key**，从而接入第三方支持 OpenAI 格式的 API。

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
