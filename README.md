# llm-practice

> 一个练习项目：把「调用大模型 → 错误分类与重试 → 文档加载 → 文本分块 → 向量化 → FAISS 检索 → 检索评测」做成**可复用、可测试、可度量**的代码。
>
> **当前已完成**：第 1 阶段（配置 / LLM 客户端 / 日志 / 命令行）、第 5 周（Markdown·PDF 解析 + 三种分块策略 + 对比实验）、第 6 周（智谱 `embedding-3` 向量化 + FAISS 索引 + 25 条评测集与 Recall 评测）。
> **尚未实现**：带引用的端到端回答、拒答、50 条正式评测集、rerank、Agent——见[第 9 节](#9-路线图与已知限制)。

## 0. 30 秒看懂现在能做什么

| 能力 | 命令 / 入口 |
| --- | --- |
| 问 DeepSeek 一个问题（日志、错误分类、按错误类型重试） | `python main.py "什么是 RAG？"` |
| 把 Markdown / PDF 解析成统一的 `Document`，再切块（fixed / recursive / semantic） | `core/loaders/` + `core/ingest.py` |
| 对 10 篇语料跑分块对比实验（5 个指标） | `python scripts/compare_chunking.py` |
| 用智谱 `embedding-3`（1024 维）建 FAISS 向量索引 | `python scripts/build_index.py` |
| 用 25 条评测集算 Recall@1/@3/@5 与 MRR | `python scripts/eval_retrieval.py` |
| 42 个单元测试（全部离线、不消耗 API 额度） | `python -m pytest` |

当前检索结果（16 条可评测样本）：**Recall@1 68.8%、Recall@3 100%、Recall@5 100%、MRR 0.844**。完整分析与局限性见 [docs/retrieval-report.md](docs/retrieval-report.md)；**Recall@5 已饱和，不能当作「检索能力达标」的证据**（原因见第 9.2 节）。

---

## 1. 这个项目解决什么问题

写 RAG 应用时会反复踩同一批坑。本项目把每个坑都收敛成一处可复用的实现：

| 常见问题 | 本项目的处理 |
| --- | --- |
| API Key 硬编码、或被误提交 | 只从 `.env` / 环境变量读；`Config`（pydantic）校验格式；`.gitignore` 忽略 `.env`，仓库只有 `.env.example` |
| 报错分不清是配置问题、代码问题还是服务端问题 | 异常层级：`AppError` → `ConfigError` / `LLMError`（下分超时、认证、限流、余额、参数、5xx） |
| 无差别重试，把不可恢复的错误也重试到超时 | 只有 `RETRYABLE`（超时 / 限流 / 5xx）才重试；401/400/402 立刻失败 |
| 重试风暴、两层重试叠加 | 分级退避 + 随机抖动，并**关闭 OpenAI SDK 自带重试**（`max_retries=0`） |
| 向量化接口不稳定、维度悄悄变了 | `EmbeddingClient` 对返回做**数量 + 维度自检**，429/5xx 才重试，4xx 立刻失败 |
| 不同格式文档各写一套解析 | `DocumentLoader` 抽象基类 + 统一 `Document` 模型（markdown / pdf） |
| 分块参数靠拍脑袋 | 三种策略可切换 + 可复现的对比实验 + 指标报告 |
| 「分块看起来不错」但检索效果未知 | 建索引 + 25 条带金标区间的评测集 + Recall/MRR 报告 |
| 换了分块策略，评测就失效 | 金标用**原文字符区间**标注，与块边界解耦，换策略仍可评测 |
| 改了代码不知道有没有坏 | 42 个 pytest 用例 + `conftest.py` 共享 fixture，全部离线可跑 |
| 出问题看不到过程 | 控制台 + 滚动文件日志（`logs/app.log`），格式含模块名与行号 |

**定位**：学习项目，重点是把「可复用的地基 + 可量化的实验」做出来，而不是功能完整的问答产品。目前完成的是 RAG 的**前半段**（数据准备 + 检索），还没有「生成带引用的回答」这一后半段。

---

## 2. 主要功能（当前仓库真实已实现）

| 功能 | 实现位置 | 说明 |
| --- | --- | --- |
| 配置校验 | `core/config.py`、`core/models.py` → `Config` | 读 `.env`，pydantic 校验：DeepSeek key 前缀与长度、智谱 key、base_url 协议、timeout、重试次数 |
| 异常层级 | `core/exceptions.py` | `AppError` → `ConfigError` / `LLMError`（超时、认证、限流、余额、参数、5xx、未知） |
| 错误分类 | `core/llm_client.py` → `classify_error()` | 把 SDK / httpx 异常按状态码翻译成项目自己的异常类型 |
| LLM 调用与重试 | `core/llm_client.py` → `LLMClient.ask()` | 单轮对话、可选 system；默认 30s 超时、最多 4 次尝试；401/400/402 不重试；限流 10/20/40s、其他 1/2/4s，加 0–0.5s 抖动 |
| 统一日志 | `core/logging_config.py` → `setup_logging()` | 控制台 + `logs/app.log`（5 MB 滚动，保留 3 个备份，UTF-8） |
| 文档加载 | `core/loaders/` | `MarkdownLoader`（`.md`/`.markdown`，取首个 `# ` 作标题）、`PDFLoader`（`pypdf` 逐页抽取，插入 `<<<PAGE n>>>` 标记并记录页偏移） |
| 固定分块 | `core/chunkers/fixed.py` | 按字符窗口切分，支持 overlap，跳过纯空白块；作为基线策略 |
| 递归分块 | `core/chunkers/recursive.py` | 按分隔符优先级递归下切（`\n## ` → `\n### ` → `\n\n` → 句末标点 → 逗号 → 字符），切完再做短块合并 |
| 语义分块 | `core/chunkers/semantic.py` | 按 Markdown 标题切段，超长段二次切分；无标题时按空行合并成段；填写 `Chunk.section` |
| 短块合并 | `core/chunkers/base.py` → `merge_short_chunks()` | 小于 100 字的块尝试与相邻块合并（上限为 `chunk_size`），减少碎片块 |
| 摄入编排 | `core/ingest.py` → `ingest()` / `pick_loader()` | 自动选 loader + 按策略分发 chunker，返回 `list[Chunk]` |
| Embedding 客户端 | `core/embedding.py` → `EmbeddingClient` | 智谱 `embedding-3`，默认 1024 维，批量 ≤64 条；维度/数量自检；429/5xx/超时最多尝试 3 次（失败后等待 1、2 秒） |
| 向量存储 | `core/vector_store.py` → `VectorStore` | FAISS `IndexFlatIP` + L2 归一化（等价余弦相似度）；`add` / `search` / `save` / `load`；索引与块元数据分两个文件落盘 |
| 检索器 | `core/retrieval.py` → `Retriever.retrieve()` | 问题 → 向量 → top-k 块（含相似度分数） |
| 建索引脚本 | `scripts/build_index.py` | 解析 `data/raw/test_corpus/*.md` → semantic 500/50 切块 → 批量向量化 → 保存 `data/index/faiss.*`，打印块数/耗时/预估成本 |
| 检索评测脚本 | `scripts/eval_retrieval.py` | 计算 Recall@1/@3/@5、MRR，按题型分组，打印失败样本；自动跳过 `should_abstain` 样本 |
| 评测集 | `data/eval/questions.json` | 25 条：fact 9 / multi_hop 7 / unanswerable 5 / trap 4；金标为文档内字符区间 |
| 评测集辅助脚本 | `scripts/generate_eval.py`、`validate_eval.py`、`fix_gold_spans.py` | 生成 20 条种子题并回填 `gold_spans`、人工核对切片、补齐 5 条缺失区间（合计 25 条） |
| 实验报告 | `docs/chunking-report.md`、`docs/retrieval-report.md` | 分块策略对比；检索指标、瓶颈诊断与优化方向 |
| 单元测试 | `tests/` + `pyproject.toml` | 42 个用例（10 个测试文件），覆盖配置、异常、重试、模型、加载器、分块、编排、embedding、向量库、命中判定 |
| 命令行入口 | `main.py` | `argparse`：可选问题（默认「用一句话解释什么是大语言模型。」）+ `--system`；成功 0、失败 1、参数错误 2 |
| Windows 启动脚本 | `run.bat` | 切 UTF-8、用 `.venv` 解释器运行 `main.py %*`、结束 `pause` |
| 阶段 1 遗留示例 | `hello_llm.py` | 硬编码问题、只读进程环境变量、无错误处理；保留作最小调用示例 |

**明确未实现**：把检索结果拼进 prompt 生成「带引用的回答」、无答案拒答、rerank / 混合检索、50 条正式评测集、流式输出、多轮对话、Agent 工具调用。

---

## 3. 使用方法

### 3.1 环境要求

- Python **3.10+**（代码用到了 `str | None` 等新语法；本机验证版本 Python 3.13.5）
- **问答**：需要 DeepSeek API Key 与网络
- **建索引 / 检索评测**：额外需要智谱 API Key 与网络（`ZHIPU_API_KEY`）
- **解析、分块、对比实验、单元测试**：全部离线可跑，不需要任何 Key
- Windows / macOS / Linux 均可；`run.bat` 仅 Windows

### 3.2 安装依赖

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` 当前包含 8 个依赖：

```text
openai>=1.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
pypdf>=4.0.0
httpx>=0.27.0
faiss-cpu>=1.7.0
numpy>=1.24.0
pytest>=8.0.0
```

> 可选开发工具（未写入 requirements）：`pytest-cov`（覆盖率）、`mypy`（类型检查，配置见 `pyproject.toml`）。

### 3.3 配置密钥

```bash
# Windows
copy .env.example .env
# macOS / Linux
# cp .env.example .env
```

编辑 `.env`：

```ini
# 问答（必填）
DEEPSEEK_API_KEY=sk-你的真实密钥

# 向量化（建索引 / 评测必填）
ZHIPU_API_KEY=你的智谱密钥

# 以下均可选
MODEL=deepseek-chat
BASE_URL=https://api.deepseek.com
TIMEOUT=30
MAX_RETRIES=4
```

优先级：**进程环境变量 > `.env`**（`load_dotenv()` 不覆盖已存在的变量）。

### 3.4 命令行问答

```powershell
python main.py                                  # 默认问题
python main.py "什么是 RAG？"                    # 指定问题（含空格要加引号）
python main.py "解释一下 Agent" --system "只用一句话回答"
python main.py --help
```

Windows 也可双击 `run.bat`，或带参调用：`run.bat "什么是 RAG？" --system "只用一句话回答"`。

### 3.5 文档解析与分块（Python API）

```python
from pathlib import Path
from core.ingest import ingest
from core.models import ChunkingConfig

chunks = ingest(
    Path("data/raw/test_corpus/01-时间管理的底层逻辑.md"),
    ChunkingConfig(strategy="semantic", chunk_size=500, overlap=50),
)
print(len(chunks), chunks[0].chunk_id, chunks[0].section)
```

- **支持后缀**：`.md`、`.markdown`、`.pdf`；其他后缀在 `pick_loader()` 抛 `ValueError`
- **`strategy`**：`fixed`（默认）/ `recursive` / `semantic`
- **`chunk_size`**：50–4000，默认 500；**`overlap`**：≥0 且小于 `chunk_size`，默认 50
- `semantic` 与 `recursive` 都会在切完后调用 `merge_short_chunks()` 合并 <100 字的碎块

`Chunk` 关键字段：`chunk_id`（`{doc_id}_{序号:04d}`）、`text`、`start_char`/`end_char`（原文区间）、`section`（仅 semantic 填写）、`page`（PDF 预留、当前未填）、`char_count`、`token_estimate`（中文约 1 字 1 token）。

新增分块策略：继承 `core.chunkers.base.Chunker` 实现 `chunk()`，再把类注册进 `core/ingest.py` 的 `CHUNKERS`。

### 3.6 建立向量索引（第 6 周）

```powershell
python scripts/build_index.py
```

流程：读取 `data/raw/test_corpus/*.md` → `semantic` 500/50 切块 → 按 64 条/批调用智谱 `embedding-3`（1024 维）→ 写入 FAISS → 保存到 `data/index/faiss.faiss` 与 `data/index/faiss.pkl`。

前置条件与退出码：

| 情况 | 输出 | 退出码 |
| --- | --- | --- |
| 正常 | 文档数 / 总块数 / 维度 / 耗时 / 预估成本 | 0 |
| 缺 `ZHIPU_API_KEY` | `错误：请在 .env 中配置 ZHIPU_API_KEY 才能进行向量化。` | 1 |
| 语料目录不存在 | `错误：找不到语料库目录 data/raw/test_corpus` | 1 |
| 向量化失败 | `向量化失败: ...` | 1 |

成本估算按 `智谱 embedding-3：0.5 元 / 100 万 tokens` 计算（脚本内写死）。

### 3.7 检索（Python API）

```python
from pathlib import Path
from core.config import load_config
from core.embedding import EmbeddingClient
from core.vector_store import VectorStore
from core.retrieval import Retriever

cfg = load_config()
store = VectorStore.load(Path("data/index/faiss.index"))      # 同时读 .faiss 与 .pkl
retriever = Retriever(EmbeddingClient(api_key=cfg.zhipu_api_key, dim=1024), store)

for chunk, score in retriever.retrieve("手冲咖啡水温多少合适？", top_k=3):
    print(f"{score:.4f}  {chunk.chunk_id}  {chunk.text[:36]}")
```

`VectorStore.load()` 的路径参数带不带后缀都可以：内部用 `with_suffix(".faiss")` / `with_suffix(".pkl")` 找文件。

### 3.8 检索评测（第 6 周）

```powershell
python scripts/eval_retrieval.py
```

- 读取 `data/index/faiss.index`（不存在会提示先跑 `build_index.py`，退出码 1）
- 读取 `data/eval/questions.json`（也兼容 `.jsonl`）
- 跳过 `should_abstain=true` 的样本（unanswerable + trap 共 9 条），实际参与评测 16 条
- 命中判定：检索块与金标字符区间**有交集**且 `doc_id` 相同（半开区间相交）
- 输出 Recall@1 / @3 / @5、MRR、按题型分组的 Recall@5、失败样本列表

### 3.9 评测集维护

| 脚本 | 作用 |
| --- | --- |
| `scripts/generate_eval.py` | 用 20 条种子题（含 `anchor` 原文片段）在原文中定位，回填精确 `gold_spans`，生成 `data/eval/questions.json` |
| `scripts/validate_eval.py` | 逐条打印 `gold_spans` 对应的切片文本，供人工核对「这段确实回答了问题」 |
| `scripts/fix_gold_spans.py` | 对 5 条需要跨段证据的题（q021–q025）写入人工核对过的多区间 `gold_spans`，使评测集达到 25 条 |

评测集中一条样本的结构：

```json
{
  "id": "q001",
  "type": "fact",
  "question": "手冲咖啡一般建议用多少度的水？",
  "gold_doc": "02-手冲咖啡入门指南",
  "should_abstain": false,
  "notes": "核心变量表格",
  "gold_spans": [[424, 436]]
}
```

### 3.10 运行测试

```bash
python -m pytest          # 完整输出
python -m pytest -q       # 精简输出
```

`pyproject.toml` 已配置 `testpaths = ["tests"]`、`pythonpath = ["."]`、`-v --tb=short`。全部用例都不访问网络：LLM / Embedding 相关测试用 mock，向量库用真实 FAISS 但只喂手工向量。

| 测试文件 | 用例数 | 覆盖内容 |
| --- | --- | --- |
| `tests/test_config.py` | 2 | 缺 key 抛 `ConfigError`、默认模型 |
| `tests/test_models.py` | 9 | key 前缀/空白、timeout 负值、重试边界、base_url 协议、`Chunk` 区间校验 |
| `tests/test_exceptions.py` | 8 | 401/403/400/429/500/502/503 与超时的分类映射 |
| `tests/test_llm_client.py` | 3 | 401 不重试、429 重试到上限、成功只调 1 次 |
| `tests/test_loaders.py` | 4 | markdown 后缀识别、标题提取、文件不存在 |
| `tests/test_chunkers.py` | 3 | 块文本必须等于原文区间、overlap 生效、短块合并 |
| `tests/test_ingest.py` | 4 | loader 选择、未知后缀、未知策略、端到端切块 |
| `tests/test_embedding.py` | 3 | 维度自检、批量顺序、401 不重试 |
| `tests/test_vector_store.py` | 4 | 写入/长度、相似度降序、空库检索、维度不匹配 |
| `tests/test_retrieval.py` | 2 | 区间相交判定、半开边界 |

> 其中 4 个用例使用 pytest 的 `tmp_path`。在权限受限的沙箱里它们会因临时目录写入被拒而报 `PermissionError`（环境问题，不是代码问题）；在普通终端下 42 个全部通过。

### 3.11 输出约定

| 通道 | 内容 | 典型用法 |
| --- | --- | --- |
| stdout | 回答（上下各一条 `=` 分隔线）；脚本统计结果 | `python main.py "…" > answer.txt` |
| stderr | 日志（时间、级别、模块:行号） | 不想看日志：`2>$null` |
| 文件 | `logs/app.log`（5 MB × 3 备份） | 事后排查 |
| 退出码 | `0` 成功 / `1` 配置或调用/构建失败 / `2` 参数错误 | 脚本判断成败 |

---

## 4. 输入输出实例

> 以下均为 2026-11-11 在本机真实运行记录；模型回答与检索分数每次会有差异。

### 4.1 命令行问答（成功）

```powershell
python main.py
```

stdout：

```text

========================================
大语言模型是一种通过海量文本数据训练、能够理解和生成自然语言的人工智能模型。
========================================

```

stderr：

```text
2026-11-02 21:07:09 [INFO    ] core.llm_client:114 - 提问：用一句话解释什么是大语言模型。
2026-11-02 21:07:10 [INFO    ] httpx2:1085 - HTTP Request: POST https://api.deepseek.com/chat/completions "HTTP/1.1 200 OK"
2026-11-02 21:07:10 [INFO    ] core.llm_client:125 - 回答成功，长度 38 字符
```

退出码：`0`

### 4.2 Key 无效（401 → 立刻失败，不重试）

```powershell
$env:DEEPSEEK_API_KEY = "sk-invalid-key-for-test"
python main.py "测试 401"
```

```text
❌ 调用失败：认证失败（401）：请检查 DEEPSEEK_API_KEY 是否正确或已过期
```

stderr 里会有一条 `不可重试的错误（LLMAuthError）` 和一个 HTTP 401 请求记录，随后退出码 `1`。

### 4.3 Key 格式非法（配置校验拦截）

```powershell
$env:DEEPSEEK_API_KEY = "not-a-valid-key"
python main.py "测试格式"
```

```text
❌ 配置错误：配置校验失败：1 validation error for Config
api_key
  Value error, API Key 应以 sk- 开头，当前是 not... [type=value_error, input_value='not-a-valid-key', input_type=str]
    For further information visit https://errors.pydantic.dev/2.14/v/value_error
```

退出码：`1`（未发出请求）

### 4.4 没有配置（无 `.env`、无环境变量）

```text
❌ 配置错误：未找到 DEEPSEEK_API_KEY。请在项目根目录创建 .env 文件，内容形如：DEEPSEEK_API_KEY=sk-xxxx
```

退出码：`1`

### 4.5 运行测试

```text
$ python -m pytest -q
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.7.0
rootdir: <项目根目录>
configfile: pyproject.toml
testpaths: tests
collected 42 items

...
42 passed in 0.99s
```

### 4.6 分块对比实验

```powershell
python scripts/compare_chunking.py
```

输出（节选，完整 15 行见报告）：

```text
成功加载 10 篇文档。

策略         | Size  | Overlap | 总块数    | 平均长    | 截断率    | 平均标题   | 长度标准差
--------------------------------------------------------------------------------
fixed      | 300   | 0       | 62     | 278.0  | 96.8 % | 0.18   | 65.5
fixed      | 500   | 50      | 41     | 458.3  | 97.6 % | 0.24   | 100.0
fixed      | 800   | 0       | 27     | 638.5  | 100.0% | 0.41   | 259.1
semantic   | 300   | 50      | 92     | 193.9  | 30.4 % | 0.11   | 74.5
semantic   | 500   | 50      | 81     | 214.7  | 24.7 % | 0.12   | 95.4
semantic   | 800   | 50      | 79     | 218.2  | 21.5 % | 0.13   | 102.1
recursive  | 300   | 50      | 97     | 177.7  | 85.6 % | 0.10   | 60.7
recursive  | 500   | 50      | 79     | 218.2  | 97.5 % | 0.13   | 87.4
recursive  | 800   | 50      | 72     | 239.4  | 100.0% | 0.14   | 110.6
```

读法：**截断率越低越好**，**平均标题数**越接近 1 说明块内结构越完整。`fixed` 不可用（91%–100% 截断且跨章节拼接），`semantic` 最好（21%–30%），`recursive` 加短块合并后块数下降但仍高度截断。完整 15 行数据见 [docs/chunking-report.md](docs/chunking-report.md)。

### 4.7 解析 + 分块（Python API）

```python
chunks = ingest(
    Path("data/raw/test_corpus/01-时间管理的底层逻辑.md"),
    ChunkingConfig(strategy="semantic", chunk_size=500, overlap=50),
)
print("chunks=", len(chunks))
c = chunks[0]
print("first:", c.chunk_id, "| section=", c.section, "| chars=", c.char_count)
print("range=", c.start_char, "-", c.end_char)
```

```text
chunks= 7
first: 01-时间管理的底层逻辑_0001 | section= 时间管理的底层逻辑：从"管理时间"到"管理注意力" | chars= 68
range= 0 - 68
```

### 4.8 建立索引

```powershell
python scripts/build_index.py
```

```text
正在解析并切块...
开始向量化，共 81 个块...
  进度: 64/81
  进度: 81/81
正在构建 FAISS 索引...
正在保存索引到磁盘...

==============================
文档数: 10
总块数: 81
向量维度: 1024
向量化耗时: 1.6 秒
预估成本: 0.0060 元 (基于 11900 tokens)
索引已保存: data\index\faiss.faiss (及 data\index\faiss.pkl)
==============================
```

退出码：`0`

### 4.9 检索评测

```powershell
python scripts/eval_retrieval.py
```

```text
开始评测，加载了 25 条样本。

=== Recall@5 总览 ===
全部参与评测数: 16 条 (已排除 unanswerable/trap)
Recall@1: 68.8%  (11/16)
Recall@3: 100.0%  (16/16)
Recall@5: 100.0%  (16/16)
MRR:      0.844

=== 按类型分组 ===
fact         Recall@5: 100.0%  (9/9)
multi_hop    Recall@5: 100.0%  (7/7)

=== 失败样本 (未命中) ===
无，全部命中！
```

退出码：`0`。**注意**：Recall@5 为 100% 主要说明评测集太容易（见第 9.2 节），真正有区分度的是 Recall@1 与 MRR。

### 4.10 检索 API

```text
0.6009  02-手冲咖啡入门指南_0001  # 手冲咖啡入门：从一杯苦涩到一杯干净  > 手冲的乐趣不在于仪式感，而
0.5927  02-手冲咖啡入门指南_0002  ## 二、四个核心变量  手冲只有四个变量在动，理解它们，就理解了全部：
0.5674  05-中国茶六大分类_0004  ## 三、怎么泡：按类给参数  | 类别 | 水温 | 投茶量（盖碗 1
```

（查询「手冲咖啡水温多少合适？」，`top_k=3`；第三名来自「茶」文档，说明单靠向量召回会混入相邻主题。）

### 4.11 帮助与参数错误

```text
usage: main.py [-h] [--system SYSTEM] [question]

问 DeepSeek 一个问题

positional arguments:
  question         要问的问题（默认问什么是大语言模型）

options:
  -h, --help       show this help message and exit
  --system SYSTEM  可选的 system 提示词
```

问题没加引号（`python main.py 什么是 RAG？`）会得到 `main.py: error: unrecognized arguments: RAG？`，退出码 `2`。

---

## 5. 项目结构

```text
.
├── main.py                     # 命令行入口：日志 → 配置 → 调 LLM → 打印
├── hello_llm.py                # 阶段 1 遗留示例（硬编码问题，只读环境变量）
├── run.bat                     # Windows 一键运行（chcp 65001 + .venv 解释器 + pause）
├── pyproject.toml              # pytest（testpaths/pythonpath/-v）与 mypy 配置
├── requirements.txt            # openai/dotenv/pydantic/pypdf/httpx/faiss-cpu/numpy/pytest
├── LICENSE                     # MIT
├── .env.example                # 环境变量模板（占位密钥需替换）
├── .env                        # 本地真实密钥（被忽略，不提交）
├── .gitignore                  # 忽略 .env、logs/、data/*、缓存、IDE、虚拟环境
├── core/
│   ├── __init__.py
│   ├── config.py               # load_config()：读 .env → pydantic Config
│   ├── models.py               # Config / Document / Chunk / ChunkingConfig / DocumentType
│   ├── exceptions.py           # AppError 异常层级
│   ├── llm_client.py           # classify_error() + LLMClient（超时、分级重试、抖动）
│   ├── logging_config.py       # setup_logging()：控制台 + logs/app.log 滚动
│   ├── embedding.py            # EmbeddingClient：智谱 embedding-3、维度自检、重试
│   ├── vector_store.py         # VectorStore：FAISS IndexFlatIP + L2 归一化 + 落盘
│   ├── retrieval.py            # Retriever：query → 向量 → top-k 块
│   ├── prompts.py              # 空占位（暂无提示词模板）
│   ├── ingest.py               # ingest() / pick_loader()，注册三种分块策略
│   ├── loaders/
│   │   ├── base.py             # DocumentLoader 抽象基类
│   │   ├── markdown.py         # MarkdownLoader：首个 # 标题 + 行数
│   │   └── pdf.py              # PDFLoader：pypdf 逐页抽取 + <<<PAGE n>>> + 页偏移
│   └── chunkers/
│       ├── base.py             # Chunker 抽象基类 + estimate_tokens + merge_short_chunks
│       ├── fixed.py            # 固定窗口 + overlap
│       ├── recursive.py        # 分隔符递归下切 + 短块合并
│       └── semantic.py         # 标题切段 + 超长二次切 + 短块合并
├── scripts/
│   ├── compare_chunking.py     # 分块对比实验（3 策略 × 3 size × overlap）
│   ├── build_index.py          # 建 FAISS 索引（智谱 embedding-3，1024 维）
│   ├── eval_retrieval.py       # Recall@1/3/5 + MRR 评测
│   ├── generate_eval.py        # 生成 20 条种子题并回填 gold_spans
│   ├── validate_eval.py        # 人工核对金标切片
│   └── fix_gold_spans.py       # 补齐 5 条跨段题，使评测集达到 25 条
├── docs/
│   ├── chunking-report.md      # 分块策略对比报告
│   └── retrieval-report.md     # 检索质量评测报告（指标、瓶颈、优化方向）
├── tests/                      # 10 个测试文件、42 个用例
│   ├── conftest.py             # fixture：fake_llm_client / no_sleep / make_http_error / make_doc / make_config
│   ├── test_config.py          # 2
│   ├── test_models.py          # 9
│   ├── test_exceptions.py      # 8
│   ├── test_llm_client.py      # 3
│   ├── test_loaders.py         # 4
│   ├── test_chunkers.py        # 3
│   ├── test_ingest.py          # 4
│   ├── test_embedding.py       # 3
│   ├── test_vector_store.py    # 4
│   └── test_retrieval.py       # 2
├── data/                       # 本地数据（除 data/eval 外都被忽略）
│   ├── raw/test_corpus/*.md    # 10 篇对比实验语料（被忽略）
│   ├── index/faiss.faiss       # FAISS 索引（被忽略）
│   ├── index/faiss.pkl         # 与索引行号对应的 Chunk 列表（pickle，被忽略）
│   ├── eval/questions.json     # 25 条评测集（**已入库**）
│   └── processed/              # 预留输出目录（当前为空）
└── logs/app.log                # 运行日志（被忽略）
```

`.venv/`、`.idea/`、`.pytest_cache/`、`.mypy_cache/`、`.coverage`、`__pycache__/` 都是本机产物，均已被忽略。

---

## 6. 配置项一览

| 变量 | 必填 | 默认值 | 校验规则 / 用途 |
| --- | --- | --- | --- |
| `DEEPSEEK_API_KEY` | 是 | — | 长度 ≥ 10、必须以 `sk-` 开头、首尾空白自动去除（问答用） |
| `ZHIPU_API_KEY` | 建索引/评测时必填 | 空 | 智谱开放平台密钥（embedding 用）；为空时 `build_index.py` 直接报错退出 1 |
| `MODEL` | 否 | `deepseek-chat` | 问答模型名 |
| `BASE_URL` | 否 | `https://api.deepseek.com` | 必须以 `http://` / `https://` 开头，末尾 `/` 自动去掉 |
| `TIMEOUT` | 否 | `30` | 浮点数，0 < 值 ≤ 300；填非数字会被拦下 |
| `MAX_RETRIES` | 否 | `4` | 整数，1 ≤ 值 ≤ 10（LLM 客户端重试次数） |

> Embedding 的模型名（`embedding-3`）、维度（1024）、base_url 与批量大小目前**写死在代码里**，没有对应环境变量。

---

## 7. 错误类型与重试策略

### 7.1 LLM（DeepSeek）

| 异常 | 触发条件 | 是否重试 | 等待 |
| --- | --- | --- | --- |
| `ConfigError` | key 缺失 / 校验不过 | 不发请求 | — |
| `LLMTimeoutError` | 超时 | 是 | 1, 2, 4 秒 + 抖动 |
| `LLMRateLimitError` | HTTP 429 | 是 | 10, 20, 40 秒 + 抖动 |
| `LLMServerError` | HTTP 5xx | 是 | 1, 2, 4 秒 + 抖动 |
| `LLMAuthError` | HTTP 401 / 403 | **否** | — |
| `LLMBadRequestError` | HTTP 400 | **否** | — |
| `LLMInsufficientQuotaError` | HTTP 402 | **否** | — |
| `LLMUnknownError` | 其他未识别异常 | **否** | — |

### 7.2 Embedding（智谱）

| 情况 | 处理 |
| --- | --- |
| 401 / 403 | 立刻抛 `EmbeddingError`（提示检查智谱 Key），不重试 |
| 400 | 立刻抛（提示可能是 `dimensions` 不被支持等参数问题） |
| 429 / 5xx | 最多尝试 3 次，失败后等待 1、2 秒再试 |
| 超时 | 最多尝试 3 次，失败后等待 1、2 秒再试 |
| 返回条数或维度不符 | 立刻抛 `EmbeddingError`（提示检查维度配置），不重试 |

> `EmbeddingError` 定义在 `core/embedding.py`，**没有**并入 `AppError` 体系，这是当前的一处设计不一致。

---

## 8. 常见问题

**控制台中文乱码？**
用 `run.bat`（内部 `chcp 65001`），或先 `chcp 65001`，或设 `$env:PYTHONIOENCODING = "utf-8"`。

**返回 401？**
`.env` 里还是占位符，或 key 已失效。注意 `TIMEOUT`/`MAX_RETRIES` 填错会报「配置校验失败」。

**`build_index.py` 说没有 `ZHIPU_API_KEY`？**
向量化用的是智谱，不是 DeepSeek。去 `.env` 加 `ZHIPU_API_KEY=...`；只做问答可以用 DeepSeek 的 key。

**`eval_retrieval.py` 提示找不到索引？**
先跑 `python scripts/build_index.py` 生成 `data/index/faiss.faiss` 与 `faiss.pkl`。索引被 `.gitignore` 忽略，新克隆的仓库必须自己建。

**Recall@5 是 100%，是不是检索很完美？**
不是。报告里已诊断：16 条可评测样本、10 篇一主题一文档、每篇仅 6–10 块，top-5 里平均 4.8 块来自金标文档，得分饱和。真正要看的指标是 Recall@1（68.8%）与 MRR（0.844）。

**为什么 top-3 里会混进不相关文档？**
单路向量召回在相近主题（咖啡 / 茶）之间会互相干扰；报告建议下一步做 rerank。

**索引文件能提交到仓库吗？**
不能，`data/index/` 被忽略。索引是由语料与模型决定的派生产物，换 embedding 模型或重新切块后必须重建。

**`faiss.pkl` 可以直接从网上下载来用吗？**
**不要**。它是 pickle 格式，加载任意来源的 pickle 等于执行任意代码；只加载自己 `build_index.py` 生成的文件。

**评测集是 25 条还是 50 条？**
目前 25 条（第 6 周过渡版本）：fact 9、multi_hop 7、unanswerable 5、trap 4。原计划的 50 条评测集属于第 8 周。

**`pytest` 有 4 个用例报 `PermissionError`？**
这 4 个用例（`test_ingest` 2 个、`test_loaders` 2 个）用 `tmp_path` 写临时目录；在权限受限的沙箱里会被拒，普通终端下 42 个用例全部通过。

**`core/prompts.py` 是空的？**
是占位文件，当前没有代码引用它——第 7 周做「拼 prompt 生成带引用回答」时会用到。

---

## 9. 路线图与已知限制

### 9.1 路线图

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 第 1 阶段 | 环境、跑通 API、配置校验、异常分类、重试、日志、命令行 | ✅ 已完成 |
| 第 5 周 | 文档解析与分块（markdown / pdf + 三种策略 + 对比实验 + 报告） | ✅ 已完成 |
| 第 6 周 | 向量化（智谱 embedding-3）+ FAISS 索引 + 检索 + 25 条评测集与 Recall/MRR 报告 | ✅ 已完成 |
| 第 7 周 | 端到端问答 + 引用溯源 + 拒答 | ⏳ 未开始 |
| 第 8 周 | 50 条评测集 + 基线数字 | ⏳ 未开始（当前 25 条为过渡集） |
| 第 9 周 | 混合检索 + rerank 优化 | ⏳ 未开始（报告已定位 rerank 为第一优先） |
| 第 11 周 | 手搓 agent loop + 工具调用 | ⏳ 未开始 |
| 第 12 周 | workflow vs agent 对照实验 | ⏳ 未开始 |
| 第 13 周 | 全链路 trace 与失败分析 | ⏳ 未开始 |

### 9.2 评测现状（第 6 周）

| 项 | 值 |
| --- | --- |
| 评测集 | 25 条：fact 9 / multi_hop 7 / unanswerable 5 / trap 4 |
| 参与召回评测 | 16 条（跳过 unanswerable + trap） |
| 索引规模 | 10 篇文档 → 81 块 → 1024 维 |
| Recall@1 | 68.8%（11/16） |
| Recall@3 / Recall@5 | 100%（16/16） |
| MRR | 0.844 |
| 建索引成本 | 约 11,900 tokens ≈ 0.006 元（1.6 秒） |

**为什么说 Recall@5 饱和**（摘自 [docs/retrieval-report.md](docs/retrieval-report.md)）：

1. 10 篇语料覆盖 10 个互不相关主题，检索实际退化成「1/10 主题分类」；
2. 每篇只有 6–10 块，top-5 已覆盖过半；
3. 出题时已知道答案位置（非盲测）。

结论：**Recall@5 不能作为检索达标的证据**；真正暴露问题的是 Recall@1 与 Recall@5 之间 31 个百分点的差距——瓶颈在排序，不在召回。原计划的 50 条评测集（事实 13 / 跨段 12 / 无法回答 13 / 陷阱 12）仍待第 8 周完成。

### 9.3 已知限制

- **只有检索，没有回答**：top-k 块不会拼进 prompt 生成答案，`core/prompts.py` 仍是空占位
- **没有引用与拒答**：`section` 只在块内保留，未进入回答；9 条拒答样本未参与任何评测
- **评测集饱和**：R@5=100% 但样本仅 16 条，无统计显著性
- **单一 embedding 模型**：只用了智谱 `embedding-3`，未做横向对比
- **没有 rerank / 混合检索**：R@1 与 R@5 的差距直接指向排序问题
- **PDF 页码未打通**：`PDFLoader` 写入的 `<<<PAGE n>>>` 与页偏移没有任何分块策略解析，`Chunk.page` 始终为空
- **分块报告正文与最新矩阵不一致**：报告的数据矩阵已更新为合并后的结果，但第 4/5 节叙述仍引用旧数字（106 块、22.6%、0.09 标题）
- **建索引成本与耗时和报告不符**：报告写 2.3 秒 / 0.0012 元，实测 1.6 秒 / 0.0060 元（11,900 tokens）——报告的成本数字偏小
- **索引用 pickle 序列化**：`VectorStore.load()` 只能加载可信来源
- **索引与语料不入库**：`data/raw`、`data/index` 被忽略，新克隆仓库无法直接复现检索评测
- **评测集规模靠脚本补齐**：`generate_eval.py` 只生成 20 条，另有 5 条由 `fix_gold_spans.py` 手工写死区间
- **测试与脚本耦合**：`tests/test_retrieval.py` 直接 `from scripts.eval_retrieval import check_hit`
- **4 个测试依赖 tmp_path**：权限受限环境下会 PermissionError
- **日志记录提问原文**：敏感问题会落在 `logs/app.log`（本地、不入库）
- **错误信息不够干净**：配置校验暴露 pydantic 原文，且会带出 DeepSeek key 前 3 个字符
- **`EmbeddingError` 不属于 `AppError` 体系**，两套异常风格并存
- **单文件摄入**：只有 `ingest(单文件)`，没有目录批量入口
- **`.txt` 未支持**：`DocumentType.TEXT` 已定义但无 loader
- **无流式、无多轮、无 token 计费统计**
- **`hello_llm.py` 与 `main.py` 功能重叠**，仅作历史示例保留

---

## 10. 技术选型与取舍

- **pydantic v2 建模**：配置、文档、块、分块参数统一建模，自动校验 + IDE 补全；后续引用与评测样本可复用同一套模型。
- **pypdf 而不是 pdfplumber**：轻量、纯 Python，普通文本型 PDF 能保留换行与符号；第 5 周只需要文本与页码。多栏排版或扫描件再考虑换（届时需要 OCR 或版面分析）。
- **自研三种分块器 + 短块合并**：代码量小、行为可读、便于写不变量测试（块文本必须等于原文区间）。`merge_short_chunks` 用「<100 字就与下一块合并（不超过 chunk_size）」来压碎片。
- **智谱 `embedding-3`（1024 维）**：支持中文、支持自定义维度；按 64 条/批调用，便于控制请求数。
- **FAISS `IndexFlatIP` + L2 归一化**：归一化后内积等价余弦相似度；当前只有 81 块，精确检索足够，不需要 ANN/HNSW，也避免引入额外近似误差。
- **评测用「字符区间」金标**：不绑定 `chunk_id`，换分块策略仍可复用评测集；命中判定为半开区间相交。
- **关闭 SDK 重试**：只保留项目层的一套退避策略，避免两层重试把一次失败放大成 9 次请求。
- **代理指标 + 诚实声明**：分块阶段用截断率等启发式指标，检索阶段用 Recall/MRR，但两份报告都明确写出「指标饱和 / 不能外推」的局限。

---

## 11. 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

---

**文档说明**：本 README 只描述仓库中**已存在**的代码；命令与输出于 2026-11-11 在本机实际核对（`build_index` 重建 81 块索引、`eval_retrieval` 得 R@1 68.8% / R@5 100% / MRR 0.844、`pytest` 42 个用例通过）。分块矩阵与 `docs/chunking-report.md` 的表格有 1–2 块的差异，原因是报告生成后合并逻辑又调整过，以脚本重跑结果为准。
