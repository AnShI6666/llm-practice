# llm-practice

> 一个练习项目：把「调用大模型 → 错误分类与重试 → 文档加载 → 文本分块 → 量化对比」做成**可复用、可测试**的代码。
>
> **当前已完成**：第 1 阶段（配置 / LLM 客户端 / 日志 / 命令行）与第 5 周（Markdown·PDF 解析 + 三种分块策略 + 对比实验）。
> **尚未实现**：向量检索、引用溯源、拒答、评测集——见[第 9 节](#9-路线图与已知限制)。

## 0. 30 秒看懂现在能做什么

| 能力 | 命令 / 入口 |
| --- | --- |
| 问 DeepSeek 一个问题（可带 system 提示词，有日志、错误分类、按错误类型重试） | `python main.py "什么是 RAG？"` |
| 把 Markdown / PDF 解析成统一的 `Document` 对象 | `core/loaders/` |
| 用 fixed / recursive / semantic 三种策略切块，每块带原文位置 | `core/chunkers/` + `core/ingest.py` |
| 对 10 篇语料跑分块对比实验，输出 5 个指标 | `python scripts/compare_chunking.py` |
| 24 个单元测试（全部离线、不消耗 API 额度） | `python -m pytest` |

分块实验的完整数据与结论在 [docs/chunking-report.md](docs/chunking-report.md)：当前选型 **semantic + chunk_size=500 / overlap=50**（截断率 22.6%，平均每块标题数 0.09；fixed 为 91%–100%，recursive 为 79%–100%）。

---

## 1. 这个项目解决什么问题

写 LLM/RAG 应用时，会反复踩同一批坑。本项目把每个坑都收敛成一处可复用的实现：

| 常见问题 | 本项目的处理 |
| --- | --- |
| API Key 硬编码、或被误提交 | 只从 `.env` / 环境变量读；`Config`（pydantic）校验格式；`.gitignore` 忽略 `.env`，仓库只有 `.env.example` |
| 报错分不清是配置问题、代码问题还是服务端问题 | 异常层级：`AppError` → `ConfigError` / `LLMError`（下分超时、认证、限流、余额、参数、5xx） |
| 无差别重试，把不可恢复的错误也重试到超时 | 只有 `RETRYABLE`（超时 / 限流 / 5xx）才重试；401/400/402 立刻失败 |
| 重试风暴、两层重试叠加 | 分级退避 + 随机抖动，并**关闭 OpenAI SDK 自带重试**（`max_retries=0`） |
| 不同格式文档各写一套解析 | `DocumentLoader` 抽象基类 + 统一 `Document` 模型（markdown / pdf） |
| 分块参数靠拍脑袋 | 三种策略可切换 + 可复现的对比实验 + 指标报告 |
| 改了代码不知道有没有坏 | 24 个 pytest 用例 + `conftest.py` 共享 fixture，全部离线可跑 |
| 出问题看不到过程 | 控制台 + 滚动文件日志（`logs/app.log`），格式含模块名与行号 |

**定位**：学习项目，重点是把「可复用的地基 + 可量化的实验」做出来，而不是功能完整的问答产品。长期目标是面向私有文档库的可溯源问答 Agent（检索 → 引用 → 拒答 → 评测），目前只完成了其中的**数据准备层**（解析 + 分块）。

---

## 2. 主要功能（当前仓库真实已实现）

| 功能 | 实现位置 | 说明 |
| --- | --- | --- |
| 配置校验 | `core/config.py`、`core/models.py` → `Config` | 读 `.env`/环境变量，交给 pydantic 校验：key 长度与前缀、base_url 协议、timeout 范围、重试次数范围；失败统一抛 `ConfigError` |
| 异常层级 | `core/exceptions.py` | `AppError` 为根；`ConfigError`；`LLMError` 下分 `LLMTimeoutError`、`LLMAuthError`、`LLMRateLimitError`、`LLMInsufficientQuotaError`、`LLMBadRequestError`、`LLMServerError`、`LLMUnknownError` |
| 错误分类 | `core/llm_client.py` → `classify_error()` | 把 SDK / httpx 的异常按状态码翻译成项目自己的异常类型 |
| LLM 调用与重试 | `core/llm_client.py` → `LLMClient.ask()` | 单轮对话、可选 system；默认 30s 超时、最多 4 次尝试；不可重试错误立刻抛；限流等 10/20/40s，其他 1/2/4s，均加 0–0.5s 抖动 |
| 统一日志 | `core/logging_config.py` → `setup_logging()` | 控制台 + `logs/app.log`（5 MB 滚动，保留 3 个备份，UTF-8） |
| 文档加载 | `core/loaders/` | `MarkdownLoader`（`.md`/`.markdown`，取首个 `# ` 作标题）、`PDFLoader`（`pypdf` 逐页抽取，插入 `<<<PAGE n>>>` 标记并记录页偏移） |
| 固定分块 | `core/chunkers/fixed.py` | 按字符窗口切分，支持 overlap，跳过纯空白块；作为基线策略 |
| 递归分块 | `core/chunkers/recursive.py` | 按分隔符优先级递归下切：`\n## ` → `\n### ` → `\n\n` → `\n` → 句末标点 → 逗号 → 字符 |
| 语义分块 | `core/chunkers/semantic.py` | 按 Markdown 标题切段，超长段二次切分；无标题时按空行合并成段；会填写 `Chunk.section` |
| 摄入编排 | `core/ingest.py` → `ingest()` / `pick_loader()` | 自动选 loader + 按策略分发 chunker，返回 `list[Chunk]` |
| 分块对比实验 | `scripts/compare_chunking.py` | 3 种策略 × 3 种 size × overlap（fixed 试 0/50/100），输出块数、平均块长、截断率、平均标题数、长度标准差 |
| 实验报告 | `docs/chunking-report.md` | 记录实验目的、指标定义、完整数据矩阵、结论与局限性 |
| 单元测试 | `tests/` + `pyproject.toml` | 24 个用例（5 个有效测试文件），覆盖配置、异常分类、重试、模型校验、分块不变量 |
| 命令行入口 | `main.py` | `argparse`：可选问题（默认「用一句话解释什么是大语言模型。」）+ `--system`；成功 0、失败 1、参数错误 2 |
| Windows 启动脚本 | `run.bat` | 切 UTF-8、用 `.venv` 解释器运行 `main.py %*`、结束 `pause` |
| 阶段 1 遗留示例 | `hello_llm.py` | 硬编码问题、只读进程环境变量、无错误处理；保留作最小调用示例 |

**明确未实现**（请不要按这些能力使用本项目）：向量化 / 向量库、检索与召回、rerank、引用溯源、无答案拒答、评测集与指标、流式输出、多轮对话、Agent 工具调用。

---

## 3. 使用方法

### 3.1 环境要求

- Python **3.10+**（代码用到了 `str | None` 等新语法；本机验证版本 Python 3.13.5）
- 只有「问答」需要网络和 API Key；**解析、分块、实验、测试全部离线可跑**
- Windows / macOS / Linux 均可；`run.bat` 仅 Windows

### 3.2 安装依赖

```bash
# 建议使用虚拟环境（仓库里已有一个 .venv，可直接复用）
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` 当前包含 6 个依赖：

```text
openai>=1.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
pypdf>=4.0.0
httpx>=0.27.0
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
DEEPSEEK_API_KEY=sk-你的真实密钥

# 以下均可选
MODEL=deepseek-chat
BASE_URL=https://api.deepseek.com
TIMEOUT=30
MAX_RETRIES=4
```

`.env.example` 里的 `sk-your-key-here` 是占位符，必须替换，否则认证会失败（401）。

优先级：**进程环境变量 > `.env`**（`load_dotenv()` 不覆盖已存在的变量）。临时换 key：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
python main.py "什么是 RAG？"
```

### 3.4 命令行问答

```powershell
# 1）默认问题
python main.py

# 2）指定问题（含空格或标点时必须加引号）
python main.py "什么是 RAG？"

# 3）指定 system 提示词
python main.py "解释一下 Agent" --system "你是一个只用一句话回答的助手"

# 4）查看帮助
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

- **支持的文件后缀**：`.md`、`.markdown`、`.pdf`；其他后缀在 `pick_loader()` 里抛 `ValueError`
- **`strategy`**：`fixed`（默认）/ `recursive` / `semantic`
- **`chunk_size`**：50–4000，默认 500；**`overlap`**：≥0 且必须小于 `chunk_size`，默认 50

`Chunk` 的字段（位置信息是后续引用溯源的基础）：

| 字段 | 含义 |
| --- | --- |
| `chunk_id` | `{doc_id}_{序号:04d}`，如 `01-时间管理的底层逻辑_0001` |
| `text` / `char_count` / `token_estimate` | 块文本、字符数、估算 token（中文约 1 字 1 token，英文约 4 字符 1 token） |
| `start_char` / `end_char` | 在原文中的字符区间，恒满足 `end_char > start_char` |
| `section` | 所属章节标题；**只有 semantic 策略会填** |
| `page` | PDF 页码字段已预留，**当前没有策略填充它**（页码线索暂存在 `Document.metadata["offsets"]`） |

新增一种分块策略只需要两步：继承 `core.chunkers.base.Chunker` 实现 `chunk()`，再把类注册进 `core/ingest.py` 的 `CHUNKERS` 字典。

### 3.6 分块对比实验

```powershell
python scripts/compare_chunking.py
```

- 脚本**必须在项目根目录运行**，固定读取 `data/raw/test_corpus/*.md`（约 10 篇）
- 语料不随仓库发布（`.gitignore` 忽略 `data/`），需要自己准备；目录不存在时脚本会打印错误提示
- 输出列：策略、`chunk_size`、`overlap`、总块数、平均块长、截断率、平均标题数、长度标准差
- 结论与完整表格已归档到 [docs/chunking-report.md](docs/chunking-report.md)

### 3.7 运行测试

```bash
python -m pytest          # 完整输出
python -m pytest -q       # 精简输出
```

`pyproject.toml` 已配置 `testpaths = ["tests"]`、`pythonpath = ["."]`、`-v --tb=short`。全部用例都不访问网络：LLM 相关测试用 mock，其余直接校验纯函数与模型。

| 测试文件 | 用例数 | 覆盖内容 |
| --- | --- | --- |
| `tests/test_config.py` | 2 | 缺 key 抛 `ConfigError`、默认模型 |
| `tests/test_exceptions.py` | 8 | 401/403/400/429/500/502/503 与超时的分类映射 |
| `tests/test_llm_client.py` | 3 | 401 不重试（1 次）、429 重试到上限（3 次）、成功只调 1 次 |
| `tests/test_models.py` | 9 | key 前缀/空白、timeout 负值、重试次数边界、base_url 协议、`Chunk` 区间校验 |
| `tests/test_chunkers.py` | 2 | **块的文本必须等于原文对应区间**、overlap 生效 |
| `tests/test_loaders.py` | 0 | 目前只是占位脚本（模块级 `print`），尚未写成测试 |

### 3.8 在代码里用 LLM 客户端

```python
from core.config import load_config
from core.llm_client import LLMClient, LLMError

cfg = load_config()
client = LLMClient(
    api_key=cfg.api_key,
    model=cfg.model,
    base_url=cfg.base_url,
    timeout=cfg.timeout,
    max_retries=cfg.max_retries,
)
try:
    print(client.ask("用一句话说明 DeepSeek 是什么", system="只用一句话回答"))
except LLMError as exc:
    print(f"失败：{type(exc).__name__} - {exc}")
```

### 3.9 输出约定

| 通道 | 内容 | 典型用法 |
| --- | --- | --- |
| stdout | 回答（上下各一条 `=` 分隔线） | `python main.py "…" > answer.txt` |
| stderr | 日志（时间、级别、模块:行号） | 不想看日志：`2>$null` |
| 文件 | `logs/app.log`（5 MB × 3 备份） | 事后排查 |
| 退出码 | `0` 成功 / `1` 配置或调用失败 / `2` 参数错误 | 脚本判断成败 |

---

## 4. 输入输出实例

> 以下均为 2026-11-02 在本机真实运行记录；模型回答每次不同，措辞会变。

### 4.1 命令行问答（成功）

输入：

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

输入：

```powershell
$env:DEEPSEEK_API_KEY = "sk-invalid-key-for-test"
python main.py "测试 401"
```

stdout：

```text
❌ 调用失败：认证失败（401）：请检查 DEEPSEEK_API_KEY 是否正确或已过期
```

stderr：

```text
2026-11-02 21:08:07 [INFO    ] core.llm_client:114 - 提问：测试 401
2026-11-02 21:08:07 [INFO    ] httpx2:1085 - HTTP Request: POST https://api.deepseek.com/chat/completions "HTTP/1.1 401 Authorization Required"
2026-11-02 21:08:07 [ERROR   ] core.llm_client:134 - 不可重试的错误（LLMAuthError）：认证失败（401）：请检查 DEEPSEEK_API_KEY 是否正确或已过期
2026-11-02 21:08:07 [ERROR   ] __main__:47 - 调用大模型失败：认证失败（401）：请检查 DEEPSEEK_API_KEY 是否正确或已过期
```

退出码：`1`（只有 1 次 HTTP 请求，没有浪费时间重试）

### 4.3 Key 格式非法（配置校验拦截）

输入：

```powershell
$env:DEEPSEEK_API_KEY = "not-a-valid-key"
python main.py "测试格式"
```

stdout（pydantic 原文较长，此处原样贴出）：

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

输入：

```powershell
python -m pytest -q
```

输出（节选）：

```text
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.7.0
rootdir: <项目根目录>
configfile: pyproject.toml
testpaths: tests
collected 24 items

tests\test_chunkers.py ..                                                [  8%]
tests\test_config.py ..                                                  [ 16%]
tests\test_exceptions.py ........                                        [ 50%]
tests\test_llm_client.py ...                                             [ 62%]
tests\test_models.py .........                                           [100%]

24 passed, 1 warning in 0.19s
```

> 这条 warning 是 `.pytest_cache` 目录写入被系统拒绝（缓存目录权限问题），不影响测试结果，详见第 8 节。

### 4.6 分块对比实验

输入：

```powershell
python scripts/compare_chunking.py
```

输出（节选，完整表格见报告）：

```text
成功加载 10 篇文档。

策略         | Size  | Overlap | 总块数    | 平均长    | 截断率    | 平均标题   | 长度标准差
--------------------------------------------------------------------------------
fixed      | 300   | 0       | 62     | 278.0  | 96.8 % | 0.18   | 65.5
fixed      | 500   | 50      | 41     | 458.3  | 97.6 % | 0.24   | 100.0
fixed      | 800   | 0       | 27     | 638.5  | 100.0% | 0.41   | 259.1
semantic   | 300   | 50      | 115    | 155.1  | 27.8 % | 0.09   | 87.6
semantic   | 500   | 50      | 106    | 164.0  | 22.6 % | 0.09   | 103.6
semantic   | 800   | 50      | 103    | 167.4  | 20.4 % | 0.10   | 108.6
recursive  | 300   | 50      | 179    | 96.3   | 79.9 % | 0.06   | 81.0
recursive  | 500   | 50      | 106    | 162.6  | 90.6 % | 0.09   | 102.4
recursive  | 800   | 50      | 83     | 207.7  | 100.0% | 0.12   | 114.6
```

读法：**截断率越低越好**（块没在句子中间被切断），**平均标题数**越接近 1 说明块内结构越完整。结论：`fixed` 完全不可用（91%–100% 截断，且经常跨章节拼接）；`semantic` 明显最好（20%–28%，块内几乎不含多个标题），报告最终选定 `size=500 / overlap=50`；`recursive` 因为实现朴素（没有小块合并）反而最碎（79%–100%）。

### 4.7 解析 + 分块（Python API）

输入：

```python
from pathlib import Path
from core.ingest import ingest
from core.models import ChunkingConfig

chunks = ingest(
    Path("data/raw/test_corpus/01-时间管理的底层逻辑.md"),
    ChunkingConfig(strategy="semantic", chunk_size=500, overlap=50),
)
print("chunks=", len(chunks))
c = chunks[0]
print("first:", c.chunk_id, "| section=", c.section, "| chars=", c.char_count)
print("range=", c.start_char, "-", c.end_char)
```

输出：

```text
chunks= 7
first: 01-时间管理的底层逻辑_0001 | section= 时间管理的底层逻辑：从"管理时间"到"管理注意力" | chars= 68
range= 0 - 68
```

### 4.8 查看帮助

```text
usage: main.py [-h] [--system SYSTEM] [question]

问 DeepSeek 一个问题

positional arguments:
  question         要问的问题（默认问什么是大语言模型）

options:
  -h, --help       show this help message and exit
  --system SYSTEM  可选的 system 提示词
```

### 4.9 参数写错（问题没加引号）

```powershell
python main.py 什么是 RAG？
```

```text
usage: main.py [-h] [--system SYSTEM] [question]
main.py: error: unrecognized arguments: RAG？
```

退出码：`2`（由 `argparse` 直接结束，未发出请求）

---

## 5. 项目结构

```text
.
├── main.py                     # 命令行入口：日志 → 配置 → 调 LLM → 打印
├── hello_llm.py                # 阶段 1 遗留示例（硬编码问题，只读环境变量）
├── run.bat                     # Windows 一键运行（chcp 65001 + .venv 解释器 + pause）
├── pyproject.toml              # pytest（testpaths/pythonpath/-v）与 mypy 配置
├── requirements.txt            # 运行依赖（openai/dotenv/pydantic/pypdf/httpx/pytest）
├── LICENSE                     # MIT
├── .env.example                # 环境变量模板（占位密钥需替换）
├── .env                        # 本地真实密钥（被忽略，不提交）
├── .gitignore                  # 忽略 .env、logs/、data/、缓存、IDE、虚拟环境
├── core/
│   ├── __init__.py             # 包标识（空文件）
│   ├── config.py               # load_config()：读 .env → pydantic Config
│   ├── models.py               # Config / Document / Chunk / ChunkingConfig / DocumentType
│   ├── exceptions.py           # AppError 异常层级
│   ├── llm_client.py           # classify_error() + LLMClient（超时、分级重试、抖动）
│   ├── logging_config.py       # setup_logging()：控制台 + logs/app.log 滚动
│   ├── prompts.py              # 空占位（暂无提示词模板）
│   ├── ingest.py               # ingest() / pick_loader()，注册三种分块策略
│   ├── loaders/
│   │   ├── __init__.py         # 包标识（空文件）
│   │   ├── base.py             # DocumentLoader 抽象基类
│   │   ├── markdown.py         # MarkdownLoader：首个 # 标题 + 行数
│   │   └── pdf.py              # PDFLoader：pypdf 逐页抽取 + <<<PAGE n>>> 标记 + 页偏移
│   └── chunkers/
│       ├── __init__.py         # 包标识（空文件）
│       ├── base.py             # Chunker 抽象基类
│       ├── fixed.py            # 固定窗口 + overlap + estimate_tokens()
│       ├── recursive.py        # 分隔符递归下切
│       └── semantic.py         # 标题切段 + 超长二次切 + 无标题时按空行合并
├── scripts/
│   └── compare_chunking.py     # 分块对比实验（3 策略 × 3 size × overlap）
├── docs/
│   └── chunking-report.md      # 实验报告：数据矩阵、结论、局限性
├── tests/
│   ├── conftest.py             # fixture：fake_llm_client / no_sleep / make_http_error / make_doc / make_config
│   ├── test_config.py          # 2 个用例
│   ├── test_exceptions.py      # 8 个用例（参数化）
│   ├── test_llm_client.py      # 3 个用例
│   ├── test_models.py          # 9 个用例（含参数化）
│   ├── test_chunkers.py        # 2 个用例
│   └── test_loaders.py         # 占位脚本（0 个用例）
├── data/                       # 本地数据，整体被 .gitignore 忽略
│   ├── raw/test_corpus/*.md    # 10 篇对比实验语料
│   ├── raw/*.pdf               # 本地 PDF 样本
│   └── processed/              # 预留输出目录（当前为空）
└── logs/app.log                # 运行日志（被忽略）
```

`.venv/`、`.idea/`、`.pytest_cache/`、`.mypy_cache/`、`.coverage`、`__pycache__/` 都是本机产物，均已被忽略。

---

## 6. 配置项一览

| 变量 | 必填 | 默认值 | 校验规则 |
| --- | --- | --- | --- |
| `DEEPSEEK_API_KEY` | 是 | — | 长度 ≥ 10，必须以 `sk-` 开头，首尾空白自动去除 |
| `MODEL` | 否 | `deepseek-chat` | 字符串 |
| `BASE_URL` | 否 | `https://api.deepseek.com` | 必须以 `http://` 或 `https://` 开头，末尾 `/` 会被去掉 |
| `TIMEOUT` | 否 | `30` | 浮点数，需满足 0 < 值 ≤ 300；填非数字会被拦下 |
| `MAX_RETRIES` | 否 | `4` | 整数，1 ≤ 值 ≤ 10 |

以上都可写在 `.env`，也可直接导出到 shell（shell 优先）。

---

## 7. 错误类型与重试策略

| 异常 | 触发条件 | 是否重试 | 等待 |
| --- | --- | --- | --- |
| `ConfigError` | key 缺失 / 校验不过 | 不发请求 | — |
| `LLMTimeoutError` | 超时（`TimeoutError`、`httpx.TimeoutException`、`APITimeoutError`） | 是 | 1, 2, 4 秒 + 抖动 |
| `LLMRateLimitError` | HTTP 429 | 是 | 10, 20, 40 秒 + 抖动 |
| `LLMServerError` | HTTP 5xx | 是 | 1, 2, 4 秒 + 抖动 |
| `LLMAuthError` | HTTP 401 / 403 | **否**，立刻抛 | — |
| `LLMBadRequestError` | HTTP 400 | **否** | — |
| `LLMInsufficientQuotaError` | HTTP 402 | **否** | — |
| `LLMUnknownError` | 其他未识别异常 | **否**（保守） | — |

默认可重试错误最多尝试 4 次；`OpenAI` 客户端本身的重试被显式关掉（`max_retries=0`），避免两层重试叠加成 9 次请求。

---

## 8. 常见问题

**控制台中文乱码？**
Windows 默认 GBK 代码页。用 `run.bat`（内部已 `chcp 65001`），或先 `chcp 65001`，或设 `$env:PYTHONIOENCODING = "utf-8"`。

**返回 401？**
`.env` 里还是占位符 `sk-your-key-here`，或 key 已失效。注意 `TIMEOUT`/`MAX_RETRIES` 填错也会报「配置校验失败」。

**为什么 401 不重试了？**
现在已经引入异常分类：认证、参数、余额类错误重试也没用，直接失败；只有超时、限流、5xx 才重试。

**日志在哪、为什么有中文问题？**
`logs/app.log`（5 MB 滚动，保留 3 个备份）。日志会记录**你提问的前 50 个字**，敏感场景注意。

**pytest 有一条 `.pytest_cache` 写入警告？**
本机上 `.pytest_cache` 目录的权限异常，pytest 写缓存失败并给出 warning，**不影响用例结果**；可以删掉该目录让其重建。

**`scripts/compare_chunking.py` 报「找不到测试文档目录」？**
它固定读取 `data/raw/test_corpus/*.md`，而 `data/` 被 `.gitignore` 忽略、不随仓库发布。自己准备约 10 篇 md 放进去，或改脚本里的路径。注意：找不到时脚本打印错误但**退出码仍是 0**（已知缺陷）。

**`python scripts/compare_chunking.py` 输出全零/块数异常？**
确认是从**项目根目录**运行（脚本用相对路径），且语料确实是 Markdown。

**`hello_llm.py` 报 `ValueError`？**
它不读 `.env`，只读进程环境变量，且不解析命令行参数。先用 `$env:DEEPSEEK_API_KEY=...`，或直接用 `main.py`。

**提示 `unrecognized arguments`？**
问题里有空格或标点却没加引号，用 `python main.py "你的问题"`。

**`core/prompts.py` 是空的？**
是占位文件，当前没有代码引用它。

---

## 9. 路线图与已知限制

### 9.1 路线图

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 第 1 阶段 | 环境、跑通 API、配置校验、异常分类、重试、日志、命令行 | ✅ 已完成 |
| 第 5 周 | 文档解析与分块（markdown / pdf + 三种策略 + 对比实验 + 报告） | ✅ 已完成 |
| 第 6 周 | 检索链路（向量化 / 向量库 / top-k 召回） | ⏳ 未开始 |
| 第 7 周 | 端到端问答 + 引用溯源 + 拒答 | ⏳ 未开始 |
| 第 8 周 | 50 条评测集 + 基线数字 | ⏳ 未开始 |
| 第 9 周 | 混合检索 + rerank 优化 | ⏳ 未开始 |
| 第 11 周 | 手搓 agent loop + 工具调用 | ⏳ 未开始 |
| 第 12 周 | workflow vs agent 对照实验 | ⏳ 未开始 |
| 第 13 周 | 全链路 trace 与失败分析 | ⏳ 未开始 |

### 9.2 评测计划（目前没有任何数据）

计划中的 50 条评测集：事实查找 13 条 / 跨段综合 12 条 / 无法回答 13 条 / 陷阱题 12 条。届时观测：Recall@5、任务成功率、引用准确率、拒答正确率、误拒率、平均延迟 / 单次成本。

> 评测集尚未建立，**以上指标全部没有数字**。拒答正确率与误拒率必须成对监控，否则「一律拒答」就能刷满拒答率。

### 9.3 已知限制

- **没有检索**：`Chunk` 只是切好放在内存里，没有向量化、没有索引、没有召回
- **没有引用与拒答**：`section` / `page` 字段目前不会出现在回答里
- **PDF 页码未打通**：`PDFLoader` 在 `raw_text` 里插入了 `<<<PAGE n>>>` 并把偏移写进 `metadata["offsets"]`，但没有任何分块策略解析它，`Chunk.page` 始终为空
- **recursive 实现朴素**：没有小块合并，实验截断率 79%–100%，报告里已如实记录
- **实验指标是启发式代理**：截断率按标点判断，不等于真实检索命中率；最终优劣要等第 6–8 周
- **语料不入库**：`data/` 被忽略，新克隆的仓库跑不了对比实验，需要自备语料
- **测试覆盖不均**：24 个用例集中在配置、异常、重试、模型、分块；`tests/test_loaders.py` 还是占位（0 用例），加载器与 `ingest()` 没有端到端测试
- **日志记录提问原文**：敏感问题会落在 `logs/app.log` 里（本地、不入库，但注意别打包外发）
- **错误信息不够干净**：配置校验直接暴露 pydantic 原文（较长），且会带出 key 前 3 个字符
- **无流式、无多轮、无 token 计费统计**
- **单文件摄入**：只有 `ingest(单文件)`，没有目录批量入口或命令行工具
- **`.txt` 未支持**：`DocumentType.TEXT` 已定义，但没有对应 loader
- **`hello_llm.py` 与 `main.py` 功能重叠**，仅作历史示例保留

---

## 10. 技术选型与取舍

- **pydantic v2 建模**（`core/models.py`）：配置、文档、块、分块参数都用同一套模型，自动校验 + IDE 补全 + 清晰的报错；后续检索结果、引用、评测样本可以复用同一套建模方式。
- **pypdf 而不是 pdfplumber**：pypdf 轻量、纯 Python，普通的文本型 PDF（例如简历）能保留换行与符号；第 5 周只需要文本与页码，够用。若之后遇到多栏排版或扫描件，再考虑换（届时要加 OCR 或版面分析）。
- **自研三种分块器，而不是直接用 LangChain 的 splitter**：代码量小、行为可读、便于写不变量测试（块文本必须等于原文区间）；也方便按实验数据调整。
- **关闭 SDK 重试**：只保留项目层的一套退避策略，避免两层重试把一次失败放大成 9 次请求。
- **实验指标用代理指标**：真正的检索收益要等向量库和评测集就绪，第 5 周先用截断率 / 标题混杂度排除明显不可用的策略（例如 fixed）。

---

## 11. 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

---

**文档说明**：本 README 只描述仓库中**已存在**的代码；命令与输出于 2026-11-02 在本机实际执行核对（pytest 24 通过、对比实验数据与 `docs/chunking-report.md` 一致）。
