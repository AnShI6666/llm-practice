# llm-practice

> 一个用来练习「把大模型接进真实项目」的最小可用骨架：**配置校验 + DeepSeek 调用封装（超时 / 重试）+ 命令行提问 + 分层报错 + 统一日志**。
>
> 当前处于第 1 阶段。仓库里**还没有**文档解析、向量检索、引用溯源、拒答和评测集——这些属于路线图中的后续阶段，见[第 8 节](#8-路线图与已知限制)。

---

## 1. 这个项目解决什么问题

随手写一段「调通就行」的 LLM 代码，几乎一定会反复踩下面这些坑。本项目把每一项都收敛成一处可复用的实现：

| 常见问题 | 本项目的处理 |
| --- | --- |
| API Key 硬编码在代码里，或被误提交到 Git | 统一从 `.env` 或进程环境变量读，不写进代码；`.gitignore` 已忽略 `.env`，仓库只提交 `.env.example` |
| 报错分不清是「没配 key」还是「接口失败」 | 拆成两种异常：`ConfigError`（配置缺失，不再发请求）与 `LLMError`（重试后仍失败） |
| 网络偶发抖动导致直接失败 | 最多尝试 3 次，失败后按 1 秒、2 秒退避再试 |
| 每个脚本各自 `new` 客户端、各自读环境变量 | `core/llm_client.py` 持有连接配置，对外只暴露 `ask()`；`main.py` 只负责命令行 |
| 出问题看不到请求过程 | 统一日志：提问内容（前 50 字）、回答长度、每次重试的原因 |
| Windows 控制台中文乱码、每次都要先激活虚拟环境 | `run.bat` 自动切 UTF-8、直接调用 `.venv` 里的解释器并透传参数 |

**定位说明**：这是一个学习项目，第 1 阶段的目标是把「读配置 → 建客户端 → 调用 → 重试 → 分类报错」做成后续阶段能直接复用的地基，而不是一个功能完整的问答产品。

**长期目标（尚未实现）**：面向私有文档库的可溯源问答 Agent —— 上传一批文档后用自然语言提问，系统检索相关资料后给出**带引用来源**的回答，查不到资料时明确拒答。

---

## 2. 主要功能（当前仓库真实已实现）

| 功能 | 实现位置 | 说明 |
| --- | --- | --- |
| 配置加载与校验 | `core/config.py` → `load_config()` | 读取 `.env`，校验必填的 `DEEPSEEK_API_KEY`；支持用 `MODEL` / `BASE_URL` 覆盖默认值；缺失时抛 `ConfigError` |
| LLM 调用封装 | `core/llm_client.py` → `LLMClient.ask()` | 用 OpenAI SDK 访问 DeepSeek 兼容接口；单轮对话，可选 `system` 提示词；默认 30 秒超时 |
| 失败重试 | `core/llm_client.py` | 最多尝试 3 次，失败后等待 1s、2s 再试；全部失败抛 `LLMError`（携带最后一次原始异常） |
| 命令行入口 | `main.py` | `argparse`：可选位置参数 `question`（默认「用一句话解释什么是大语言模型。」）与 `--system`；成功返回 0，失败返回 1，参数错误返回 2 |
| 统一日志 | `main.py` 里的 `logging.basicConfig` + 客户端内 `logger` | INFO 级，输出时间、级别、模块名：提问内容、回答长度、每次重试的警告；写入 **stderr** |
| Windows 一键启动 | `run.bat` | `chcp 65001` 切 UTF-8 → 切到脚本目录 → 用 `.venv\Scripts\python.exe` 运行 `main.py %*` → 结束时 `pause` 停留 |
| 阶段 1 最小示例 | `hello_llm.py` | 硬编码一个问题，演示最直接的调用方式；不读 `.env`、不解析命令行参数、不做错误处理 |
| 提示词模块占位 | `core/prompts.py` | 目前是空文件，计划放提示词模板，暂未被任何代码引用 |

**明确未实现**（请不要按这些能力使用本项目）：文档上传与解析、文本分块、向量库 / 检索、引用溯源、无答案拒答、评测集与指标、流式输出、多轮对话上下文、Agent 工具调用。

---

## 3. 使用方法

### 3.1 环境要求

- Python **3.10+**（代码使用了 `str | None` 等新语法；本机验证版本为 Python 3.13.5）
- 可访问 `https://api.deepseek.com` 的网络，以及一个可用的 DeepSeek API Key
- Windows / macOS / Linux 均可运行 `main.py`；`run.bat` 仅适用于 Windows

### 3.2 安装依赖

```bash
# 建议用虚拟环境（仓库里已有一个 .venv，可直接复用）
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` 只有三项：`openai>=1.0.0`、`python-dotenv>=1.0.0`、`pytest>=8.0.0`。

> 下面命令里的 `python` 都指**已激活虚拟环境**的解释器；不想激活时可直接写 `.\.venv\Scripts\python.exe`。

### 3.3 配置密钥

```bash
# Windows
copy .env.example .env
# macOS / Linux
# cp .env.example .env
```

然后编辑 `.env`：

```ini
DEEPSEEK_API_KEY=sk-你的真实密钥

# 下面两项可选，不写则使用默认值
MODEL=deepseek-chat
BASE_URL=https://api.deepseek.com
```

注意：`.env.example` 中的 `sk-your-key-here` 是占位符，**必须替换**，否则接口会返回 401。

配置优先级：**进程环境变量 > `.env`**（`load_dotenv()` 默认不覆盖已存在的环境变量）。因此临时换 key 可以直接在 shell 里设：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
python main.py "什么是 RAG？"
```

### 3.4 运行

```powershell
# 1）默认问题
python main.py

# 2）指定问题（含空格或标点时必须加引号）
python main.py "什么是 RAG？"

# 3）指定 system 提示词，控制回答风格
python main.py "解释一下 Agent" --system "你是一个只用一句话回答的助手"

# 4）查看帮助
python main.py --help
```

Windows 下也可以直接双击 `run.bat`，或在命令行传参：

```bat
run.bat
run.bat "什么是 RAG？" --system "只用一句话回答"
```

`run.bat` 会把参数原样透传给 `main.py`，并在结束时停在 `pause`，适合双击运行后查看结果。

### 3.5 在 Python 代码里调用

```python
from core.config import load_config
from core.llm_client import LLMClient

cfg = load_config()
client = LLMClient(
    api_key=cfg["api_key"],
    model=cfg["model"],
    base_url=cfg["base_url"],
)
print(client.ask("用一句话说明 DeepSeek 是什么", system="只用一句话回答"))
```

`LLMClient` 的构造参数：`api_key`（必填）、`model`、`base_url`、`timeout=30.0`、`max_retries=3`。

### 3.6 输出约定

| 通道 | 内容 | 典型用法 |
| --- | --- | --- |
| stdout | 仅模型回答文本 | `python main.py "…" > answer.txt` |
| stderr | 日志（提问、HTTP 请求、回答长度、重试警告） | 不想看日志：`2>$null` |
| 退出码 | `0` 成功 / `1` 配置缺失或调用失败 / `2` 参数错误 | 便于脚本判断成败 |

---

## 4. 输入输出实例

> 以下输出均为在本机对 `deepseek-chat` 的真实运行记录；模型回答每次都不完全相同，措辞会有变化。

### 4.1 默认问题

输入：

```powershell
python main.py
```

stdout（回答）：

```text
大语言模型是一种基于海量文本数据训练、通过预测下一个词来生成连贯语言的人工智能模型。
```

stderr（日志）：

```text
2026-10-21 19:25:35,410 [INFO] core.llm_client: 提问：用一句话解释什么是大语言模型。
2026-10-21 19:25:35,722 [INFO] httpx2: HTTP Request: POST https://api.deepseek.com/chat/completions "HTTP/1.1 200 OK"
2026-10-21 19:25:36,245 [INFO] core.llm_client: 回答长度：42 字符
```

退出码：`0`

### 4.2 指定问题 + system 提示词

输入：

```powershell
python main.py "什么是 RAG？" --system "你是一个只用一句话回答的助手"
```

输出：

```text
RAG（检索增强生成）是一种将外部知识库的检索能力与大语言模型生成能力相结合的技术，让模型在回答前先检索相关资料，从而提高回答的准确性和时效性。
```

对应日志：`提问：什么是 RAG？` / `回答长度：72 字符`，退出码 `0`。

### 4.3 查看帮助

输入：

```powershell
python main.py --help
```

输出：

```text
usage: main.py [-h] [--system SYSTEM] [question]

问 DeepSeek 一个问题

positional arguments:
  question         要问的问题（默认问什么是大语言模型）

options:
  -h, --help       show this help message and exit
  --system SYSTEM  可选的 system 提示词
```

### 4.4 配置缺失（没有可用的 API Key）

输入：项目根目录不存在 `.env`，且环境变量里也没有 `DEEPSEEK_API_KEY`。

```powershell
python main.py
```

输出：

```text
配置错误：未找到 DEEPSEEK_API_KEY。请在项目根目录创建 .env 文件，内容形如：DEEPSEEK_API_KEY=sk-xxxx
```

退出码：`1`（此时不会发出任何请求）。

### 4.5 Key 无效（演示重试与最终失败）

输入：

```powershell
$env:DEEPSEEK_API_KEY = "sk-invalid-key-for-test"
python main.py "测试重试"
```

stderr（每次失败都会警告，等待 1 秒、2 秒后重试；共 3 次尝试）：

```text
2026-10-21 19:25:47,001 [WARNING] core.llm_client: 第 1/3 次调用失败：Error code: 401 - {'error': {'message': 'Authentication Fails, Your api key: ****test is invalid ...'}}
2026-10-21 19:25:47,001 [INFO] core.llm_client: 1 秒后重试
2026-10-21 19:25:48,114 [WARNING] core.llm_client: 第 2/3 次调用失败：Error code: 401 - {...}
2026-10-21 19:25:48,114 [INFO] core.llm_client: 2 秒后重试
2026-10-21 19:25:50,194 [WARNING] core.llm_client: 第 3/3 次调用失败：Error code: 401 - {...}
```

stdout（最终失败提示）：

```text
调用失败：调用失败，已重试 3 次：Error code: 401 - {'error': {'message': 'Authentication Fails, Your api key: ****test is invalid (request_id: ...)', 'type': 'authentication_error', 'param': None, 'code': 'invalid_request_error'}}
```

退出码：`1`

### 4.6 参数写错（问题没加引号）

输入：

```powershell
python main.py 什么是 RAG？
```

stderr：

```text
usage: main.py [-h] [--system SYSTEM] [question]
main.py: error: unrecognized arguments: RAG？
```

退出码：`2`（由 `argparse` 直接结束进程，未发出请求）。

### 4.7 阶段 1 脚本 `hello_llm.py`

它只读**进程环境变量**，不读 `.env`，也没有错误处理。未设置变量时：

```text
Traceback (most recent call last):
  File "<项目根目录>\hello_llm.py", line 7, in <module>
    raise ValueError("请先设置 DEEPSEEK_API_KEY 环境变量")
ValueError: 请先设置 DEEPSEEK_API_KEY 环境变量
```

正确用法（先在 shell 里导出变量，问题固定写在代码里，命令行参数会被忽略）：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
python hello_llm.py
```

```text
大语言模型是一种基于海量文本数据训练、通过预测下一个词来生成连贯自然语言的人工智能模型。
```

### 4.8 作为库调用

输入：第 3.5 节的代码片段。

输出：

```text
DeepSeek是由深度求索公司创造的AI助手，具备强大的语言理解和生成能力，支持多种任务处理。
```

---

## 5. 项目结构

```text
.
├── main.py              # 命令行入口：解析参数 → 读配置 → 调 LLM → 打印回答
├── hello_llm.py         # 阶段 1 最小示例：硬编码问题、直接调接口
├── run.bat              # Windows 启动脚本：切 UTF-8、用 .venv 解释器、透传参数
├── core/
│   ├── __init__.py      # 空文件，标识 Python 包
│   ├── config.py        # load_config()：读 .env、校验 key、支持 MODEL / BASE_URL 覆盖
│   ├── llm_client.py    # LLMClient.ask()：调用、超时、指数退避重试、异常封装
│   └── prompts.py       # 空占位，计划放提示词模板，当前无逻辑
├── requirements.txt     # openai / python-dotenv / pytest
├── .env.example         # 环境变量模板（占位密钥，需替换）
├── .env                 # 本地真实密钥，已被 .gitignore 忽略，不提交
├── .gitignore           # 忽略 .env、虚拟环境、缓存、数据产物等
└── LICENSE              # MIT 许可证
```

`.idea/` 是 IDE 本地配置，`.venv/` 是虚拟环境，两者都不属于项目代码。

---

## 6. 配置项一览

| 变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `DEEPSEEK_API_KEY` | 是 | — | DeepSeek 密钥；缺失时 `load_config()` 抛 `ConfigError` |
| `MODEL` | 否 | `deepseek-chat` | 请求的模型名；写错会返回 400 |
| `BASE_URL` | 否 | `https://api.deepseek.com` | 接口地址，可指向兼容 OpenAI 协议的代理 |

以上变量既可以写在 `.env`，也可以直接导出到 shell（shell 优先）。

---

## 7. 常见问题

**控制台中文乱码？**
Windows 默认代码页是 GBK。用 `run.bat`（内部已 `chcp 65001`），或手动执行 `chcp 65001`，或设置 `$env:PYTHONIOENCODING = "utf-8"` 后再运行。

**返回 401？**
`.env` 里的 key 还是占位符 `sk-your-key-here`，或 key 已失效。检查 `.env`，也可先确认 shell 里的环境变量是否覆盖了它。

**返回 400 并提示模型名不被支持？**
`MODEL` 写错了，删掉该行使用默认的 `deepseek-chat`。

**为什么报错前要等 3 秒多？**
目前所有异常都会触发重试，包括 401、400 这类重试也无用的错误（`llm_client.py` 里已标注后续再细分异常类型）。`max_retries` 目前只能在自己实例化 `LLMClient` 时调小，`main.py` 没有对外暴露这个参数。

**`hello_llm.py` 报 `ValueError`？**
它不读 `.env`，只读进程环境变量。先 `export` / `$env:DEEPSEEK_API_KEY=...`，或者改用 `main.py`。

**提示 `unrecognized arguments`？**
问题里有空格或标点却没加引号。用 `python main.py "你的问题"`。

**没有测试可以跑吗？**
`requirements.txt` 里装了 `pytest`，但仓库目前还没有测试文件。

---

## 8. 路线图与已知限制

### 8.1 路线图

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 第 1 阶段 | 环境搭建、跑通模型 API、配置校验、重试封装、命令行入口（即本 README 描述的全部能力） | ✅ 已完成 |
| 第 5 周 | 文档解析与分块 | ⏳ 未开始 |
| 第 6 周 | 检索链路 | ⏳ 未开始 |
| 第 7 周 | 端到端问答 + 引用溯源 + 拒答 | ⏳ 未开始 |
| 第 8 周 | 50 条评测集 + 基线数字 | ⏳ 未开始 |
| 第 9 周 | 混合检索 + rerank 优化 | ⏳ 未开始 |
| 第 11 周 | 手搓 agent loop + 工具调用 | ⏳ 未开始 |
| 第 12 周 | workflow vs agent 对照实验 | ⏳ 未开始 |
| 第 13 周 | 全链路 trace 与失败分析 | ⏳ 未开始 |

### 8.2 评测计划（目前没有任何数据）

固定的 50 条评测集计划由**事实查找 13 条 / 跨段综合 12 条 / 无法回答 13 条 / 陷阱题 12 条**组成，每次改动后重跑。届时计划观测的指标：

| 指标 | 定义 |
| --- | --- |
| 检索召回率 Recall@5 | 正确证据出现在检索前 5 条的比例 |
| 任务成功率 | 回答正确且完整的比例 |
| 引用准确率 | 引用来源确实支持该句的比例 |
| 拒答正确率 | 无答案的问题中被正确拒答的比例 |
| 误拒率 | 有答案却被拒答的比例 |
| 平均延迟 / 单次成本 | 端到端耗时与 token 费用 |

> 评测集尚未建立，因此以上指标**暂无基线数字**。另外拒答正确率与误拒率必须成对监控，否则模型学会「一律拒答」就能刷满拒答率。

### 8.3 已知限制

- 单轮对话：没有历史上下文，每次提问都是独立请求
- 非流式：`stream=False`，必须等完整回答才返回
- 不支持 `temperature`、`max_tokens` 等采样参数
- 不统计 token 用量与费用，也没有 trace / 落盘日志
- 异常未分类：认证失败、参数错误也会重试 3 次，浪费约 3 秒
- 存在两层重试：OpenAI SDK 自身对连接错误 / 5xx / 429 默认再重试 2 次，叠加外层 3 次尝试后，最坏情况一次提问会发出约 9 个 HTTP 请求（401 这类错误 SDK 不重试，所以正好 3 次）
- `LLMError` 的文案本身含「调用失败」，与 `main.py` 的外层前缀重复，提示语观感略啰嗦
- 无自动化测试；`core/prompts.py` 仍为空；原计划的 `DECISIONS.md` 技术决策日志尚未创建
- `hello_llm.py` 与 `main.py` 功能重叠，仅作为阶段 1 的最小示例保留，后续可删除

---

## 9. 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

---

**文档说明**：本 README 只描述仓库中**已存在**的代码；文中的命令与输出均在 2026-10-21 于本机实际执行核对。
