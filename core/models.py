"""项目的数据模型。

用 pydantic 的好处：
- 自动校验类型和约束（如字符串非空、数值范围）
- IDE 能补全字段名
- 出错时报错信息很明确（哪一行、哪个字段、为什么错）
- 未来做 RAG 时，检索结果/引用/评测样本都用同样方式建模
"""
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

class Config(BaseModel):
    """运行配置。"""

    api_key: str = Field(..., min_length=10, description="DeepSeek API Key")
    model: str = Field(default="deepseek-chat")
    base_url: str = Field(default="https://api.deepseek.com")
    timeout: float = Field(default=30.0, gt=0, le=300)
    max_retries: int = Field(default=4, ge=1, le=10)

    @field_validator("api_key")
    @classmethod
    def _check_key_prefix(cls, v: str) -> str:
        """Key 必须以 sk- 开头，且不能有空白字符。"""
        v = v.strip()  #去除api_key前面的空白
        if not v.startswith("sk-"):
            raise ValueError(f"API Key 应以 sk- 开头，当前是 {v[:3]}...")
        return v

    @field_validator("base_url")
    @classmethod
    def _check_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url 必须以 http:// 或 https:// 开头")
        return v.rstrip("/")

class DocumentType(str, Enum):
    """文档类型。用枚举而不是裸字符串，避免拼错。"""

    MARKDOWN = "markdown"
    PDF = "pdf"
    TEXT = "text"


class Document(BaseModel):
    """一篇原始文档。"""

    doc_id: str = Field(..., description="文档唯一标识，建议用文件名去扩展名")
    source_path: Path = Field(..., description="原始文件路径")
    doc_type: DocumentType
    title: str | None = Field(default=None, description="文档标题（若能解析出来）")
    raw_text: str = Field(..., min_length=1, description="全文纯文本")
    metadata: dict[str, str] = Field(default_factory=dict, description="额外元信息")

    @field_validator("doc_id")
    @classmethod
    def _no_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("doc_id 不能为空")
        if any(c.isspace() for c in v):
            raise ValueError("doc_id 不应包含空白字符，建议用下划线")
        return v


class Chunk(BaseModel):
    """一个文本块。这是 RAG 系统最小的检索单位。"""

    chunk_id: str = Field(..., description="块唯一标识，格式 {doc_id}_{序号:04d}")
    doc_id: str
    text: str = Field(..., min_length=1)
    # ---- 位置信息（引用溯源必需）----
    start_char: int = Field(..., ge=0, description="在原文中的字符起始位置")
    end_char: int = Field(..., ge=0, description="在原文中的字符结束位置")
    # ---- 结构信息（过滤与展示用）----
    section: str | None = Field(default=None, description="所属章节标题")
    page: int | None = Field(default=None, ge=1, description="PDF 页码")
    # ---- 度量信息（成本控制与评估用）----
    char_count: int = Field(..., gt=0)
    token_estimate: int = Field(..., gt=0, description="token 估算值")
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("end_char")
    @classmethod
    def _end_after_start(cls, v: int, info) -> int:
        start = info.data.get("start_char")
        if start is not None and v <= start:
            raise ValueError(f"end_char({v}) 必须大于 start_char({start})")
        return v

    def __str__(self) -> str:
        return f"[{self.chunk_id}] {self.text[:40]}..."


class ChunkingConfig(BaseModel):
    """分块参数。做成模型，便于实验时网格搜索。"""

    strategy: str = Field(default="fixed")
    chunk_size: int = Field(default=500, ge=50, le=4000, description="目标字符数")
    overlap: int = Field(default=50, ge=0, description="相邻块重叠字符数")

    @field_validator("overlap")
    @classmethod
    def _overlap_less_than_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError(f"overlap({v}) 必须小于 chunk_size({size})")
        return v