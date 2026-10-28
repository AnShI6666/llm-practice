"""项目的数据模型。

用 pydantic 的好处：
- 自动校验类型和约束（如字符串非空、数值范围）
- IDE 能补全字段名
- 出错时报错信息很明确（哪一行、哪个字段、为什么错）
- 未来做 RAG 时，检索结果/引用/评测样本都用同样方式建模
"""
from pydantic import BaseModel, Field, field_validator


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
        v = v.strip()
        if not v.startswith("sk-"):
            raise ValueError(f"API Key 应以 sk- 开头，当前是 {v[:3]}...")
        return v

    @field_validator("base_url")
    @classmethod
    def _check_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url 必须以 http:// 或 https:// 开头")
        return v.rstrip("/")