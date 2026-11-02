"""pytest 共享配置与 fixture。

conftest.py 会被 pytest 自动发现，里面的 fixture 所有测试文件都能用。
"""
import pytest
from core.llm_client import LLMClient
from core.models import Document, DocumentType, ChunkingConfig
from pathlib import Path
from unittest.mock import MagicMock

@pytest.fixture
def fake_llm_client() -> LLMClient:
    """一个把底层 OpenAI 客户端换成 mock 的 LLMClient。

    用途：测试重试逻辑时不打真实 API，快且免费。
    """
    client = LLMClient(api_key="sk-test-key-123456", max_retries=3)
    client._client = MagicMock()
    return client


@pytest.fixture
def no_sleep(monkeypatch):
    """把 time.sleep 变成空操作，让重试测试瞬间完成。"""
    monkeypatch.setattr("core.llm_client.time.sleep", lambda s: None)


@pytest.fixture
def make_http_error():
    """构造带指定状态码的 HTTP 异常。"""
    import httpx

    def _make(status: int) -> Exception:
        req = httpx.Request("POST", "http://test")
        resp = httpx.Response(status_code=status, request=req)
        return httpx.HTTPStatusError("err", request=req, response=resp)

    return _make

@pytest.fixture
def make_doc():
    def _make(text="这是一段测试文本。包含多个句子。用来验证切块。", doc_id="test_doc"):
        return Document(
            doc_id=doc_id, source_path=Path(f"{doc_id}.md"),
            doc_type=DocumentType.MARKDOWN, raw_text=text
        )
    return _make

@pytest.fixture
def make_config():
    def _make(size=50, overlap=10, strategy="fixed"):
        return ChunkingConfig(chunk_size=size, overlap=overlap, strategy=strategy)
    return _make