"""测试 LLMClient 的重试逻辑。"""
import pytest
from unittest.mock import MagicMock, patch
import httpx

from core.llm_client import LLMClient
from core.exceptions import LLMAuthError, LLMRateLimitError


# --- 1. 辅助函数（解决 _http_error 报错） ---
def _http_error(status: int) -> Exception:
    """构造一个带状态码的 httpx 异常。"""
    resp = httpx.Response(status_code=status, request=httpx.Request("POST", "http://x"))
    return httpx.HTTPStatusError("err", request=resp.request, response=resp)


# --- 2. 辅助函数（解决 _make_client 报错） ---
def _make_client(max_retries: int) -> LLMClient:
    """构造一个底层 OpenAI 客户端被 Mock 掉的 LLMClient。"""
    with patch("core.llm_client.OpenAI") as mock_openai:
        client = LLMClient(api_key="sk-fake", max_retries=max_retries)
        return client


# --- 3. 测试用例（解决 pytest、LLMAuthError 报错） ---
def test_auth_error_not_retried():
    """认证失败（401）应该只调用 1 次就放弃，不能重试。"""
    client = _make_client(max_retries=3)
    client._client.chat.completions.create.side_effect = _http_error(401)

    with pytest.raises(LLMAuthError):
        client.ask("hi")

    # ★ 核心断言：只调了 1 次
    assert client._client.chat.completions.create.call_count == 1


def test_rate_limit_is_retried(monkeypatch):
    """限流（429）应该重试，直到达到最大次数。"""
    # 把 time.sleep 替换为空函数，防止测试真的等 10 秒、20 秒！
    monkeypatch.setattr("core.llm_client.time.sleep", lambda s: None)

    client = _make_client(max_retries=3)
    client._client.chat.completions.create.side_effect = _http_error(429)

    with pytest.raises(LLMRateLimitError):
        client.ask("hi")

    # ★ 核心断言：调用了 3 次（1次初始 + 2次重试）
    assert client._client.chat.completions.create.call_count == 3


def test_successful_call():
    """成功的调用应该只调 1 次，并返回内容。"""
    client = _make_client(max_retries=3)

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "我是回答"
    client._client.chat.completions.create.return_value = mock_response

    answer = client.ask("hi")
    assert answer == "我是回答"
    assert client._client.chat.completions.create.call_count == 1