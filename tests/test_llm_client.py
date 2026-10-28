"""测试 LLMClient 的重试逻辑。"""
import pytest
from unittest.mock import MagicMock

from core.exceptions import LLMAuthError, LLMRateLimitError

# --- 3. 测试用例（解决 pytest、LLMAuthError 报错） ---
def test_auth_error_not_retried(fake_llm_client, make_http_error):
    """认证失败应该只调 1 次，不重试。"""
    fake_llm_client._client.chat.completions.create.side_effect = make_http_error(401)

    with pytest.raises(LLMAuthError):
        fake_llm_client.ask("hi")

    assert fake_llm_client._client.chat.completions.create.call_count == 1


def test_rate_limit_is_retried(fake_llm_client, make_http_error, no_sleep):
    """限流应该重试到上限。"""
    fake_llm_client._client.chat.completions.create.side_effect = make_http_error(429)

    with pytest.raises(LLMRateLimitError):
        fake_llm_client.ask("hi")

    assert fake_llm_client._client.chat.completions.create.call_count == 3


def test_successful_call(fake_llm_client):
    """成功的调用应该只调 1 次，并返回内容。"""

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "我是回答"
    fake_llm_client._client.chat.completions.create.return_value = mock_response

    answer = fake_llm_client.ask("hi")
    assert answer == "我是回答"
    assert fake_llm_client._client.chat.completions.create.call_count == 1