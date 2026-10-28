"""测试异常分类是否正确。关键是：不同错误要分到不同类。"""
import httpx
import pytest

from core.exceptions import (
    LLMAuthError, LLMBadRequestError, LLMRateLimitError, LLMServerError,
    LLMTimeoutError,
)
from core.llm_client import classify_error


@pytest.mark.parametrize("status,expected", [
    (401, LLMAuthError),
    (403, LLMAuthError),
    (400, LLMBadRequestError),
    (429, LLMRateLimitError),
    (500, LLMServerError),
    (502, LLMServerError),
    (503, LLMServerError),
])
def test_http_status_classification(make_http_error, status, expected):
    """HTTP 状态码 → 异常类型的映射表。"""
    assert isinstance(classify_error(make_http_error(status)), expected)


def test_timeout_classification():
    """超时应该被识别。"""
    assert isinstance(classify_error(TimeoutError("timed out")), LLMTimeoutError)
