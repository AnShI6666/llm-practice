"""测试异常分类是否正确。关键是：不同错误要分到不同类。"""
import httpx
import pytest

from core.exceptions import (
    LLMAuthError, LLMBadRequestError, LLMRateLimitError, LLMServerError,
    LLMTimeoutError,
)
from core.llm_client import classify_error


def _http_error(status: int) -> Exception:
    """构造一个带状态码的异常。"""
    resp = httpx.Response(status_code=status, request=httpx.Request("POST", "http://x"))
    return httpx.HTTPStatusError("err", request=resp.request, response=resp)


def test_401_is_auth_error():
    assert isinstance(classify_error(_http_error(401)), LLMAuthError)


def test_403_is_auth_error():
    assert isinstance(classify_error(_http_error(403)), LLMAuthError)


def test_429_is_rate_limit():
    assert isinstance(classify_error(_http_error(429)), LLMRateLimitError)


def test_400_is_bad_request():
    assert isinstance(classify_error(_http_error(400)), LLMBadRequestError)


def test_500_is_server_error():
    assert isinstance(classify_error(_http_error(500)), LLMServerError)


def test_timeout_detected():
    assert isinstance(classify_error(TimeoutError("timed out")), LLMTimeoutError)
