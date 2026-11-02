"""数据模型校验测试。重点是：非法输入要能被拦住。"""
import pytest
from pydantic import ValidationError

from core.models import Config


def test_valid_config():
    """正常配置应该通过。"""
    cfg = Config(api_key="sk-1234567890abcdef")
    assert cfg.model == "deepseek-chat"
    assert cfg.timeout == 30.0


def test_key_without_prefix_rejected():
    """Key 不以 sk- 开头应该被拒绝。"""
    with pytest.raises(ValidationError, match="sk-"):
        Config(api_key="wrong-key-1234567890")


def test_key_with_whitespace_stripped():
    """Key 前后的空白应该被自动去掉。"""
    cfg = Config(api_key="  sk-1234567890abcdef  ")
    assert cfg.api_key == "sk-1234567890abcdef"


def test_negative_timeout_rejected():
    """timeout 不能是负数。"""
    with pytest.raises(ValidationError):
        Config(api_key="sk-1234567890abcdef", timeout=-1)


def test_max_retries_bounds():
    """max_retries 必须在 1~10 之间。"""
    with pytest.raises(ValidationError):
        Config(api_key="sk-1234567890abcdef", max_retries=0)
    with pytest.raises(ValidationError):
        Config(api_key="sk-1234567890abcdef", max_retries=100)


@pytest.mark.parametrize("bad_url", ["ftp://x", "example.com", ""])
def test_bad_base_url_rejected(bad_url):
    """base_url 必须是 http(s)://。"""
    with pytest.raises(ValidationError):
        Config(api_key="sk-1234567890abcdef", base_url=bad_url)

def test_invalid_chunk_rejected():
    """验证 end_char 必须大于 start_char。"""
    from core.models import Chunk
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="c1", doc_id="d1", text="abc",
            start_char=10, end_char=5, # 故意写错
            char_count=3, token_estimate=1
        )