import pytest
from unittest.mock import MagicMock, patch
from core.embedding import EmbeddingClient, EmbeddingError

def test_embedding_self_check():
    """维度自检：如果模型返回维度不对，应该立刻抛错。"""
    client = EmbeddingClient(api_key="sk-test", dim=1024)
    # 伪造一个长度为 3 的向量
    with pytest.raises(EmbeddingError, match="向量维度不符"):
        client._self_check([[1.0, 2.0, 3.0]], expected_count=1)

def test_embedding_batch_order():
    """验证批量接口返回的顺序与输入一致。"""
    client = EmbeddingClient(api_key="sk-test", dim=2)
    texts = ["A", "B"]

    # 这里用简单的 mock 代替真实 API
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"embedding": [1.0, 0.0]},  # 对应 A
                {"embedding": [0.0, 1.0]}  # 对应 B
            ]
        }
        mock_post.return_value = mock_resp

        vectors = client.embed_batch(texts)
        assert vectors == [[1.0, 0.0], [0.0, 1.0]]

def test_embedding_401_no_retry():
    """401 认证失败不重试。"""
    client = EmbeddingClient(api_key="sk-bad", max_retries=3)
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_post.return_value = mock_resp

        with pytest.raises(EmbeddingError, match="认证失败"):
            client.embed("test")
        assert mock_post.call_count == 1  # 只调 1 次