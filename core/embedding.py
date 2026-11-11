"""Embedding 客户端封装：统一处理调用、超时、重试、维度自检。"""
import logging
import time
from typing import Sequence
import httpx

logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """向量化失败。"""

class EmbeddingClient:
    def __init__(self,api_key: str,model: str = "embedding-3",
                 base_url: str = "https://open.bigmodel.cn/api/paas/v4",dim: int = 1024,
                 timeout: float = 30.0,max_retries: int = 3,) -> None:
        self.model = model
        self.dim = dim  # ★ 记录维度，用于自检
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def embed(self, text: str) -> list[float]:
        """单条向量化。"""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化。返回顺序与输入一致。失败时抛出 EmbeddingError。"""
        if not texts:
            return []

        url = f"{self._base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": list(texts),
            # 智谱支持自定义维度，显式传入以保证与 self.dim 一致
            "dimensions": self.dim,
        }

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)

                # 不可重试的错误：4xx（除429外）立刻失败
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    self._raise_for_http_error(resp)

                # 可重试的错误：429 或 5xx
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    last_error = EmbeddingError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    wait = 2 ** (attempt - 1)
                    logger.warning(
                        "Embedding 第 %d/%d 次失败（HTTP %d），%d 秒后重试",
                        attempt, self.max_retries, resp.status_code, wait,
                    )
                    if attempt < self.max_retries:
                        time.sleep(wait)
                    continue

                # 2xx 成功
                data = resp.json()
                vectors = [item["embedding"] for item in data["data"]]

                # ★ 核心：自检维度与数量
                self._self_check(vectors, len(texts))

                logger.info("Embedding 成功，%d 条，维度 %d", len(vectors), self.dim)
                return vectors

            except httpx.TimeoutException as exc:
                last_error = exc
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "Embedding 超时（第 %d/%d 次），%d 秒后重试",
                    attempt, self.max_retries, wait,
                )
                if attempt < self.max_retries:
                    time.sleep(wait)
            except EmbeddingError:
                raise  # 自检失败等业务异常，不重试
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Embedding 未知错误（第 %d/%d 次）：%s",
                    attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))

        raise EmbeddingError(
            f"向量化失败，已重试 {self.max_retries} 次：{last_error}"
        ) from last_error

    def _self_check(self, vectors: list[list[float]], expected_count: int) -> None:
        """校验返回的向量数量和维度。"""
        if len(vectors) != expected_count:
            raise EmbeddingError(
                f"向量数量不符：期望 {expected_count}，实际 {len(vectors)}。"
            )
        for i, v in enumerate(vectors):
            if len(v) != self.dim:
                raise EmbeddingError(
                    f"第 {i} 条向量维度不符：期望 {self.dim}，实际 {len(v)}。"
                    f"请检查 EMBEDDING_DIM 配置是否与模型匹配。"
                )

    def _raise_for_http_error(self, resp: httpx.Response) -> None:
        """将不可重试的 HTTP 错误转换为带诊断信息的 EmbeddingError。"""
        detail = resp.text[:200]
        if resp.status_code in (401, 403):
            msg = f"认证失败（{resp.status_code}）：请检查智谱 API Key 是否正确或已过期。{detail}"
        elif resp.status_code == 400:
            msg = f"请求非法（400）：可能是参数格式错误（如 dimensions 不被支持）。{detail}"
        else:
            msg = f"Embedding 请求失败（HTTP {resp.status_code}）：{detail}"
        raise EmbeddingError(msg)