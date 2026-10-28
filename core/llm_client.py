"""LLM 客户端封装：统一处理调用、超时、重试、错误分类。

设计要点：
- 用类持有「一组相关状态」：client / model / timeout / max_retries
- 外部只需调 ask()，不用每次传 client 和 model
- 错误分类后只重试「可重试」的（超时 / 限流 / 5xx）
- 限流等更久（10s→20s→40s），其他指数退避（1s→2s→4s），都加抖动
"""
import logging
import random
import time
from typing import Final, Literal, TypedDict  # 新增类型导入
import httpx
from openai import OpenAI

from core.exceptions import (
    LLMAuthError,
    LLMBadRequestError,
    LLMError,
    LLMInsufficientQuotaError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMUnknownError,
)

logger: Final = logging.getLogger(__name__)

# 向后兼容：老代码 from core.llm_client import LLMError 仍然可用
__all__ = ["LLMClient", "LLMError", "classify_error", "RETRYABLE"]

# 可重试的异常类型（与 exceptions.py 里的注释一一对应）
RETRYABLE = (LLMTimeoutError, LLMRateLimitError, LLMServerError)

class ChatMessage(TypedDict):
    """消息字典的精确类型。"""
    role: Literal["system", "user", "assistant"]
    content: str

def _build_messages(question: str, system: str | None) -> list[ChatMessage]:
    """构造消息列表。

    抽成独立函数的好处：可以单独测试，不用启动整个 client。
    """
    messages: list[ChatMessage] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    return messages

def classify_error(exc: Exception) -> LLMError:
    """把底层 SDK 的各种异常翻译成项目自己的异常类型。

    调用方只需要认识 LLMError 一族，不需要知道 openai/httpx 的内部结构。
    """
    # 超时：SDK 的 APITimeoutError 底层是 httpx.TimeoutException
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return LLMTimeoutError(str(exc))
    # openai SDK 的 APITimeoutError 不一定能让上面的 isinstance 命中，按类名兜底
    if type(exc).__name__ == "APITimeoutError":
        return LLMTimeoutError(str(exc))

    # 有 HTTP 状态码的，按状态码分类
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)

    if status is not None:
        if status in (401, 403):
            return LLMAuthError(f"认证失败（{status}）：请检查 DEEPSEEK_API_KEY 是否正确或已过期")
        if status == 402:
            return LLMInsufficientQuotaError(f"余额不足（{status}）：请去平台充值")
        if status == 429:
            return LLMRateLimitError(f"触发限流（{status}）")
        if status == 400:
            return LLMBadRequestError(f"请求非法（{status}）：可能是代码 bug")
        if 500 <= status < 600:
            return LLMServerError(f"服务端错误（{status}）")

    return LLMUnknownError(str(exc))


class LLMClient:
    """DeepSeek 客户端封装。持有连接配置，对外只暴露 ask()。"""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout: float = 30.0,
        max_retries: int = 4,   # 限流是 10/20/40，至少 4 次尝试才完整覆盖
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        # 关键：关掉 SDK 自带的重试（默认 2），避免两层重试叠加
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    def ask(self, question: str, system: str | None = None) -> str:
        """发一个问题，返回回答文本。

        Raises:
            LLMError: 其子类；不可重试的立刻抛，可重试的重试完仍失败才抛。
        """
        messages = _build_messages(question, system)

        logger.info("提问：%s", question[:50])
        last_error: LLMError | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore
                    stream=False,
                )
                answer = resp.choices[0].message.content or ""  # type: ignore
                logger.info("回答成功，长度 %d 字符", len(answer))
                return answer

            except Exception as raw_exc:  # noqa: BLE001 —— 在边界处统一翻译
                exc = classify_error(raw_exc)
                last_error = exc

                # 不可重试的错误，立刻失败
                if not isinstance(exc, RETRYABLE):
                    logger.error("不可重试的错误（%s）：%s", type(exc).__name__, exc)
                    raise exc from raw_exc

                if attempt >= self.max_retries:
                    logger.error("已达最大重试次数 %d，放弃", self.max_retries)
                    break

                wait = self._calc_wait(attempt, exc)
                logger.warning(
                    "第 %d/%d 次失败（%s），%.1f 秒后重试",
                    attempt, self.max_retries, type(exc).__name__, wait,
                )
                time.sleep(wait)

        raise last_error or LLMUnknownError("未知失败")

    def _calc_wait(self, attempt: int, exc: LLMError) -> float:
        """计算等待秒数。

        - 限流：等更久（10, 20, 40 秒）
        - 其他可重试错误：指数退避（1, 2, 4 秒）
        - 都加一点随机抖动，避免多实例同时重试打爆服务
        """
        if isinstance(exc, LLMRateLimitError):
            base = 10.0 * (2 ** (attempt - 1))
        else:
            base = 1.0 * (2 ** (attempt - 1))
        jitter = random.uniform(0, 0.5)
        return base + jitter