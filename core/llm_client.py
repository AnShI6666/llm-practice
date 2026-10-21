"""LLM 客户端封装：统一处理调用、超时、重试。
设计要点：
- 用类持有「一组相关状态」：client / model / timeout / max_retries
- 外部只需调 ask()，不用每次传 client 和 model
- 指数退避重试：1s → 2s → 4s，避免瞬时故障时一起打服务
"""
import logging
import time
from openai import OpenAI
logger = logging.getLogger(__name__)
class LLMError(Exception):
    """LLM 调用失败（已重试完）时抛出。"""
class LLMClient:
    """DeepSeek 客户端封装。持有连接配置，对外只暴露 ask()。"""
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    def ask(self, question: str, system: str | None = None) -> str:
        """发一个问题，返回回答文本。
        Args:
            question: 用户问题。
            system: 可选的 system 提示词，用于设定回答风格。
        Raises:
            LLMError: 重试 max_retries 次后仍失败。
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})
        logger.info("提问：%s", question[:50])
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                )
                answer = resp.choices[0].message.content or ""
                logger.info("回答长度：%d 字符", len(answer))
                return answer
            except Exception as exc:  # noqa: BLE001 —— 第 3 周会细分异常类型
                last_error = exc
                wait = 2 ** (attempt - 1)  # 1, 2, 4 秒
                logger.warning(
                    "第 %d/%d 次调用失败：%s", attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    logger.info("%d 秒后重试", wait)
                    time.sleep(wait)
        raise LLMError(
            f"调用失败，已重试 {self.max_retries} 次：{last_error}"
        ) from last_error