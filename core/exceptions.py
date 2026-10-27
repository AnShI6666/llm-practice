"""项目的异常层级。

设计原则：**每一个异常类型都应该对应一种不同的处理动作**。
如果两个异常的处理方式完全一样，它们就不该是两个类型。
"""


class AppError(Exception):
    """本项目所有异常的基类。

    好处：调用方可以用 except AppError 一把兜住所有"我自己的"错误，
    而不会误抓 KeyError、TypeError 这类编程 bug。
    """


# ---------------- 配置类（不可恢复，必须人介入）----------------

class ConfigError(AppError):
    """配置缺失或非法（如没填 API Key）。"""


# ---------------- 大模型调用类 ----------------

class LLMError(AppError):
    """LLM 调用失败的基类。"""


class LLMTimeoutError(LLMError):
    """请求超时。**可重试。**"""


class LLMAuthError(LLMError):
    """认证失败（401/403）：Key 错误、过期、无权限。**不可重试**，重试只会浪费时间和配额。"""


class LLMRateLimitError(LLMError):
    """触发限流（429）。**可重试，但需要等待更久。**"""


class LLMBadRequestError(LLMError):
    """请求参数非法（400）。**不可重试**——这通常说明代码有 bug。"""


class LLMInsufficientQuotaError(LLMError):
    """余额不足（402）。**不可重试**，需要去充值。"""


class LLMServerError(LLMError):
    """服务端错误（5xx）。**可重试。**"""


class LLMUnknownError(LLMError):
    """无法归类的错误。默认保守处理：不重试（或按配置重试）。"""