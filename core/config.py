"""配置管理：从 .env 读取并校验。"""
import os

from dotenv import load_dotenv

from core.exceptions import ConfigError
from core.models import Config


def load_config() -> Config:
    """读取配置，返回已校验的 Config 对象。

    Raises:
        ConfigError: 缺少环境变量或校验失败时。
    """
    load_dotenv()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ConfigError(
            "未找到 DEEPSEEK_API_KEY。请在项目根目录创建 .env 文件，"
            "内容形如：DEEPSEEK_API_KEY=sk-xxxx"
        )

    try:
        return Config(api_key=api_key,
            model=os.environ.get("MODEL", "deepseek-chat"),
            base_url=os.environ.get("BASE_URL", "https://api.deepseek.com"),
            # 注意：如果用户填了 TIMEOUT=abc，float() 会报错，现在它会被下面的 except 捕获
            timeout=float(os.environ.get("TIMEOUT", "30")),
            max_retries=int(os.environ.get("MAX_RETRIES", "4")),)
    except Exception as exc:
        # 把 pydantic 的校验错误翻译成我们的 ConfigError
        raise ConfigError(f"配置校验失败：{exc}") from exc