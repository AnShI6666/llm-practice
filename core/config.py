"""配置管理：从 .env 读取并校验运行所需的环境变量。
设计要点：
- 把「读环境变量 + 校验 + 报错」从主逻辑里分离出去
- 用自定义异常 ConfigError 区分「配置问题」和「网络问题」
- 所有可配置项都允许用环境变量覆盖（MODEL / BASE_URL）
"""
import os
from dotenv import load_dotenv
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"
class ConfigError(Exception):
    """配置缺失或非法时抛出。"""
def load_config() -> dict:
    """读取配置，返回 {api_key, model, base_url}。
    Raises:
        ConfigError: 缺少 DEEPSEEK_API_KEY 时。
    """
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ConfigError(
            "未找到 DEEPSEEK_API_KEY。请在项目根目录创建 .env 文件，"
            "内容形如：DEEPSEEK_API_KEY=sk-xxxx"
        )
    return {
        "api_key": api_key,
        "model": os.environ.get("MODEL", DEFAULT_MODEL),
        "base_url": os.environ.get("BASE_URL", DEFAULT_BASE_URL),
    }