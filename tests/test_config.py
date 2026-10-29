import pytest
from core.config import ConfigError, load_config


def test_missing_key_raises(monkeypatch):
    """没有 API Key 时应该抛 ConfigError。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)     #delenv删除 raising=False 表示“没有就算了，不报错”。
    monkeypatch.setattr("core.config.load_dotenv", lambda *a, **k: None)   #setattr替换
    with pytest.raises(ConfigError):
        load_config()

def test_default_model(monkeypatch):
    """没设 MODEL 时应该用默认模型。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test123456")   #setenv设置
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.setattr("core.config.load_dotenv", lambda *a, **k: None)
    cfg = load_config()
    assert cfg.model == "deepseek-chat"