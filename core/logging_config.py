"""日志配置：一次配置，全项目生效。

要点：
- 同时输出到控制台和文件
- 文件按大小滚动，避免日志把磁盘写满
- 格式包含时间、级别、模块名、行号，方便定位
"""
import logging
import logging.handlers
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.log"

_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO, log_to_file: bool = True) -> None:
    """配置根 logger。整个程序只需调用一次。"""
    root = logging.getLogger()
    root.setLevel(level)

    # 避免重复添加 handler（多次调用 setup_logging 时会重复输出）
    if root.handlers:
        return

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    # 控制台
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # 文件（按大小滚动：单个文件最大 5MB，保留 3 个备份）
    if log_to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)