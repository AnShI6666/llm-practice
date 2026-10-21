"""程序入口：接收一个问题，调用 LLM 并打印回答。
用法：
    python main.py
    python main.py "什么是 RAG？"
    python main.py "解释一下 Agent" --system "你是一个只用一句话回答的助手"
"""
import argparse
import logging
import sys
from core.config import ConfigError, load_config
from core.llm_client import LLMClient, LLMError
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="问 DeepSeek 一个问题")
    parser.add_argument(
        "question",
        nargs="?",
        default="用一句话解释什么是大语言模型。",
        help="要问的问题（默认问什么是大语言模型）",
    )
    parser.add_argument("--system", default=None, help="可选的 system 提示词")
    return parser
def main() -> int:
    args = build_parser().parse_args()
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"配置错误：{exc}")
        return 1
    client = LLMClient(
        api_key=cfg["api_key"],
        model=cfg["model"],
        base_url=cfg["base_url"],
    )
    try:
        answer = client.ask(args.question, system=args.system)
    except LLMError as exc:
        print(f"调用失败：{exc}")
        return 1
    print(answer)
    return 0
if __name__ == "__main__":
    sys.exit(main())