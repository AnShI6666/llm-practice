"""命令行 RAG 问答：python scripts/ask.py "你的问题" """
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config, ConfigError
from core.context import ContextBudget
from core.embedding import EmbeddingClient, EmbeddingError
from core.exceptions import LLMError
from core.generator import Generator
from core.llm_client import LLMClient
from core.logging_config import setup_logging
from core.rag import RAGPipeline
from core.retrieval import Retriever
from core.vector_store import VectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于本地文档的 RAG 问答")
    parser.add_argument("question", help="要问的问题")
    parser.add_argument("--top-k", type=int, default=8, help="检索候选数")
    parser.add_argument(
        "--strategy",
        default="by_budget_with_floor",
        choices=["by_score", "by_budget", "by_budget_with_floor"],
    )
    parser.add_argument("--budget", type=int, default=8000, help="总 token 预算")
    parser.add_argument("--no-trace", action="store_true", help="不写 trace")
    return parser


def main() -> int:
    setup_logging()
    args = build_parser().parse_args()

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"配置错误：{exc}")
        return 1

    if not cfg.zhipu_api_key:
        print("错误：请在 .env 中配置 ZHIPU_API_KEY（检索需要向量化）")
        return 1

    index_path = Path("data/index/faiss.index")
    if not index_path.with_suffix(".faiss").exists():
        print("错误：找不到索引，请先运行 scripts/build_index.py")
        return 1

    store = VectorStore.load(index_path)
    embed_client = EmbeddingClient(api_key=cfg.zhipu_api_key, dim=1024)
    retriever = Retriever(embed_client, store)

    llm = LLMClient(
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )
    generator = Generator(llm)

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        budget=ContextBudget(total=args.budget),
        top_k=args.top_k,
        strategy=args.strategy,
        trace_dir=None if args.no_trace else Path("data/traces"),
    )

    try:
        result = pipeline.answer(args.question)
    except (LLMError, EmbeddingError) as exc:
        print(f"调用失败：{exc}")
        return 1

    print(result.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())