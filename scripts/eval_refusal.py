"""拒答测试：跑 9 条 unanswerable + trap。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config
from core.context import ContextBudget
from core.embedding import EmbeddingClient
from core.generator import Generator
from core.llm_client import LLMClient
from core.rag import RAGPipeline
from core.retrieval import Retriever
from core.vector_store import VectorStore


def main() -> int:
    cfg = load_config()

    store = VectorStore.load(Path("data/index/faiss.index"))
    retriever = Retriever(
        EmbeddingClient(api_key=cfg.zhipu_api_key, dim=1024),
        store,
    )
    llm = LLMClient(
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=Generator(llm),
        budget=ContextBudget(total=8000),
        top_k=8,
        strategy="by_budget_with_floor",
        trace_dir=None,
    )

    items = json.loads(
        Path("data/eval/questions.json").read_text(encoding="utf-8")
    )
    items = [x for x in items if x.get("should_abstain")]

    rows: list[dict[str, object]] = []
    for item in items:
        result = pipeline.answer(item["question"])
        rows.append({
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "abstained": result.abstained,
            "answer": result.answer,
            "citations": [c.model_dump() for c in result.citations],
            "invalid_markers": result.invalid_markers,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_ms": result.latency_ms,
        })
        print(
            f"{item['id']} {item['type']:<12} "
            f"abstained={result.abstained} "
            f"answer={result.answer[:70]!r}"
        )

    out = Path("data/experiments/refusal.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    unans = [r for r in rows if r["type"] == "unanswerable"]
    traps = [r for r in rows if r["type"] == "trap"]
    correct_refusal = sum(1 for r in unans if r["abstained"])

    print()
    print(f"unanswerable: {correct_refusal}/{len(unans)} 正确拒答")
    print(f"trap: {len(traps)} 条，需要人工按「纠正 / 只拒答 / 顺着答」分类")
    print(f"结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())