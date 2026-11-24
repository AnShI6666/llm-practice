"""上下文管理 A/B/C 对比实验。"""
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config
from core.context import ContextBudget
from core.embedding import EmbeddingClient
from core.generator import Generator
from core.llm_client import LLMClient
from core.rag import RAGPipeline
from core.retrieval import Retriever
from core.vector_store import VectorStore


@dataclass
class ExperimentConfig:
    name: str
    top_k: int
    budget: int
    strategy: str

class ExperimentRow(TypedDict):
    config: str
    id: str
    type: str
    question: str
    answer: str
    citations: list[dict[str, Any]]
    invalid_markers: list[int]
    abstained: bool
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int

CONFIGS = [
    ExperimentConfig("A_top5", 5, 8000, "by_score"),
    ExperimentConfig("B_8000", 8, 8000, "by_budget_with_floor"),
    ExperimentConfig("C_4000", 8, 4000, "by_budget_with_floor"),
]


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
    generator = Generator(llm)

    items = json.loads(
        Path("data/eval/questions.json").read_text(encoding="utf-8")
    )
    items = [x for x in items if not x.get("should_abstain")]

    all_rows: list[ExperimentRow] = []

    for exp in CONFIGS:
        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            budget=ContextBudget(total=exp.budget),
            top_k=exp.top_k,
            strategy=exp.strategy,
            trace_dir=None,
        )

        rows: list[ExperimentRow] = []
        for item in items:
            result = pipeline.answer(item["question"])
            rows.append({
                "config": exp.name,
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "answer": result.answer,
                "citations": [c.model_dump() for c in result.citations],
                "invalid_markers": result.invalid_markers,
                "abstained": result.abstained,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_ms": result.latency_ms,
            })

        all_rows.extend(rows)

        prompt_mean = statistics.mean(
            int(r["prompt_tokens"]) for r in rows
        )
        latency_mean = statistics.mean(
            int(r["latency_ms"]) for r in rows
        )
        print(
            f"{exp.name}: prompt_mean={prompt_mean:.0f} "
            f"latency_mean={latency_mean:.0f}ms"
        )

    out = Path("data/experiments/context_experiment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())