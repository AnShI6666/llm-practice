"""RAG 端到端编排：问题 → 检索 → 裁剪 → 生成 → 带引用的答案。"""
import json
import logging
from datetime import datetime
from pathlib import Path

from core.context import ContextBudget, trim_by_budget
from core.generator import Generator
from core.models import Chunk, RagAnswer
from core.retrieval import Retriever

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        budget: ContextBudget,
        top_k: int = 8,
        strategy: str = "by_budget_with_floor",
        trace_dir: Path | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._budget = budget
        self._top_k = top_k
        self._strategy = strategy
        self._trace_dir = trace_dir

    def answer(self, question: str) -> RagAnswer:
        # 1. 检索更多候选（top_k 比最终塞进 prompt 的多）
        candidates = self._retriever.retrieve(question, top_k=self._top_k)

        # 2. 按预算裁剪
        chosen = trim_by_budget(
            [chunk for chunk, _ in candidates],
            self._budget,
            strategy=self._strategy,
        )

        # 3. 生成
        result = self._generator.generate(question, chosen)

        # 4. 记录 trace
        self._trace(question, candidates, chosen, result)
        return result

    def _trace(
        self,
        question: str,
        candidates: list[tuple[Chunk, float]],
        chosen: list[Chunk],
        result: RagAnswer,
    ) -> None:
        if self._trace_dir is None:
            return

        self._trace_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = self._trace_dir / f"{ts}.json"

        payload = {
            "question": question,
            "top_k": self._top_k,
            "strategy": self._strategy,
            "budget": {
                "total": self._budget.total,
                "system_reserve": self._budget.system_reserve,
                "answer_reserve": self._budget.answer_reserve,
                "question_reserve": self._budget.question_reserve,
                "available_for_chunks": self._budget.available_for_chunks,
            },
            "candidates": [
                {
                    "chunk_id": chunk.chunk_id,
                    "score": score,
                    "tokens": chunk.token_estimate,
                }
                for chunk, score in candidates
            ],
            "chosen": [
                {"chunk_id": chunk.chunk_id, "tokens": chunk.token_estimate}
                for chunk in chosen
            ],
            "answer": result.model_dump(),
        }

        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("trace 已写入 %s", path)