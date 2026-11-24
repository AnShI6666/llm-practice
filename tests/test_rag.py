"""测试 RAGPipeline 的端到端编排（全部 mock）。"""
import json
from unittest.mock import MagicMock

from core.context import ContextBudget
from core.generator import Generator
from core.models import Chunk, RagAnswer
from core.rag import RAGPipeline
from core.retrieval import Retriever


def make_chunk(cid: str, tokens: int = 100) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="d",
        text="x" * tokens,
        start_char=0,
        end_char=tokens,
        char_count=tokens,
        token_estimate=tokens,
    )


def make_pipeline(tmp_path, candidates):
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve.return_value = candidates

    generator = MagicMock(spec=Generator)
    expected = RagAnswer(
        question="q",
        answer="答 [1]",
        citations=[],
        prompt_tokens=100,
        completion_tokens=20,
        latency_ms=10,
    )
    generator.generate.return_value = expected

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        budget=ContextBudget(total=8000),
        top_k=8,
        strategy="by_budget_with_floor",
        trace_dir=tmp_path,
    )
    return pipeline, retriever, generator, expected


def test_pipeline_end_to_end(tmp_path):
    c1 = make_chunk("c1", 100)
    c2 = make_chunk("c2", 100)
    pipeline, retriever, generator, expected = make_pipeline(
        tmp_path, [(c1, 0.9), (c2, 0.8)]
    )

    result = pipeline.answer("q")

    assert result is expected
    retriever.retrieve.assert_called_once_with("q", top_k=8)
    generator.generate.assert_called_once_with("q", [c1, c2])

    traces = list(tmp_path.glob("*.json"))
    assert len(traces) == 1
    payload = json.loads(traces[0].read_text(encoding="utf-8"))
    assert payload["question"] == "q"
    assert [x["chunk_id"] for x in payload["chosen"]] == ["c1", "c2"]


def test_pipeline_passes_empty_chunks_when_no_candidates(tmp_path):
    pipeline, _, generator, _ = make_pipeline(tmp_path, [])

    pipeline.answer("q")

    generator.generate.assert_called_once_with("q", [])