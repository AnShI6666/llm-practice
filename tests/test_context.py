"""测试上下文预算与裁剪。"""
import pytest

from core.context import ContextBudget, trim_by_budget
from core.models import Chunk


def make_chunk(cid: str, tokens: int) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="d",
        text="x" * tokens,
        start_char=0,
        end_char=tokens,
        char_count=tokens,
        token_estimate=tokens,
    )


def budget_with_available(available: int) -> ContextBudget:
    # total - 100 - 200 - 100 = available
    return ContextBudget(
        total=available + 400,
        system_reserve=100,
        answer_reserve=200,
        question_reserve=100,
    )


def test_available_for_chunks():
    budget = ContextBudget(
        total=1000,
        system_reserve=100,
        answer_reserve=200,
        question_reserve=100,
    )
    assert budget.available_for_chunks == 600


def test_by_budget_does_not_exceed():
    budget = budget_with_available(600)
    chunks = [make_chunk("a", 300), make_chunk("b", 300),
              make_chunk("c", 300), make_chunk("d", 300)]

    chosen = trim_by_budget(chunks, budget, strategy="by_budget")

    assert [c.chunk_id for c in chosen] == ["a", "b"]
    assert sum(c.token_estimate for c in chosen) <= 600


def test_by_budget_returns_empty_if_first_too_big():
    budget = budget_with_available(600)
    chunks = [make_chunk("big", 700), make_chunk("small", 100)]

    assert trim_by_budget(chunks, budget, strategy="by_budget") == []


def test_floor_keeps_min_chunks_even_over_budget():
    budget = budget_with_available(100)
    chunks = [make_chunk("a", 300), make_chunk("b", 300), make_chunk("c", 300)]

    chosen = trim_by_budget(
        chunks, budget, strategy="by_budget_with_floor", min_chunks=3
    )

    assert len(chosen) == 3
    assert sum(c.token_estimate for c in chosen) > 100


def test_by_score_returns_all():
    budget = budget_with_available(100)
    chunks = [make_chunk("a", 300), make_chunk("b", 300)]
    assert len(trim_by_budget(chunks, budget, strategy="by_score")) == 2


def test_empty_chunks():
    assert trim_by_budget([], budget_with_available(600)) == []


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        trim_by_budget([make_chunk("a", 10)], budget_with_available(600),
                       strategy="unknown")