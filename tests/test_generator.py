"""测试 Generator 的引用解析与拒答识别。"""
from unittest.mock import MagicMock

import pytest

from core.exceptions import LLMError
from core.generator import Generator
from core.llm_client import LLMClient
from core.models import Chunk


def make_chunk(i: int, text: str = "内容") -> Chunk:
    return Chunk(
        chunk_id=f"doc_{i:04d}",
        doc_id="doc",
        text=text,
        start_char=i * 10,
        end_char=i * 10 + len(text),
        char_count=len(text),
        token_estimate=max(1, len(text)),
    )


def make_generator(answer: str) -> tuple[Generator, MagicMock]:
    llm = MagicMock(spec=LLMClient)
    llm.ask_with_usage.return_value = (answer, 100, 20)
    return Generator(llm), llm


def test_single_marker_maps_to_first_chunk():
    g, _ = make_generator("建议 88–94℃ [1]。")
    chunks = [make_chunk(1, "水温 88–94℃"), make_chunk(2, "其他")]
    result = g.generate("水温？", chunks)

    assert [c.marker for c in result.citations] == [1]
    assert result.citations[0].chunk_id == chunks[0].chunk_id
    assert result.citations[0].quote == chunks[0].text
    assert result.invalid_markers == []
    assert result.abstained is False


def test_multiple_markers_all_parsed():
    g, _ = make_generator("A [1][3]。")
    chunks = [make_chunk(1), make_chunk(2), make_chunk(3)]
    result = g.generate("q", chunks)

    assert [c.marker for c in result.citations] == [1, 3]


def test_duplicate_marker_only_one_citation():
    g, _ = make_generator("[1] 再强调一次 [1]。")
    chunks = [make_chunk(1), make_chunk(2)]
    result = g.generate("q", chunks)

    assert len(result.citations) == 1
    assert result.citations[0].marker == 1


def test_invalid_marker_is_flagged():
    """模型引用了不存在的编号，必须记录，不能静默忽略。"""
    g, _ = make_generator("建议 [9]。")
    chunks = [make_chunk(1), make_chunk(2), make_chunk(3)]
    result = g.generate("q", chunks)

    assert result.citations == []
    assert result.invalid_markers == [9]
    assert "[9]" in result.answer


def test_abstain_is_detected():
    g, _ = make_generator("未找到相关内容。缺少：作者信息。")
    result = g.generate("作者是谁？", [make_chunk(1)])

    assert result.abstained is True


def test_empty_chunks_skip_llm():
    """没有检索到任何块时，应该直接拒答，不该浪费一次 API 调用。"""
    llm = MagicMock(spec=LLMClient)
    g = Generator(llm)

    result = g.generate("q", [])

    assert result.abstained is True
    assert "未找到相关内容" in result.answer
    llm.ask_with_usage.assert_not_called()


def test_empty_answer_raises_llm_error():
    g, _ = make_generator("   ")
    with pytest.raises(LLMError):
        g.generate("q", [make_chunk(1)])