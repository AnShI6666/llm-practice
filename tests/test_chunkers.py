import pytest
from core.chunkers.fixed import FixedChunker
from core.chunkers.semantic import SemanticChunker
from core.chunkers.recursive import RecursiveChunker


def test_chunk_text_matches_source(make_doc, make_config):
    """★★★ 最关键的测试：块的文本必须等于原文对应区间。"""
    doc = make_doc("第一句话。第二句话。第三句话。第四句话。")
    config = make_config(size=60, overlap=5)

    for chunker in [FixedChunker(), SemanticChunker(), RecursiveChunker()]:
        chunks = chunker.chunk(doc, config)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.text == doc.raw_text[chunk.start_char:chunk.end_char]
            assert chunk.start_char >= 0
            assert chunk.end_char <= len(doc.raw_text)


def test_overlap_works(make_doc, make_config):
    """验证 overlap 生效：相邻块有重叠。"""
    doc = make_doc("0123456789" * 10)  # 100个字符
    config = make_config(size=80, overlap=10)
    chunks = FixedChunker().chunk(doc, config)
    assert chunks[0].end_char > chunks[1].start_char