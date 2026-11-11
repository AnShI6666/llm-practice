"""测试检索逻辑。"""
from core.models import Chunk
from scripts.eval_retrieval import check_hit  # 假设你用的是 check_hit

def _make_chunk(start: int, end: int) -> Chunk:
    return Chunk(
        chunk_id="c1", doc_id="d1", text="x",
        start_char=start, end_char=end, char_count=1, token_estimate=1
    )

def test_chunks_covering_intersection():
    """区间有交集才命中。"""
    c = _make_chunk(0, 10)
    # [0,10) 与 [5,15) 相交
    assert check_hit(c, "d1", [[5, 15]]) is True
    # [0,10) 与 [10,20) 不相交（半开区间）
    assert check_hit(c, "d1", [[10, 20]]) is False

def test_chunks_covering_boundary():
    """区间是半开区间：块 [0,10) 与 span [10,20) 不相交。"""
    c1 = _make_chunk(0, 10)
    c2 = _make_chunk(10, 20)
    gold_spans = [[10, 20]]

    # 块 1 的 end 刚好等于 span 的 start，不应该算命中
    assert check_hit(c1, "d1", gold_spans) is False
    # 块 2 刚好完全覆盖 span，应该算命中
    assert check_hit(c2, "d1", gold_spans) is True