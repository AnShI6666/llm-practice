import pytest
from core.vector_store import VectorStore
from core.models import Chunk

def _make_chunk(cid: str, start: int) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="d", text="x",
        start_char=start, end_char=start + 1, char_count=1, token_estimate=1
    )

def test_store_add_and_len():
    store = VectorStore(dim=3)
    c1, c2 = _make_chunk("1", 0), _make_chunk("2", 1)
    store.add([c1, c2], [[1.0, 0, 0], [0, 1.0, 0]])
    assert len(store) == 2

def test_store_search_order():
    store = VectorStore(dim=3)
    c1, c2 = _make_chunk("1", 0), _make_chunk("2", 1)
    # 写入两个向量
    store.add([c1, c2], [[1.0, 0, 0], [0.8, 0.2, 0]])

    # 搜索 [1, 0, 0]，最相似的应该是 c1
    results = store.search([1.0, 0, 0], top_k=2)
    assert len(results) == 2
    assert results[0][0].chunk_id == "1"
    assert results[1][0].chunk_id == "2"
    assert results[0][1] > results[1][1]  # 相似度降序

def test_store_empty_search():
    store = VectorStore(dim=3)
    assert store.search([1.0, 0, 0]) == []

def test_store_dimension_mismatch():
    store = VectorStore(dim=3)
    c1 = _make_chunk("1", 0)
    # 写入维度为 4 的向量应该报错
    with pytest.raises(ValueError, match="向量维度"):
        store.add([c1], [[1.0, 0, 0, 0]])