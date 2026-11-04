"""编排层测试。"""
import pytest

from core.ingest import ingest, pick_loader
from core.loaders.markdown import MarkdownLoader
from core.models import ChunkingConfig


def test_pick_loader_markdown():
    assert isinstance(pick_loader(__import__("pathlib").Path("a.md")), MarkdownLoader)


def test_pick_loader_unsupported():
    from pathlib import Path
    with pytest.raises(ValueError, match="不支持"):
        pick_loader(Path("a.docx"))


def test_ingest_unknown_strategy(tmp_path):
    from pathlib import Path
    p = tmp_path / "a.md"
    p.write_text("# T\n\n内容。", encoding="utf-8")
    with pytest.raises(ValueError, match="未知分块策略"):
        ingest(p, ChunkingConfig(strategy="不存在的策略"))


def test_ingest_end_to_end(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# 标题\n\n" + "内容句子。" * 100, encoding="utf-8")
    chunks = ingest(p, ChunkingConfig(strategy="fixed", chunk_size=200, overlap=20))
    assert len(chunks) >= 2
    assert all(c.doc_id == "a" for c in chunks)
    assert all(c.text for c in chunks)