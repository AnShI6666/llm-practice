"""加载器测试。"""
from pathlib import Path

import pytest

from core.loaders.markdown import MarkdownLoader


@pytest.fixture
def md_file(tmp_path):
    """在临时目录造一个 Markdown 文件。"""
    p = tmp_path / "note.md"
    p.write_text("# 我的标题\n\n第一段内容。\n\n第二段内容。", encoding="utf-8")
    return p


def test_supports_markdown():
    ldr = MarkdownLoader()
    assert ldr.supports("a.md") is True
    assert ldr.supports("a.markdown") is True
    assert ldr.supports("a.PDF") is False
    assert ldr.supports("a.txt") is False


def test_load_returns_document(md_file):
    doc = MarkdownLoader().load(md_file)
    assert doc.doc_id == "note"
    assert doc.title == "我的标题"
    assert "第一段内容" in doc.raw_text
    assert doc.metadata["line_count"]


def test_load_resource_not_found():
    with pytest.raises(FileNotFoundError):
        MarkdownLoader().load(Path("不存在的文件.md"))


def test_extract_title_without_heading(tmp_path):
    p = tmp_path / "plain.md"
    p.write_text("没有标题的纯文本。", encoding="utf-8")
    assert MarkdownLoader().load(p).title is None