# core/loaders/markdown.py
"""Markdown 文档加载器。

解析策略：除了取纯文本，还要**保留标题层级**——因为分块时要用它判断
"这里是不是一个语义边界"。
"""
from pathlib import Path

from core.models import Document, DocumentType
from core.loaders.base import DocumentLoader


class MarkdownLoader(DocumentLoader):
    def supports(self, path) -> bool:
        return Path(path).suffix.lower() in (".md", ".markdown")

    def load(self, path) -> Document:
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        title = self._extract_title(raw)
        return Document(
            doc_id=p.stem,
            source_path=p,
            doc_type=DocumentType.MARKDOWN,
            title=title,
            raw_text=raw,
            metadata={"line_count": str(raw.count("\n") + 1)},
        )

    @staticmethod
    def _extract_title(text: str) -> str | None:
        """取第一个一级标题作为文档标题。"""
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("# "):
                return s[2:].strip()
        return None