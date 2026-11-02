# core/loaders/pdf.py
"""PDF 文档加载器。

关键：要记录每页的字符区间，这样分块时才能把「块」映射回「页码」，
第 7 周做引用溯源时才能说「来自第 3 页」。
"""
from pathlib import Path

from pypdf import PdfReader

from core.models import Document, DocumentType
from core.loaders.base import DocumentLoader


class PDFLoader(DocumentLoader):
    def supports(self, path) -> bool:
        return Path(path).suffix.lower() == ".pdf"

    def load(self, path) -> Document:
        p = Path(path)
        reader = PdfReader(str(p))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")

        # 用换页符拼接，并在 metadata 里记录每页的字符区间
        parts, offsets, cursor = [], [], 0
        for i, text in enumerate(pages, start=1):
            part = f"\n\n<<<PAGE {i}>>>\n\n{text}"
            parts.append(part)
            offsets.append(f"page{i}_start={cursor}")
            cursor += len(part)

        return Document(
            doc_id=p.stem,
            source_path=p,
            doc_type=DocumentType.PDF,
            title=p.stem,
            raw_text="".join(parts),
            metadata={
                "page_count": str(len(pages)),
                "offsets": ",".join(offsets),
            },
        )