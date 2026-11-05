"""按语义结构分块：优先在标题、段落边界处切开。

核心思路：**先按大结构切，再对超长的部分做二次切分**。
这比"从头开始数 500 个字"更符合人类阅读习惯。
"""
import re

from core.chunkers.base import Chunker, merge_short_chunks
from core.chunkers.fixed import estimate_tokens
from core.models import Chunk, ChunkingConfig, Document

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class SemanticChunker(Chunker):
    def chunk(self, doc: Document, config: ChunkingConfig) -> list[Chunk]:
        text = doc.raw_text
        # 找到所有标题位置作为候选切分点
        marks = [(m.start(), m.group(2).strip(), len(m.group(1)))
                 for m in _HEADING.finditer(text)]
        chunks: list[Chunk] = []
        idx = 0

        if not marks:
            # 没有标题：退化为按空行分段
            return merge_short_chunks(self._by_paragraph(doc, config), doc.raw_text, min_chars=100, max_chars=config.chunk_size)

        # 按标题切段，过长再二次切
        for i, (pos, title, level) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
            seg = text[pos:end]
            if not seg.strip():
                continue
            for sub_start, sub_end, sub_text in self._split_long(seg, config):
                idx += 1
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_{idx:04d}",
                    doc_id=doc.doc_id,
                    text=sub_text,
                    start_char=pos + sub_start,
                    end_char=pos + sub_end,
                    section=title,
                    char_count=len(sub_text),
                    token_estimate=estimate_tokens(sub_text),
                ))
        return merge_short_chunks(chunks, doc.raw_text, min_chars=100, max_chars=config.chunk_size)

    def _split_long(self, seg: str, config: ChunkingConfig):
        """段落超过 chunk_size 时二次切分。"""
        size = config.chunk_size
        if len(seg) <= size:
            yield 0, len(seg), seg
            return
        ov = config.overlap
        s = 0
        while s < len(seg):
            e = min(s + size, len(seg))
            yield s, e, seg[s:e]
            if e >= len(seg):
                break
            s = e - ov

    def _by_paragraph(self, doc: Document, config: ChunkingConfig) -> list[Chunk]:
        """无标题时按空行分段，合并过短的段。"""
        chunks: list[Chunk] = []
        idx = 0
        buf, buf_start = "", 0
        pos = 0
        for para in doc.raw_text.split("\n\n"):
            if not para.strip():
                pos += len(para) + 2
                continue
            if not buf:
                buf_start = pos
            if len(buf) + len(para) > config.chunk_size and buf:
                idx += 1
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_{idx:04d}", doc_id=doc.doc_id, text=buf,
                    start_char=buf_start, end_char=buf_start + len(buf),
                    char_count=len(buf), token_estimate=estimate_tokens(buf),
                ))
                buf, buf_start = "", pos
            buf += ("\n\n" if buf else "") + para
            pos += len(para) + 2
        if buf.strip():
            idx += 1
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_{idx:04d}", doc_id=doc.doc_id, text=buf,
                start_char=buf_start, end_char=buf_start + len(buf),
                char_count=len(buf), token_estimate=estimate_tokens(buf),
            ))
        return chunks