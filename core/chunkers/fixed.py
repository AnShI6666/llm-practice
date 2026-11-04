"""固定长度分块：每 chunk_size 字符一块，相邻块重叠 overlap 字符。

这是**基线策略**——其他策略必须证明自己比它好，否则就用它。
"""
from core.chunkers.base import Chunker, estimate_tokens
from core.models import Chunk, ChunkingConfig, Document


class FixedChunker(Chunker):
    def chunk(self, doc: Document, config: ChunkingConfig) -> list[Chunk]:
        text = doc.raw_text                                  #全文
        size, ov = config.chunk_size, config.overlap         #间隔 重叠
        chunks: list[Chunk] = []                             #存放文本块
        start, idx = 0, 0                                    #游标 序号
        while start < len(text):
            end = min(start + size, len(text))
            piece = text[start:end]
            if piece.strip():
                idx += 1
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_{idx:04d}",
                    doc_id=doc.doc_id,
                    text=piece,
                    start_char=start,
                    end_char=end,
                    char_count=len(piece),
                    token_estimate=estimate_tokens(piece),
                ))
            if end >= len(text):
                break
            start = end - ov
        return chunks
