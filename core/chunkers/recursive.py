"""递归分块：先大结构，再句子，最后字符。"""
from core.chunkers.base import Chunker
from core.chunkers.fixed import estimate_tokens
from core.models import Chunk, ChunkingConfig, Document

class RecursiveChunker(Chunker):
    def __init__(self):
        self.separators = ["\n## ", "\n### ", "\n\n", "\n", "。", "！", "？", "，", ""]

    def chunk(self, doc: Document, config: ChunkingConfig) -> list[Chunk]:
        chunks: list[Chunk] = []
        self._split_text(doc.raw_text, 0, config, self.separators, doc, chunks)
        for idx, chunk in enumerate(chunks, start=1):
            chunk.chunk_id = f"{doc.doc_id}_{idx:04d}"
        return chunks

    def _split_text(self, text: str, start_offset: int, config: ChunkingConfig, separators: list[str], doc: Document, chunks: list[Chunk]):
        if len(text) <= config.chunk_size:
            if text.strip():
                chunks.append(Chunk(
                    chunk_id="temp", doc_id=doc.doc_id, text=text,
                    start_char=start_offset, end_char=start_offset + len(text),
                    char_count=len(text), token_estimate=estimate_tokens(text)
                ))
            return

        separator = separators[0]
        next_separators = separators[1:]

        if separator:
            parts = text.split(separator)
            current_offset = start_offset
            for i, part in enumerate(parts):
                piece = part if i == len(parts) - 1 else part + separator
                self._split_text(piece, current_offset, config, next_separators, doc, chunks)
                current_offset += len(piece)
        else:
            for i in range(0, len(text), config.chunk_size):
                piece = text[i:i + config.chunk_size]
                self._split_text(piece, start_offset + i, config, [], doc, chunks)