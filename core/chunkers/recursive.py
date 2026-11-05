"""递归分块：先大结构，再句子，最后字符。"""
from core.chunkers.base import Chunker, merge_short_chunks, estimate_tokens
from core.models import Chunk, ChunkingConfig, Document

class RecursiveChunker(Chunker):
    def __init__(self) -> None:
        self.separators = ["\n## ", "\n### ", "\n\n", "。", "！", "？", "，", ""]

    def chunk(self, doc: Document, config: ChunkingConfig) -> list[Chunk]:
        raw_pieces: list[tuple[str, int]] = []
        self._split_text(doc.raw_text, 0, config, self.separators, raw_pieces)

        chunks: list[Chunk] = []
        for idx, (text, start) in enumerate(raw_pieces, start=1):
            if text.strip():
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_{idx:04d}",
                    doc_id=doc.doc_id,
                    text=text,
                    start_char=start,
                    end_char=start + len(text),
                    char_count=len(text),
                    token_estimate=estimate_tokens(text)
                ))
        return merge_short_chunks(chunks, doc.raw_text, min_chars=100, max_chars=config.chunk_size)

    def _split_text(
        self,
        text: str,
        start_offset: int,
        config: ChunkingConfig,
        separators: list[str],
        out_pieces: list[tuple[str, int]]
    ) -> None:
        if len(text) <= config.chunk_size or not separators:
            if text.strip():
                out_pieces.append((text, start_offset))
            return

        separator = separators[0]
        next_separators = separators[1:]

        if separator:
            parts = text.split(separator)
            current_offset = start_offset
            for i, part in enumerate(parts):
                piece = part if i == len(parts) - 1 else part + separator
                self._split_text(piece, current_offset, config, next_separators, out_pieces)
                current_offset += len(piece)
        else:
            for i in range(0, len(text), config.chunk_size):
                piece = text[i:i + config.chunk_size]
                self._split_text(piece, start_offset + i, config, [], out_pieces)