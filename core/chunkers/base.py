from abc import ABC, abstractmethod
from core.models import Chunk, ChunkingConfig, Document

def estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文约 1 字 = 1 token，英文约 4 字符 = 1 token。"""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - cjk
    return max(1, cjk + other // 4)

class Chunker(ABC):
    @abstractmethod
    def chunk(self, doc: Document, config: ChunkingConfig) -> list[Chunk]:
        """把文档切成块。"""

def merge_short_chunks(chunks: list[Chunk], min_chars: int = 100, max_chars: int = 800) -> list[Chunk]:
    """合并过短的相邻块：如果某块 < min_chars，尝试与后一块合并。

    合并后必须更新 text, start_char, end_char, char_count, token_estimate。
    """
    if not chunks:
        return []

    merged: list[Chunk] = []
    current = chunks[0]

    for next_chunk in chunks[1:]:
        # 如果当前块太短，且合并后不会撑爆（这里限制最大 800 字，可根据 config 调整）
        if current.char_count < min_chars and (current.char_count + next_chunk.char_count) <= max_chars:
            # 拼接文本
            new_text = current.text + "\n\n" + next_chunk.text
            current = Chunk(
                chunk_id=current.chunk_id,  # 沿用前一个的 ID
                doc_id=current.doc_id,
                text=new_text,
                start_char=current.start_char,
                end_char=next_chunk.end_char,  # 终点延伸到下一块的终点
                section=current.section or next_chunk.section,
                page=current.page or next_chunk.page,
                char_count=len(new_text),
                token_estimate=estimate_tokens(new_text),
            )
        else:
            # 当前块够长，直接存下，把 next_chunk 作为新的 current
            merged.append(current)
            current = next_chunk

    merged.append(current)

    # 重新编号，保持 ID 的连续性
    for idx, c in enumerate(merged, start=1):
        c.chunk_id = f"{c.doc_id}_{idx:04d}"

    return merged