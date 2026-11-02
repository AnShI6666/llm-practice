"""编排：加载文档 → 分块 → 返回 Chunk 列表。"""
from pathlib import Path

from core.chunkers.base import Chunker
from core.chunkers.fixed import FixedChunker
from core.chunkers.recursive import RecursiveChunker
from core.chunkers.semantic import SemanticChunker
from core.loaders.base import DocumentLoader
from core.loaders.markdown import MarkdownLoader
from core.loaders.pdf import PDFLoader
from core.models import Chunk, ChunkingConfig

LOADERS: list[DocumentLoader] = [MarkdownLoader(), PDFLoader()]
CHUNKERS: dict[str, type[Chunker]] = {
    "fixed": FixedChunker,
    "semantic": SemanticChunker,
    "recursive": RecursiveChunker,
}


def pick_loader(path: Path) -> DocumentLoader:
    for ldr in LOADERS:
        if ldr.supports(path):
            return ldr
    raise ValueError(f"不支持的文件类型：{path.suffix}")


def ingest(path: Path, config: ChunkingConfig) -> list[Chunk]:
    """单文件摄入。"""
    doc = pick_loader(path).load(path)
    chunker_cls = CHUNKERS.get(config.strategy)
    if chunker_cls is None:
        raise ValueError(f"未知分块策略：{config.strategy}")
    return chunker_cls().chunk(doc, config)