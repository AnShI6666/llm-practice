from abc import ABC, abstractmethod

from core.models import Chunk, ChunkingConfig, Document


class Chunker(ABC):
    @abstractmethod
    def chunk(self, doc: Document, config: ChunkingConfig) -> list[Chunk]:
        """把文档切成块。"""