"""向量存储与检索（基于 FAISS）。"""
import pickle
from pathlib import Path

import faiss
import numpy as np

from core.models import Chunk


class VectorStore:
    """向量索引。负责「写入」和「按向量查最近邻」。"""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        # IndexFlatIP = 内积索引；配合归一化向量等价于余弦相似度
        self._index:faiss.Index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []  # 与索引行号一一对应

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """写入。chunks 与 vectors 必须等长且顺序对应。"""
        if len(chunks) != len(vectors):
            raise ValueError(f"块数({len(chunks)})与向量数({len(vectors)})不一致")
        arr = np.asarray(vectors, dtype="float32")
        if arr.shape[1] != self.dim:
            raise ValueError(f"向量维度 {arr.shape[1]} != 索引维度 {self.dim}")

        # ★ 归一化：让内积等价于余弦相似度
        faiss.normalize_L2(arr)
        self._index.add(arr)
        self._chunks.extend(chunks)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        """返回 [(块, 相似度)]，按相似度降序。"""
        if self._index.ntotal == 0:
            return []
        q = np.asarray([query_vector], dtype="float32")
        faiss.normalize_L2(q)

        # 确保不会超过库里的总数
        actual_k = min(top_k, self._index.ntotal)
        scores, idxs = self._index.search(q, actual_k)

        out = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:  # FAISS 查不到时会返回 -1
                continue
            out.append((self._chunks[i], float(score)))
        return out

    def save(self, path: Path) -> None:
        """保存索引与块元数据。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        # 1. 保存 FAISS 索引
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        # 2. 保存 Chunk 列表（用 pickle）
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self._chunks, f)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """加载索引与块元数据。"""
        # 1. 读入 FAISS 索引
        index = faiss.read_index(str(path.with_suffix(".faiss")))
        # 2. 读入 Chunk 列表
        with open(path.with_suffix(".pkl"), "rb") as f:
            chunks = pickle.load(f)

        store = cls(dim=index.d)
        store._index = index
        store._chunks = chunks
        return store

    def __len__(self) -> int:
        return self._index.ntotal