"""检索模块：把 Embedding 和 VectorStore 组合起来。"""
from core.embedding import EmbeddingClient
from core.vector_store import VectorStore
from core.models import Chunk

class Retriever:
    def __init__(self, embed_client: EmbeddingClient, vector_store: VectorStore):
        self.embed_client = embed_client
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """对外接口：输入字符串问题，返回最相似的 K 个块。"""
        # 1. 把问题变成向量
        query_vector = self.embed_client.embed(query)
        # 2. 去向量库搜
        return self.vector_store.search(query_vector, top_k=top_k)