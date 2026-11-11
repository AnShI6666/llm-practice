"""把 data/raw 下的文档切块、向量化、建索引。"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import load_config, ConfigError
from core.models import ChunkingConfig
from core.ingest import ingest
from core.embedding import EmbeddingClient, EmbeddingError
from core.vector_store import VectorStore

def main() -> int:
    # 1. 加载配置
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"配置错误：{exc}")
        return 1

    if not cfg.zhipu_api_key:
        print("错误：请在 .env 中配置 ZHIPU_API_KEY 才能进行向量化。")
        return 1

    corpus = Path("data/raw/test_corpus")
    if not corpus.exists():
        print(f"错误：找不到语料库目录 {corpus}")
        return 1

    files = sorted(corpus.glob("*.md"))
    if not files:
        print(f"错误：目录 {corpus} 下没有 .md 文件。")
        return 1
    # 2. 切块
    config = ChunkingConfig(strategy="semantic", chunk_size=500, overlap=50)
    all_chunks = []

    print("正在解析并切块...")
    for f in files:
        try:
            chunks = ingest(f, config)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"解析文件 {f.name} 失败: {e}")
            return 1

    if not all_chunks:
        print("错误：没有切出任何文本块。")
        return 1
    # 3. 初始化 Embedding 客户端和向量库
    dim = 1024
    embed_client = EmbeddingClient(
        api_key=cfg.zhipu_api_key, model="embedding-3", dim=dim
    )
    vector_store = VectorStore(dim=dim)
    # 4. 批量向量化 (智谱 API 单次最大 64 条)
    batch_size = 64
    total_tokens = 0
    all_vectors: list[list[float]] = []

    print(f"开始向量化，共 {len(all_chunks)} 个块...")
    start_time = time.time()

    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_texts = [c.text for c in batch_chunks]

        try:
            # 调用 API 获取向量
            vectors = embed_client.embed_batch(batch_texts)
            all_vectors.extend(vectors)
            # 统计 token 消耗
            total_tokens += sum(c.token_estimate for c in batch_chunks)

            print(f"  进度: {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")
        except EmbeddingError as e:
            print(f"向量化失败: {e}")
            return 1

    end_time = time.time()
    elapsed = end_time - start_time
    # 5. 写入 FAISS 索引
    print("正在构建 FAISS 索引...")
    try:
        vector_store.add(all_chunks, all_vectors)
    except Exception as e:
        print(f"写入索引失败: {e}")
        return 1
    # 6. 保存索引
    index_dir = Path("data/index")
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "faiss.index"

    print("正在保存索引到磁盘...")
    try:
        vector_store.save(index_path)
    except Exception as e:
        print(f"保存索引失败: {e}")
        return 1
    # 7. 打印统计与成本
    # 智谱 embedding-3 价格：0.5 元 / 100万 tokens
    cost = (total_tokens / 1_000_000) * 0.5
    print("\n" + "=" * 30)
    print(f"文档数: {len(files)}")
    print(f"总块数: {len(all_chunks)}")
    print(f"向量维度: {dim}")
    print(f"向量化耗时: {elapsed:.1f} 秒")
    print(f"预估成本: {cost:.4f} 元 (基于 {total_tokens} tokens)")
    print(f"索引已保存: {index_path.with_suffix('.faiss')} (及 {index_path.with_suffix('.pkl')})")
    print("=" * 30)
    return 0

if __name__ == "__main__":
    sys.exit(main())