"""分块策略对比实验。"""
import sys
import statistics
from pathlib import Path

# 确保能导入 core 模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.models import ChunkingConfig
from core.loaders.markdown import MarkdownLoader
from core.chunkers.fixed import FixedChunker
from core.chunkers.semantic import SemanticChunker
from core.chunkers.recursive import RecursiveChunker


def is_broken_boundary(chunk_text: str) -> bool:
    """判断块是否在句子中间被切断（True 表示被切断）。"""
    t = chunk_text.strip()
    if not t:
        return True
    head_ok = not t[0] in "，。！？、；：）」』”\"’\"‘"
    head_ok = head_ok or t.startswith("#")
    tail_ok = t[-1] in "。！？.!?；;：:」』）\n"
    return not (head_ok and tail_ok)

def evaluate_chunks(chunks):
    """计算 4 个核心指标。"""
    if not chunks:
        return 0, 0, 0, 0
    # 1. 块数与平均块长
    count = len(chunks)
    avg_len = sum(c.char_count for c in chunks) / count
    # 2. 边界完整性（越低越好，这里计算被切断的百分比）
    broken_count = sum(1 for c in chunks if is_broken_boundary(c.text))
    broken_rate = broken_count / count
    # 3. 语义凝聚度（块内包含的标题数量，越少越好，理想是 1 个）
    # 用块内 "# " 的数量近似
    heading_counts = [c.text.count("\n# ") + (1 if c.text.startswith("# ") else 0) for c in chunks]
    avg_headings = sum(heading_counts) / count
    # 4. 长度标准差（越小越均匀）
    lengths = [c.char_count for c in chunks]
    std_dev = statistics.stdev(lengths) if len(lengths) > 1 else 0
    return count, avg_len, broken_rate, avg_headings, std_dev

def main():
    corpus_dir = Path("data/raw/test_corpus")
    if not corpus_dir.exists():
        print(f"错误：找不到测试文档目录 {corpus_dir}，请先准备 10 篇 md 文件。")
        return

    loader = MarkdownLoader()
    docs = []
    for p in corpus_dir.glob("*.md"):
        docs.append(loader.load(p))
    print(f"成功加载 {len(docs)} 篇文档。\n")
    strategies = {
        "fixed": FixedChunker(),
        "semantic": SemanticChunker(),
        "recursive": RecursiveChunker()
    }

    # 实验矩阵
    results = []
    print(
        f"{'策略':<10} | {'Size':<5} | {'Overlap':<7} | {'总块数':<6} | {'平均长':<6} | {'截断率':<6} | {'平均标题':<6} | {'长度标准差':<8}")
    print("-" * 80)

    for name, chunker in strategies.items():
        # fixed 测三种 overlap，其他测一种
        overlaps = [0, 50, 100] if name == "fixed" else [50]
        for size in [300, 500, 800]:
            for ov in overlaps:
                config = ChunkingConfig(chunk_size=size, overlap=ov, strategy=name)
                all_chunks = []
                for doc in docs:
                    all_chunks.extend(chunker.chunk(doc, config))
                count, avg_len, broken_rate, avg_head, std = evaluate_chunks(all_chunks)
                results.append((name, size, ov, count, avg_len, broken_rate, avg_head, std))
                print(f"{name:<10} | {size:<5} | {ov:<7} | {count:<6} | {avg_len:<6.1f} | {broken_rate * 100:<5.1f}% | {avg_head:<6.2f} | {std:<8.1f}")
if __name__ == "__main__":
    main()