"""检索评测：算 Recall@1 / Recall@3 / Recall@5 / MRR。"""
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import DefaultDict,Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config, ConfigError
from core.embedding import EmbeddingClient
from core.vector_store import VectorStore
from core.retrieval import Retriever
from core.models import Chunk

def check_hit(chunk: Chunk, gold_doc: str, gold_spans: list[list[int]]) -> bool:
    """判断检索出的块是否命中了金标区间（区间有交集）。"""
    if chunk.doc_id != gold_doc:
        return False
    for s, e in gold_spans:
        # 半开区间相交判定：[c.start, c.end) 与 [s, e)
        if chunk.start_char < e and chunk.end_char > s:
            return True
    return False

def main() -> int:
    # 1. 初始化配置和检索器
    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"配置错误: {e}")
        return 1
    index_path = Path("data/index/faiss.index")
    if not index_path.with_suffix(".faiss").exists():
        print("错误：找不到索引文件，请先运行 build_index.py")
        return 1
    store = VectorStore.load(index_path)
    embed_client = EmbeddingClient(api_key=cfg.zhipu_api_key, model="embedding-3", dim=1024)
    retriever = Retriever(embed_client, store)

    # 2. 加载评测集 (兼容 .json 和 .jsonl)
    eval_path_json = Path("data/eval/questions.json")
    eval_path_jsonl = Path("data/eval/questions.jsonl")
    items = []
    if eval_path_json.exists():
        with open(eval_path_json, "r", encoding="utf-8") as f:
            items = json.load(f)
    elif eval_path_jsonl.exists():
        with open(eval_path_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    else:
        print("错误：找不到评测集文件 (data/eval/questions.json 或 .jsonl)")
        return 1

    # 3. 评测参数
    max_k = 5
    metrics = {"R@1": 0, "R@3": 0, "R@5": 0, "MRR": 0.0}
    total_eval_count = 0
    group_stats: DefaultDict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "R@5": 0, "hit_details": []})
    print(f"开始评测，加载了 {len(items)} 条样本。\n")

    # 4. 遍历评测集
    for item in items:
        # 跳过拒答类样本
        if item.get("should_abstain", False):
            continue
        total_eval_count += 1
        q_type = item.get("type", "unknown")
        group_stats[q_type]["total"] += 1
        gold_doc = item["gold_doc"]
        gold_spans = item["gold_spans"]
        # 检索前 5 个
        results = retriever.retrieve(item["question"], top_k=max_k)
        # 找出第一个命中的位置
        first_hit_rank = None
        hit_chunks = []
        for rank, (chunk, score) in enumerate(results, start=1):
            if check_hit(chunk, gold_doc, gold_spans):
                hit_chunks.append((rank, chunk.chunk_id))
                if first_hit_rank is None:
                    first_hit_rank = rank

        # 计算指标
        if first_hit_rank is not None:
            if first_hit_rank == 1: metrics["R@1"] += 1
            if first_hit_rank <= 3: metrics["R@3"] += 1
            metrics["R@5"] += 1
            metrics["MRR"] += 1.0 / first_hit_rank
            group_stats[q_type]["R@5"] += 1
        else:
            # 记录失败样本
            group_stats[q_type]["hit_details"].append({
                "id": item["id"], "question": item["question"],
                "reason": "未命中任何金标区间"
            })

    # 5. 打印报告
    print("=== Recall@5 总览 ===")
    print(f"全部参与评测数: {total_eval_count} 条 (已排除 unanswerable/trap)")
    print(f"Recall@1: {metrics['R@1'] / total_eval_count * 100:.1f}%  ({metrics['R@1']}/{total_eval_count})")
    print(f"Recall@3: {metrics['R@3'] / total_eval_count * 100:.1f}%  ({metrics['R@3']}/{total_eval_count})")
    print(f"Recall@5: {metrics['R@5'] / total_eval_count * 100:.1f}%  ({metrics['R@5']}/{total_eval_count})")
    print(f"MRR:      {metrics['MRR'] / total_eval_count:.3f}")
    print("\n=== 按类型分组 ===")
    for q_type, stats in group_stats.items():
        if stats["total"] > 0:
            print(
                f"{q_type:<12} Recall@5: {stats['R@5'] / stats['total'] * 100:.1f}%  ({stats['R@5']}/{stats['total']})")
    print("\n=== 失败样本 (未命中) ===")
    failed_count = 0
    for q_type, stats in group_stats.items():
        for fail in stats["hit_details"]:
            print(f"[{fail['id']}] {q_type:<10} {fail['reason']} - 「{fail['question'][:20]}...」")
            failed_count += 1
    if failed_count == 0:
        print("无，全部命中！")
    return 0

if __name__ == "__main__":
    sys.exit(main())