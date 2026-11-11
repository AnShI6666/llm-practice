"""验证评测集里的 gold_spans 是否精准。"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.loaders.markdown import MarkdownLoader

def main():
    eval_file = Path("data/eval/questions.json")
    if not eval_file.exists():
        print(f"找不到评测集文件 {eval_file}")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"共加载 {len(items)} 条评测数据。\n")

    loader = MarkdownLoader()
    doc_cache = {}  # 缓存加载过的文档，避免重复读盘

    for item in items:
        print(f"[{item['id']}] 类型: {item['type']}")
        print(f"  问题: {item['question']}")

        if not item.get("gold_spans"):
            print(f"  (无 gold_spans, 预期拒答: {item['should_abstain']})")
            print("-" * 40)
            continue

        doc_id = item["gold_doc"]
        if doc_id not in doc_cache:
            path = Path(f"data/raw/test_corpus/{doc_id}.md")
            if not path.exists():
                print(f"  ❌ 找不到文档: {path}")
                continue
            doc_cache[doc_id] = loader.load(path)

        raw_text = doc_cache[doc_id].raw_text

        for start, end in item["gold_spans"]:
            span_text = raw_text[start:end]
            print(f"  -> 切片[{start}:{end}] : \"{span_text}\"")

        print("-" * 40)

if __name__ == "__main__":
    main()