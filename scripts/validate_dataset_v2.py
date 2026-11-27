"""校验 eval/dataset.jsonl：字段、区间、gold_answer。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.loaders.markdown import MarkdownLoader

PATH = Path("eval/dataset.jsonl")
REQUIRED = {"id", "type", "question", "gold_doc", "gold_spans",
            "gold_answer", "should_abstain"}
TYPES = {"fact", "multi_hop", "unanswerable", "trap"}


def main() -> int:
    loader = MarkdownLoader()
    cache: dict[str, str] = {}
    errors: list[str] = []
    ids: set[str] = set()
    counts: dict[str, int] = {}

    for line_no, line in enumerate(PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        missing = REQUIRED - set(item)
        if missing:
            errors.append(f"line {line_no}: 缺字段 {missing}")
            continue

        qid = item["id"]
        if qid in ids:
            errors.append(f"line {line_no}: id 重复 {qid}")
        ids.add(qid)

        qtype = item["type"]
        counts[qtype] = counts.get(qtype, 0) + 1
        if qtype not in TYPES:
            errors.append(f"{qid}: 未知 type {qtype}")

        doc_id = item["gold_doc"]
        if doc_id not in cache:
            p = Path("data/raw/test_corpus") / f"{doc_id}.md"
            if not p.exists():
                errors.append(f"{qid}: 找不到文档 {p}")
                continue
            cache[doc_id] = loader.load(p).raw_text
        raw = cache[doc_id]

        spans = item["gold_spans"]
        if qtype in ("fact", "multi_hop"):
            if not spans:
                errors.append(f"{qid}: {qtype} 的 gold_spans 为空")
            for s, e in spans:
                if not (0 <= s < e <= len(raw)):
                    errors.append(f"{qid}: 区间非法 [{s}, {e}]")
                elif not raw[s:e].strip():
                    errors.append(f"{qid}: 区间 [{s}, {e}] 是空文本")
            if not item["gold_answer"].strip():
                errors.append(f"{qid}: gold_answer 为空")

        if qtype in ("unanswerable", "trap") and not item["should_abstain"]:
            errors.append(f"{qid}: {qtype} 应设 should_abstain=true")

    print("类型统计:", counts)
    print("总数:", sum(counts.values()))
    if errors:
        print(f"\n发现 {len(errors)} 个问题：")
        for e in errors:
            print(" -", e)
        return 1
    print("校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())