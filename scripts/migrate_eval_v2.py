"""把 data/eval/questions.json 迁移成 eval/dataset.jsonl。"""
import json
from pathlib import Path

SRC = Path("data/eval/questions.json")
DST = Path("eval/dataset.jsonl")


def main() -> int:
    items = json.loads(SRC.read_text(encoding="utf-8"))
    DST.parent.mkdir(parents=True, exist_ok=True)

    with DST.open("w", encoding="utf-8") as f:
        for item in items:
            row = {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "gold_doc": item["gold_doc"],
                "gold_spans": item["gold_spans"],
                "gold_answer": item.get("gold_answer", ""),
                "should_abstain": item.get("should_abstain", False),
                "notes": item.get("notes", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"写入 {len(items)} 条到 {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())