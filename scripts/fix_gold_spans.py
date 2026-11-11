"""修复评测集中 gold_spans 为空（[[0,0]]）的 5 条样本。

用法（在项目根目录执行）：
    .\.venv\Scripts\python.exe scripts/fix_gold_spans.py

修复后请重跑：
    .\.venv\Scripts\python.exe scripts/eval_retrieval.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.loaders.markdown import MarkdownLoader  # noqa: E402

QUESTIONS = ROOT / "data/eval/questions.json"

# 经人工核对过原文的正确答案区间
# 判定依据：逐条打开原文确认「这段文字确实回答了该问题」
FIX = {
    "q021": {
        "gold_doc": "02-手冲咖啡入门指南",
        "gold_spans": [[921, 957], [1035, 1068]],
        # 太苦→水温降低 / 涩→水温过高
    },
    "q022": {
        "gold_doc": "03-Git分支策略对比",
        "gold_spans": [[1399, 1425], [1512, 1568]],
        # 一天部署多次→主干开发 / GitFlow 与每天上线矛盾
    },
    "q023": {
        "gold_doc": "04-睡眠的科学",
        "gold_spans": [[468, 505], [591, 628]],
        # 总时长八小时但恢复差→连续性 / 规律>连续>时长
    },
    "q024": {
        "gold_doc": "09-家庭网络优化",
        "gold_spans": [[536, 570], [645, 692]],
        # Cat5 是千兆杀手 / 卡在 94Mbps 是百兆链路特征
    },
    "q025": {
        "gold_doc": "02-手冲咖啡入门指南",
        "gold_spans": [[246, 285], [1253, 1323]],
        # 钱花在磨豆机 / 别急着换器具，先把流程重复二十次
    },
}


def main() -> int:
    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    ldr = MarkdownLoader()
    changed = 0

    for item in items:
        qid = item.get("id")
        if qid not in FIX:
            continue
        fix = FIX[qid]
        item["gold_doc"] = fix["gold_doc"]
        item["gold_spans"] = fix["gold_spans"]

        # 立刻验证：区间必须落在文档内且非空
        raw = ldr.load(ROOT / "data/raw/test_corpus" / f"{fix['gold_doc']}.md").raw_text
        for s, e in fix["gold_spans"]:
            assert 0 <= s < e <= len(raw), f"{qid} 区间非法: [{s}, {e}] (文档长 {len(raw)})"
            assert raw[s:e].strip(), f"{qid} 区间 [{s}, {e}] 是空文本"
        changed += 1
        print(f"✅ {qid} 已修复 → {fix['gold_doc']} {fix['gold_spans']}")

    if changed == 0:
        print("⚠️ 没有匹配到需要修复的样本（可能已修过）")
        return 1

    QUESTIONS.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n共修复 {changed} 条，已写入 {QUESTIONS}")
    print("下一步：重跑 scripts/eval_retrieval.py 获取真实 Recall@5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())