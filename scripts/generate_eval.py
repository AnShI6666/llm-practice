"""自动生成 20 条评测集，并计算精准的 gold_spans。"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.loaders.markdown import MarkdownLoader

# 20 条题目的种子数据
SEED_DATA = [
    {
        "id": "q001", "type": "fact", "question": "手冲咖啡一般建议用多少度的水？",
        "gold_doc": "02-手冲咖啡入门指南", "anchor": "水温 | 88–94 ℃", "should_abstain": False, "notes": "核心变量表格"
    },
    {
        "id": "q002", "type": "fact", "question": "在 Git 的哪种分支策略下，开发人员会直接向 main 提交或者使用短分支？",
        "gold_doc": "03-Git分支策略对比", "anchor": "所有人直接往 `main` 提交，或者用存活不超过一天的短分支。",
        "should_abstain": False, "notes": "主干开发特征"
    },
    {
        "id": "q003", "type": "fact", "question": "睡眠周期中，哪个阶段主要负责身体修复和免疫？",
        "gold_doc": "04-睡眠的科学", "anchor": "N3 深睡（慢波） | 15–25% | 身体修复、免疫、生长激素分泌",
        "should_abstain": False, "notes": "深睡作用"
    },
    {
        "id": "q004", "type": "fact", "question": "摄影构图中的三分法，建议把地平线放在什么位置？",
        "gold_doc": "06-摄影构图笔记", "anchor": "横线放地平线：**不要放在正中间**，放在上三分之一或下三分之一。",
        "should_abstain": False, "notes": "三分法避坑"
    },
    {
        "id": "q005", "type": "fact", "question": "家庭网络中，网线至少要达到什么级别才能满足千兆带宽的需求？",
        "gold_doc": "09-家庭网络优化", "anchor": "千兆需要 **Cat5e 及以上**（Cat5e / Cat6）", "should_abstain": False,
        "notes": "网线选型"
    },
    {
        "id": "q006", "type": "fact", "question": "财务规划中，应急金的目标金额通常设定为几个月的生存成本？",
        "gold_doc": "07-个人财务第一性原理", "anchor": "第一层：应急金 | 失业、生病、突发 | 3–6 个月生存成本",
        "should_abstain": False, "notes": "应急金标准"
    },
    {
        "id": "q007", "type": "multi_hop", "question": "为什么在下午三点喝咖啡可能会导致晚上入睡困难？",
        "gold_doc": "04-睡眠的科学",
        "anchor": "咖啡因的半衰期约 5–6 小时，下午 3 点的一杯咖啡，到晚上 9 点还有约四分之一的量在起作用。",
        "should_abstain": False, "notes": "跨段落：咖啡因半衰期 + 睡眠压力"
    },
    {
        "id": "q008", "type": "multi_hop", "question": "根据文中所述，如何从咖啡的味道反推出是哪个手冲变量出了问题？",
        "gold_doc": "02-手冲咖啡入门指南",
        "anchor": "**太酸、像柠檬水** → 萃取不足。研磨调细，或水温升高 2 ℃。",
        "should_abstain": False, "notes": "需要结合四个核心变量与味道反向调试"
    },
    {
        "id": "q009", "type": "multi_hop", "question": "为什么在 Git Flow 中，长期存在的 develop 分支会有风险？",
        "gold_doc": "03-Git分支策略对比",
        "anchor": "**缺点**：分支多、合并路径长，长期存在的 `develop` 与 `main` 容易漂移。",
        "should_abstain": False, "notes": "需要理解分支流向与合并冲突"
    },
    {
        "id": "q010", "type": "multi_hop", "question": "作者建议采用哪些具体的策略来对抗“时间不够”这个假象？",
        "gold_doc": "01-时间管理的底层逻辑",
        "anchor": "关键动作只有一个：**先保护深度时间，再用浅层时间去填剩下的缝隙**。",
        "should_abstain": False, "notes": "需要汇总深读、浅层、恢复时间的策略"
    },
    {
        "id": "q011", "type": "multi_hop",
        "question": "如果发现无论换什么设备测速都在百兆左右，可能是什么原因且如何解决？",
        "gold_doc": "09-家庭网络优化",
        "anchor": "一个典型症状：无论怎么换设备，测速都卡在 **94 Mbps** 左右。这就是百兆链路的特征。",
        "should_abstain": False, "notes": "需要结合排查顺序的第一、二步"
    },
    {
        "id": "q012", "type": "unanswerable", "question": "这篇文章的作者本人平时有喝手冲咖啡的习惯吗？",
        "gold_doc": "02-手冲咖啡入门指南", "anchor": None, "should_abstain": True, "notes": "文档未提及作者背景"
    },
    {
        "id": "q013", "type": "unanswerable",
        "question": "个人财务文章中提到的“沪深300指数基金”，其具体的年度管理费率是多少？",
        "gold_doc": "07-个人财务第一性原理", "anchor": None, "should_abstain": True, "notes": "文中只提到整体主动基金费率，未具体到沪深300"
    },
    {
        "id": "q014", "type": "unanswerable", "question": "Git 主干开发策略中，要求开发团队每天至少提交多少个 commit？",
        "gold_doc": "03-Git分支策略对比", "anchor": None, "should_abstain": True, "notes": "文中只说“小而频繁”，无具体数字"
    },
    {
        "id": "q015", "type": "unanswerable",
        "question": "城市漫步指南里提到的那家“开了三十年、招牌已经掉色的修表店”，位于哪一条街？",
        "gold_doc": "10-城市漫步指南", "anchor": None, "should_abstain": True, "notes": "只是举例子，无具体地点"
    },
    {
        "id": "q016", "type": "unanswerable", "question": "睡眠科学里提到的“建设性担忧”方法，是由哪位心理学家提出的？",
        "gold_doc": "04-睡眠的科学", "anchor": None, "should_abstain": True, "notes": "文中未提到人名"
    },
    {
        "id": "q017", "type": "trap", "question": "手冲咖啡时，为了萃取更充分，最好把水温和研磨度同时调高，对吗？",
        "gold_doc": "02-手冲咖啡入门指南", "anchor": None, "should_abstain": True, "notes": "原文说“一次只动一个变量”，问题故意说反"
    },
    {
        "id": "q018", "type": "trap",
        "question": "在家庭网络设置中，如果家里有承重墙挡住5G信号，应该把 2.4G 频段关掉来强迫手机连 5G，对吗？",
        "gold_doc": "09-家庭网络优化", "anchor": None, "should_abstain": True, "notes": "原文说“问题不是频段，是需要加节点”"
    },
    {
        "id": "q019", "type": "trap",
        "question": "文章认为，像 Git Flow 这种多分支模型，因为严谨，所以更适合一天部署多次的团队，对吗？",
        "gold_doc": "03-Git分支策略对比", "anchor": None, "should_abstain": True, "notes": "原文说“在持续部署场景下属于过度设计”"
    },
    {
        "id": "q020", "type": "trap",
        "question": "投资时，如果想要获得超额收益，就应该频繁买卖不同的主动型基金来择时，对吗？",
        "gold_doc": "07-个人财务第一性原理", "anchor": None, "should_abstain": True, "notes": "原文推荐指数基金、定投，反对频繁换标的、择时"
    },
]


def main():
    corpus_dir = Path("data/raw/test_corpus")
    eval_dir = Path("data/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    loader = MarkdownLoader()
    docs = {}
    for p in corpus_dir.glob("*.md"):
        doc_id = p.stem
        docs[doc_id] = loader.load(p)

    results = []
    for item in SEED_DATA:
        item_copy = item.copy()

        if item["anchor"]:
            doc = docs.get(item["gold_doc"])
            if not doc:
                print(f"❌ 找不到文档 {item['gold_doc']}，跳过 {item['id']}")
                continue

            start = doc.raw_text.find(item["anchor"])
            if start != -1:
                end = start + len(item["anchor"])
                item_copy["gold_spans"] = [[start, end]]
            else:
                print(f"❌ 在 {item['gold_doc']} 中找不到锚点: '{item['anchor']}'")
                item_copy["gold_spans"] = []
        else:
            item_copy["gold_spans"] = []

        # 删掉临时的 anchor 字段
        del item_copy["anchor"]
        results.append(item_copy)

    out_file = eval_dir / "questions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 成功生成 {len(results)} 条评测数据！已保存至 {out_file}")
    print("你可以运行 `python scripts/validate_eval.py` 进行人工核对。")


if __name__ == "__main__":
    main()