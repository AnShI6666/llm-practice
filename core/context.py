from dataclasses import dataclass
from core.models import Chunk
from core.chunkers.base import estimate_tokens

@dataclass
class ContextBudget:
    """一次请求的 token 预算分配。"""

    total: int = 8000          # 总预算
    system_reserve: int = 400  # system prompt 预留
    answer_reserve: int = 400  # 留给模型输出的
    question_reserve: int = 200

    @property
    def available_for_chunks(self) -> int:
        return max(
            0,
            self.total - self.system_reserve
            - self.answer_reserve - self.question_reserve,
        )

def trim_by_budget(
    chunks: list[Chunk],
    budget: ContextBudget,
    strategy: str = "by_score",
    min_chunks: int = 3,
) -> list[Chunk]:
    """按预算裁剪检索结果。

    - by_score: 保持检索顺序，原样返回（条数由调用方的 top_k 控制）
    - by_budget: 累加 token，超预算就停；严格不超限，可能返回空
    - by_budget_with_floor: 至少保留 min_chunks 条，可能超预算
    """
    if not chunks:
        return []

    if strategy == "by_score":
        return list(chunks)

    if strategy not in ("by_budget", "by_budget_with_floor"):
        raise ValueError(f"未知裁剪策略: {strategy}")

    available = budget.available_for_chunks
    chosen: list[Chunk] = []
    used = 0

    for chunk in chunks:
        tok = chunk.token_estimate

        # 保底策略：先无条件拿够 min_chunks 条
        if strategy == "by_budget_with_floor" and len(chosen) < min_chunks:
            chosen.append(chunk)
            used += tok
            continue

        if used + tok > available:
            break

        chosen.append(chunk)
        used += tok

    return chosen