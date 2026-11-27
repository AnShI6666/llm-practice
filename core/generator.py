"""RAG 生成器：调用 LLM 并解析引用。"""
import re
import time
import logging
from core.exceptions import LLMError
from core.llm_client import LLMClient
from core.models import Chunk, Citation, RagAnswer
from core.prompts import ABSTAIN_PREFIX, SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

# 匹配 [1] [2] [1][3] 这类标记
_MARKER = re.compile(r"\[(\d+)\]")
# 允许的拒答表述：标准话术 + 常见改写（只用于判定，不写进 prompt）
_ABSTAIN_PHRASES: tuple[str, ...] = (
    ABSTAIN_PREFIX,
    "资料中没有提到",
    "资料中未提到",
    "资料未涉及",
    "没有相关资料",
    "资料不足",
    "无法回答",
)

# 真拒答的长度上限：话术本身很短，超过这个长度说明实质上是回答了
_MAX_ABSTAIN_LEN = 80


def judge_abstained(answer: str) -> bool:
    """判定一次回答是否属于「拒答」。
    两条规则**同时满足**才算拒答：
    1. **开头匹配**：去掉首尾空白后，必须以某种拒答话术开头。
       解决 q023 假阳性——完整回答之后追加拒答话术，不该算拒答。
    2. **有效长度约束**：去掉话术后剩余内容 < 80 字。
       解决"先说未找到、再长篇补充"的伪拒答——那种实质上是回答了。
    Args:
        answer: 模型原始回答。
    Returns:
        True 表示拒答。
    Examples:
        >>> judge_abstained("未找到相关内容。缺少：作者背景。")
        True
        >>> judge_abstained("资料中没有提到那位修表店的具体街道。")
        True
        >>> judge_abstained("水温 88-94℃ [1]。\\n\\n未找到相关内容。缺少：个人情况")
        False
    """
    text = answer.strip()
    if not text:
        return True
    head = next((p for p in _ABSTAIN_PHRASES if text.startswith(p)), None)
    if head is None:
        return False
    rest = text[len(head):].strip(" 　。，、：:；;\n")
    return len(rest) < _MAX_ABSTAIN_LEN

class Generator:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def generate(self, question: str, chunks: list[Chunk]) -> RagAnswer:
        if not chunks:
            return RagAnswer(
                question=question,
                answer=f"{ABSTAIN_PREFIX}。缺少：没有检索到任何相关资料。",
                abstained=True,
            )

        t0 = time.perf_counter()
        raw, prompt_tokens, completion_tokens = self._llm.ask_with_usage(
            build_user_prompt(question, chunks),
            system=SYSTEM_PROMPT,
        )
        latency = int((time.perf_counter() - t0) * 1000)

        if not raw.strip():
            logger.warning("模型返回空回答，question=%r", question)
            raise LLMError("模型返回空回答")

        citations, invalid = self._parse_citations(raw, chunks)
        abstained = judge_abstained(raw)

        if invalid:
            logger.warning(
                "模型引用了不存在的编号: %s（只有 %d 条资料）",
                invalid, len(chunks),
            )

        # 一致性提示：判定为拒答却带了引用，说明模型在"边拒答边引用"，
        # 需要人工复核（不改变判定，只留痕）
        if abstained and citations:
            logger.info(
                "拒答样本却含 %d 条引用，请复核：%s",
                len(citations), question[:40],
            )

        return RagAnswer(
            question=question,
            answer=raw,
            citations=citations,
            abstained=abstained,
            invalid_markers=invalid,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency,
        )

    def _parse_citations(
        self, answer: str, chunks: list[Chunk]
    ) -> tuple[list[Citation], list[int]]:
        """把答案里的 [n] 还原成 Citation，并找出非法编号。"""
        cited: list[Citation] = []
        invalid: list[int] = []
        seen: set[int] = set()

        for m in _MARKER.finditer(answer):
            n = int(m.group(1))
            if n in seen:
                continue
            seen.add(n)
            if 1 <= n <= len(chunks):
                c = chunks[n - 1]
                cited.append(Citation(
                    marker=n,
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    start_char=c.start_char,
                    end_char=c.end_char,
                    quote=c.text,
                ))
            else:
                invalid.append(n)

        return cited, invalid