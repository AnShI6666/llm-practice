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
# 匹配拒答
_ABSTAIN = re.compile(re.escape(ABSTAIN_PREFIX))


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
        abstained = bool(_ABSTAIN.search(raw))

        if invalid:
            logger.warning(
                "模型引用了不存在的编号: %s（只有 %d 条资料）",
                invalid, len(chunks),
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