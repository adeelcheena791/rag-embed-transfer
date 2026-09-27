"""Exact prompt text (design v0.2 Appendix A and B, addendum v0.2.1 A.3 and B.2).

The strings below are the pre-registered prompts, copied verbatim; PROMPT_HASH changes if any
character changes, and every generation / judgement record stores it, so a silent edit is detectable.
"""
from __future__ import annotations

import hashlib

# ---------------- Appendix A: reader ----------------
A1_SYSTEM = ("You are a question answering assistant. Answer the question using the passages provided. "
             "If the passages do not contain the answer, give your best short answer anyway. "
             "Reply with the answer only, as a short phrase of a few words, with no explanation.")
A2_SYSTEM = ("You are a question answering assistant. "
             "Reply with the answer only, as a short phrase of a few words, with no explanation.")
A3_SYSTEM = ("You are a question answering assistant. Answer the question using only the passages provided. "
             "If the passages do not contain the answer, reply exactly: I don't know. "
             "Otherwise reply with the answer only, as a short phrase of a few words, with no explanation.")


def format_passages(passages: list[tuple[str, str]]) -> str:
    """[(title, text), ...] in rank order -> '[1] title\\ntext\\n\\n[2] ...' (Appendix A.1)."""
    return "\n\n".join(f"[{i}] {title}\n{text}" for i, (title, text) in enumerate(passages, start=1))


def reader_messages(question: str, passages: list[tuple[str, str]] | None, variant: str) -> list[dict]:
    if variant == "A2":
        if passages:
            raise ValueError("closed-book prompt must not receive passages")
        return [{"role": "system", "content": A2_SYSTEM},
                {"role": "user", "content": f"Question: {question}\nAnswer:"}]
    if variant not in ("A1", "A3"):
        raise ValueError(f"unknown reader prompt {variant}")
    if not passages:
        raise ValueError(f"{variant} needs >= 1 passage")
    system = A1_SYSTEM if variant == "A1" else A3_SYSTEM
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Passages:\n{format_passages(passages)}\n\nQuestion: {question}\nAnswer:"}]


# ---------------- Appendix B: judge ----------------
B_HEADER = ("You will be given PASSAGES, a QUESTION, and an ANSWER.\n"
            "Decide whether the ANSWER is fully supported by the PASSAGES.\n"
            "An answer is SUPPORTED only if the passages state it or directly imply it.\n"
            "Do not use outside knowledge. Do not judge whether the answer is true in the real world.\n\n")
B_FEWSHOT = """Here are three examples.

Example 1
PASSAGES:
[1] Lake Arvel
Lake Arvel is a freshwater lake in the fictional region of Tolmark. Its deepest point is 212 metres.
QUESTION: how deep is lake arvel
ANSWER: 212 metres
Verdict: SUPPORTED

Example 2
PASSAGES:
[1] Orsin Bridge
The Orsin Bridge was completed in 1874 and spans the river Dellet.
QUESTION: who designed the orsin bridge
ANSWER: Marta Keel
Verdict: NOT_SUPPORTED

Example 3
PASSAGES:
[1] Valen Cup
The Valen Cup is an annual rowing race. The 2019 race was won by the Harrow Point crew.
[2] Harrow Point
Harrow Point is a coastal town known for its rowing club.
QUESTION: who won the 2019 valen cup
ANSWER: Harrow Point
Verdict: SUPPORTED

Now the item to judge.

"""
B_TAIL = "\n\nQUESTION: {question}\nANSWER: {answer}\n\nReply with exactly one word: SUPPORTED or NOT_SUPPORTED."


def judge_messages(question: str, passages: list[tuple[str, str]], answer: str, variant: str) -> list[dict]:
    """Single user message, no system role (Appendix B). variant: zero_shot | few_shot (F2)."""
    if variant not in ("zero_shot", "few_shot"):
        raise ValueError(f"unknown judge variant {variant}")
    body = B_HEADER + (B_FEWSHOT if variant == "few_shot" else "") + "PASSAGES:\n" + format_passages(passages)
    return [{"role": "user", "content": body + B_TAIL.format(question=question, answer=answer)}]


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()[:12]


PROMPT_HASH = {"A1": _hash(A1_SYSTEM), "A2": _hash(A2_SYSTEM), "A3": _hash(A3_SYSTEM),
               "zero_shot": _hash(B_HEADER, B_TAIL), "few_shot": _hash(B_HEADER, B_FEWSHOT, B_TAIL)}
