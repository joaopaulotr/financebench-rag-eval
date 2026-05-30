"""Judge v2: stricter A|GT with numerical extraction + tolerance rules."""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Phase02"))

from judge import (
    judge_context_relevance,
    judge_answer_faithfulness,
    judge_answer_relevance,
    _call,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_GT_PROMPT_V2 = """\
You are evaluating a financial RAG system. Your job is to score whether the model answer matches the ground truth.

QUESTION:
{question}

EXPECTED ANSWER (ground truth):
{expected}

MODEL ANSWER:
{answer}

## MANDATORY EVALUATION PROCESS — follow in order:

**Step 1 — Extract numbers**
- Extract the final numerical value(s) from the ground truth.
- Extract the final numerical value(s) from the model answer.
- Normalize to the same format before comparing: 0.8 = 80%; $2.018B = $2,018M; ratio 0.02 = 2%.

**Step 2 — Compute relative difference**
- relative_diff = |model_value - gt_value| / |gt_value|
- If gt_value is 0, use absolute difference.

**Step 3 — Apply scoring rules**

For NUMERICAL questions (any answer with a specific number):
- relative_diff ≤ 5%  AND reasoning correct → score 5
- relative_diff ≤ 5%  but minor explanation issue → score 4
- relative_diff 5–15% with correct methodology → score 3
- relative_diff > 15% regardless of reasoning quality → score 1 or 2
- A well-explained wrong number is STILL WRONG. Do not reward methodology that produces incorrect results.
- Matching direction only ("yes it increased") without matching the specific numbers → score 2 max.

For YES/NO or qualitative questions (no specific number in ground truth):
- Direction must match exactly (yes vs no = score 1)
- Key entities/facts must match (wrong segment, wrong company = score 1-2)
- Partial match on key facts → score 3

**Step 4 — Final check**
- "I don't know" or refusal → score 1 always
- Ignore formatting differences, focus on numerical accuracy and factual correctness

Score scale:
5 = Numbers match within 5% AND reasoning correct
4 = Numbers match within 5% but minor explanation issue
3 = Numbers off by 5–15% with correct method, OR qualitative partial match
2 = Numbers off >15% but direction/approach correct
1 = Wrong numbers (>15% off), wrong direction, hallucinated values, or refusal

Respond with valid JSON only:
{{"score": <1-5>, "reasoning": "<one sentence explaining the numerical comparison and why you chose this score>"}}"""


def judge_answer_correctness_v2(question: str, expected: str, answer: str) -> dict:
    """A|GT v2 — strict numerical comparison before qualitative evaluation."""
    return _call(_GT_PROMPT_V2.format(question=question, expected=expected, answer=answer))
