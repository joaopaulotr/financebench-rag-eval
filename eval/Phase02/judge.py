import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CQ_PROMPT = """\
You are evaluating a retrieval-augmented generation system for financial documents (SEC filings).

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Task: Does the retrieved context contain the information needed to answer the question?

For financial questions, consider:
- Is the context from the correct company?
- Does it cover the correct fiscal year / period?
- Does it contain the specific financial metric being asked about?

Score on a scale of 1 to 5:
1 = Completely irrelevant (wrong company, wrong period, wrong metric)
2 = Mentions the company but lacks the specific data needed
3 = Partially relevant (related data, but not the exact answer)
4 = Contains the answer but requires significant inference or calculation
5 = Directly and explicitly contains data to answer the question

Respond with valid JSON only:
{{"score": <1-5>, "reasoning": "<one sentence>"}}"""

_AC_PROMPT = """\
You are evaluating a retrieval-augmented generation system for financial documents (SEC filings).

RETRIEVED CONTEXT:
{context}

MODEL ANSWER:
{answer}

Task: Is every factual claim in the model answer supported by the retrieved context?

Pay special attention to:
- Specific numbers, percentages, dollar amounts — are they in the context?
- Company names, fiscal years — do they match?
- Claims that sound plausible but are not in the context (hallucination)

Score on a scale of 1 to 5:
1 = Major claims are not in the context (clear hallucination)
2 = Most claims are not supported by the context
3 = Some claims supported, some not — mixed faithfulness
4 = Almost all claims supported; minor unsupported details
5 = Every factual claim is directly traceable to the provided context

Respond with valid JSON only:
{{"score": <1-5>, "reasoning": "<one sentence>"}}"""

_AQ_PROMPT = """\
You are evaluating a retrieval-augmented generation system for financial documents (SEC filings).

QUESTION:
{question}

MODEL ANSWER:
{answer}

Task: Does the model answer address the question that was asked?

For financial questions, consider:
- Does it provide the specific metric requested (e.g., revenue, capex, net income)?
- Does it cover the correct time period (fiscal year, quarter)?
- Is the unit correct (USD millions, percentage, ratio)?
- A response of "I don't know" or "not found in context" scores 1 even if technically honest.

Score on a scale of 1 to 5:
1 = Does not answer the question (refuses, goes off-topic, or says "I don't know")
2 = Addresses the topic but not the specific question asked
3 = Partially answers (right direction but incomplete, imprecise, or wrong period)
4 = Answers the question with minor gaps or imprecision
5 = Directly, completely, and precisely answers the question as asked

Respond with valid JSON only:
{{"score": <1-5>, "reasoning": "<one sentence>"}}"""

# ---------------------------------------------------------------------------
# Judge functions
# ---------------------------------------------------------------------------

def _call(prompt: str) -> dict:
    response = llm.invoke(prompt)
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def judge_context_relevance(question: str, context: str) -> dict:
    """C|Q — is the retrieved context relevant to the question?"""
    return _call(_CQ_PROMPT.format(question=question, context=context))


def judge_answer_faithfulness(context: str, answer: str) -> dict:
    """A|C — is the answer grounded in the context (no hallucination)?"""
    return _call(_AC_PROMPT.format(context=context, answer=answer))


def judge_answer_relevance(question: str, answer: str) -> dict:
    """A|Q — does the answer address the question?"""
    return _call(_AQ_PROMPT.format(question=question, answer=answer))


_GT_PROMPT = """\
You are evaluating a retrieval-augmented generation system for financial documents (SEC filings).

QUESTION:
{question}

EXPECTED ANSWER (ground truth):
{expected}

MODEL ANSWER:
{answer}

Task: Does the model answer convey the same information as the expected answer?

Guidelines:
- For numerical answers: allow minor rounding differences (e.g. $1,577M vs $1,577.00M is correct). Wrong number = score 1-2.
- For yes/no answers: direction must match. "Yes" vs "No" = score 1.
- For qualitative answers: key facts must match. Partial match = score 3.
- "I don't know" or refusal always scores 1, even if the question is hard.
- Ignore formatting differences — focus on substance.

Score on a scale of 1 to 5:
1 = Wrong or refused to answer
2 = Right direction but significantly off (wrong number, missing key entity)
3 = Partially correct (some key facts match, others missing or wrong)
4 = Mostly correct with minor imprecision
5 = Correct — matches expected answer in all key facts

Respond with valid JSON only:
{{"score": <1-5>, "reasoning": "<one sentence>"}}"""


def judge_answer_correctness(question: str, expected: str, answer: str) -> dict:
    """A|GT — does the answer match the ground truth expected answer?"""
    return _call(_GT_PROMPT.format(question=question, expected=expected, answer=answer))
