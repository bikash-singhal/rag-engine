import pytest

from src.retrieval.adaptive.adaptive_retriever import AdaptiveRetriever


@pytest.fixture
def adaptive_retriever() -> AdaptiveRetriever:
    return AdaptiveRetriever()


@pytest.mark.parametrize(
    "question",
    [
        "What is the leave policy?",
        "Who is eligible for reimbursement?",
        "How many annual leave days are allowed?",
        "Define probation period.",
    ],
)
def test_selects_simple_strategy(
    adaptive_retriever: AdaptiveRetriever,
    question: str,
) -> None:
    config = adaptive_retriever.get_retrieval_config(question)

    assert config.strategy == "simple"
    assert config.retrieval_top_k == 8
    assert config.final_top_k == 4


@pytest.mark.parametrize(
    "question",
    [
        "Compare annual leave and sick leave.",
        "What is the difference between leave types?",
        "Annual leave versus casual leave",
        "Explain the pros and cons of remote work.",
    ],
)
def test_selects_comparative_strategy(
    adaptive_retriever: AdaptiveRetriever,
    question: str,
) -> None:
    config = adaptive_retriever.get_retrieval_config(question)

    assert config.strategy == "comparative"
    assert config.retrieval_top_k == 15
    assert config.final_top_k == 6


@pytest.mark.parametrize(
    "question",
    [
        "Provide an overview of the security guidelines.",
        "Explain everything about the employee policies.",
        "Summarize all travel procedures.",
        (
            "Explain the complete process that employees must follow "
            "when submitting different categories of expense claims "
            "across departments and approval levels."
        ),
    ],
)
def test_selects_broad_strategy(
    adaptive_retriever: AdaptiveRetriever,
    question: str,
) -> None:
    config = adaptive_retriever.get_retrieval_config(question)

    assert config.strategy == "broad"
    assert config.retrieval_top_k == 25
    assert config.final_top_k == 8


def test_selects_default_strategy(
    adaptive_retriever: AdaptiveRetriever,
) -> None:
    config = adaptive_retriever.get_retrieval_config(
        "Tell me about employee reimbursement.",
    )

    assert config.strategy == "default"
    assert config.retrieval_top_k == 12
    assert config.final_top_k == 5


def test_empty_question_uses_default_strategy(
    adaptive_retriever: AdaptiveRetriever,
) -> None:
    config = adaptive_retriever.get_retrieval_config("   ")

    assert config.strategy == "default"
