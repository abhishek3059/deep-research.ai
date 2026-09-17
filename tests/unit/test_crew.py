"""Unit tests for CriticAgent in src.agents.crew."""

from __future__ import annotations


class TestCriticAgentInitialization:
    """Tests for CriticAgent initialization."""

    def test_critic_agent_initialization(self):
        """Test that CriticAgent initializes correctly."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()
        assert agent._agent is not None
        assert agent._agent.role == "Quality Critic"


class TestParseReview:
    """Tests for _parse_review method."""

    def test_parse_review_with_good_scores(self):
        """Test parsing a review with high scores."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        raw_result = (
            "Faithfulness: 0.9\n"
            "Relevancy: 0.85\n"
            "Completeness: 0.8\n"
            "The answer is well-grounded in the context."
        )

        result = agent._parse_review(raw_result, "test answer", ["context1"])
        assert result["passed"] is True
        assert result["faithfulness"] == 0.9
        assert result["relevancy"] == 0.85
        assert result["completeness"] == 0.8

    def test_parse_review_with_low_scores(self):
        """Test parsing a review with low scores."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        raw_result = (
            "Faithfulness: 0.3\n"
            "Relevancy: 0.4\n"
            "Completeness: 0.2\n"
            "Contains hallucination and irrelevant content."
        )

        result = agent._parse_review(raw_result, "test answer", ["context1"])
        assert result["passed"] is False
        assert result["faithfulness"] == 0.3
        assert result["relevancy"] == 0.4
        assert result["completeness"] == 0.2
        assert "Potential hallucination detected" in result["issues"]
        assert "Contains irrelevant information" in result["issues"]

    def test_parse_review_with_incomplete_flag(self):
        """Test that 'incomplete' keyword is detected as an issue."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        raw_result = (
            "Faithfulness: 0.7\n"
            "Relevancy: 0.6\n"
            "Completeness: 0.5\n"
            "Answer may be incomplete."
        )

        result = agent._parse_review(raw_result, "answer", ["context"])
        assert "Answer may be incomplete" in result["issues"]

    def test_parse_review_summary_truncated(self):
        """Test that summary is truncated to 500 chars."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        long_result = "x" * 600
        result = agent._parse_review(long_result, "answer", ["context"])
        assert len(result["summary"]) == 500

    def test_parse_review_empty_result(self):
        """Test parsing with empty raw result."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        result = agent._parse_review("", "answer", ["context"])
        assert result["summary"] == "Review completed"


class TestExtractScore:
    """Tests for _extract_score method."""

    def test_extract_score_with_fraction(self):
        """Test extracting score in X/10 format."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        score = agent._extract_score(
            "faithfulness: 8/10", "faithfulness", "answer", ["context"]
        )
        assert score == 0.8

    def test_extract_score_with_decimal(self):
        """Test extracting score in decimal format."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        score = agent._extract_score(
            "faithfulness: 0.75", "faithfulness", "answer", ["context"]
        )
        assert score == 0.75

    def test_extract_score_with_percentage(self):
        """Test extracting score in percentage format."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        score = agent._extract_score(
            "faithfulness: 80%", "faithfulness", "answer", ["context"]
        )
        assert score == 0.8

    def test_extract_score_case_insensitive(self):
        """Test that dimension matching is case-insensitive."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        score = agent._extract_score(
            "FAITHFULNESS: 0.9", "faithfulness", "answer", ["context"]
        )
        assert score == 0.9

    def test_extract_score_fallback_with_context(self):
        """Test fallback scoring uses keyword overlap when no pattern found."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        score = agent._extract_score(
            "no scores here", "faithfulness", "the answer is correct", ["the answer is correct"]
        )
        # Fallback computes overlap: {"the","answer","is","correct"} vs {"the","answer","is","correct"}
        # overlap = 4/4 = 1.0
        assert score == 1.0

    def test_extract_score_fallback_partial_overlap(self):
        """Test fallback scoring with partial keyword overlap."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        score = agent._extract_score(
            "no scores here", "faithfulness", "the answer", ["the answer is correct"]
        )
        # overlap = {"the","answer"} & {"the","answer","is","correct"} = 2/2 = 1.0
        assert score == 1.0

    def test_extract_score_fallback_no_overlap(self):
        """Test fallback scoring with zero keyword overlap."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        score = agent._extract_score(
            "no scores here", "faithfulness", "xyz abc", ["the answer is correct"]
        )
        # overlap = {"xyz","abc"} & {"the","answer","is","correct"} = 0/2 = 0.0
        assert score == 0.0

    def test_extract_score_fallback_no_context(self):
        """Test fallback returns 0.5 when no context provided."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        score = agent._extract_score(
            "no scores here", "faithfulness", "answer", []
        )
        assert score == 0.5

    def test_extract_score_fallback_empty_answer(self):
        """Test fallback returns 0.0 when answer is empty."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        score = agent._extract_score(
            "no scores here", "faithfulness", "", ["context"]
        )
        assert score == 0.0

    def test_extract_score_clamped_above_one(self):
        """Test that scores are clamped to max 1.0."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        score = agent._extract_score(
            "faithfulness: 11/10", "faithfulness", "answer", ["context"]
        )
        assert score == 1.0

    def test_extract_score_clamped_below_zero(self):
        """Test that scores are clamped to min 0.0."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        score = agent._extract_score(
            "faithfulness: -5", "faithfulness", "answer", ["context"]
        )
        assert score == 0.0


class TestPassFailLogic:
    """Tests for pass/fail determination."""

    def test_passed_when_all_scores_high(self):
        """Test that passed is True when all scores >= 0.5."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        raw_result = "Faithfulness: 0.6 Relevancy: 0.7 Completeness: 0.8"
        result = agent._parse_review(raw_result, "answer", ["context"])
        assert result["passed"] is True

    def test_failed_when_any_score_low(self):
        """Test that passed is False when any score < 0.5."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        raw_result = "Faithfulness: 0.3 Relevancy: 0.8 Completeness: 0.9"
        result = agent._parse_review(raw_result, "answer", ["context"])
        assert result["passed"] is False

    def test_passed_at_boundary(self):
        """Test that passed is True when all scores exactly 0.5."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        raw_result = "Faithfulness: 0.5 Relevancy: 0.5 Completeness: 0.5"
        result = agent._parse_review(raw_result, "answer", ["context"])
        assert result["passed"] is True

    def test_failed_when_just_below_boundary(self):
        """Test that passed is False when one score is 0.49."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        raw_result = "Faithfulness: 0.49 Relevancy: 0.8 Completeness: 0.9"
        result = agent._parse_review(raw_result, "answer", ["context"])
        assert result["passed"] is False

    def test_passed_all_scores_perfect(self):
        """Test that passed is True when all scores are 1.0."""
        from src.agents.crew import CriticAgent

        agent = CriticAgent()

        raw_result = "Faithfulness: 1.0 Relevancy: 1.0 Completeness: 1.0"
        result = agent._parse_review(raw_result, "answer", ["context"])
        assert result["passed"] is True
