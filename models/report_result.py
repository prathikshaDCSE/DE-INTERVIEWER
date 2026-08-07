"""
models/report_result.py

Structured, immutable representation of a single completed hiring
report. ``ReportService`` is the only thing that should construct
this object -- callers just read it.

Design notes
------------
* Kept as a plain ``@dataclass(frozen=True)`` rather than a pydantic
  model, mirroring ``models/evaluation_result.py`` and
  ``models/gemini_response.py``: this is an internal, in-process value
  object produced entirely from data already validated by
  ``ReportService._validate_report``, so no further validation is
  needed on the read path.
* This model represents exactly one thing: the outcome of generating
  the final hiring report for one completed interview. It never
  smuggles a raw ``InterviewSession``, raw ``EvaluationResult`` list,
  or raw Gemini response into itself -- ``ReportService`` always
  returns this normalized shape.
* Mutable inputs (lists) are converted to tuples in ``__post_init__``
  so the dataclass is genuinely immutable/hashable-safe, not merely
  frozen at the attribute level. ``competency_scores`` is a tuple of
  small immutable ``CompetencyScoreSummary`` records rather than a
  raw dict, so each competency's aggregate (score, confidence,
  strengths, weaknesses) travels together as one typed unit instead of
  four parallel structures the caller has to zip up themselves.
* ``generated_at`` is kept as a ``datetime`` object rather than a
  pre-formatted string, mirroring ``EvaluationResult.timestamp`` --
  sorting, comparison, and any future BigQuery/serialization use are
  all cleaner against a real ``datetime``. It is converted to
  ISO-8601 only at the ``to_dict()`` boundary.
* ``decision`` and ``confidence_level`` are deliberately left as plain
  ``str`` rather than a ``Literal`` -- unlike ``EvaluationResult``'s
  fixed three-value ``next_action``, both of these come from
  configuration (``thresholds.yaml`` ``decision_thresholds`` /
  ``confidence_scoring.confidence_levels``) and their legal value sets
  can change without a code change, so a compile-time-checked
  ``Literal`` would be actively wrong here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class CompetencyScoreSummary:
    """
    Aggregated evaluation performance for a single competency across
    every question asked in that competency during the interview.

    Attributes:
        competency: Competency id (e.g. ``"C1"``).
        competency_name: Human-readable competency name (e.g.
            ``"SQL & Query Optimization"``), resolved from
            ``competencies.yaml`` at aggregation time so the report
            prompt and rendered report never have to re-resolve it.
        average_score: Mean raw score (0-5 scale) across every
            evaluation recorded for this competency.
        percentage: ``average_score`` expressed as a 0-100 percentage
            of the configured maximum score.
        average_confidence: Mean ``EvaluationResult.confidence``
            (0-100) across every evaluation recorded for this
            competency.
        question_count: Number of questions (including follow-ups)
            evaluated for this competency.
        strengths: Deduplicated, order-preserved strengths drawn from
            this competency's evaluations only.
        weaknesses: Deduplicated, order-preserved weaknesses drawn
            from this competency's evaluations only.
    """

    competency: str
    competency_name: str
    average_score: float
    percentage: float
    average_confidence: float
    question_count: int
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "strengths", tuple(self.strengths))
        object.__setattr__(self, "weaknesses", tuple(self.weaknesses))

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of this competency summary."""
        return {
            "competency": self.competency,
            "competency_name": self.competency_name,
            "average_score": self.average_score,
            "percentage": self.percentage,
            "average_confidence": self.average_confidence,
            "question_count": self.question_count,
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
        }


@dataclass(frozen=True)
class ReportResult:
    """
    Result of generating the final hiring report for one completed
    interview.

    Attributes:
        candidate_name: The candidate's name, as recorded on the
            ``InterviewSession``.
        overall_score: Weighted overall score, 0-100, computed by
            ``ReportService.calculate_overall_score`` from every
            ``EvaluationResult`` in the session -- never computed by
            Gemini.
        overall_percentage: ``overall_score`` expressed as a
            percentage of the configured maximum
            (``competency_scoring.maximum_score`` in
            ``thresholds.yaml``). Distinct from ``overall_score``
            only when that configured maximum is not 100; identical
            to it otherwise.
        decision: Hiring decision string (e.g. ``"SELECT"``,
            ``"HOLD"``, ``"REJECT"``), resolved entirely from
            ``thresholds.yaml`` ``decision_thresholds`` -- never
            hardcoded.
        confidence_level: Confidence label (e.g. ``"HIGH"``,
            ``"MEDIUM"``, ``"LOW"``), resolved from ``thresholds.yaml``
            ``confidence_scoring.confidence_levels``.
        confidence_percentage: Mean ``EvaluationResult.confidence``
            across every evaluation in the interview, 0-100.
        competency_scores: Aggregated per-competency performance, one
            ``CompetencyScoreSummary`` per competency assessed during
            the interview.
        competency_summary: The same aggregate data as
            ``competency_scores``, pre-formatted as the exact text
            block passed to ``PromptService.build_report_prompt`` --
            retained here so the report's numeric claims can be
            audited against precisely what the model was given,
            without needing to recompute the formatting.
        strengths: Deduplicated, order-preserved strengths merged
            across every ``EvaluationResult`` in the interview.
        weaknesses: Deduplicated, order-preserved weaknesses merged
            across every ``EvaluationResult`` in the interview.
        recommendations: Actionable development recommendations,
            generated only from ``weaknesses`` -- never fabricated
            beyond what the evaluation data supports.
        report_markdown: The full rendered hiring report, as returned
            by ``GeminiService.generate_text()`` and validated by
            ``ReportService.validate_report``.
        generated_at: UTC ``datetime`` of when this report was
            constructed.
        request_id: Correlation ID assigned by ``ReportService`` at
            the start of the call, threaded through logs and any
            exception raised for this request.
        metadata: Small, known set of contextual fields about how this
            report was produced (e.g. ``session_id``,
            ``questions_evaluated``, ``competencies_assessed``) --
            never a dumping ground for raw session/evaluation data.
            Excluded from ``repr()`` to keep log lines short.
    """

    candidate_name: str
    overall_score: float
    overall_percentage: float
    decision: str
    confidence_level: str
    confidence_percentage: float
    competency_scores: tuple[CompetencyScoreSummary, ...]
    competency_summary: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    recommendations: tuple[str, ...]
    report_markdown: str
    generated_at: datetime
    request_id: str
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # Frozen dataclass: bypass __setattr__ to coerce any
        # list-typed inputs into tuples so instances stay immutable
        # even if a caller passed lists.
        object.__setattr__(self, "competency_scores", tuple(self.competency_scores))
        object.__setattr__(self, "strengths", tuple(self.strengths))
        object.__setattr__(self, "weaknesses", tuple(self.weaknesses))
        object.__setattr__(self, "recommendations", tuple(self.recommendations))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ReportResult("
            f"candidate_name={self.candidate_name!r}, "
            f"overall_score={self.overall_score!r}, "
            f"decision={self.decision!r}, "
            f"confidence_level={self.confidence_level!r}, "
            f"request_id={self.request_id!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the report result."""
        return {
            "candidate_name": self.candidate_name,
            "overall_score": self.overall_score,
            "overall_percentage": self.overall_percentage,
            "decision": self.decision,
            "confidence_level": self.confidence_level,
            "confidence_percentage": self.confidence_percentage,
            "competency_scores": [
                summary.to_dict() for summary in self.competency_scores
            ],
            "competency_summary": self.competency_summary,
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
            "recommendations": list(self.recommendations),
            "report_markdown": self.report_markdown,
            "generated_at": self.generated_at.isoformat(),
            "request_id": self.request_id,
        }