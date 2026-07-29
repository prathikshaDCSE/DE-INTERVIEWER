from unittest.mock import MagicMock

import pytest

from repository.question_repository import (
    QuestionBankError,
    QuestionNotFoundError,
)
from repository.user_repository import InvalidUserError, UserNotFoundError
from services.question_service import (
    CandidateValidationError,
    InterviewCompletedError,
    InterviewSessionError,
    QuestionSelectionError,
    QuestionService,
    QuestionServiceError,
)


def create_service(question_side_effect=None):
    question_repo = MagicMock()
    user_repo = MagicMock()
    audit_repo = MagicMock()

    user_repo.user_exists.return_value = True

    if question_side_effect is not None:
        question_repo.get_question.side_effect = question_side_effect
    else:
        question_repo.get_question.side_effect = [
            {"question_id": "Q1", "question": "Explain normalization."},
            {"question_id": "Q2", "question": "Explain indexing."},
            None,
        ]

    question_repo.get_followup_question.return_value = "What are normal forms?"
    question_repo.mark_question_used.return_value = None
    audit_repo.log_event.return_value = None

    service = QuestionService(
        question_repository=question_repo,
        user_repository=user_repo,
        audit_repository=audit_repo,
    )
    return service, question_repo, user_repo, audit_repo


def start_session(service, candidate_id="candidate-1"):
    return service.start_interview(
        candidate_id=candidate_id,
        competency="C2",
        stage="S1",
        candidate_level="Junior",
        experience=1,
    )


# ============================================================
# Interview lifecycle
# ============================================================


def test_start_interview_rejects_unknown_candidate():
    service, question_repo, user_repo, audit_repo = create_service()
    user_repo.user_exists.return_value = False

    with pytest.raises(CandidateValidationError):
        start_session(service)


def test_start_interview_translates_user_not_found_error():
    service, question_repo, user_repo, audit_repo = create_service()
    user_repo.user_exists.side_effect = UserNotFoundError("no such user")

    with pytest.raises(CandidateValidationError):
        start_session(service)


def test_start_interview_translates_invalid_user_error():
    service, question_repo, user_repo, audit_repo = create_service()
    user_repo.user_exists.side_effect = InvalidUserError("bad id")

    with pytest.raises(CandidateValidationError):
        start_session(service)


def test_start_interview_rejects_duplicate_active_session():
    service, *_ = create_service()
    start_session(service)

    with pytest.raises(InterviewSessionError):
        start_session(service)


def test_start_interview_allows_new_session_after_previous_ended():
    service, *_ = create_service()
    session = start_session(service)
    service.end_interview(session.session_id)

    new_session = start_session(service)
    assert new_session.session_id != session.session_id
    assert new_session.active is True


def test_resume_interview_returns_active_session():
    service, *_ = create_service()
    session = start_session(service)

    resumed = service.resume_interview(session.session_id)
    assert resumed.session_id == session.session_id


def test_resume_interview_unknown_session_raises():
    service, *_ = create_service()

    with pytest.raises(InterviewSessionError):
        service.resume_interview("does-not-exist")


def test_resume_interview_completed_session_raises():
    service, *_ = create_service()
    session = start_session(service)
    service.end_interview(session.session_id)

    with pytest.raises(InterviewCompletedError):
        service.resume_interview(session.session_id)


def test_end_interview_marks_inactive_and_sets_completed_at():
    service, *_ = create_service()
    session = start_session(service)

    service.end_interview(session.session_id)

    assert session.active is False
    assert session.completed_at is not None


def test_end_interview_already_completed_raises():
    service, *_ = create_service()
    session = start_session(service)
    service.end_interview(session.session_id)

    with pytest.raises(InterviewCompletedError):
        service.end_interview(session.session_id)


def test_cancel_interview_marks_inactive_without_completed_at():
    service, *_ = create_service()
    session = start_session(service)

    service.cancel_interview(session.session_id)

    assert session.active is False
    assert session.completed_at is None


def test_cancel_interview_unknown_session_raises():
    service, *_ = create_service()

    with pytest.raises(InterviewSessionError):
        service.cancel_interview("nope")


# ============================================================
# get_next_question / get_followup_question
# ============================================================


def test_get_next_question_on_inactive_session_raises():
    service, *_ = create_service()
    session = start_session(service)
    service.end_interview(session.session_id)

    with pytest.raises(InterviewCompletedError):
        service.get_next_question(session.session_id)


def test_get_next_question_no_match_raises_selection_error():
    service, question_repo, *_ = create_service(question_side_effect=[None])
    session = start_session(service)

    with pytest.raises(QuestionSelectionError):
        service.get_next_question(session.session_id)


def test_get_next_question_translates_question_bank_error():
    service, question_repo, *_ = create_service()
    question_repo.get_question.side_effect = QuestionBankError("bank broken")
    session = start_session(service)

    with pytest.raises(QuestionSelectionError):
        service.get_next_question(session.session_id)


def test_get_next_question_translates_mark_used_error():
    service, question_repo, *_ = create_service()
    question_repo.get_question.side_effect = [
        {"question_id": "Q1", "question": "Explain normalization."}
    ]
    question_repo.mark_question_used.side_effect = QuestionNotFoundError("missing")
    session = start_session(service)

    with pytest.raises(QuestionSelectionError):
        service.get_next_question(session.session_id)


def test_get_next_question_increments_total_questions_across_calls():
    service, *_ = create_service()
    session = start_session(service)

    service.get_next_question(session.session_id)
    service.get_next_question(session.session_id)

    assert session.total_questions == 2
    assert session.completed_questions == 2
    assert session.asked_questions == ["Q1", "Q2"]


def test_get_followup_question_no_current_question_raises():
    service, *_ = create_service()
    session = start_session(service)

    with pytest.raises(QuestionSelectionError):
        service.get_followup_question(session.session_id)


def test_get_followup_question_translates_repository_error():
    service, question_repo, *_ = create_service()
    session = start_session(service)
    service.get_next_question(session.session_id)
    question_repo.get_followup_question.side_effect = QuestionBankError("boom")

    with pytest.raises(QuestionSelectionError):
        service.get_followup_question(session.session_id)


def test_get_followup_question_unknown_session_raises():
    service, *_ = create_service()

    with pytest.raises(InterviewSessionError):
        service.get_followup_question("bad-session")


# ============================================================
# mark_question_used (service-level, direct)
# ============================================================


def test_mark_question_used_is_idempotent_for_same_question():
    service, *_ = create_service()
    session = start_session(service)

    service.mark_question_used(session=session, question_id="Q100")
    service.mark_question_used(session=session, question_id="Q100")

    assert session.asked_questions.count("Q100") == 1
    assert session.completed_questions == 1


# ============================================================
# Score validation and aggregation
# ============================================================


def test_add_score_rejects_non_integer():
    service, *_ = create_service()
    session = start_session(service)

    with pytest.raises(QuestionServiceError):
        service.add_score(session.session_id, 3.5)


def test_add_score_rejects_bool():
    service, *_ = create_service()
    session = start_session(service)

    with pytest.raises(QuestionServiceError):
        service.add_score(session.session_id, True)


@pytest.mark.parametrize("score", [-1, 6])
def test_add_score_rejects_out_of_range(score):
    service, *_ = create_service()
    session = start_session(service)

    with pytest.raises(QuestionServiceError):
        service.add_score(session.session_id, score)


def test_add_score_on_inactive_session_raises():
    service, *_ = create_service()
    session = start_session(service)
    service.end_interview(session.session_id)

    with pytest.raises(InterviewCompletedError):
        service.add_score(session.session_id, 3)


def test_score_aggregates_with_no_scores():
    service, *_ = create_service()
    session = start_session(service)

    assert service.get_total_score(session.session_id) == 0
    assert service.get_average_score(session.session_id) == 0.0
    assert service.get_highest_score(session.session_id) is None
    assert service.get_lowest_score(session.session_id) is None


def test_score_aggregates_with_multiple_scores():
    service, *_ = create_service()
    session = start_session(service)

    for score in (2, 4, 5):
        service.add_score(session.session_id, score)

    assert service.get_total_score(session.session_id) == 11
    assert service.get_average_score(session.session_id) == pytest.approx(11 / 3)
    assert service.get_highest_score(session.session_id) == 5
    assert service.get_lowest_score(session.session_id) == 2


def test_clear_scores_resets_history():
    service, *_ = create_service()
    session = start_session(service)
    service.add_score(session.session_id, 4)

    service.clear_scores(session.session_id)

    assert session.score_history == []
    assert service.get_average_score(session.session_id) == 0.0


# ============================================================
# Difficulty engine
# ============================================================


@pytest.mark.parametrize(
    "current_difficulty,score,expected",
    [
        ("Easy", 0, "Easy"),
        ("Medium", 2, "Easy"),
        ("Medium", 3, "Medium"),
        ("Medium", 4, "Hard"),
        ("Expert", 5, "Expert"),
        ("Easy", 5, "Medium"),
    ],
)
def test_calculate_next_difficulty(current_difficulty, score, expected):
    service, *_ = create_service()
    assert (
        service.calculate_next_difficulty(current_difficulty, score) == expected
    )


def test_calculate_next_difficulty_unknown_level_raises():
    service, *_ = create_service()

    with pytest.raises(QuestionServiceError):
        service.calculate_next_difficulty("Nightmare", 3)


def test_calculate_next_difficulty_invalid_score_raises():
    service, *_ = create_service()

    with pytest.raises(QuestionServiceError):
        service.calculate_next_difficulty("Easy", 10)


def test_promote_difficulty_caps_at_expert():
    service, *_ = create_service()
    session = start_session(service)
    session.current_difficulty = "Expert"

    result = service.promote_difficulty(session.session_id)
    assert result == "Expert"


def test_promote_difficulty_moves_up_one_level():
    service, *_ = create_service()
    session = start_session(service)

    result = service.promote_difficulty(session.session_id)
    assert result == "Medium"


def test_demote_difficulty_floors_at_easy():
    service, *_ = create_service()
    session = start_session(service)

    result = service.demote_difficulty(session.session_id)
    assert result == "Easy"


def test_demote_difficulty_moves_down_one_level():
    service, *_ = create_service()
    session = start_session(service)
    session.current_difficulty = "Hard"

    result = service.demote_difficulty(session.session_id)
    assert result == "Medium"


def test_update_difficulty_on_inactive_session_raises():
    service, *_ = create_service()
    session = start_session(service)
    service.end_interview(session.session_id)

    with pytest.raises(InterviewCompletedError):
        service.update_difficulty(session.session_id, 5)


def test_update_difficulty_applies_new_level():
    service, *_ = create_service()
    session = start_session(service)

    result = service.update_difficulty(session.session_id, 5)
    assert result == "Medium"
    assert session.current_difficulty == "Medium"


def test_record_score_adds_score_and_updates_difficulty():
    service, *_ = create_service()
    session = start_session(service)

    result_difficulty = service.record_score(session.session_id, 5)

    assert session.score_history == [5]
    assert result_difficulty == "Medium"
    assert session.current_difficulty == "Medium"


# ============================================================
# Progress tracking
# ============================================================


def test_progress_defaults_to_default_target():
    service, *_ = create_service()
    session = start_session(service)

    progress = service.get_progress(session.session_id)

    assert progress["total_questions"] == QuestionService.DEFAULT_TARGET_QUESTIONS
    assert progress["answered_questions"] == 0
    assert progress["remaining_questions"] == QuestionService.DEFAULT_TARGET_QUESTIONS
    assert progress["completion_percentage"] == 0.0
    assert progress["difficulty"] == "Easy"


def test_progress_respects_metadata_target_override():
    service, *_ = create_service()
    session = start_session(service)
    session.metadata["target_question_count"] = 4
    service.mark_question_used(session=session, question_id="Q1")
    service.mark_question_used(session=session, question_id="Q2")

    progress = service.get_progress(session.session_id)

    assert progress["total_questions"] == 4
    assert progress["answered_questions"] == 2
    assert progress["remaining_questions"] == 2
    assert progress["completion_percentage"] == 50.0


def test_completion_percentage_caps_at_100():
    service, *_ = create_service()
    session = start_session(service)
    session.metadata["target_question_count"] = 2
    for question_id in ("Q1", "Q2", "Q3"):
        service.mark_question_used(session=session, question_id=question_id)

    assert service.get_completion_percentage(session.session_id) == 100.0
    assert service.get_remaining_questions(session.session_id) == 0


def test_completion_percentage_zero_target_returns_zero():
    service, *_ = create_service()
    session = start_session(service)
    session.metadata["target_question_count"] = 0

    assert service.get_completion_percentage(session.session_id) == 0.0


# ============================================================
# Statistics and summary
# ============================================================


def test_get_statistics_reflects_session_state():
    service, *_ = create_service()
    session = start_session(service)
    service.add_score(session.session_id, 3)
    service.add_score(session.session_id, 5)
    service.get_next_question(session.session_id)

    stats = service.get_statistics(session.session_id)

    assert stats["questions_asked"] == 1
    assert stats["questions_completed"] == 1
    assert stats["average_score"] == 4.0
    assert stats["highest_score"] == 5
    assert stats["lowest_score"] == 3
    assert stats["active"] is True
    assert stats["duration_seconds"] >= 0


def test_generate_summary_contains_expected_fields():
    service, *_ = create_service()
    session = start_session(service)
    service.add_score(session.session_id, 4)

    summary = service.generate_summary(session.session_id)

    assert summary["candidate_id"] == "candidate-1"
    assert summary["session_id"] == session.session_id
    assert summary["average_score"] == 4.0
    assert summary["questions_answered"] == 0
    assert summary["completed_at"] is None
    assert summary["duration_seconds"] >= 0


# ============================================================
# Recommendation thresholds
# ============================================================


@pytest.mark.parametrize(
    "scores,expected",
    [
        ([5, 5, 4], QuestionService.RECOMMENDATION_STRONG_HIRE),
        ([4, 3], QuestionService.RECOMMENDATION_HIRE),
        ([3, 2], QuestionService.RECOMMENDATION_BORDERLINE),
        ([2, 1], QuestionService.RECOMMENDATION_TRAINING_REQUIRED),
        ([0, 1], QuestionService.RECOMMENDATION_REJECT),
        ([], QuestionService.RECOMMENDATION_REJECT),
    ],
)
def test_get_recommendation_thresholds(scores, expected):
    service, *_ = create_service()
    session = start_session(service)
    for score in scores:
        service.add_score(session.session_id, score)

    assert service.get_recommendation(session.session_id) == expected


def test_get_recommendation_unknown_session_raises():
    service, *_ = create_service()

    with pytest.raises(InterviewSessionError):
        service.get_recommendation("missing-session")