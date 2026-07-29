from repository.question_repository import QuestionRepository
import pytest

def test_repository_loads():
    repo = QuestionRepository()
    assert repo is not None

def test_get_total_questions():
    repo = QuestionRepository()

    assert repo.get_total_questions() > 0

def test_get_statistics():
    repo = QuestionRepository()

    stats = repo.get_statistics()

    assert isinstance(stats, dict)
    assert stats["total_questions"] > 0

def test_get_questions_by_competency():
    repo = QuestionRepository()

    questions = repo.get_questions_by_competency("C2")

    assert len(questions) > 0

def test_get_questions_by_stage():
    repo = QuestionRepository()

    questions = repo.get_questions_by_stage("S1")

    assert len(questions) > 0

def test_get_questions_by_difficulty():
    repo = QuestionRepository()

    questions = repo.get_questions_by_difficulty("Easy")

    assert len(questions) > 0

def test_get_question():
    repo = QuestionRepository()

    question = repo.get_question(
        competency="C2",
        stage="S1",
        difficulty="Easy",
        candidate_level="Fresher",
        experience=0,
    )

    assert question is not None

def test_mark_question_used():
    repo = QuestionRepository()

    question = repo.get_question(
        competency="C2",
        stage="S1",
        difficulty="Easy",
        candidate_level="Fresher",
        experience=0,
    )

    repo.mark_question_used(question["question_id"])

    assert question["question_id"] in repo.used_questions

def test_reset_session():
    repo = QuestionRepository()

    question = repo.get_question(
        competency="C2",
        stage="S1",
        difficulty="Easy",
        candidate_level="Fresher",
        experience=0,
    )

    repo.mark_question_used(question["question_id"])

    repo.reset_session()

    assert len(repo.used_questions) == 0

def test_reset_session():
    repo = QuestionRepository()

    question = repo.get_question(
        competency="C2",
        stage="S1",
        difficulty="Easy",
        candidate_level="Fresher",
        experience=0,
    )

    repo.mark_question_used(question["question_id"])

    repo.reset_session()

    assert len(repo.used_questions) == 0

def test_get_followup_question():
    repo = QuestionRepository()

    question = repo.get_question(
        competency="C2",
        stage="S1",
        difficulty="Easy",
        candidate_level="Fresher",
        experience=0,
    )

    followup = repo.get_followup_question(
        question["question_id"],
        1,
    )

    assert followup is None or isinstance(followup, str)

def test_get_questions_by_candidate_level():
    repo = QuestionRepository()

    questions = repo.get_questions_by_candidate_level("Fresher")

    assert isinstance(questions, list)
    assert len(questions) > 0
def test_get_questions_by_experience():
    repo = QuestionRepository()

    questions = repo.get_questions_by_experience(0)

    assert isinstance(questions, list)
def test_invalid_competency():
    repo = QuestionRepository()

    with pytest.raises(Exception):
        repo.get_questions_by_competency("INVALID")

def test_invalid_stage():
    repo = QuestionRepository()

    with pytest.raises(Exception):
        repo.get_questions_by_stage("INVALID")

def test_invalid_difficulty():
    repo = QuestionRepository()

    with pytest.raises(Exception):
        repo.get_questions_by_difficulty("INVALID")

def test_invalid_candidate_level():
    repo = QuestionRepository()

    with pytest.raises(Exception):
        repo.get_questions_by_candidate_level("")
