from __future__ import annotations

import logging
import random
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

COLUMN_QUESTION_ID = "Question ID"
COLUMN_COMPETENCY = "Competency"
COLUMN_STAGE = "Stage"
COLUMN_DIFFICULTY = "Difficulty"
COLUMN_BLOOM_LEVEL = "Bloom Level"
COLUMN_INTERVIEW_TYPE = "Interview Type"
COLUMN_CANDIDATE_LEVELS = "Candidate Levels"
COLUMN_MIN_EXPERIENCE = "Minimum Experience (Years)"
COLUMN_MAX_EXPERIENCE = "Maximum Experience (Years)"
COLUMN_TARGET_ROLES = "Target Roles"
COLUMN_QUESTION = "Question"
COLUMN_LEARNING_OBJECTIVE = "Learning Objective"
COLUMN_BUSINESS_CONTEXT = "Business Context"
COLUMN_EXPECTED_CONCEPTS = "Expected Concepts"
COLUMN_EVALUATOR_KEYWORDS = "Keywords"
COLUMN_EVALUATOR_NOTES = "Expected Answer Notes"
COLUMN_POSITIVE_INDICATORS = "Positive Indicators"
COLUMN_NEGATIVE_INDICATORS = "Negative Indicators"
COLUMN_COMMON_MISTAKES = "Common Mistakes"
COLUMN_RELATED_COMPETENCIES = "Related Competencies"
COLUMN_PREREQUISITE_CONCEPTS = "Prerequisite Concepts"
COLUMN_FOLLOWUP_1 = "Follow-up Question 1"
COLUMN_FOLLOWUP_2 = "Follow-up Question 2"
COLUMN_SCORE_0_GUIDANCE = "Score 0 Guidance"
COLUMN_SCORE_1_GUIDANCE = "Score 1 Guidance"
COLUMN_SCORE_2_GUIDANCE = "Score 2 Guidance"
COLUMN_SCORE_3_GUIDANCE = "Score 3 Guidance"
COLUMN_SCORE_4_GUIDANCE = "Score 4 Guidance"
COLUMN_SCORE_5_GUIDANCE = "Score 5 Guidance"
COLUMN_EVALUATION_WEIGHT = "Evaluation Weight (%)"
COLUMN_ESTIMATED_TIME = "Estimated Time (min)"
COLUMN_NEXT_DIFF_GE_4 = "Next Difficulty if Score ≥4"
COLUMN_NEXT_DIFF_LE_2 = "Next Difficulty if Score ≤2"
COLUMN_CONFIDENCE_WEIGHT = "Confidence Weight"
COLUMN_QUESTION_STATUS = "Question Status"
COLUMN_ACTIVE = "Active"
COLUMN_VERSION = "Version"
COLUMN_CREATED_BY = "Created By"
COLUMN_REVIEWED_BY = "Reviewed By"
COLUMN_REVIEW_DATE = "Review Date"
COLUMN_LAST_UPDATED = "Last Updated"

COLUMN_SOURCE_WORKBOOK = "source_workbook"
COLUMN_CANDIDATE_LEVELS_PARSED = "__parsed_candidate_levels"

REQUIRED_COLUMNS = [
    COLUMN_QUESTION_ID,
    COLUMN_COMPETENCY,
    COLUMN_STAGE,
    COLUMN_DIFFICULTY,
    COLUMN_BLOOM_LEVEL,
    COLUMN_INTERVIEW_TYPE,
    COLUMN_CANDIDATE_LEVELS,
    COLUMN_MIN_EXPERIENCE,
    COLUMN_MAX_EXPERIENCE,
    COLUMN_TARGET_ROLES,
    COLUMN_QUESTION,
    COLUMN_LEARNING_OBJECTIVE,
    COLUMN_BUSINESS_CONTEXT,
    COLUMN_EXPECTED_CONCEPTS,
    COLUMN_EVALUATOR_KEYWORDS,
    COLUMN_EVALUATOR_NOTES,
    COLUMN_POSITIVE_INDICATORS,
    COLUMN_NEGATIVE_INDICATORS,
    COLUMN_COMMON_MISTAKES,
    COLUMN_RELATED_COMPETENCIES,
    COLUMN_PREREQUISITE_CONCEPTS,
    COLUMN_FOLLOWUP_1,
    COLUMN_FOLLOWUP_2,
    COLUMN_SCORE_0_GUIDANCE,
    COLUMN_SCORE_1_GUIDANCE,
    COLUMN_SCORE_2_GUIDANCE,
    COLUMN_SCORE_3_GUIDANCE,
    COLUMN_SCORE_4_GUIDANCE,
    COLUMN_SCORE_5_GUIDANCE,
    COLUMN_EVALUATION_WEIGHT,
    COLUMN_ESTIMATED_TIME,
    COLUMN_NEXT_DIFF_GE_4,
    COLUMN_NEXT_DIFF_LE_2,
    COLUMN_CONFIDENCE_WEIGHT,
    COLUMN_QUESTION_STATUS,
    COLUMN_ACTIVE,
    COLUMN_VERSION,
    COLUMN_CREATED_BY,
    COLUMN_REVIEWED_BY,
    COLUMN_REVIEW_DATE,
    COLUMN_LAST_UPDATED,
]

NUMERIC_COLUMNS = [
    COLUMN_MIN_EXPERIENCE,
    COLUMN_MAX_EXPERIENCE,
    COLUMN_EVALUATION_WEIGHT,
    COLUMN_ESTIMATED_TIME,
    COLUMN_CONFIDENCE_WEIGHT,
]

CONFIG_COMPETENCIES = "competencies.yaml"
CONFIG_STAGES = "stages.yaml"
CONFIG_THRESHOLDS = "thresholds.yaml"

ACTIVE_TRUE_VALUES = {"true", "t", "yes", "y", "1"}
ACTIVE_FALSE_VALUES = {"false", "f", "no", "n", "0"}


class QuestionBankError(Exception):
    """Raised when the question repository encounters a general failure."""


class WorkbookValidationError(QuestionBankError):
    """Raised when a workbook fails validation."""


class QuestionNotFoundError(QuestionBankError):
    """Raised when a requested question is not found."""


@dataclass(frozen=True)
class QuestionRecord:
    """Structured representation of a single question bank item."""

    question_id: str
    competency: str
    stage: str
    difficulty: str
    bloom_level: str | None
    interview_type: str | None
    candidate_levels: str | None
    minimum_experience_years: float | None
    maximum_experience_years: float | None
    target_roles: str | None
    question: str | None
    learning_objective: str | None
    business_context: str | None
    expected_concepts: str | None
    evaluator_keywords: str | None
    evaluator_notes: str | None
    positive_indicators: str | None
    negative_indicators: str | None
    common_mistakes: str | None
    related_competencies: str | None
    prerequisite_concepts: str | None
    followup_question_1: str | None
    followup_question_2: str | None
    score_0_guidance: str | None
    score_1_guidance: str | None
    score_2_guidance: str | None
    score_3_guidance: str | None
    score_4_guidance: str | None
    score_5_guidance: str | None
    evaluation_weight: float | None
    estimated_time: float | None
    next_difficulty_if_score_ge_4: str | None
    next_difficulty_if_score_le_2: str | None
    confidence_weight: float | None
    question_status: str | None
    active: bool
    version: str | None
    created_by: str | None
    reviewed_by: str | None
    review_date: str | None
    last_updated: str | None
    source_workbook: str

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the question record."""
        return asdict(self)


class QuestionRepository:
    """Repository layer for loading and filtering the question bank."""

    def __init__(
        self,
        question_bank_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the repository.

        Args:
            question_bank_path: Optional path to the question_bank directory.
            config_path: Optional path to the config directory.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initializing QuestionRepository")

        project_root = Path(__file__).resolve().parent.parent
        self.question_bank_path = (
            Path(question_bank_path)
            if question_bank_path is not None
            else project_root / "question_bank"
        )
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else project_root / "config"
        )

        self.questions_df = pd.DataFrame()
        self.used_questions: set[str] = set()
        self._session_lock = threading.RLock()
        self._rng = random.Random()

        self.reference_data: dict[str, Any] = {}
        self.valid_competencies: set[str] = set()
        self.valid_stages: set[str] = set()
        self.valid_difficulties: set[str] = set()

        self.load_question_bank()

    def load_question_bank(self) -> None:
        """Discover, validate, and load all workbooks from the question bank."""
        self.logger.info("Loading question bank from %s", self.question_bank_path)
        self._load_reference_values()

        workbook_paths = self.discover_workbooks()
        if not workbook_paths:
            raise QuestionBankError(
                f"No workbook files found in question bank path: {self.question_bank_path}"
            )

        frames: list[pd.DataFrame] = []
        for workbook_path in workbook_paths:
            self.logger.debug("Loading workbook %s", workbook_path.name)
            validated_frame = self.validate_workbook(workbook_path)
            validated_frame[COLUMN_SOURCE_WORKBOOK] = workbook_path.name
            frames.append(validated_frame)

        self.questions_df = pd.concat(frames, ignore_index=True)
        self._validate_unique_question_ids()
        self._materialize_question_dataframe()
        self.logger.info(
            "Loaded %d questions from %d workbook(s)",
            len(self.questions_df),
            len(workbook_paths),
        )

    def discover_workbooks(self) -> list[Path]:
        """
        Discover all .xlsx files inside the question_bank directory.

        Returns:
            A sorted list of workbook paths.
        """
        self.logger.debug("Discovering workbooks in %s", self.question_bank_path)
        if not self.question_bank_path.exists() or not self.question_bank_path.is_dir():
            raise QuestionBankError(
                f"Question bank directory not found: {self.question_bank_path}"
            )

        workbook_paths = sorted(
            p
            for p in self.question_bank_path.iterdir()
            if p.is_file() and p.suffix.lower() == ".xlsx"
        )
        self.logger.debug("Discovered %d workbook(s)", len(workbook_paths))
        return workbook_paths

    def validate_workbook(self, workbook_path: Path | str) -> pd.DataFrame:
        """
        Validate a workbook and return a normalized DataFrame.

        Args:
            workbook_path: Workbook file path.

        Returns:
            Validated DataFrame containing required columns.
        """
        path = Path(workbook_path)
        self.logger.debug("Validating workbook %s", path)

        if not path.exists() or not path.is_file():
            raise WorkbookValidationError(f"Workbook not found: {path}")

        if path.suffix.lower() != ".xlsx":
            raise WorkbookValidationError(
                f"Workbook {path.name} must be an .xlsx file"
            )

        try:
            workbook_data = pd.read_excel(path, sheet_name=None)
        except Exception as exc:
            raise WorkbookValidationError(
                f"Failed to read workbook {path.name}: {exc}"
            ) from exc

        if not workbook_data:
            raise WorkbookValidationError(f"Workbook {path.name} contains no worksheets")

        if len(workbook_data) != 1:
            raise WorkbookValidationError(
                f"Workbook {path.name} must contain exactly one worksheet"
            )

        sheet_name = next(iter(workbook_data))
        dataframe = workbook_data[sheet_name].copy()

        missing_columns = set(REQUIRED_COLUMNS) - set(dataframe.columns)
        if missing_columns:
            raise WorkbookValidationError(
                f"Workbook {path.name} is missing required columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

        dataframe = dataframe.loc[:, REQUIRED_COLUMNS].copy()
        dataframe[COLUMN_QUESTION_ID] = dataframe[COLUMN_QUESTION_ID].astype(str).str.strip()
        if dataframe[COLUMN_QUESTION_ID].eq("").any():
            raise WorkbookValidationError(
                f"Workbook {path.name} contains blank Question ID values"
            )

        duplicate_rows = dataframe[
            dataframe[COLUMN_QUESTION_ID].duplicated(keep=False)
        ]
        if not duplicate_rows.empty:
            duplicate_ids = ", ".join(
                sorted(set(duplicate_rows[COLUMN_QUESTION_ID].tolist()))
            )
            raise WorkbookValidationError(
                f"Workbook {path.name} contains duplicate Question ID values: {duplicate_ids}"
            )

        self._normalize_numeric_columns(dataframe)
        self._normalize_active_column(dataframe)
        self._normalize_candidate_levels(dataframe)
        self._validate_column_values(dataframe, path.name)

        return dataframe

    def get_question(
        self,
        competency: str,
        stage: str,
        difficulty: str,
        candidate_level: str,
        experience: int | float,
        exclude_used: bool = True,
    ) -> dict[str, Any] | None:
        """
        Select a random active question that matches all filters.

        Args:
            competency: Competency identifier.
            stage: Stage identifier.
            difficulty: Difficulty label.
            candidate_level: Candidate level string.
            experience: Candidate experience in years.
            exclude_used: If True, exclude questions already marked used.

        Returns:
            A question dictionary or None when no match is found.
        """
        self._ensure_loaded()
        self._validate_query_values(
            competency=competency,
            stage=stage,
            difficulty=difficulty,
            candidate_level=candidate_level,
            experience=experience,
        )

        filtered = self._filter_questions(
            competency=competency,
            stage=stage,
            difficulty=difficulty,
            candidate_level=candidate_level,
            experience=experience,
            exclude_used=exclude_used,
            active_only=True,
        )

        if filtered.empty:
            self.logger.info(
                "No matching question found for competency=%s stage=%s difficulty=%s candidate_level=%s experience=%s",
                competency,
                stage,
                difficulty,
                candidate_level,
                experience,
            )
            return None

        selected_index = self._rng.choice(filtered.index.to_list())
        selected_row = filtered.loc[selected_index]

        self.logger.debug(
            "Selected question %s from %d eligible candidates",
            selected_row[COLUMN_QUESTION_ID],
            len(filtered),
        )
        return self._row_to_dict(selected_row)

    def get_followup_question(
        self, question_id: str, followup_number: int
    ) -> str | None:
        """
        Return the follow-up question for the specified question.

        Args:
            question_id: Question identifier.
            followup_number: 1 or 2.

        Returns:
            The follow-up question string, or None.
        """
        self._ensure_loaded()
        if followup_number not in {1, 2}:
            raise QuestionBankError("followup_number must be 1 or 2")

        question_id_clean = str(question_id).strip()
        matches = self.questions_df[
            self.questions_df[COLUMN_QUESTION_ID] == question_id_clean
        ]
        if matches.empty:
            raise QuestionNotFoundError(f"Question ID not found: {question_id}")

        column_name = COLUMN_FOLLOWUP_1 if followup_number == 1 else COLUMN_FOLLOWUP_2
        value = matches.iloc[0].get(column_name)
        return None if pd.isna(value) else str(value).strip()

    def get_questions_by_stage(self, stage: str) -> list[dict[str, Any]]:
        """
        Return all active questions for a stage.
        """
        self._ensure_loaded()
        if stage not in self.valid_stages:
            raise QuestionBankError(f"Invalid stage value: {stage}")

        filtered = self._filter_questions(stage=stage, active_only=True, exclude_used=False)
        return self._rows_to_list(filtered)

    def get_questions_by_competency(self, competency: str) -> list[dict[str, Any]]:
        """
        Return all active questions for a competency.
        """
        self._ensure_loaded()
        if competency not in self.valid_competencies:
            raise QuestionBankError(f"Invalid competency value: {competency}")

        filtered = self._filter_questions(competency=competency, active_only=True, exclude_used=False)
        return self._rows_to_list(filtered)

    def get_questions_by_difficulty(self, difficulty: str) -> list[dict[str, Any]]:
        """
        Return all active questions for a difficulty.
        """
        self._ensure_loaded()
        if difficulty not in self.valid_difficulties:
            raise QuestionBankError(f"Invalid difficulty value: {difficulty}")

        filtered = self._filter_questions(difficulty=difficulty, active_only=True, exclude_used=False)
        return self._rows_to_list(filtered)

    def get_questions_by_candidate_level(
        self, candidate_level: str
    ) -> list[dict[str, Any]]:
        """
        Return all active questions matching a candidate level.
        """
        self._ensure_loaded()
        if not candidate_level or not str(candidate_level).strip():
            raise QuestionBankError("candidate_level must be a non-empty string")

        filtered = self._filter_questions(
            candidate_level=candidate_level,
            active_only=True,
            exclude_used=False,
        )
        return self._rows_to_list(filtered)

    def get_questions_by_experience(
        self, experience: int | float
    ) -> list[dict[str, Any]]:
        """
        Return all active questions matching candidate experience.
        """
        self._ensure_loaded()
        if experience is None or not isinstance(experience, (int, float)):
            raise QuestionBankError("experience must be a numeric value")

        filtered = self._filter_questions(
            experience=experience,
            active_only=True,
            exclude_used=False,
        )
        return self._rows_to_list(filtered)

    def mark_question_used(self, question_id: str) -> None:
        """
        Mark a question as used within the current session.
        """
        self._ensure_loaded()
        question_id_clean = str(question_id).strip()

        if not self.questions_df[COLUMN_QUESTION_ID].eq(question_id_clean).any():
            raise QuestionNotFoundError(f"Question ID not found: {question_id}")

        with self._session_lock:
            self.used_questions.add(question_id_clean)
        self.logger.debug("Marked question %s as used", question_id_clean)

    def reset_session(self) -> None:
        """Reset the in-memory used question set for the current repository."""
        with self._session_lock:
            self.used_questions.clear()
        self.logger.info("Session reset; used questions cleared")

    def get_total_questions(self) -> int:
        """
        Return the total number of loaded questions.
        """
        self._ensure_loaded()
        return int(self.questions_df.shape[0])

    def get_statistics(self) -> dict[str, Any]:
        """
        Return question bank statistics.
        """
        self._ensure_loaded()
        return {
            "total_questions": self.get_total_questions(),
            "questions_per_competency": self._count_by(COLUMN_COMPETENCY),
            "questions_per_difficulty": self._count_by(COLUMN_DIFFICULTY),
            "questions_per_stage": self._count_by(COLUMN_STAGE),
            "questions_per_workbook": self._count_by(COLUMN_SOURCE_WORKBOOK),
        }

    def _ensure_loaded(self) -> None:
        if self.questions_df.empty:
            raise QuestionBankError("Question bank has not been loaded")

    def _rows_to_list(self, dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        return [self._row_to_dict(row) for _, row in dataframe.iterrows()]

    def _row_to_dict(self, row: pd.Series) -> dict[str, Any]:
        record = QuestionRecord(
            question_id=str(row[COLUMN_QUESTION_ID]),
            competency=str(row[COLUMN_COMPETENCY]),
            stage=str(row[COLUMN_STAGE]),
            difficulty=str(row[COLUMN_DIFFICULTY]),
            bloom_level=self._safe_str(row.get(COLUMN_BLOOM_LEVEL)),
            interview_type=self._safe_str(row.get(COLUMN_INTERVIEW_TYPE)),
            candidate_levels=self._safe_str(row.get(COLUMN_CANDIDATE_LEVELS)),
            minimum_experience_years=self._safe_float(row.get(COLUMN_MIN_EXPERIENCE)),
            maximum_experience_years=self._safe_float(row.get(COLUMN_MAX_EXPERIENCE)),
            target_roles=self._safe_str(row.get(COLUMN_TARGET_ROLES)),
            question=self._safe_str(row.get(COLUMN_QUESTION)),
            learning_objective=self._safe_str(row.get(COLUMN_LEARNING_OBJECTIVE)),
            business_context=self._safe_str(row.get(COLUMN_BUSINESS_CONTEXT)),
            expected_concepts=self._safe_str(row.get(COLUMN_EXPECTED_CONCEPTS)),
            evaluator_keywords=self._safe_str(row.get(COLUMN_EVALUATOR_KEYWORDS)),
            evaluator_notes=self._safe_str(row.get(COLUMN_EVALUATOR_NOTES)),
            positive_indicators=self._safe_str(row.get(COLUMN_POSITIVE_INDICATORS)),
            negative_indicators=self._safe_str(row.get(COLUMN_NEGATIVE_INDICATORS)),
            common_mistakes=self._safe_str(row.get(COLUMN_COMMON_MISTAKES)),
            related_competencies=self._safe_str(row.get(COLUMN_RELATED_COMPETENCIES)),
            prerequisite_concepts=self._safe_str(row.get(COLUMN_PREREQUISITE_CONCEPTS)),
            followup_question_1=self._safe_str(row.get(COLUMN_FOLLOWUP_1)),
            followup_question_2=self._safe_str(row.get(COLUMN_FOLLOWUP_2)),
            score_0_guidance=self._safe_str(row.get(COLUMN_SCORE_0_GUIDANCE)),
            score_1_guidance=self._safe_str(row.get(COLUMN_SCORE_1_GUIDANCE)),
            score_2_guidance=self._safe_str(row.get(COLUMN_SCORE_2_GUIDANCE)),
            score_3_guidance=self._safe_str(row.get(COLUMN_SCORE_3_GUIDANCE)),
            score_4_guidance=self._safe_str(row.get(COLUMN_SCORE_4_GUIDANCE)),
            score_5_guidance=self._safe_str(row.get(COLUMN_SCORE_5_GUIDANCE)),
            evaluation_weight=self._safe_float(row.get(COLUMN_EVALUATION_WEIGHT)),
            estimated_time=self._safe_float(row.get(COLUMN_ESTIMATED_TIME)),
            next_difficulty_if_score_ge_4=self._safe_str(row.get(COLUMN_NEXT_DIFF_GE_4)),
            next_difficulty_if_score_le_2=self._safe_str(row.get(COLUMN_NEXT_DIFF_LE_2)),
            confidence_weight=self._safe_float(row.get(COLUMN_CONFIDENCE_WEIGHT)),
            question_status=self._safe_str(row.get(COLUMN_QUESTION_STATUS)),
            active=bool(row[COLUMN_ACTIVE]),
            version=self._safe_str(row.get(COLUMN_VERSION)),
            created_by=self._safe_str(row.get(COLUMN_CREATED_BY)),
            reviewed_by=self._safe_str(row.get(COLUMN_REVIEWED_BY)),
            review_date=self._safe_str(row.get(COLUMN_REVIEW_DATE)),
            last_updated=self._safe_str(row.get(COLUMN_LAST_UPDATED)),
            source_workbook=str(row[COLUMN_SOURCE_WORKBOOK]),
        )
        return record.to_dict()

    def _filter_questions(
        self,
        competency: str | None = None,
        stage: str | None = None,
        difficulty: str | None = None,
        candidate_level: str | None = None,
        experience: int | float | None = None,
        exclude_used: bool = True,
        active_only: bool = True,
    ) -> pd.DataFrame:
        mask = pd.Series(True, index=self.questions_df.index)

        if active_only:
            mask &= self.questions_df[COLUMN_ACTIVE] == True  # noqa: E712

        if competency is not None:
            mask &= self.questions_df[COLUMN_COMPETENCY] == competency

        if stage is not None:
            mask &= self.questions_df[COLUMN_STAGE] == stage

        if difficulty is not None:
            mask &= self.questions_df[COLUMN_DIFFICULTY] == difficulty

        if candidate_level is not None:
            normalized_level = self._normalize_search_term(candidate_level)
            mask &= self.questions_df[COLUMN_CANDIDATE_LEVELS_PARSED].map(
                lambda parsed: normalized_level in parsed
            )

        if experience is not None:
            mask &= (
                self.questions_df[COLUMN_MIN_EXPERIENCE].le(experience)
                & self.questions_df[COLUMN_MAX_EXPERIENCE].ge(experience)
            )

        if exclude_used:
            with self._session_lock:
                if self.used_questions:
                    mask &= ~self.questions_df[COLUMN_QUESTION_ID].isin(
                        self.used_questions
                    )

        filtered = self.questions_df.loc[mask]
        self.logger.debug(
            "Filtered questions: %d rows; competency=%s stage=%s difficulty=%s candidate_level=%s experience=%s exclude_used=%s active_only=%s",
            len(filtered),
            competency,
            stage,
            difficulty,
            candidate_level,
            experience,
            exclude_used,
            active_only,
        )
        return filtered

    def _validate_unique_question_ids(self) -> None:
        duplicates = self.questions_df[
            self.questions_df[COLUMN_QUESTION_ID].duplicated(keep=False)
        ][COLUMN_QUESTION_ID]
        if not duplicates.empty:
            duplicate_values = ", ".join(sorted(set(duplicates.tolist())))
            raise QuestionBankError(
                f"Duplicate Question ID values across workbooks: {duplicate_values}"
            )

    def _validate_column_values(self, dataframe: pd.DataFrame, workbook_name: str) -> None:
        errors: list[str] = []

        competency_values = {
            str(value).strip()
            for value in dataframe[COLUMN_COMPETENCY].dropna().unique()
        }
        invalid_competencies = sorted(competency_values - self.valid_competencies)
        if invalid_competencies:
            errors.append(
                f"Invalid Competency values in {workbook_name}: "
                f"{', '.join(invalid_competencies)}"
            )

        stage_values = {
            str(value).strip() for value in dataframe[COLUMN_STAGE].dropna().unique()
        }
        invalid_stages = sorted(stage_values - self.valid_stages)
        if invalid_stages:
            errors.append(
                f"Invalid Stage values in {workbook_name}: "
                f"{', '.join(invalid_stages)}"
            )

        difficulty_values = {
            str(value).strip()
            for value in dataframe[COLUMN_DIFFICULTY].dropna().unique()
        }
        invalid_difficulties = sorted(difficulty_values - self.valid_difficulties)
        if invalid_difficulties:
            errors.append(
                f"Invalid Difficulty values in {workbook_name}: "
                f"{', '.join(invalid_difficulties)}"
            )

        if errors:
            self.logger.error(
                "Workbook validation failed for %s: %s",
                workbook_name,
                " | ".join(errors),
            )
            raise WorkbookValidationError(" | ".join(errors))

    def _normalize_numeric_columns(self, dataframe: pd.DataFrame) -> None:
        for column in NUMERIC_COLUMNS:
            if column in dataframe.columns:
                dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    def _normalize_active_column(self, dataframe: pd.DataFrame) -> None:
        def parse_active(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if pd.isna(value):
                raise ValueError("Active value is missing")
            normalized = str(value).strip().lower()
            if normalized in ACTIVE_TRUE_VALUES:
                return True
            if normalized in ACTIVE_FALSE_VALUES:
                return False
            raise ValueError(f"Invalid Active value: {value}")

        normalized_values: list[bool] = []
        errors: list[str] = []
        for index, raw_value in dataframe[COLUMN_ACTIVE].items():
            try:
                normalized_values.append(parse_active(raw_value))
            except ValueError as exc:
                errors.append(f"Row {index + 2}: {exc}")
                normalized_values.append(False)

        if errors:
            raise WorkbookValidationError(
                "Active column validation failed: " + "; ".join(errors)
            )

        dataframe[COLUMN_ACTIVE] = normalized_values

    def _normalize_candidate_levels(self, dataframe: pd.DataFrame) -> None:
        dataframe[COLUMN_CANDIDATE_LEVELS_PARSED] = dataframe[
            COLUMN_CANDIDATE_LEVELS
        ].map(self._parse_candidate_levels_value)

    def _parse_candidate_levels_value(self, value: Any) -> frozenset[str]:
        if pd.isna(value):
            return frozenset()

        normalized_value = str(value).strip()
        if not normalized_value:
            return frozenset()

        parts = [
            part.strip()
            for part in normalized_value.split(",")
            if part.strip()
        ]
        return frozenset(parts)

    def _validate_query_values(
        self,
        competency: str,
        stage: str,
        difficulty: str,
        candidate_level: str,
        experience: int | float,
    ) -> None:
        if competency not in self.valid_competencies:
            raise QuestionBankError(f"Invalid competency value: {competency}")

        if stage not in self.valid_stages:
            raise QuestionBankError(f"Invalid stage value: {stage}")

        if difficulty not in self.valid_difficulties:
            raise QuestionBankError(f"Invalid difficulty value: {difficulty}")

        if not candidate_level or not str(candidate_level).strip():
            raise QuestionBankError("candidate_level must be a non-empty string")

        if experience is None or not isinstance(experience, (int, float)):
            raise QuestionBankError("experience must be a numeric value")

    def _load_reference_values(self) -> None:
        self.reference_data["competencies"] = self._load_yaml_file(
            self.config_path / CONFIG_COMPETENCIES
        )
        self.reference_data["stages"] = self._load_yaml_file(
            self.config_path / CONFIG_STAGES
        )
        self.reference_data["thresholds"] = self._load_yaml_file(
            self.config_path / CONFIG_THRESHOLDS
        )

        self.valid_competencies = {
            item["id"].strip()
            for item in self.reference_data["competencies"].get("competencies", [])
            if isinstance(item, dict) and item.get("id")
        }
        self.valid_stages = {
            item["id"].strip()
            for item in self.reference_data["stages"].get("stages", [])
            if isinstance(item, dict) and item.get("id")
        }
        self.valid_difficulties = set()
        for item in self.reference_data["competencies"].get("competencies", []):
            if isinstance(item, dict):
                for difficulty in item.get("difficulty_levels", []) or []:
                    if isinstance(difficulty, str):
                        self.valid_difficulties.add(difficulty.strip())

        if not self.valid_competencies:
            raise QuestionBankError("No competency reference values loaded")
        if not self.valid_stages:
            raise QuestionBankError("No stage reference values loaded")
        if not self.valid_difficulties:
            raise QuestionBankError("No difficulty reference values loaded")

    def _load_yaml_file(self, path: Path) -> dict[str, Any]:
        if yaml is None:
            raise QuestionBankError(
                "PyYAML is required to load config files; install pyyaml"
            )

        if not path.exists() or not path.is_file():
            raise QuestionBankError(f"Config file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as handle:
                content = yaml.safe_load(handle)
        except Exception as exc:
            raise QuestionBankError(
                f"Failed to load YAML config file {path.name}: {exc}"
            ) from exc

        if not isinstance(content, dict):
            raise QuestionBankError(
                f"Config file {path.name} must contain a YAML mapping at the root"
            )
        return content

    def _materialize_question_dataframe(self) -> None:
        self.questions_df = self.questions_df.assign(
            **{
                COLUMN_COMPETENCY: self.questions_df[COLUMN_COMPETENCY].astype(
                    "category"
                ),
                COLUMN_STAGE: self.questions_df[COLUMN_STAGE].astype("category"),
                COLUMN_DIFFICULTY: self.questions_df[COLUMN_DIFFICULTY].astype(
                    "category"
                ),
                COLUMN_ACTIVE: self.questions_df[COLUMN_ACTIVE].astype(bool),
                COLUMN_MIN_EXPERIENCE: pd.to_numeric(
                    self.questions_df[COLUMN_MIN_EXPERIENCE], errors="coerce"
                ),
                COLUMN_MAX_EXPERIENCE: pd.to_numeric(
                    self.questions_df[COLUMN_MAX_EXPERIENCE], errors="coerce"
                ),
            }
        )

    def _count_by(self, column_name: str) -> dict[str, int]:
        counts = self.questions_df[column_name].fillna("UNKNOWN").value_counts()
        return {str(label): int(count) for label, count in counts.items()}

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_search_term(value: str) -> str:
        return str(value).strip()