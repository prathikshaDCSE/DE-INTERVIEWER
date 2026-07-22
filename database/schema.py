"""
database/schema.py
===================
Idempotent BigQuery schema initializer for the DE-Interviewer platform.

Creates the `de_interviewer` dataset (if missing) and all 14 tables with
their partitioning and clustering configuration, matching
`de_interviewer_schema.sql` exactly. Safe to run repeatedly: dataset and
table creation both use exists_ok=True, so re-running this script on an
already-initialized project is a no-op.

Usage:
    python -m database.schema

Configuration is sourced from config/settings.py (GCP_PROJECT_ID,
BIGQUERY_DATASET, SERVICE_ACCOUNT_FILE), which itself reads the
underlying .env / Streamlit Secrets values. BIGQUERY_LOCATION is read
directly via os.getenv, defaulting to "asia-south1".

Auth: uses an explicit Service Account JSON key file (not ADC), loaded via
google.oauth2.service_account. Locally, download a key for a service
account with BigQuery Data Editor + Job User roles and point
SERVICE_ACCOUNT_FILE (in config/settings.py or its .env source) at it.
On Streamlit Community Cloud, write the JSON contents to that path from
Streamlit Secrets at app startup (see deployment docs). Never commit the
key file — keep secrets/ in .gitignore.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from google.api_core.exceptions import GoogleAPICallError
from google.cloud import bigquery
from google.cloud.bigquery import SchemaField as F
from google.oauth2 import service_account

from config.settings import settings

logger = logging.getLogger("de_interviewer.schema")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

PROJECT_ID = settings.GCP_PROJECT_ID
DATASET_ID = settings.BIGQUERY_DATASET
LOCATION = os.getenv("BIGQUERY_LOCATION", "asia-south1")
SERVICE_ACCOUNT_FILE = settings.SERVICE_ACCOUNT_FILE


@dataclass
class TableSpec:
    name: str
    schema: list
    description: str
    partition_field: str | None = None
    clustering_fields: list | None = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Authentication & RBAC
# ---------------------------------------------------------------------------

USERS = TableSpec(
    name="users",
    description="Every authenticated user (candidate, reviewer, admin)",
    clustering_fields=["user_id"],
    schema=[
        F("user_id", "STRING", "REQUIRED", description="UUID, primary key"),
        F("email", "STRING", "REQUIRED", description="Google OAuth email, unique"),
        F("full_name", "STRING", "NULLABLE"),
        F("picture_url", "STRING", "NULLABLE"),
        F("created_at", "TIMESTAMP", "REQUIRED"),
        F("last_login", "TIMESTAMP", "NULLABLE"),
        F("status", "STRING", "REQUIRED", description="ACTIVE | DISABLED"),
    ],
)

USER_ROLES = TableSpec(
    name="user_roles",
    description="Role assignment history; current role = latest active=TRUE row",
    clustering_fields=["user_id", "role"],
    schema=[
        F("role_id", "STRING", "REQUIRED"),
        F("user_id", "STRING", "REQUIRED"),
        F("role", "STRING", "REQUIRED", description="CANDIDATE | REVIEWER | ADMIN"),
        F("assigned_by", "STRING", "NULLABLE", description="user_id or 'SYSTEM'"),
        F("assigned_at", "TIMESTAMP", "REQUIRED"),
        F("active", "BOOLEAN", "REQUIRED"),
    ],
)

USER_SESSIONS = TableSpec(
    name="user_sessions",
    description="Login/logout audit trail",
    partition_field="login_time",
    clustering_fields=["user_id"],
    schema=[
        F("session_id", "STRING", "REQUIRED"),
        F("user_id", "STRING", "REQUIRED"),
        F("login_time", "TIMESTAMP", "REQUIRED"),
        F("logout_time", "TIMESTAMP", "NULLABLE"),
        F("ip_address", "STRING", "NULLABLE"),
        F("browser", "STRING", "NULLABLE"),
        F("device", "STRING", "NULLABLE"),
    ],
)

REVIEWER_REQUESTS = TableSpec(
    name="reviewer_requests",
    description="Candidate -> Reviewer access requests awaiting Admin approval",
    partition_field="requested_at",
    clustering_fields=["status", "user_id"],
    schema=[
        F("request_id", "STRING", "REQUIRED"),
        F("user_id", "STRING", "REQUIRED"),
        F("full_name", "STRING", "REQUIRED"),
        F("email", "STRING", "REQUIRED"),
        F("organization", "STRING", "NULLABLE"),
        F("designation", "STRING", "NULLABLE"),
        F("department", "STRING", "NULLABLE"),
        F("linkedin_url", "STRING", "NULLABLE"),
        F("requested_role", "STRING", "REQUIRED"),
        F("business_justification", "STRING", "REQUIRED"),
        F("status", "STRING", "REQUIRED", description="PENDING | APPROVED | REJECTED"),
        F("approved_by", "STRING", "NULLABLE"),
        F("approved_at", "TIMESTAMP", "NULLABLE"),
        F("comments", "STRING", "NULLABLE"),
        F("requested_at", "TIMESTAMP", "REQUIRED"),
    ],
)

# ---------------------------------------------------------------------------
# 2. Interview lifecycle
# ---------------------------------------------------------------------------

CANDIDATE_SESSIONS = TableSpec(
    name="candidate_sessions",
    description="One row per interview attempt; intake + final scoring/decision",
    partition_field="started_at",
    clustering_fields=["user_id", "decision"],
    schema=[
        F("session_id", "STRING", "REQUIRED"),
        F("user_id", "STRING", "REQUIRED"),
        F("experience_years", "FLOAT64", "NULLABLE"),
        F("target_role", "STRING", "NULLABLE"),
        F("skills", "STRING", "REPEATED"),
        F("resume_summary", "STRING", "NULLABLE"),
        F("started_at", "TIMESTAMP", "REQUIRED"),
        F("completed_at", "TIMESTAMP", "NULLABLE"),
        F("status", "STRING", "REQUIRED", description="IN_PROGRESS | COMPLETED | ABANDONED"),
        F("technical_score", "FLOAT64", "NULLABLE"),
        F("soft_skill_score", "FLOAT64", "NULLABLE"),
        F("overall_score", "FLOAT64", "NULLABLE"),
        F("decision", "STRING", "NULLABLE", description="SELECT | HOLD | REJECT"),
        F("confidence_level", "STRING", "NULLABLE"),
        F("confidence_percentage", "FLOAT64", "NULLABLE"),
        F("competency_coverage_pct", "FLOAT64", "NULLABLE"),
        F("prompt_version", "STRING", "NULLABLE"),
        F("model_version", "STRING", "NULLABLE"),
        F("interview_mode", "STRING", "NULLABLE", description="STANDARD | PRACTICE | CALIBRATION"),
        F("langgraph_session_id", "STRING", "NULLABLE"),
        F("interviewer_agent_version", "STRING", "NULLABLE"),
        F("evaluator_agent_version", "STRING", "NULLABLE"),
    ],
)

INTERVIEW_QUESTIONS = TableSpec(
    name="interview_questions",
    description="Every question/follow-up instance actually asked in a session",
    partition_field="asked_at",
    clustering_fields=["session_id", "competency"],
    schema=[
        F("question_instance_id", "STRING", "REQUIRED"),
        F("session_id", "STRING", "REQUIRED"),
        F("question_id", "STRING", "REQUIRED"),
        F("competency", "STRING", "REQUIRED"),
        F("difficulty", "STRING", "NULLABLE"),
        F("stage", "STRING", "REQUIRED", description="S0 | S1 | S2 | S3 | S4"),
        F("question_text", "STRING", "REQUIRED"),
        F("followup_level", "INT64", "REQUIRED"),
        F("parent_question_instance_id", "STRING", "NULLABLE"),
        F("asked_at", "TIMESTAMP", "REQUIRED"),
    ],
)

INTERVIEW_RESPONSES = TableSpec(
    name="interview_responses",
    description="Raw candidate answers, 1:1 with interview_questions",
    partition_field="submitted_at",
    clustering_fields=["question_instance_id"],
    schema=[
        F("response_id", "STRING", "REQUIRED"),
        F("question_instance_id", "STRING", "REQUIRED"),
        F("candidate_answer", "STRING", "NULLABLE"),
        F("answer_summary", "STRING", "NULLABLE"),
        F("response_type", "STRING", "NULLABLE",
          description="TEXT | EMPTY | I_DONT_KNOW | OFF_TOPIC | SPAM"),
        F("response_time_seconds", "INT64", "NULLABLE"),
        F("submitted_at", "TIMESTAMP", "REQUIRED"),
    ],
)

# ---------------------------------------------------------------------------
# 3. AI evaluation
# ---------------------------------------------------------------------------

ANSWER_EVALUATIONS = TableSpec(
    name="answer_evaluations",
    description="Evaluator Agent structured JSON output, 1:1 with interview_responses",
    partition_field="evaluation_time",
    clustering_fields=["competency", "score"],
    schema=[
        F("evaluation_id", "STRING", "REQUIRED"),
        F("response_id", "STRING", "REQUIRED"),
        F("score", "INT64", "REQUIRED", description="0-5 scale"),
        F("competency", "STRING", "REQUIRED"),
        F("feedback", "STRING", "REQUIRED"),
        F("reasoning", "STRING", "REQUIRED"),
        F("followup_required", "BOOLEAN", "REQUIRED"),
        F("expected_concepts", "STRING", "REPEATED"),
        F("detected_concepts", "STRING", "REPEATED"),
        F("strengths", "STRING", "REPEATED"),
        F("weaknesses", "STRING", "REPEATED"),
        F("evidence", "STRING", "NULLABLE",
          description="Excerpt from the answer supporting the score"),
        F("confidence", "FLOAT64", "NULLABLE", description="Evaluator's confidence, 0-1"),
        F("temperature", "FLOAT64", "REQUIRED", description="Always 0"),
        F("model", "STRING", "REQUIRED"),
        F("prompt_version", "STRING", "REQUIRED"),
        F("evaluation_time", "TIMESTAMP", "REQUIRED"),
    ],
)

COMPETENCY_SCORES = TableSpec(
    name="competency_scores",
    description="Aggregated per-competency rollup per session",
    clustering_fields=["session_id", "competency"],
    schema=[
        F("competency_score_id", "STRING", "REQUIRED"),
        F("session_id", "STRING", "REQUIRED"),
        F("competency", "STRING", "REQUIRED"),
        F("average_score", "FLOAT64", "REQUIRED"),
        F("competency_percentage", "FLOAT64", "REQUIRED"),
        F("weight_pct", "FLOAT64", "REQUIRED"),
        F("question_count", "INT64", "REQUIRED"),
        F("calculated_at", "TIMESTAMP", "REQUIRED"),
    ],
)

FINAL_REPORTS = TableSpec(
    name="final_reports",
    description="Final immutable report snapshot backing the PDF and Candidate Report page",
    clustering_fields=["session_id", "decision"],
    schema=[
        F("report_id", "STRING", "REQUIRED"),
        F("session_id", "STRING", "REQUIRED"),
        F("technical_score", "FLOAT64", "REQUIRED"),
        F("soft_skill_score", "FLOAT64", "REQUIRED"),
        F("overall_score", "FLOAT64", "REQUIRED"),
        F("decision", "STRING", "REQUIRED"),
        F("confidence_level", "STRING", "REQUIRED"),
        F("confidence_percentage", "FLOAT64", "REQUIRED"),
        F("strengths", "STRING", "REPEATED"),
        F("development_areas", "STRING", "REPEATED"),
        F("recommendations", "STRING", "REPEATED"),
        F("communication_score", "FLOAT64", "NULLABLE"),
        F("structured_thinking_score", "FLOAT64", "NULLABLE"),
        F("problem_solving_score", "FLOAT64", "NULLABLE"),
        F("ambiguity_handling_score", "FLOAT64", "NULLABLE"),
        F("followup_quality_score", "FLOAT64", "NULLABLE"),
        F("evidence_rationale", "STRING", "NULLABLE"),
        F("pdf_report_path", "STRING", "NULLABLE"),
        F("report_version", "STRING", "REQUIRED"),
        F("generated_at", "TIMESTAMP", "REQUIRED"),
    ],
)

# ---------------------------------------------------------------------------
# 4. Governance & audit
# ---------------------------------------------------------------------------

AUDIT_LOGS = TableSpec(
    name="audit_logs",
    description="Immutable, append-only audit trail for every security-relevant event",
    partition_field="timestamp",
    clustering_fields=["user_id", "action"],
    schema=[
        F("audit_id", "STRING", "REQUIRED"),
        F("user_id", "STRING", "NULLABLE"),
        F("action", "STRING", "REQUIRED"),
        F("entity", "STRING", "NULLABLE"),
        F("entity_id", "STRING", "NULLABLE"),
        F("description", "STRING", "NULLABLE"),
        F("ip_address", "STRING", "NULLABLE"),
        F("status", "STRING", "REQUIRED", description="SUCCESS | FAILURE"),
        F("severity", "STRING", "REQUIRED", description="INFO | WARNING | ERROR | SECURITY"),
        F("timestamp", "TIMESTAMP", "REQUIRED"),
    ],
)

# ---------------------------------------------------------------------------
# 5. AI reproducibility / versioning
# ---------------------------------------------------------------------------

PROMPT_VERSIONS = TableSpec(
    name="prompt_versions",
    description="Version-controlled Interviewer/Evaluator prompts for reproducibility",
    clustering_fields=["prompt_type", "is_active"],
    schema=[
        F("version_id", "STRING", "REQUIRED"),
        F("prompt_name", "STRING", "REQUIRED"),
        F("prompt_type", "STRING", "REQUIRED", description="INTERVIEWER | EVALUATOR"),
        F("prompt_text", "STRING", "REQUIRED"),
        F("created_at", "TIMESTAMP", "REQUIRED"),
        F("created_by", "STRING", "NULLABLE"),
        F("is_active", "BOOLEAN", "REQUIRED"),
    ],
)

MODEL_VERSIONS = TableSpec(
    name="model_versions",
    description="Locks each session to a specific model config for reproducibility",
    clustering_fields=["model_name", "is_active"],
    schema=[
        F("model_version_id", "STRING", "REQUIRED"),
        F("model_name", "STRING", "REQUIRED"),
        F("temperature", "FLOAT64", "REQUIRED"),
        F("top_p", "FLOAT64", "REQUIRED"),
        F("max_tokens", "INT64", "NULLABLE"),
        F("created_at", "TIMESTAMP", "REQUIRED"),
        F("is_active", "BOOLEAN", "REQUIRED"),
    ],
)

# ---------------------------------------------------------------------------
# 6. Question repository
# ---------------------------------------------------------------------------

QUESTION_BANK = TableSpec(
    name="question_bank",
    description="Static/curated question bank; source of truth loaded from questions.yaml",
    clustering_fields=["competency", "difficulty"],
    schema=[
        F("question_id", "STRING", "REQUIRED"),
        F("competency", "STRING", "REQUIRED"),
        F("difficulty", "STRING", "REQUIRED"),
        F("question_text", "STRING", "REQUIRED"),
        F("expected_concepts", "STRING", "REPEATED"),
        F("keywords", "STRING", "REPEATED"),
        F("followups", "STRING", "REPEATED"),
        F("scoring_guidance", "STRING", "NULLABLE"),
        F("expected_answer_notes", "STRING", "NULLABLE"),
        F("estimated_time_seconds", "INT64", "NULLABLE"),
        F("created_at", "TIMESTAMP", "REQUIRED"),
        F("updated_at", "TIMESTAMP", "NULLABLE"),
        F("active", "BOOLEAN", "REQUIRED"),
    ],
)

ALL_TABLES: list[TableSpec] = [
    USERS,
    USER_ROLES,
    USER_SESSIONS,
    REVIEWER_REQUESTS,
    CANDIDATE_SESSIONS,
    INTERVIEW_QUESTIONS,
    INTERVIEW_RESPONSES,
    ANSWER_EVALUATIONS,
    COMPETENCY_SCORES,
    FINAL_REPORTS,
    AUDIT_LOGS,
    PROMPT_VERSIONS,
    MODEL_VERSIONS,
    QUESTION_BANK,
]


def get_client() -> bigquery.Client:
    """
    Create and return a BigQuery client using the Service Account JSON.
    """

    if not settings.GCP_PROJECT_ID:
        raise RuntimeError(
            "GCP_PROJECT_ID is not set. Please configure it in config/settings.py "
            "or the underlying .env file."
        )

    if not os.path.exists(settings.SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(
            f"Service Account JSON not found: {settings.SERVICE_ACCOUNT_FILE}"
        )

    credentials = service_account.Credentials.from_service_account_file(
        settings.SERVICE_ACCOUNT_FILE
    )

    return bigquery.Client(
        project=settings.GCP_PROJECT_ID,
        credentials=credentials
    )


def ensure_dataset(client: bigquery.Client) -> bigquery.Dataset:
    dataset_ref = bigquery.DatasetReference(PROJECT_ID, DATASET_ID)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = LOCATION
    dataset.description = "DE-Interviewer: AI-powered Data Engineering interview platform"
    dataset = client.create_dataset(dataset, exists_ok=True)
    logger.info("Dataset ready: %s.%s (%s)", PROJECT_ID, DATASET_ID, LOCATION)
    return dataset


def build_table(spec: TableSpec) -> bigquery.Table:
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{spec.name}"
    table = bigquery.Table(table_ref, schema=spec.schema)
    table.description = spec.description
    if spec.partition_field:
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=spec.partition_field,
        )
    if spec.clustering_fields:
        table.clustering_fields = spec.clustering_fields
    return table


def create_all_tables(client: bigquery.Client) -> None:
    for spec in ALL_TABLES:
        table = build_table(spec)
        try:
            client.create_table(table, exists_ok=True)
            logger.info("Table ready: %s", spec.name)
        except GoogleAPICallError as exc:
            logger.error("Failed to create table %s: %s", spec.name, exc)
            raise


def initialize_schema() -> None:
    client = get_client()
    ensure_dataset(client)
    create_all_tables(client)
    logger.info(
        "Schema initialization complete: %d tables in %s.%s",
        len(ALL_TABLES), PROJECT_ID, DATASET_ID,
    )


if __name__ == "__main__":
    initialize_schema()