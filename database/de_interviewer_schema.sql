-- ============================================================================
-- DE-INTERVIEWER  |  BigQuery Sandbox (Free-Tier) Production Schema
-- Dataset: de_interviewer
-- 14 tables | Partitioned + Clustered | BQ-native constraints (NOT ENFORCED)
-- ============================================================================
-- Notes on BigQuery specifics used throughout this file:
--   * BigQuery PK/FK constraints are metadata-only (NOT ENFORCED) — they help
--     the query optimizer and document relationships, but the application/
--     repository layer must enforce integrity (BQ has no native FK checks).
--   * Only ONE partition column allowed per table, and it must be
--     DATE/TIMESTAMP/DATETIME or an INT64 range. We partition by DATE(<ts>).
--   * Clustering supports up to 4 columns, chosen by query-filter frequency.
--   * IDs are STRING (UUIDs generated in the app with uuid4() / GENERATE_UUID()
--     at insert time) — simplest, portable, free-tier friendly.
--   * BigQuery Sandbox tables auto-expire after 60 days of no edits unless you
--     add a billing account — fine for an internship/demo project.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS `de_interviewer`
OPTIONS (
  description = "DE-Interviewer: AI-powered Data Engineering interview platform",
  location = "asia-south1"
);

-- ============================================================================
-- 1. AUTHENTICATION & RBAC
-- ============================================================================

CREATE TABLE IF NOT EXISTS `de_interviewer.users` (
  user_id       STRING    NOT NULL OPTIONS(description="UUID, primary key"),
  email         STRING    NOT NULL OPTIONS(description="Google OAuth email, unique"),
  full_name     STRING    OPTIONS(description="Display name from OAuth profile"),
  picture_url   STRING    OPTIONS(description="OAuth profile picture URL"),
  created_at    TIMESTAMP NOT NULL OPTIONS(description="First login timestamp"),
  last_login    TIMESTAMP OPTIONS(description="Most recent login timestamp"),
  status        STRING    NOT NULL OPTIONS(description="ACTIVE | DISABLED"),
  PRIMARY KEY (user_id) NOT ENFORCED
)
CLUSTER BY user_id
OPTIONS (description = "Every authenticated user (candidate, reviewer, admin)");

CREATE TABLE IF NOT EXISTS `de_interviewer.user_roles` (
  role_id       STRING    NOT NULL OPTIONS(description="UUID, primary key"),
  user_id       STRING    NOT NULL OPTIONS(description="FK -> users.user_id"),
  role          STRING    NOT NULL OPTIONS(description="CANDIDATE | REVIEWER | ADMIN"),
  assigned_by   STRING    OPTIONS(description="user_id or 'SYSTEM' (admins.yaml bootstrap)"),
  assigned_at   TIMESTAMP NOT NULL,
  active        BOOL      NOT NULL OPTIONS(description="Only one active role row per user"),
  PRIMARY KEY (role_id) NOT ENFORCED,
  FOREIGN KEY (user_id) REFERENCES `de_interviewer.users`(user_id) NOT ENFORCED
)
CLUSTER BY user_id, role
OPTIONS (description = "Role assignment history; current role = latest active=TRUE row");

CREATE TABLE IF NOT EXISTS `de_interviewer.user_sessions` (
  session_id    STRING    NOT NULL OPTIONS(description="UUID, primary key"),
  user_id       STRING    NOT NULL OPTIONS(description="FK -> users.user_id"),
  login_time    TIMESTAMP NOT NULL,
  logout_time   TIMESTAMP,
  ip_address    STRING,
  browser       STRING,
  device        STRING,
  PRIMARY KEY (session_id) NOT ENFORCED,
  FOREIGN KEY (user_id) REFERENCES `de_interviewer.users`(user_id) NOT ENFORCED
)
PARTITION BY DATE(login_time)
CLUSTER BY user_id
OPTIONS (description = "Login/logout audit trail, separate from interview candidate_sessions");

CREATE TABLE IF NOT EXISTS `de_interviewer.reviewer_requests` (
  request_id              STRING    NOT NULL OPTIONS(description="UUID, primary key"),
  user_id                 STRING    NOT NULL OPTIONS(description="FK -> users.user_id"),
  full_name               STRING    NOT NULL,
  email                   STRING    NOT NULL,
  organization            STRING,
  designation             STRING    OPTIONS(description="Recruiter, Hiring Manager, etc."),
  department              STRING,
  linkedin_url            STRING,
  requested_role          STRING    NOT NULL OPTIONS(description="Always REVIEWER"),
  business_justification  STRING    NOT NULL,
  status                  STRING    NOT NULL OPTIONS(description="PENDING | APPROVED | REJECTED"),
  approved_by             STRING    OPTIONS(description="Admin user_id who actioned it"),
  approved_at             TIMESTAMP,
  comments                STRING    OPTIONS(description="Admin's approval/rejection note"),
  requested_at            TIMESTAMP NOT NULL,
  PRIMARY KEY (request_id) NOT ENFORCED,
  FOREIGN KEY (user_id) REFERENCES `de_interviewer.users`(user_id) NOT ENFORCED
)
PARTITION BY DATE(requested_at)
CLUSTER BY status, user_id
OPTIONS (description = "Candidate -> Reviewer access requests awaiting Admin approval");

-- ============================================================================
-- 2. INTERVIEW LIFECYCLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS `de_interviewer.candidate_sessions` (
  session_id              STRING    NOT NULL OPTIONS(description="UUID, primary key"),
  user_id                 STRING    NOT NULL OPTIONS(description="FK -> users.user_id"),
  experience_years        FLOAT64   OPTIONS(description="Candidate-declared years of experience"),
  target_role             STRING,
  skills                  ARRAY<STRING>,
  resume_summary          STRING    OPTIONS(description="Optional free-text resume summary"),
  started_at              TIMESTAMP NOT NULL,
  completed_at            TIMESTAMP,
  status                  STRING    NOT NULL OPTIONS(description="IN_PROGRESS | COMPLETED | ABANDONED"),
  technical_score         FLOAT64   OPTIONS(description="0-100, FR8"),
  soft_skill_score        FLOAT64   OPTIONS(description="0-100, FR7"),
  overall_score           FLOAT64   OPTIONS(description="0-100, FR8 weighted formula"),
  decision                STRING    OPTIONS(description="SELECT | HOLD | REJECT (FR9)"),
  confidence_level        STRING    OPTIONS(description="LOW | MEDIUM | HIGH"),
  confidence_percentage   FLOAT64,
  competency_coverage_pct FLOAT64   OPTIONS(description="% of 8 competencies assessed, min 6 required"),
  prompt_version          STRING    OPTIONS(description="FK -> prompt_versions.version_id, reproducibility"),
  model_version            STRING   OPTIONS(description="FK -> model_versions.model_version_id"),
  interview_mode              STRING OPTIONS(description="e.g. STANDARD | PRACTICE | CALIBRATION"),
  langgraph_session_id        STRING OPTIONS(description="LangGraph run/thread ID for tracing this workflow execution"),
  interviewer_agent_version   STRING OPTIONS(description="Version tag of the Interviewer Agent used"),
  evaluator_agent_version     STRING OPTIONS(description="Version tag of the Evaluator Agent used"),
  PRIMARY KEY (session_id) NOT ENFORCED,
  FOREIGN KEY (user_id) REFERENCES `de_interviewer.users`(user_id) NOT ENFORCED
)
PARTITION BY DATE(started_at)
CLUSTER BY user_id, decision
OPTIONS (description = "One row per interview attempt; FR1 intake + FR8/FR9 final scoring/decision");

CREATE TABLE IF NOT EXISTS `de_interviewer.interview_questions` (
  question_instance_id        STRING    NOT NULL OPTIONS(description="UUID, primary key"),
  session_id                  STRING    NOT NULL OPTIONS(description="FK -> candidate_sessions.session_id"),
  question_id                 STRING    NOT NULL OPTIONS(description="FK -> question_bank.question_id"),
  competency                  STRING    NOT NULL OPTIONS(description="C1..C8"),
  difficulty                  STRING    OPTIONS(description="EASY | MEDIUM | HARD"),
  stage                       STRING    NOT NULL OPTIONS(description="S0 | S1 | S2 | S3 | S4"),
  question_text                STRING   NOT NULL OPTIONS(description="Rendered question text (incl. dynamic follow-up)"),
  followup_level               INT64    NOT NULL OPTIONS(description="0 = original, 1-2 = follow-up depth (FR3 max 2)"),
  parent_question_instance_id STRING    OPTIONS(description="Self-FK; NULL for original questions"),
  asked_at                    TIMESTAMP NOT NULL,
  PRIMARY KEY (question_instance_id) NOT ENFORCED,
  FOREIGN KEY (session_id) REFERENCES `de_interviewer.candidate_sessions`(session_id) NOT ENFORCED,
  FOREIGN KEY (question_id) REFERENCES `de_interviewer.question_bank`(question_id) NOT ENFORCED
)
PARTITION BY DATE(asked_at)
CLUSTER BY session_id, competency
OPTIONS (description = "Every question/follow-up instance actually asked in a session");

CREATE TABLE IF NOT EXISTS `de_interviewer.interview_responses` (
  response_id             STRING    NOT NULL OPTIONS(description="UUID, primary key"),
  question_instance_id    STRING    NOT NULL OPTIONS(description="FK -> interview_questions.question_instance_id"),
  candidate_answer        STRING    OPTIONS(description="Raw candidate answer text, may be empty/null"),
  answer_summary          STRING    OPTIONS(description="LLM-condensed summary for reporting"),
  response_type           STRING    OPTIONS(description="TEXT | EMPTY | I_DONT_KNOW | OFF_TOPIC | SPAM"),
  response_time_seconds   INT64,
  submitted_at            TIMESTAMP NOT NULL,
  PRIMARY KEY (response_id) NOT ENFORCED,
  FOREIGN KEY (question_instance_id) REFERENCES `de_interviewer.interview_questions`(question_instance_id) NOT ENFORCED
)
PARTITION BY DATE(submitted_at)
CLUSTER BY question_instance_id
OPTIONS (description = "Raw candidate answers, 1:1 with interview_questions");

-- ============================================================================
-- 3. AI EVALUATION (Evaluator Agent output — matches strict JSON contract)
-- ============================================================================

CREATE TABLE IF NOT EXISTS `de_interviewer.answer_evaluations` (
  evaluation_id       STRING    NOT NULL OPTIONS(description="UUID, primary key"),
  response_id         STRING    NOT NULL OPTIONS(description="FK -> interview_responses.response_id"),
  score                INT64    NOT NULL OPTIONS(description="0-5 scale, FR5"),
  competency           STRING   NOT NULL,
  feedback             STRING   NOT NULL,
  reasoning             STRING  NOT NULL OPTIONS(description="Evaluator chain-of-justification, explainability"),
  followup_required    BOOL     NOT NULL,
  expected_concepts    ARRAY<STRING>,
  detected_concepts    ARRAY<STRING>,
  strengths            ARRAY<STRING>,
  weaknesses            ARRAY<STRING>,
  evidence              STRING  OPTIONS(description="Verbatim/paraphrased excerpt from the answer supporting the score"),
  confidence             FLOAT64 OPTIONS(description="Evaluator's self-reported confidence in this score, 0-1"),
  temperature           FLOAT64 NOT NULL OPTIONS(description="Always 0 per reproducibility controls"),
  model                 STRING  NOT NULL OPTIONS(description="e.g. gemini-2.5-flash"),
  prompt_version        STRING  NOT NULL OPTIONS(description="FK -> prompt_versions.version_id"),
  evaluation_time        TIMESTAMP NOT NULL,
  PRIMARY KEY (evaluation_id) NOT ENFORCED,
  FOREIGN KEY (response_id) REFERENCES `de_interviewer.interview_responses`(response_id) NOT ENFORCED
)
PARTITION BY DATE(evaluation_time)
CLUSTER BY competency, score
OPTIONS (description = "Evaluator Agent structured JSON output, 1:1 with interview_responses");

CREATE TABLE IF NOT EXISTS `de_interviewer.competency_scores` (
  competency_score_id     STRING  NOT NULL OPTIONS(description="UUID, primary key"),
  session_id              STRING  NOT NULL OPTIONS(description="FK -> candidate_sessions.session_id"),
  competency               STRING NOT NULL OPTIONS(description="C1..C8"),
  average_score            FLOAT64 NOT NULL OPTIONS(description="mean(question_scores), 0-5"),
  competency_percentage    FLOAT64 NOT NULL OPTIONS(description="FR6: (mean/5)*100"),
  weight_pct               FLOAT64 NOT NULL OPTIONS(description="Configured weight, e.g. C3=20"),
  question_count           INT64   NOT NULL,
  calculated_at            TIMESTAMP NOT NULL,
  PRIMARY KEY (competency_score_id) NOT ENFORCED,
  FOREIGN KEY (session_id) REFERENCES `de_interviewer.candidate_sessions`(session_id) NOT ENFORCED
)
CLUSTER BY session_id, competency
OPTIONS (description = "Aggregated per-competency rollup per session (FR4/FR6)");

CREATE TABLE IF NOT EXISTS `de_interviewer.final_reports` (
  report_id                STRING    NOT NULL OPTIONS(description="UUID, primary key"),
  session_id               STRING    NOT NULL OPTIONS(description="FK -> candidate_sessions.session_id"),
  technical_score          FLOAT64   NOT NULL,
  soft_skill_score         FLOAT64   NOT NULL,
  overall_score            FLOAT64   NOT NULL,
  decision                 STRING    NOT NULL OPTIONS(description="SELECT | HOLD | REJECT"),
  confidence_level         STRING    NOT NULL,
  confidence_percentage    FLOAT64   NOT NULL,
  strengths                ARRAY<STRING>,
  development_areas        ARRAY<STRING>,
  recommendations          ARRAY<STRING>,
  communication_score          FLOAT64 OPTIONS(description="FR7 soft skill sub-score, 0-100"),
  structured_thinking_score    FLOAT64 OPTIONS(description="FR7 soft skill sub-score, 0-100"),
  problem_solving_score        FLOAT64 OPTIONS(description="FR7 soft skill sub-score, 0-100"),
  ambiguity_handling_score     FLOAT64 OPTIONS(description="FR7 soft skill sub-score, 0-100"),
  followup_quality_score       FLOAT64 OPTIONS(description="FR7 soft skill sub-score, 0-100"),
  evidence_rationale       STRING    OPTIONS(description="Evidence-based narrative justifying the decision"),
  pdf_report_path          STRING    OPTIONS(description="Path/URI of generated ReportLab PDF"),
  report_version           STRING    NOT NULL OPTIONS(description="Report template/format version that generated this row"),
  generated_at             TIMESTAMP NOT NULL,
  PRIMARY KEY (report_id) NOT ENFORCED,
  FOREIGN KEY (session_id) REFERENCES `de_interviewer.candidate_sessions`(session_id) NOT ENFORCED
)
CLUSTER BY session_id, decision
OPTIONS (description = "FR10: final immutable report snapshot backing the PDF and Candidate Report page");

-- ============================================================================
-- 4. GOVERNANCE & AUDIT
-- ============================================================================

CREATE TABLE IF NOT EXISTS `de_interviewer.audit_logs` (
  audit_id      STRING    NOT NULL OPTIONS(description="UUID, primary key"),
  user_id       STRING    OPTIONS(description="Actor; NULL for anonymous/system events"),
  action        STRING    NOT NULL OPTIONS(description="LOGIN, LOGOUT, INTERVIEW_STARTED, ROLE_CHANGED, ACCESS_DENIED, etc."),
  entity        STRING    OPTIONS(description="Entity type acted upon, e.g. candidate_sessions"),
  entity_id     STRING    OPTIONS(description="ID of the affected row"),
  description   STRING,
  ip_address    STRING,
  status        STRING    NOT NULL OPTIONS(description="SUCCESS | FAILURE"),
  severity      STRING    NOT NULL OPTIONS(description="INFO | WARNING | ERROR | SECURITY"),
  timestamp     TIMESTAMP NOT NULL,
  PRIMARY KEY (audit_id) NOT ENFORCED
)
PARTITION BY DATE(timestamp)
CLUSTER BY user_id, action
OPTIONS (description = "Immutable, append-only audit trail for every security-relevant event");

-- ============================================================================
-- 5. AI REPRODUCIBILITY / VERSIONING
-- ============================================================================

CREATE TABLE IF NOT EXISTS `de_interviewer.prompt_versions` (
  version_id     STRING    NOT NULL OPTIONS(description="UUID, primary key"),
  prompt_name    STRING    NOT NULL,
  prompt_type    STRING    NOT NULL OPTIONS(description="INTERVIEWER | EVALUATOR"),
  prompt_text    STRING    NOT NULL,
  created_at     TIMESTAMP NOT NULL,
  created_by     STRING,
  is_active      BOOL      NOT NULL,
  PRIMARY KEY (version_id) NOT ENFORCED
)
CLUSTER BY prompt_type, is_active
OPTIONS (description = "Version-controlled Interviewer/Evaluator prompts for reproducibility");

CREATE TABLE IF NOT EXISTS `de_interviewer.model_versions` (
  model_version_id  STRING    NOT NULL OPTIONS(description="UUID, primary key"),
  model_name        STRING    NOT NULL OPTIONS(description="e.g. gemini-2.5-flash"),
  temperature       FLOAT64   NOT NULL OPTIONS(description="Must be 0 for evaluator determinism"),
  top_p             FLOAT64   NOT NULL,
  max_tokens        INT64,
  created_at        TIMESTAMP NOT NULL,
  is_active         BOOL      NOT NULL,
  PRIMARY KEY (model_version_id) NOT ENFORCED
)
CLUSTER BY model_name, is_active
OPTIONS (description = "Locks each session to a specific model config for reproducibility");

-- ============================================================================
-- 6. QUESTION REPOSITORY
-- ============================================================================

CREATE TABLE IF NOT EXISTS `de_interviewer.question_bank` (
  question_id             STRING    NOT NULL OPTIONS(description="UUID, primary key"),
  competency               STRING   NOT NULL OPTIONS(description="C1..C8"),
  difficulty                STRING  NOT NULL OPTIONS(description="EASY | MEDIUM | HARD"),
  question_text             STRING  NOT NULL,
  expected_concepts        ARRAY<STRING>,
  keywords                  ARRAY<STRING>,
  followups                 ARRAY<STRING>  OPTIONS(description="Pre-authored follow-up prompt templates"),
  scoring_guidance          STRING  OPTIONS(description="How the evaluator should map answer quality to 0-5"),
  expected_answer_notes     STRING,
  estimated_time_seconds    INT64   OPTIONS(description="Expected time to answer, used to project total interview duration"),
  created_at                TIMESTAMP NOT NULL,
  updated_at                TIMESTAMP,
  active                    BOOL    NOT NULL,
  PRIMARY KEY (question_id) NOT ENFORCED
)
CLUSTER BY competency, difficulty
OPTIONS (description = "Static/curated question bank; source of truth loaded from questions.yaml");

-- ============================================================================
-- PARTITIONING SUMMARY (high-volume, time-series tables)
-- ============================================================================
-- candidate_sessions   -> DATE(started_at)
-- interview_questions  -> DATE(asked_at)
-- interview_responses  -> DATE(submitted_at)
-- answer_evaluations   -> DATE(evaluation_time)
-- audit_logs           -> DATE(timestamp)
-- user_sessions        -> DATE(login_time)
-- reviewer_requests    -> DATE(requested_at)
--
-- Reference/config tables (users, user_roles, competency_scores, final_reports,
-- prompt_versions, model_versions, question_bank) are small and low-write —
-- not partitioned, clustered only, to keep the free-tier query-scan footprint low.
--
-- CLUSTERING SUMMARY
-- Chosen by the most frequent WHERE/JOIN filters in the app's repository layer:
--   user_id, session_id, competency, decision, action, status
-- ============================================================================