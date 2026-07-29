from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from google.api_core.exceptions import GoogleAPICallError, RetryError
from google.cloud import bigquery
from google.cloud.bigquery import dbapi
from google.cloud.bigquery.job import QueryJob
from google.cloud.bigquery.table import TableReference
from utils.retry import RetryConfig, RetryError as RepositoryRetryError, execute_with_retry


class BigQueryRepositoryError(Exception):
    """Base exception raised by BigQueryRepository."""


class ConnectionError(BigQueryRepositoryError):
    """Raised when BigQuery client connection cannot be established."""


class QueryExecutionError(BigQueryRepositoryError):
    """Raised when a query cannot be executed."""


class InsertError(BigQueryRepositoryError):
    """Raised when row insertion fails."""


class UpdateError(BigQueryRepositoryError):
    """Raised when an update operation fails."""


class DeleteError(BigQueryRepositoryError):
    """Raised when a delete operation fails."""


class ValidationError(BigQueryRepositoryError):
    """Raised when supplied input is invalid."""


@dataclass(frozen=True)
class BigQuerySchemaField:
    """Represents a BigQuery schema field."""

    name: str
    field_type: str
    mode: str
    description: str | None


class BigQueryRepository:
    """A generic BigQuery repository for data access."""

    ENV_PROJECT = "GOOGLE_CLOUD_PROJECT"
    ENV_DATASET = "BIGQUERY_DATASET"
    ENV_CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"
    DEFAULT_QUERY_TIMEOUT = 30.0

    def __init__(
        self,
        project: str | None = None,
        dataset: str | None = None,
        credentials_path: str | None = None,
        env_path: str | None = None,
        client: bigquery.Client | None = None,
        logger: logging.Logger | None = None,
        retry_config: RetryConfig | None = None,
        query_timeout: float = DEFAULT_QUERY_TIMEOUT,
    ) -> None:
        """
        Initialize the BigQuery repository.

        Args:
            project: Optional Google Cloud project id.
            dataset: Optional BigQuery dataset id.
            credentials_path: Optional path to service account json.
            env_path: Optional path to .env file.
            client: Optional BigQuery client for dependency injection.
            logger: Optional logger.
            retry_config: Optional retry configuration for transient failures.
            query_timeout: Default query timeout in seconds.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initializing BigQueryRepository")
        self._client_lock = threading.RLock()
        self._transaction_lock = threading.RLock()

        self._env_path = Path(env_path) if env_path else Path(".env")
        self._load_env_file(self._env_path)

        self.project = project or os.getenv(self.ENV_PROJECT)
        self.dataset = dataset or os.getenv(self.ENV_DATASET)
        self.credentials_path = credentials_path or os.getenv(self.ENV_CREDENTIALS)
        self._validate_configuration()
        self.query_timeout = query_timeout
        self.retry_config = retry_config or RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=8.0,
            backoff_multiplier=2.0,
            jitter_ratio=0.1,
            retryable_exceptions=(GoogleAPICallError, RetryError),
            operation_name="bigquery_operation",
        )

        self._client: bigquery.Client | None = client
        self._transaction_connection: dbapi.Connection | None = None
        self._transaction_active = False

        if self.credentials_path:
            os.environ[self.ENV_CREDENTIALS] = self.credentials_path

    def _load_env_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            return

        self.logger.debug("Loading environment variables from %s", path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except OSError as exc:
            raise ConnectionError(f"Unable to read .env file at {path}: {exc}") from exc

    def _validate_configuration(self) -> None:
        if not self.project:
            raise ValidationError(
                f"Missing BigQuery project. Set {self.ENV_PROJECT}."
            )
        if not self.dataset:
            raise ValidationError(
                f"Missing BigQuery dataset. Set {self.ENV_DATASET}."
            )
        if self.credentials_path:
            credentials_file = Path(self.credentials_path)
            if not credentials_file.exists() or not credentials_file.is_file():
                raise ValidationError(
                    f"Credentials file does not exist: {self.credentials_path}"
                )

    def connect(self) -> bigquery.Client:
        """
        Establish or reuse a BigQuery client.

        Returns:
            A BigQuery client instance.
        """
        with self._client_lock:
            if self._client is not None:
                return self._client

            self.logger.info(
                "Establishing BigQuery client for project=%s dataset=%s",
                self.project,
                self.dataset,
            )
            try:
                self._client = self._with_retry(
                    lambda: bigquery.Client(project=self.project),
                    operation_name="bigquery_connect",
                )
                self.logger.debug("BigQuery client established")
                return self._client
            except GoogleAPICallError as exc:
                raise ConnectionError(
                    f"Failed to create BigQuery client for project {self.project}"
                ) from exc
            except RetryError as exc:
                raise ConnectionError(
                    f"Retry failure while creating BigQuery client for project {self.project}"
                ) from exc
            except RepositoryRetryError as exc:
                raise ConnectionError(
                    f"Retry exhaustion while creating BigQuery client for project {self.project}"
                ) from exc

    def disconnect(self) -> None:
        """
        Close the BigQuery client and release resources.
        """
        with self._client_lock:
            if self._client is None:
                return

            self.logger.info("Disconnecting BigQuery client")
            close_method = getattr(self._client, "close", None)
            if callable(close_method):
                close_method()
            self._client = None

    def health_check(self) -> dict[str, Any]:
        """
        Perform a health check against BigQuery.

        Returns:
            Health status details.
        """
        self._ensure_connected()
        start = time.monotonic()
        dataset_exists = self.dataset_exists(self.dataset)
        tables = []
        connection_status = "OK" if dataset_exists else "DATASET_MISSING"
        if dataset_exists:
            tables = self.list_tables(self.dataset)
        latency_ms = round((time.monotonic() - start) * 1000, 2)
        status = "healthy" if dataset_exists else "unhealthy"

        health = {
            "status": status,
            "project": self.project,
            "dataset": self.dataset,
            "connection": connection_status,
            "latency_ms": latency_ms,
            "tables": tables,
        }
        self.logger.debug("Health check result: %s", health)
        return health

    def execute_query(
        self,
        sql: str,
        parameters: dict[str, Any] | Sequence[Any] | None = None,
        job_config: bigquery.QueryJobConfig | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a parameterized query and return result rows.

        Args:
            sql: SQL query text.
            parameters: Query parameters.
            job_config: Optional BigQuery job config.
            timeout: Query timeout in seconds.

        Returns:
            Query result rows as list of dictionaries.
        """
        if not sql or not sql.strip():
            raise ValidationError("SQL query text must not be empty")

        if self._transaction_active:
            return self._execute_query_in_transaction(sql, parameters)

        effective_timeout = timeout if timeout is not None else self.query_timeout
        return self._execute_query_with_client(
            sql,
            parameters,
            job_config,
            effective_timeout,
        )

    def fetch_one(
        self,
        sql: str,
        parameters: dict[str, Any] | Sequence[Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Execute a query and return a single row.

        Args:
            sql: SQL query text.
            parameters: Query parameters.
            timeout: Query timeout in seconds.

        Returns:
            First row as a dictionary, or None if no rows exist.
        """
        rows = self.execute_query(sql, parameters=parameters, timeout=timeout)
        return rows[0] if rows else None

    def fetch_all(
        self,
        sql: str,
        parameters: dict[str, Any] | Sequence[Any] | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a query and return all rows.

        Args:
            sql: SQL query text.
            parameters: Query parameters.
            timeout: Query timeout in seconds.

        Returns:
            List of row dictionaries.
        """
        return self.execute_query(sql, parameters=parameters, timeout=timeout)

    def insert(
        self,
        table: str,
        row: dict[str, Any],
        schema: Iterable[str] | None = None,
    ) -> int:
        """
        Insert a single row into a table.

        Args:
            table: Table reference in dataset.table or project.dataset.table form.
            row: Row payload.
            schema: Optional schema field names for validation.

        Returns:
            Number of inserted rows.
        """
        if not isinstance(row, dict):
            raise ValidationError("Row payload must be a dictionary")

        return self.insert_many(table, [row], schema=schema)

    def insert_many(
        self,
        table: str,
        rows: Sequence[dict[str, Any]],
        schema: Iterable[str] | None = None,
    ) -> int:
        """
        Insert multiple rows into a table.

        Args:
            table: Table reference.
            rows: Sequence of row payloads.
            schema: Optional schema field names for validation.

        Returns:
            Number of inserted rows.
        """
        if not rows:
            return 0
        if not all(isinstance(row, dict) for row in rows):
            raise ValidationError("All rows must be dictionaries")

        table_ref = self._resolve_table_reference(table)
        self._validate_payload_schema(rows, table_ref, schema)

        self.logger.info(
            "Inserting %d row(s) into %s", len(rows), self._format_table_reference(table_ref)
        )
        try:
            errors = self._with_retry(
                lambda: self.connect().insert_rows_json(table_ref, list(rows)),
                operation_name="bigquery_insert_rows_json",
            )
        except GoogleAPICallError as exc:
            raise InsertError(
                f"BigQuery insert failed for table {self._format_table_reference(table_ref)}"
            ) from exc
        except RepositoryRetryError as exc:
            raise InsertError(
                f"BigQuery insert retry exhaustion for table {self._format_table_reference(table_ref)}"
            ) from exc

        if errors:
            self.logger.error("Insert errors: %s", errors)
            raise InsertError(
                f"Insert failed for table {self._format_table_reference(table_ref)}: {errors}"
            )

        return len(rows)

    def update(
        self,
        sql: str,
        parameters: dict[str, Any] | Sequence[Any] | None = None,
        timeout: float | None = None,
    ) -> int:
        """
        Execute a parameterized UPDATE statement.

        Args:
            sql: UPDATE SQL text.
            parameters: Query parameters.
            timeout: Query timeout in seconds.

        Returns:
            Number of affected rows.
        """
        return self._execute_dml(sql, parameters, timeout, operation="UPDATE")

    def delete(
        self,
        sql: str,
        parameters: dict[str, Any] | Sequence[Any] | None = None,
        timeout: float | None = None,
    ) -> int:
        """
        Execute a parameterized DELETE statement.

        Args:
            sql: DELETE SQL text.
            parameters: Query parameters.
            timeout: Query timeout in seconds.

        Returns:
            Number of affected rows.
        """
        return self._execute_dml(sql, parameters, timeout, operation="DELETE")

    def upsert(
        self,
        table: str,
        key_columns: Sequence[str],
        payload: dict[str, Any] | Sequence[dict[str, Any]],
        schema: Iterable[str] | None = None,
        timeout: float | None = None,
    ) -> int:
        """
        Perform an upsert using MERGE.

        Args:
            table: Table reference.
            key_columns: Key columns used for matching.
            payload: Single row or list of rows to upsert.
            schema: Optional schema field names for validation.
            timeout: Job timeout in seconds.

        Returns:
            Number of affected rows.
        """
        rows = [payload] if isinstance(payload, dict) else list(payload)
        if not rows:
            return 0
        if not key_columns:
            raise ValidationError("key_columns must contain at least one column")
        if not all(isinstance(key, str) and key.strip() for key in key_columns):
            raise ValidationError("key_columns must contain non-empty strings")

        table_ref = self._resolve_table_reference(table)
        self._validate_payload_schema(rows, table_ref, schema)

        query, parameters = self._build_merge_statement(table_ref, key_columns, rows)
        affected_rows = self._execute_dml(query, parameters, timeout, operation="MERGE")
        self.logger.info(
            "Upserted %d row(s) into %s", len(rows), self._format_table_reference(table_ref)
        )
        return affected_rows

    def table_exists(self, table: str) -> bool:
        """
        Check whether a table exists.

        Args:
            table: Table reference.
        """
        table_ref = self._resolve_table_reference(table)
        try:
            self._with_retry(
                lambda: self.connect().get_table(table_ref),
                operation_name="bigquery_get_table",
            )
            return True
        except GoogleAPICallError:
            return False
        except RepositoryRetryError:
            return False

    def dataset_exists(self, dataset: str) -> bool:
        """
        Check whether a dataset exists.

        Args:
            dataset: Dataset id.
        """
        self._validate_dataset_name(dataset)
        try:
            dataset_ref = bigquery.DatasetReference(self.project, dataset)
            self._with_retry(
                lambda: self.connect().get_dataset(dataset_ref),
                operation_name="bigquery_get_dataset",
            )
            return True
        except GoogleAPICallError:
            return False
        except RepositoryRetryError:
            return False

    def create_dataset_if_missing(
        self, dataset: str, location: str = "US"
    ) -> None:
        """
        Create a dataset if it does not exist.

        Args:
            dataset: Dataset id.
            location: Data location.
        """
        self._validate_dataset_name(dataset)
        dataset_ref = bigquery.DatasetReference(self.project, dataset)
        if self.dataset_exists(dataset):
            self.logger.debug("Dataset %s already exists", dataset)
            return

        self.logger.info("Creating dataset %s in location %s", dataset, location)
        dataset_obj = bigquery.Dataset(dataset_ref)
        dataset_obj.location = location
        try:
            self._with_retry(
                lambda: self.connect().create_dataset(dataset_obj, exists_ok=True),
                operation_name="bigquery_create_dataset",
            )
        except GoogleAPICallError as exc:
            raise ConnectionError(
                f"Failed to create dataset {dataset} in project {self.project}"
            ) from exc
        except RepositoryRetryError as exc:
            raise ConnectionError(
                f"Retry exhaustion while creating dataset {dataset} in project {self.project}"
            ) from exc

    def count_rows(
        self,
        table: str,
        where_clause: str | None = None,
        parameters: dict[str, Any] | Sequence[Any] | None = None,
        timeout: float | None = None,
    ) -> int:
        """
        Count rows in a table.

        Args:
            table: Table reference.
            where_clause: Optional WHERE clause without leading WHERE.
            parameters: Query parameters.
            timeout: Query timeout in seconds.
        """
        table_ref = self._resolve_table_reference(table)
        table_sql = self._format_table_reference(table_ref)
        sql = f"SELECT COUNT(1) AS row_count FROM {table_sql}"
        if where_clause and where_clause.strip():
            sql += f" WHERE {where_clause.strip()}"

        row = self.fetch_one(sql, parameters=parameters, timeout=timeout)
        return int(row["row_count"]) if row and row.get("row_count") is not None else 0

    def get_schema(self, table: str) -> list[BigQuerySchemaField]:
        """
        Retrieve table schema information.

        Args:
            table: Table reference.
        """
        table_ref = self._resolve_table_reference(table)
        table_obj = self._with_retry(
            lambda: self.connect().get_table(table_ref),
            operation_name="bigquery_get_schema",
        )
        return [
            BigQuerySchemaField(
                name=field.name,
                field_type=field.field_type,
                mode=field.mode,
                description=field.description,
            )
            for field in table_obj.schema
        ]

    def list_tables(self, dataset: str) -> list[str]:
        """
        List tables in a dataset.

        Args:
            dataset: Dataset id.
        """
        self._validate_dataset_name(dataset)
        dataset_ref = bigquery.DatasetReference(self.project, dataset)
        tables = self._with_retry(
            lambda: self.connect().list_tables(dataset_ref),
            operation_name="bigquery_list_tables",
        )
        return [table.table_id for table in tables]

    def begin_transaction(self) -> None:
        """
        Begin a BigQuery transaction using DB API.

        Raises:
            ValidationError: If a transaction is already active.
        """
        with self._transaction_lock:
            if self._transaction_active:
                raise ValidationError("Transaction already active")

            self.logger.info("Beginning BigQuery transaction")
            try:
                connection = dbapi.connect(project=self.project)
                connection.autocommit = False
                self._transaction_connection = connection
                self._transaction_active = True
            except GoogleAPICallError as exc:
                raise ConnectionError("Failed to begin BigQuery transaction") from exc

    def commit(self) -> None:
        """
        Commit the active BigQuery transaction.
        """
        with self._transaction_lock:
            if not self._transaction_active or self._transaction_connection is None:
                raise ValidationError("No active transaction to commit")

            self.logger.info("Committing BigQuery transaction")
            try:
                self._transaction_connection.commit()
            except GoogleAPICallError as exc:
                raise ConnectionError("Failed to commit BigQuery transaction") from exc
            finally:
                self._close_transaction()

    def rollback(self) -> None:
        """
        Roll back the active BigQuery transaction.
        """
        with self._transaction_lock:
            if not self._transaction_active or self._transaction_connection is None:
                raise ValidationError("No active transaction to rollback")

            self.logger.info("Rolling back BigQuery transaction")
            try:
                self._transaction_connection.rollback()
            except GoogleAPICallError as exc:
                raise ConnectionError("Failed to rollback BigQuery transaction") from exc
            finally:
                self._close_transaction()

    def _execute_query_with_client(
        self,
        sql: str,
        parameters: dict[str, Any] | Sequence[Any] | None,
        job_config: bigquery.QueryJobConfig | None,
        timeout: float,
    ) -> list[dict[str, Any]]:
        self._ensure_connected()
        query_job_config = job_config or bigquery.QueryJobConfig()
        if parameters:
            query_job_config.query_parameters = self._build_query_parameters(parameters)

        self.logger.debug("Executing query: %s", sql)
        start_time = time.monotonic()
        try:
            query_job = self._with_retry(
                lambda: self.connect().query(
                    sql,
                    job_config=query_job_config,
                    timeout=timeout,
                ),
                operation_name="bigquery_query_submit",
            )
            result = self._with_retry(
                lambda: query_job.result(timeout=timeout),
                operation_name="bigquery_query_result",
            )
            rows = self._convert_rows_to_dict(result)
            self._log_duration("Query executed", start_time)
            return rows
        except (GoogleAPICallError, RetryError) as exc:
            raise QueryExecutionError("BigQuery query execution failed") from exc
        except RepositoryRetryError as exc:
            raise QueryExecutionError("BigQuery query retry exhaustion") from exc

    def _execute_query_in_transaction(
        self,
        sql: str,
        parameters: dict[str, Any] | Sequence[Any] | None,
    ) -> list[dict[str, Any]]:
        with self._transaction_lock:
            if not self._transaction_active or self._transaction_connection is None:
                raise ValidationError("No active transaction")

            self.logger.debug("Executing transactional query: %s", sql)
            cursor = self._transaction_connection.cursor()
            try:
                cursor.execute(sql, parameters or ())
                rows = self._convert_dbapi_rows_to_dict(cursor)
                return rows
            except Exception as exc:
                raise QueryExecutionError("Transactional query execution failed") from exc

    def _execute_dml(
        self,
        sql: str,
        parameters: dict[str, Any] | Sequence[Any] | None,
        timeout: float,
        operation: str,
    ) -> int:
        if not sql or not sql.strip():
            raise ValidationError(f"{operation} SQL must not be empty")

        if self._transaction_active:
            rows = self.execute_query(sql, parameters=parameters, timeout=timeout)
            return len(rows)

        self._ensure_connected()
        if not sql.strip().upper().startswith(operation):
            raise ValidationError(f"{operation} statement must begin with {operation}")

        self.logger.debug("%s statement: %s", operation, sql)
        start_time = time.monotonic()
        try:
            job_config = bigquery.QueryJobConfig()
            if parameters:
                job_config.query_parameters = self._build_query_parameters(parameters)

            query_job = self._with_retry(
                lambda: self.connect().query(
                    sql,
                    job_config=job_config,
                    timeout=timeout,
                ),
                operation_name=f"bigquery_{operation.lower()}_submit",
            )
            result = self._with_retry(
                lambda: query_job.result(timeout=timeout),
                operation_name=f"bigquery_{operation.lower()}_result",
            )
            affected = int(getattr(result, "num_dml_affected_rows", 0) or 0)
            self._log_duration(f"{operation} executed", start_time)
            return affected
        except (GoogleAPICallError, RetryError) as exc:
            if operation == "UPDATE":
                raise UpdateError(f"BigQuery update failed") from exc
            if operation == "DELETE":
                raise DeleteError(f"BigQuery delete failed") from exc
            raise QueryExecutionError(f"BigQuery {operation} failed") from exc
        except RepositoryRetryError as exc:
            if operation == "UPDATE":
                raise UpdateError("BigQuery update retry exhaustion") from exc
            if operation == "DELETE":
                raise DeleteError("BigQuery delete retry exhaustion") from exc
            raise QueryExecutionError(f"BigQuery {operation} retry exhaustion") from exc

    def _resolve_table_reference(self, table: str) -> TableReference:
        parts = table.strip().split(".")
        if len(parts) == 1:
            self._validate_table_name(parts[0])
            return bigquery.DatasetReference(self.project, self.dataset).table(parts[0])
        if len(parts) == 2:
            dataset_name, table_name = parts
            self._validate_dataset_name(dataset_name)
            self._validate_table_name(table_name)
            return bigquery.DatasetReference(self.project, dataset_name).table(table_name)
        if len(parts) == 3:
            project_name, dataset_name, table_name = parts
            self._validate_project_name(project_name)
            self._validate_dataset_name(dataset_name)
            self._validate_table_name(table_name)
            return bigquery.DatasetReference(project_name, dataset_name).table(table_name)

        raise ValidationError(
            "Table reference must be formatted as table, dataset.table, or project.dataset.table"
        )

    def _format_table_reference(self, table_ref: TableReference) -> str:
        project = table_ref.project
        dataset_id = table_ref.dataset_id
        table_id = table_ref.table_id
        return f"`{project}`.`{dataset_id}`.`{table_id}`"

    def _validate_table_name(self, value: str) -> None:
        if not value or not value.strip():
            raise ValidationError("Table name must not be empty")
        if not value.replace("_", "").replace("-", "").isalnum():
            raise ValidationError(f"Invalid table name: {value}")

    def _validate_dataset_name(self, value: str) -> None:
        if not value or not value.strip():
            raise ValidationError("Dataset name must not be empty")
        if not value.replace("_", "").replace("-", "").isalnum():
            raise ValidationError(f"Invalid dataset name: {value}")

    def _validate_project_name(self, value: str) -> None:
        if not value or not value.strip():
            raise ValidationError("Project name must not be empty")
        if not value.replace("-", "").replace("_", "").isalnum():
            raise ValidationError(f"Invalid project name: {value}")

    def _validate_payload_schema(
        self,
        rows: Sequence[dict[str, Any]],
        table_ref: TableReference,
        schema: Iterable[str] | None = None,
    ) -> None:
        if schema is not None:
            schema_fields = {field.strip() for field in schema}
        else:
            table_obj = self._with_retry(
                lambda: self.connect().get_table(table_ref),
                operation_name="bigquery_validate_schema",
            )
            schema_fields = {field.name for field in table_obj.schema}

        for row in rows:
            missing_columns = [key for key in row.keys() if key not in schema_fields]
            if missing_columns:
                raise ValidationError(
                    f"Payload contains invalid columns: {', '.join(sorted(missing_columns))}"
                )

    def _build_merge_statement(
        self,
        table_ref: TableReference,
        key_columns: Sequence[str],
        rows: Sequence[dict[str, Any]],
    ) -> tuple[str, list[bigquery.ScalarQueryParameter]]:
        if not rows:
            raise ValidationError("Upsert payload must contain at least one row")

        field_names = list(rows[0].keys())
        if not field_names:
            raise ValidationError("Upsert payload rows must contain at least one field")

        for key in key_columns:
            if key not in field_names:
                raise ValidationError(
                    f"key_columns must be present in payload fields: {key}"
                )

        merge_rows: list[str] = []
        parameters: list[bigquery.ScalarQueryParameter] = []
        for idx, row in enumerate(rows):
            column_expressions = []
            for field in field_names:
                param_name = f"p_{idx}_{field}"
                column_expressions.append(f"@{param_name} AS `{field}`")
                parameters.append(self._build_query_parameter(param_name, row[field]))
            merge_rows.append("SELECT " + ", ".join(column_expressions))
        source_query = " UNION ALL ".join(merge_rows)
        target = self._format_table_reference(table_ref)
        key_filters = " AND ".join(
            f"T.`{key}` = S.`{key}`" for key in key_columns
        )
        update_assignments = ", ".join(
            f"T.`{field}` = S.`{field}`" for field in field_names if field not in key_columns
        )
        insert_columns = ", ".join(f"`{field}`" for field in field_names)
        insert_values = ", ".join(f"S.`{field}`" for field in field_names)

        merge_sql = (
            f"MERGE {target} T\n"
            f"USING ({source_query}) S\n"
            f"ON {key_filters}\n"
            "WHEN MATCHED THEN\n"
            f"  UPDATE SET {update_assignments}\n"
            "WHEN NOT MATCHED THEN\n"
            f"  INSERT ({insert_columns}) VALUES ({insert_values})"
        )
        return merge_sql, parameters

    def _build_query_parameters(
        self, parameters: dict[str, Any] | Sequence[Any]
    ) -> list[bigquery.ScalarQueryParameter]:
        query_parameters: list[bigquery.ScalarQueryParameter] = []
        if isinstance(parameters, dict):
            for name, value in parameters.items():
                query_parameters.append(self._build_query_parameter(name, value))
        else:
            for idx, value in enumerate(parameters):
                query_parameters.append(self._build_query_parameter(str(idx), value))
        return query_parameters

    def _build_query_parameter(
        self, name: str, value: Any
    ) -> bigquery.ScalarQueryParameter:
        if isinstance(value, bool):
            data_type = "BOOL"
        elif isinstance(value, int) and not isinstance(value, bool):
            data_type = "INT64"
        elif isinstance(value, float):
            data_type = "FLOAT64"
        elif isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
            data_type = "STRING"
        else:
            data_type = "STRING"
        return bigquery.ScalarQueryParameter(name, data_type, value)

    def _convert_rows_to_dict(
        self, rows: Iterable[bigquery.Row]
    ) -> list[dict[str, Any]]:
        return [dict(row) for row in rows]

    def _convert_dbapi_rows_to_dict(self, cursor: dbapi.Cursor) -> list[dict[str, Any]]:
        columns = [column[0] for column in cursor.description] if cursor.description else []
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _close_transaction(self) -> None:
        if self._transaction_connection is not None:
            try:
                self._transaction_connection.close()
            except Exception:
                pass
        self._transaction_connection = None
        self._transaction_active = False

    def _ensure_connected(self) -> None:
        try:
            self.connect()
        except BigQueryRepositoryError:
            raise

    def _log_duration(self, message: str, start_time: float) -> None:
        self.logger.debug("%s in %.2fms", message, (time.monotonic() - start_time) * 1000)

    def _with_retry(self, func: Any, operation_name: str) -> Any:
        retry_config = RetryConfig(
            max_attempts=self.retry_config.max_attempts,
            initial_delay=self.retry_config.initial_delay,
            max_delay=self.retry_config.max_delay,
            backoff_multiplier=self.retry_config.backoff_multiplier,
            jitter_ratio=self.retry_config.jitter_ratio,
            retryable_exceptions=self.retry_config.retryable_exceptions,
            retryable_predicate=self.retry_config.retryable_predicate,
            operation_name=operation_name,
        )
        return execute_with_retry(
            func,
            config=retry_config,
            logger=self.logger,
        )
