import pytest
from unittest.mock import MagicMock

from repository.bigquery_repository import (
    BigQueryRepository,
    ValidationError,
)


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def repository(mock_client):
    return BigQueryRepository(
        project="test-project",
        dataset="test_dataset",
        client=mock_client,
    )


# -----------------------------------------------------
# Initialization
# -----------------------------------------------------

def test_repository_initialization(repository):
    assert repository.project == "test-project"
    assert repository.dataset == "test_dataset"


# -----------------------------------------------------
# Connection
# -----------------------------------------------------

def test_connect_returns_existing_client(repository, mock_client):
    client = repository.connect()
    assert client == mock_client


def test_disconnect(repository, mock_client):
    repository.disconnect()
    mock_client.close.assert_called_once()


# -----------------------------------------------------
# Validation
# -----------------------------------------------------

def test_execute_query_empty_sql(repository):
    with pytest.raises(ValidationError):
        repository.execute_query("")


def test_invalid_table_name(repository):
    with pytest.raises(ValidationError):
        repository._validate_table_name("")


def test_invalid_dataset_name(repository):
    with pytest.raises(ValidationError):
        repository._validate_dataset_name("")


def test_invalid_project_name(repository):
    with pytest.raises(ValidationError):
        repository._validate_project_name("")


# -----------------------------------------------------
# Insert Validation
# -----------------------------------------------------

def test_insert_invalid_payload(repository):
    with pytest.raises(ValidationError):
        repository.insert("users", [])


def test_insert_many_empty(repository):
    assert repository.insert_many("users", []) == 0


# -----------------------------------------------------
# Query Parameters
# -----------------------------------------------------

def test_build_query_parameter_int(repository):
    parameter = repository._build_query_parameter("age", 25)

    assert parameter.name == "age"
    assert parameter.type_ == "INT64"


def test_build_query_parameter_string(repository):
    parameter = repository._build_query_parameter("name", "John")

    assert parameter.name == "name"
    assert parameter.type_ == "STRING"


def test_build_query_parameter_bool(repository):
    parameter = repository._build_query_parameter("active", True)

    assert parameter.type_ == "BOOL"


# -----------------------------------------------------
# Convert Rows
# -----------------------------------------------------

def test_convert_rows_to_dict(repository):
    rows = [{"id": 1, "name": "Alice"}]

    result = repository._convert_rows_to_dict(rows)

    assert result == rows


# -----------------------------------------------------
# Count
# -----------------------------------------------------

def test_count_rows(repository):
    repository.fetch_one = MagicMock(return_value={"row_count": 15})

    count = repository.count_rows("users")

    assert count == 15


# -----------------------------------------------------
# Dataset Exists
# -----------------------------------------------------

def test_dataset_exists(repository, mock_client):
    mock_client.get_dataset.return_value = MagicMock()

    assert repository.dataset_exists("test_dataset") is True


# -----------------------------------------------------
# Table Exists
# -----------------------------------------------------

def test_table_exists(repository, mock_client):
    mock_client.get_table.return_value = MagicMock()

    assert repository.table_exists("users") is True


# -----------------------------------------------------
# List Tables
# -----------------------------------------------------

def test_list_tables(repository, mock_client):
    table = MagicMock()
    table.table_id = "users"

    mock_client.list_tables.return_value = [table]

    tables = repository.list_tables("test_dataset")

    assert tables == ["users"]


# -----------------------------------------------------
# Fetch One
# -----------------------------------------------------

def test_fetch_one(repository):
    repository.execute_query = MagicMock(
        return_value=[{"id": 1}]
    )

    result = repository.fetch_one("SELECT *")

    assert result == {"id": 1}


# -----------------------------------------------------
# Fetch All
# -----------------------------------------------------

def test_fetch_all(repository):
    repository.execute_query = MagicMock(
        return_value=[{"id": 1}, {"id": 2}]
    )

    result = repository.fetch_all("SELECT *")

    assert len(result) == 2


# -----------------------------------------------------
# Health Check
# -----------------------------------------------------

def test_health_check(repository):
    repository.dataset_exists = MagicMock(return_value=True)
    repository.list_tables = MagicMock(return_value=["users"])

    result = repository.health_check()

    assert result["status"] == "healthy"


# -----------------------------------------------------
# Schema
# -----------------------------------------------------

def test_get_schema(repository, mock_client):
    field = MagicMock()
    field.name = "id"
    field.field_type = "STRING"
    field.mode = "NULLABLE"
    field.description = None

    table = MagicMock()
    table.schema = [field]

    mock_client.get_table.return_value = table

    schema = repository.get_schema("users")

    assert len(schema) == 1
    assert schema[0].name == "id"