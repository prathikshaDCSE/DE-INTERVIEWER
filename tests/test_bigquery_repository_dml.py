import pytest
from unittest.mock import MagicMock, patch

from repository.bigquery_repository import (
    BigQueryRepository,
    ValidationError,
)


# ----------------------------------------------------------
# Fixtures
# ----------------------------------------------------------

@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def repo(mock_client):
    return BigQueryRepository(
        project="test-project",
        dataset="test_dataset",
        client=mock_client,
    )


# ----------------------------------------------------------
# execute_query()
# ----------------------------------------------------------

def test_execute_query_success(repo):
    repo._execute_query_with_client = MagicMock(
        return_value=[{"id": 1, "name": "Alice"}]
    )

    result = repo.execute_query("SELECT * FROM users")

    assert len(result) == 1
    assert result[0]["name"] == "Alice"


def test_execute_query_empty_sql(repo):
    with pytest.raises(ValidationError):
        repo.execute_query("")


def test_execute_query_transaction(repo):
    repo._transaction_active = True
    repo._execute_query_in_transaction = MagicMock(
        return_value=[{"id": 1}]
    )

    result = repo.execute_query("SELECT 1")

    assert result == [{"id": 1}]
    repo._execute_query_in_transaction.assert_called_once()


# ----------------------------------------------------------
# insert()
# ----------------------------------------------------------

def test_insert_calls_insert_many(repo):
    repo.insert_many = MagicMock(return_value=1)

    row = {
        "user_id": "U001",
        "name": "Alice"
    }

    result = repo.insert("users", row)

    assert result == 1
    repo.insert_many.assert_called_once()


def test_insert_invalid_payload(repo):
    with pytest.raises(ValidationError):
        repo.insert("users", [])


# ----------------------------------------------------------
# insert_many()
# ----------------------------------------------------------

def test_insert_many_empty(repo):
    assert repo.insert_many("users", []) == 0


def test_insert_many_invalid(repo):
    with pytest.raises(ValidationError):
        repo.insert_many("users", [1, 2, 3])


def test_insert_many_success(repo, mock_client):

    table_ref = MagicMock()

    repo._resolve_table_reference = MagicMock(
        return_value=table_ref
    )

    repo._validate_payload_schema = MagicMock()

    mock_client.insert_rows_json.return_value = []

    rows = [
        {
            "user_id": "U001",
            "name": "Alice"
        }
    ]

    result = repo.insert_many(
        "users",
        rows,
    )

    assert result == 1

    mock_client.insert_rows_json.assert_called_once()


# ----------------------------------------------------------
# update()
# ----------------------------------------------------------

def test_update_success(repo):

    repo._execute_dml = MagicMock(
        return_value=3
    )

    result = repo.update(
        "UPDATE users SET name=@name",
        {"name": "Bob"},
    )

    assert result == 3


def test_update_empty_sql(repo):

    with pytest.raises(ValidationError):
        repo.update("")


# ----------------------------------------------------------
# delete()
# ----------------------------------------------------------

def test_delete_success(repo):

    repo._execute_dml = MagicMock(
        return_value=2
    )

    result = repo.delete(
        "DELETE FROM users WHERE id=@id",
        {"id": "1"},
    )

    assert result == 2


def test_delete_empty_sql(repo):

    with pytest.raises(ValidationError):
        repo.delete("")


# ----------------------------------------------------------
# upsert()
# ----------------------------------------------------------

def test_upsert_success(repo):

    repo._resolve_table_reference = MagicMock(
        return_value=MagicMock()
    )

    repo._validate_payload_schema = MagicMock()

    repo._build_merge_statement = MagicMock(
        return_value=("MERGE SQL", [])
    )

    repo._execute_dml = MagicMock(
        return_value=1
    )

    payload = {
        "user_id": "U001",
        "name": "Alice"
    }

    result = repo.upsert(
        table="users",
        key_columns=["user_id"],
        payload=payload,
    )

    assert result == 1


def test_upsert_empty_payload(repo):

    result = repo.upsert(
        table="users",
        key_columns=["user_id"],
        payload=[],
    )

    assert result == 0


def test_upsert_invalid_key(repo):

    with pytest.raises(ValidationError):

        repo.upsert(
            table="users",
            key_columns=[],
            payload={
                "user_id": "1"
            },
        )


# ----------------------------------------------------------
# begin_transaction()
# ----------------------------------------------------------

@patch("repository.bigquery_repository.dbapi.connect")
def test_begin_transaction(mock_connect, repo):

    connection = MagicMock()

    mock_connect.return_value = connection

    repo.begin_transaction()

    assert repo._transaction_active is True

    mock_connect.assert_called_once()


# ----------------------------------------------------------
# commit()
# ----------------------------------------------------------

def test_commit(repo):

    connection = MagicMock()

    repo._transaction_connection = connection

    repo._transaction_active = True

    repo.commit()

    connection.commit.assert_called_once()

    assert repo._transaction_active is False


# ----------------------------------------------------------
# rollback()
# ----------------------------------------------------------

def test_rollback(repo):

    connection = MagicMock()

    repo._transaction_connection = connection

    repo._transaction_active = True

    repo.rollback()

    connection.rollback.assert_called_once()

    assert repo._transaction_active is False


# ----------------------------------------------------------
# transaction validation
# ----------------------------------------------------------

def test_commit_without_transaction(repo):

    with pytest.raises(ValidationError):
        repo.commit()


def test_rollback_without_transaction(repo):

    with pytest.raises(ValidationError):
        repo.rollback()


def test_begin_transaction_twice(repo):

    repo._transaction_active = True

    with pytest.raises(ValidationError):
        repo.begin_transaction()