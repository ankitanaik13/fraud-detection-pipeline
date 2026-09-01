from unittest.mock import MagicMock, patch

import pytest

from src.storage.db_writer import ensure_schema, jdbc_config, write_batch_idempotent


def test_jdbc_config_decodes_credentials():
    url, props = jdbc_config("postgresql://user:p%40ss@db.example:5544/fraud")
    assert url == "jdbc:postgresql://db.example:5544/fraud"
    assert props["user"] == "user"
    assert props["password"] == "p@ss"


def test_jdbc_config_rejects_non_postgres_urls():
    with pytest.raises(ValueError, match="valid postgres"):
        jdbc_config("sqlite:///tmp/test.db")


def fake_connection():
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = connection.cursor.return_value
    return connection


def test_ensure_schema_creates_tables_and_indexes():
    connection = fake_connection()
    with patch("src.storage.db_writer.psycopg2.connect", return_value=connection):
        ensure_schema("postgresql://user:pass@db/fraud")
    assert connection.cursor.return_value.execute.call_count == 12


def test_idempotent_writer_stages_merges_and_drops():
    batch_df = MagicMock()
    connection = fake_connection()
    with patch("src.storage.db_writer.psycopg2.connect", return_value=connection):
        write_batch_idempotent(
            batch_df,
            batch_id=7,
            postgres_url="postgresql://user:pass@db/fraud",
            jdbc_url="jdbc:postgresql://db/fraud",
            jdbc_props={"user": "user"},
        )

    batch_df.write.jdbc.assert_called_once()
    assert connection.cursor.return_value.execute.call_count == 3


def test_idempotent_writer_rejects_invalid_batch_id():
    with pytest.raises(ValueError, match="non-negative"):
        write_batch_idempotent(
            MagicMock(),
            batch_id=-1,
            postgres_url="postgresql://user:pass@db/fraud",
            jdbc_url="jdbc:postgresql://db/fraud",
            jdbc_props={},
        )
