import pytest
import asyncio
import os
from unittest.mock import Mock, patch
from typing import Dict, Any, List
from sqlalchemy import text
from fastapi.testclient import TestClient

# Import application modules
from app.database.db_connector import DatabaseConnector
from app.agent.db_agent_connector import DBAgentConnector, ConversationSession
from app.api.routes import app, get_db_agent


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_db_config():
    """Test database configuration."""
    return {
        "host": os.getenv("TEST_DB_HOST", "localhost"),
        "port": os.getenv("TEST_DB_PORT", "5432"),
        "user": os.getenv("TEST_DB_USER", "postgres"),
        "password": os.getenv("TEST_DB_PASSWORD", "postgres"),
        "database": os.getenv("TEST_DB_NAME", "test_db")
    }


@pytest.fixture
def test_connection_string(test_db_config):
    """Create test database connection string."""
    return f"postgresql://{test_db_config['user']}:{test_db_config['password']}@{test_db_config['host']}:{test_db_config['port']}/{test_db_config['database']}"


@pytest.fixture
def mock_db_connector():
    """Mock database connector for unit tests."""
    mock_connector = Mock(spec=DatabaseConnector)
    mock_connector.test_connection.return_value = True
    mock_connector.get_table_names.return_value = ["actor", "film", "customer", "rental"]
    mock_connector.get_table_schema.return_value = {
        "table_name": "actor",
        "columns": [
            {"name": "actor_id", "type": "INTEGER", "nullable": False},
            {"name": "first_name", "type": "VARCHAR(45)", "nullable": False},
            {"name": "last_name", "type": "VARCHAR(45)", "nullable": False}
        ],
        "primary_keys": ["actor_id"],
        "foreign_keys": [],
        "indices": []
    }
    mock_connector.execute_query.return_value = {
        "success": True,
        "data": [
            {"actor_id": 1, "first_name": "John", "last_name": "Doe"},
            {"actor_id": 2, "first_name": "Jane", "last_name": "Smith"}
        ],
        "affected_rows": 2
    }
    mock_connector.get_database_schema.return_value = {
        "actor": {
            "table_name": "actor",
            "columns": [
                {"name": "actor_id", "type": "INTEGER", "nullable": False},
                {"name": "first_name", "type": "VARCHAR(45)", "nullable": False},
                {"name": "last_name", "type": "VARCHAR(45)", "nullable": False}
            ],
            "primary_keys": ["actor_id"],
            "foreign_keys": [],
            "indices": []
        }
    }
    return mock_connector


@pytest.fixture
def sample_actor_data():
    """Sample actor data for testing."""
    return [
        {"actor_id": 1, "first_name": "John", "last_name": "Doe"},
        {"actor_id": 2, "first_name": "Jane", "last_name": "Smith"},
        {"actor_id": 3, "first_name": "Bob", "last_name": "Johnson"}
    ]


@pytest.fixture
def sample_film_data():
    """Sample film data for testing."""
    return [
        {"film_id": 1, "title": "Test Movie", "rating": "PG-13", "length": 120},
        {"film_id": 2, "title": "Another Film", "rating": "R", "length": 90}
    ]


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for agent testing."""
    mock_response = Mock()
    mock_response.content = "SELECT * FROM actor WHERE first_name = 'John'"
    return mock_response


@pytest.fixture
def mock_agent_result():
    """Mock agent result for testing."""
    return {
        "response": "I found 2 actors named John in the database.",
        "context": {
            "sql_query": "SELECT * FROM actor WHERE first_name = 'John'",
            "execution_plan": "select operation on actor table",
            "operation_details": "Retrieve actors with first name John"
        },
        "execution_details": {
            "success": True,
            "operation": "select",
            "data": []
        }
    }


@pytest.fixture
def conversation_session():
    """Create a test conversation session."""
    session = ConversationSession("test_session")
    return session


@pytest.fixture
def mock_db_agent_connector(mock_db_connector):
    """Mock DB agent connector with properly configured database schema."""
    # Create a mock that behaves like DBAgentConnector but with mocked dependencies
    mock_agent = Mock(spec=DBAgentConnector)

    # Setup the db_connector attribute
    mock_agent.db_connector = mock_db_connector

    # Setup database_schema as a dictionary, not a Mock
    mock_agent.database_schema = {
        "actor": {
            "table_name": "actor",
            "columns": [
                {"name": "actor_id", "type": "INTEGER", "nullable": False},
                {"name": "first_name", "type": "VARCHAR(45)", "nullable": False},
                {"name": "last_name", "type": "VARCHAR(45)", "nullable": False}
            ],
            "primary_keys": ["actor_id"],
            "foreign_keys": [],
            "indices": []
        }
    }

    # Setup conversation sessions
    mock_agent.conversation_sessions = {}

    # Setup method return values
    mock_agent.execute_natural_language_query.return_value = {
        "success": True,
        "agent_response": "Test response",
        "data": [{"actor_id": 1, "first_name": "John"}],
        "affected_rows": 1,
        "session_id": "test_session"
    }

    mock_agent.get_session_info.return_value = {
        "session_id": "test_session",
        "created_at": "2024-01-01T12:00:00",
        "last_activity": "2024-01-01T12:30:00",
        "query_count": 1,
        "last_table": "actor",
        "last_operation": "select",
        "context_summary": "Test session"
    }

    mock_agent.get_active_sessions.return_value = ["test_session"]
    mock_agent.clear_session.return_value = True

    return mock_agent


@pytest.fixture
def test_client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_agent_dependency(mock_db_agent_connector):
    """Mock the DB agent dependency for API testing."""

    def override_get_db_agent():
        return mock_db_agent_connector

    app.dependency_overrides[get_db_agent] = override_get_db_agent
    yield mock_db_agent_connector
    app.dependency_overrides.clear()


@pytest.fixture
def sakila_schema():
    """Sakila database schema for testing."""
    return {
        "actor": {
            "table_name": "actor",
            "columns": [
                {"name": "actor_id", "type": "INTEGER", "nullable": False},
                {"name": "first_name", "type": "VARCHAR(45)", "nullable": False},
                {"name": "last_name", "type": "VARCHAR(45)", "nullable": False},
                {"name": "last_update", "type": "TIMESTAMP", "nullable": False}
            ],
            "primary_keys": ["actor_id"],
            "foreign_keys": [],
            "indices": []
        },
        "film": {
            "table_name": "film",
            "columns": [
                {"name": "film_id", "type": "INTEGER", "nullable": False},
                {"name": "title", "type": "VARCHAR(255)", "nullable": False},
                {"name": "description", "type": "TEXT", "nullable": True},
                {"name": "release_year", "type": "INTEGER", "nullable": True},
                {"name": "rating", "type": "VARCHAR(10)", "nullable": True},
                {"name": "length", "type": "INTEGER", "nullable": True}
            ],
            "primary_keys": ["film_id"],
            "foreign_keys": [],
            "indices": []
        }
    }


@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables."""
    test_env = {
        "GOOGLE_API_KEY": "test_api_key",
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_USER": "test_user",
        "DB_PASSWORD": "test_pass",
        "DB_NAME": "test_db"
    }

    with patch.dict(os.environ, test_env):
        yield


class DatabaseTestHelper:
    """Helper class for database testing operations."""

    @staticmethod
    def create_test_tables(engine):
        """Create test tables for integration testing."""
        with engine.connect() as conn:
            conn.execute(text("""
                              CREATE TABLE IF NOT EXISTS test_actor
                              (
                                  actor_id
                                  SERIAL
                                  PRIMARY
                                  KEY,
                                  first_name
                                  VARCHAR
                              (
                                  45
                              ) NOT NULL,
                                  last_name VARCHAR
                              (
                                  45
                              ) NOT NULL,
                                  last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                  )
                              """))
            conn.commit()

    @staticmethod
    def insert_test_data(engine, table_name: str, data: List[Dict[str, Any]]):
        """Insert test data into specified table."""
        with engine.connect() as conn:
            for row in data:
                columns = ", ".join(row.keys())
                placeholders = ", ".join([f":{k}" for k in row.keys()])
                query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                conn.execute(text(query), row)
            conn.commit()

    @staticmethod
    def cleanup_test_tables(engine):
        """Clean up test tables after testing."""
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS test_actor CASCADE"))
            conn.commit()


@pytest.fixture
def db_test_helper():
    """Database test helper fixture."""
    return DatabaseTestHelper
