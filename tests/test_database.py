import pytest
from unittest.mock import Mock, patch
from sqlalchemy.exc import SQLAlchemyError
from app.database.db_connector import DatabaseConnector


class TestDatabaseConnector:
    """Test cases for DatabaseConnector class."""

    def test_init_with_connection_string(self):
        """Test initialization with explicit connection string."""
        conn_str = "postgresql://user:pass@localhost:5432/testdb"
        connector = DatabaseConnector(conn_str)
        assert connector.connection_string == conn_str

    @patch.dict('os.environ', {
        'DB_USER': 'test_user',
        'DB_PASSWORD': 'test_pass',
        'DB_HOST': 'test_host',
        'DB_PORT': '5432',
        'DB_NAME': 'test_db'
    })
    def test_init_with_environment_variables(self):
        """Test initialization using environment variables."""
        connector = DatabaseConnector()
        expected = "postgresql://test_user:test_pass@test_host:5432/test_db"
        assert connector.connection_string == expected

    @patch('app.database.db_connector.create_engine')
    def test_test_connection_success(self, mock_create_engine):
        """Test successful database connection."""
        mock_engine = Mock()
        mock_conn = Mock()
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        mock_create_engine.return_value = mock_engine

        connector = DatabaseConnector("test://connection")
        result = connector.test_connection()

        assert result is True
        mock_conn.execute.assert_called_once()

    @patch('app.database.db_connector.create_engine')
    def test_test_connection_failure(self, mock_create_engine):
        """Test database connection failure."""
        mock_engine = Mock()
        mock_engine.connect.side_effect = SQLAlchemyError("Connection failed")
        mock_create_engine.return_value = mock_engine

        connector = DatabaseConnector("test://connection")
        result = connector.test_connection()

        assert result is False

    @patch('app.database.db_connector.create_engine')
    def test_execute_query_select_success(self, mock_create_engine):
        """Test successful SELECT query execution."""
        # Setup mocks
        mock_engine = Mock()
        mock_conn = Mock()
        mock_result = Mock()

        mock_result.returns_rows = True
        mock_result.keys.return_value = ['id', 'name']
        mock_result.fetchall.return_value = [(1, 'John'), (2, 'Jane')]
        mock_result.rowcount = 2

        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        mock_create_engine.return_value = mock_engine

        connector = DatabaseConnector("test://connection")
        result = connector.execute_query("SELECT * FROM users")

        assert result["success"] is True
        assert len(result["data"]) == 2
        assert result["data"][0] == {"id": 1, "name": "John"}
        assert result["affected_rows"] == 2

    @patch('app.database.db_connector.create_engine')
    def test_execute_query_insert_success(self, mock_create_engine):
        """Test successful INSERT query execution."""
        mock_engine = Mock()
        mock_conn = Mock()
        mock_result = Mock()

        mock_result.returns_rows = False
        mock_result.rowcount = 1

        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        mock_create_engine.return_value = mock_engine

        connector = DatabaseConnector("test://connection")
        result = connector.execute_query("INSERT INTO users (name) VALUES ('John')")

        assert result["success"] is True
        assert "data" not in result
        assert result["affected_rows"] == 1

    @patch('app.database.db_connector.create_engine')
    def test_execute_query_with_parameters(self, mock_create_engine):
        """Test query execution with parameters."""
        mock_engine = Mock()
        mock_conn = Mock()
        mock_result = Mock()

        mock_result.returns_rows = True
        mock_result.keys.return_value = ['id', 'name']
        mock_result.fetchall.return_value = [(1, 'John')]
        mock_result.rowcount = 1

        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        mock_create_engine.return_value = mock_engine

        connector = DatabaseConnector("test://connection")
        result = connector.execute_query(
            "SELECT * FROM users WHERE name = :name",
            {"name": "John"}
        )

        assert result["success"] is True

    @patch('app.database.db_connector.create_engine')
    def test_execute_query_failure(self, mock_create_engine):
        """Test query execution failure."""
        mock_engine = Mock()
        mock_conn = Mock()
        mock_conn.execute.side_effect = SQLAlchemyError("Query failed")
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        mock_create_engine.return_value = mock_engine

        connector = DatabaseConnector("test://connection")
        result = connector.execute_query("INVALID SQL")

        assert result["success"] is False
        assert "error" in result

    @patch('app.database.db_connector.inspect')
    @patch('app.database.db_connector.create_engine')
    def test_get_table_names(self, mock_create_engine, mock_inspect):
        """Test getting table names."""
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["users", "products", "orders"]
        mock_inspect.return_value = mock_inspector

        connector = DatabaseConnector("test://connection")
        tables = connector.get_table_names()

        assert tables == ["users", "products", "orders"]

    @patch('app.database.db_connector.inspect')
    @patch('app.database.db_connector.create_engine')
    def test_get_table_schema(self, mock_create_engine, mock_inspect):
        """Test getting table schema."""
        mock_inspector = Mock()
        mock_inspector.get_columns.return_value = [
            {"name": "id", "type": "INTEGER", "nullable": False, "default": None},
            {"name": "name", "type": "VARCHAR(255)", "nullable": True, "default": None}
        ]
        mock_inspector.get_pk_constraint.return_value = {"constrained_columns": ["id"]}
        mock_inspector.get_foreign_keys.return_value = []
        mock_inspector.get_indexes.return_value = []
        mock_inspect.return_value = mock_inspector

        connector = DatabaseConnector("test://connection")
        schema = connector.get_table_schema("users")

        assert schema["table_name"] == "users"
        assert len(schema["columns"]) == 2
        assert schema["primary_keys"] == ["id"]
        assert schema["foreign_keys"] == []

    @patch('app.database.db_connector.inspect')
    @patch('app.database.db_connector.create_engine')
    def test_get_table_schema_with_cache(self, mock_create_engine, mock_inspect):
        """Test table schema caching."""
        mock_inspector = Mock()
        mock_inspect.return_value = mock_inspector

        connector = DatabaseConnector("test://connection")
        connector._metadata_cache["users"] = {"cached": True}

        schema = connector.get_table_schema("users")

        assert schema == {"cached": True}
        mock_inspector.get_columns.assert_not_called()

    @patch('app.database.db_connector.create_engine')
    def test_execute_operation_select(self, mock_create_engine):
        """Test execute_operation with SELECT."""
        # Create a real DatabaseConnector instance with mocked engine
        mock_engine = Mock()
        mock_conn = Mock()
        mock_result = Mock()

        mock_result.returns_rows = True
        mock_result.keys.return_value = ['id', 'name']
        mock_result.fetchall.return_value = [(1, 'John')]
        mock_result.rowcount = 1

        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        mock_create_engine.return_value = mock_engine

        connector = DatabaseConnector("test://connection")

        params = {
            "table": "users",
            "columns": ["id", "name"],
            "where": "age > 18",
            "order_by": "name",
            "limit": "10"
        }

        result = connector.execute_operation("select", params)

        assert result["success"] is True
        assert len(result["data"]) == 1

    @patch('app.database.db_connector.create_engine')
    def test_execute_operation_unsupported(self, mock_create_engine):
        """Test execute_operation with unsupported operation."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        connector = DatabaseConnector("test://connection")
        result = connector.execute_operation("unsupported", {})

        assert result["success"] is False
        assert "Unsupported operation type" in result["error"]


@pytest.mark.integration
class TestDatabaseConnectorIntegration:
    """Integration tests for DatabaseConnector with real database."""

    @pytest.fixture(autouse=True)
    def setup_test_db(self, test_connection_string, db_test_helper):
        """Setup test database for integration tests."""
        try:
            from sqlalchemy import create_engine
            engine = create_engine(test_connection_string)
            db_test_helper.create_test_tables(engine)
            yield engine
        except Exception as e:
            pytest.skip(f"Test database not available: {e}")
        finally:
            try:
                db_test_helper.cleanup_test_tables(engine)
            except:
                pass

    def test_real_database_operations(self, test_connection_string, sample_actor_data, setup_test_db):
        """Test real database operations."""
        connector = DatabaseConnector(test_connection_string)

        # Test connection
        assert connector.test_connection() is True

        # Insert test data
        for actor in sample_actor_data:
            result = connector.execute_operation("insert", {
                "table": "test_actor",
                "values": {
                    "first_name": actor["first_name"],
                    "last_name": actor["last_name"]
                }
            })
            assert result["success"] is True

        # Query data
        result = connector.execute_operation("select", {
            "table": "test_actor",
            "columns": ["*"]
        })
        assert result["success"] is True
        assert len(result["data"]) == len(sample_actor_data)
