import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from app.agent.db_agent_connector import (
    DBAgentConnector,
    ConversationSession
)


class TestConversationSession:
    """Test cases for ConversationSession class."""

    def test_session_initialization(self):
        """Test ConversationSession initialization."""
        session = ConversationSession("test_session", max_history=5)

        assert session.session_id == "test_session"
        assert session.history == []
        assert session.context == {}
        assert session.last_table is None
        assert session.last_operation is None
        assert session.max_history == 5
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)

    def test_add_query_success(self):
        """Test adding successful query to session."""
        session = ConversationSession("test_session")

        query = "SELECT * FROM actor"
        result = {
            "success": True,
            "sql_query": "SELECT * FROM actor",
            "data": [{"actor_id": 1, "name": "John"}],
            "affected_rows": 1,
            "operation_type": "select"
        }

        session.add_query(query, result)

        assert len(session.history) == 1
        assert session.history[0]["query"] == query
        assert session.history[0]["success"] is True
        assert session.last_table == "actor"
        assert session.last_operation == "select"

    def test_add_query_failure(self):
        """Test adding failed query to session."""
        session = ConversationSession("test_session")

        query = "INVALID SQL"
        result = {
            "success": False,
            "error": "Syntax error"
        }

        session.add_query(query, result)

        assert len(session.history) == 1
        assert session.history[0]["success"] is False

    def test_history_limit(self):
        """Test that session history respects max_history limit."""
        session = ConversationSession("test_session", max_history=3)

        # Add more queries than the limit
        for i in range(5):
            session.add_query(f"query {i}", {"success": True})

        assert len(session.history) == 3
        # Should keep the last 3 queries
        assert session.history[0]["query"] == "query 2"
        assert session.history[2]["query"] == "query 4"

    def test_extract_table_from_sql(self):
        """Test table extraction from SQL queries."""
        test_cases = [
            ("SELECT * FROM actor", "actor"),
            ("SELECT * FROM film WHERE rating = 'PG'", "film"),
            ("INSERT INTO customer VALUES (1, 'John')", "customer"),
            ("UPDATE rental SET return_date = NOW()", "rental")
        ]

        # Test each case with a fresh session to avoid state pollution
        for sql, expected_table in test_cases:
            session = ConversationSession("test_session")  # Fresh session for each test
            session._extract_table_from_sql(sql)
            assert session.last_table == expected_table, f"Failed for SQL: {sql}, expected: {expected_table}, got: {session.last_table}"

    def test_extract_filter_from_sql(self):
        """Test filter extraction from SQL queries."""
        session = ConversationSession("test_session")

        test_cases = [
            ("SELECT * FROM actor WHERE first_name = 'John'", "first_name = 'john'"),
            ("SELECT * FROM film WHERE rating = 'PG' AND length > 120", "rating = 'pg' and length > 120"),
            ("SELECT * FROM customer", None)
        ]

        for sql, expected_filter in test_cases:
            # Reset filter state for each test
            session.last_filter = None
            session._extract_filter_from_sql(sql)
            if expected_filter:
                assert expected_filter in session.last_filter.lower()
            else:
                assert session.last_filter is None

    def test_get_context_summary(self):
        """Test context summary generation."""
        session = ConversationSession("test_session")

        # Empty session
        summary = session.get_context_summary()
        assert "No previous conversation history" in summary

        # Session with history
        session.last_table = "actor"
        session.last_operation = "select"
        session.add_query("Show actors", {"success": True})

        summary = session.get_context_summary()
        assert "actor" in summary
        assert "select" in summary

    def test_session_expiration(self):
        """Test session expiration logic."""
        session = ConversationSession("test_session")

        # Fresh session should not be expired
        assert not session.is_expired(60)

        # Manually set old last_activity
        session.last_activity = datetime.now() - timedelta(minutes=90)
        assert session.is_expired(60)


class TestDBAgentConnector:
    """Test cases for DBAgentConnector class."""

    @patch('app.agent.db_agent_connector.DatabaseConnector')
    def test_init(self, mock_db_connector_class):
        """Test DBAgentConnector initialization."""
        mock_db_connector = Mock()
        mock_db_connector.get_database_schema.return_value = {"actor": {}}
        mock_db_connector_class.return_value = mock_db_connector

        connector = DBAgentConnector("test://connection")

        assert connector.db_connector == mock_db_connector
        assert connector.conversation_sessions == {}
        assert connector.database_schema == {"actor": {}}

    @patch('app.agent.db_agent_connector.DatabaseConnector')
    def test_get_or_create_session_new(self, mock_db_connector_class):
        """Test creating a new session."""
        mock_db_connector = Mock()
        mock_db_connector.get_database_schema.return_value = {}
        mock_db_connector_class.return_value = mock_db_connector

        connector = DBAgentConnector("test://connection")

        session = connector._get_or_create_session("new_session")

        assert session.session_id == "new_session"
        assert "new_session" in connector.conversation_sessions

    @patch('app.agent.db_agent_connector.DatabaseConnector')
    def test_get_or_create_session_existing(self, mock_db_connector_class):
        """Test getting an existing session."""
        mock_db_connector = Mock()
        mock_db_connector.get_database_schema.return_value = {}
        mock_db_connector_class.return_value = mock_db_connector

        connector = DBAgentConnector("test://connection")

        # Create session first
        first_session = connector._get_or_create_session("existing_session")
        first_session.last_table = "actor"

        # Get same session
        second_session = connector._get_or_create_session("existing_session")

        assert first_session is second_session
        assert second_session.last_table == "actor"

    @pytest.mark.parametrize("query,expected", [
        ("show tables", True),
        ("list tables", True),
        ("what tables are there", True),
        ("show me all actors", False),
        ("that table", False),
        ("from this table", False)
    ])
    @patch('app.agent.db_agent_connector.DatabaseConnector')
    def test_is_table_listing_query(self, mock_db_connector_class, query, expected):
        """Test table listing query detection."""
        mock_db_connector = Mock()
        mock_db_connector.get_database_schema.return_value = {}
        mock_db_connector_class.return_value = mock_db_connector

        connector = DBAgentConnector("test://connection")
        result = connector._is_table_listing_query(query)
        assert result == expected

    @patch('app.agent.db_agent_connector.DatabaseConnector')
    def test_resolve_contextual_references(self, mock_db_connector_class):
        """Test contextual reference resolution."""
        mock_db_connector = Mock()
        mock_db_connector.get_database_schema.return_value = {}
        mock_db_connector_class.return_value = mock_db_connector

        connector = DBAgentConnector("test://connection")
        session = ConversationSession("test_session")
        session.last_table = "actor"
        session.last_filter = "first_name = 'John'"

        test_cases = [
            ("show me that table", "show me the actor table"),
            ("how many?", "Count records in the actor table where first_name = 'John'"),
            ("show me the first 5", "Show me the first 5 records from the actor table where first_name = 'John'")
        ]

        for input_query, expected_output in test_cases:
            result = connector._resolve_contextual_references(input_query, session)
            assert "actor" in result.lower()

    @patch('app.agent.db_agent_connector.DatabaseConnector')
    def test_create_context_aware_prompt(self, mock_db_connector_class, sakila_schema):
        """Test context-aware prompt creation."""
        mock_db_connector = Mock()
        mock_db_connector.get_database_schema.return_value = {}
        mock_db_connector_class.return_value = mock_db_connector

        connector = DBAgentConnector("test://connection")
        session = ConversationSession("test_session")
        session.last_table = "actor"
        session.add_query("Show actors", {"success": True})

        result = connector._create_context_aware_prompt(
            "Show more actors", session, sakila_schema
        )

        assert "conversation_context" in result
        assert result["conversation_context"]["last_table"] == "actor"

    @patch('app.agent.db_agent_connector.DatabaseConnector')
    def test_extract_sql_from_response(self, mock_db_connector_class):
        """Test SQL extraction from agent responses."""
        mock_db_connector = Mock()
        mock_db_connector.get_database_schema.return_value = {}
        mock_db_connector_class.return_value = mock_db_connector

        connector = DBAgentConnector("test://connection")

        test_cases = [
            # SQL in code block
            ("Here's the query:\n```sql\nSELECT * FROM actor\n```", {}, "SELECT * FROM actor"),
            # SQL in context
            ("", {"sql_query": "SELECT * FROM film"}, "SELECT * FROM film"),
            # No SQL found
            ("Just a regular response", {}, None)
        ]

        for response, context, expected in test_cases:
            result = connector._extract_sql_from_response(response, context)
            assert result == expected

    @patch('app.agent.db_agent_connector.DatabaseConnector')
    def test_get_session_info(self, mock_db_connector_class):
        """Test getting session information."""
        mock_db_connector = Mock()
        mock_db_connector.get_database_schema.return_value = {}
        mock_db_connector_class.return_value = mock_db_connector

        connector = DBAgentConnector("test://connection")
        session = connector._get_or_create_session("test_session")
        session.last_table = "actor"
        session.add_query("test", {"success": True})

        info = connector.get_session_info("test_session")

        assert info["session_id"] == "test_session"
        assert info["query_count"] == 1
        assert info["last_table"] == "actor"

    @patch('app.agent.db_agent_connector.DatabaseConnector')
    def test_get_session_info_not_found(self, mock_db_connector_class):
        """Test getting info for non-existent session."""
        mock_db_connector = Mock()
        mock_db_connector.get_database_schema.return_value = {}
        mock_db_connector_class.return_value = mock_db_connector

        connector = DBAgentConnector("test://connection")

        info = connector.get_session_info("nonexistent_session")

        assert "error" in info

    @patch('app.agent.db_agent_connector.DatabaseConnector')
    def test_clear_session(self, mock_db_connector_class):
        """Test clearing a session."""
        mock_db_connector = Mock()
        mock_db_connector.get_database_schema.return_value = {}
        mock_db_connector_class.return_value = mock_db_connector

        connector = DBAgentConnector("test://connection")
        connector._get_or_create_session("test_session")

        assert "test_session" in connector.conversation_sessions

        result = connector.clear_session("test_session")

        assert result is True
        assert "test_session" not in connector.conversation_sessions

    @patch('app.agent.db_agent_connector.DatabaseConnector')
    def test_clear_session_not_found(self, mock_db_connector_class):
        """Test clearing non-existent session."""
        mock_db_connector = Mock()
        mock_db_connector.get_database_schema.return_value = {}
        mock_db_connector_class.return_value = mock_db_connector

        connector = DBAgentConnector("test://connection")

        result = connector.clear_session("nonexistent_session")

        assert result is False

    @patch('app.agent.db_agent_connector.DatabaseConnector')
    def test_get_active_sessions(self, mock_db_connector_class):
        """Test getting active sessions list."""
        mock_db_connector = Mock()
        mock_db_connector.get_database_schema.return_value = {}
        mock_db_connector_class.return_value = mock_db_connector

        connector = DBAgentConnector("test://connection")
        connector._get_or_create_session("session1")
        connector._get_or_create_session("session2")

        sessions = connector.get_active_sessions()

        assert len(sessions) == 2
        assert "session1" in sessions
        assert "session2" in sessions
