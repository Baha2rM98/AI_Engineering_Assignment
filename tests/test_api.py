from unittest.mock import Mock


class TestAPIRoutes:
    """Test cases for API routes."""

    def test_read_root(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "LangGraph Database Agent" in data["name"]

    def test_health_check_success(self, test_client, mock_agent_dependency):
        """Test successful health check."""
        mock_agent_dependency.db_connector.test_connection.return_value = True
        mock_agent_dependency.conversation_sessions = {"session1": Mock()}

        response = test_client.get("/db_connection")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "connected"
        assert data["database_connection"] == "ok"
        assert data["active_sessions"] == 1

    def test_health_check_failure(self, test_client, mock_agent_dependency):
        """Test health check with database connection failure."""
        mock_agent_dependency.db_connector.test_connection.return_value = False

        response = test_client.get("/db_connection")

        assert response.status_code == 503
        data = response.json()
        assert "Database connection failed" in data["detail"]

    def test_process_query_success(self, test_client, mock_agent_dependency):
        """Test successful query processing."""
        # Mock methods properly return the expected values
        mock_agent_dependency.execute_natural_language_query.return_value = {
            "success": True,
            "agent_response": "Found 2 actors",
            "data": [
                {"actor_id": 1, "first_name": "John"},
                {"actor_id": 2, "first_name": "Jane"}
            ],
            "affected_rows": 2
        }
        mock_agent_dependency.get_session_info.return_value = {
            "session_id": "test_session",
            "query_count": 1
        }

        request_data = {
            "query": "Show me all actors",
            "session_id": "test_session"
        }

        response = test_client.post("/query", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Found 2 actors"
        assert len(data["data"]) == 2
        assert data["session_id"] == "test_session"

    def test_process_query_empty_query(self, test_client, mock_agent_dependency):
        """Test processing empty query."""
        request_data = {
            "query": "",
            "session_id": "test_session"
        }

        response = test_client.post("/query", json=request_data)

        assert response.status_code == 400
        assert "Query cannot be empty" in response.json()["detail"]

    def test_get_active_sessions(self, test_client, mock_agent_dependency):
        """Test getting active sessions."""
        # This should work now with the proper mock setup
        response = test_client.get("/sessions")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_session_info_success(self, test_client, mock_agent_dependency):
        """Test getting session information."""
        # This should work now with the proper mock setup  
        response = test_client.get("/sessions/test_session")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session"

    def test_clear_session_success(self, test_client, mock_agent_dependency):
        """Test successful session clearing."""
        # This should work now with the proper mock setup
        response = test_client.delete("/sessions/test_session")

        assert response.status_code == 200
        data = response.json()
        assert "cleared successfully" in data["message"]


class TestRequestValidation:
    """Test request validation and edge cases."""

    def test_query_request_validation(self, test_client, mock_agent_dependency):
        """Test QueryRequest model validation."""
        # Missing query field
        response = test_client.post("/query", json={"session_id": "test"})
        assert response.status_code == 422

        # Invalid JSON
        response = test_client.post("/query", data="invalid json")
        assert response.status_code == 422

    def test_query_with_special_characters(self, test_client, mock_agent_dependency):
        """Test query with special characters and Unicode."""
        special_queries = [
            "Find actors with names containing 'ñ'",
            "Search for films with rating >= 'PG-13'",
            "Show customers where email contains '@'",
            "Find actors named 'José' or 'François'"
        ]

        for query in special_queries:
            request_data = {"query": query, "session_id": "test"}
            response = test_client.post("/query", json=request_data)
            assert response.status_code == 200
