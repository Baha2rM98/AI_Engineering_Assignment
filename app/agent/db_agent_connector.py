from typing import Dict, Any, List
import json
from app.database.db_connector import DatabaseConnector
from app.agent.langraph_agent import query_database


class DBAgentConnector:
    """
    Connector class to integrate the LangGraph agent with the database connector.
    This class bridges the natural language processing capabilities of the agent
    with the database operations.
    """

    def __init__(self, connection_string=None):
        """Initialize the connector with database connection."""
        self.db_connector = DatabaseConnector(connection_string)
        self.database_schema = None
        self._refresh_schema()

    def _refresh_schema(self):
        """Refresh the database schema information."""
        try:
            self.database_schema = self.db_connector.get_database_schema()
        except Exception as e:
            self.database_schema = {"error": f"Failed to fetch schema: {str(e)}"}

    def _parse_operation_details(self, operation_details: str) -> Dict[str, Any]:
        """Parse the operation details from the agent into executable parameters."""
        try:
            # Try to extract JSON from the text
            # Look for content between triple backticks if it's formatted that way
            if "```json" in operation_details and "```" in operation_details.split("```json")[1]:
                json_str = operation_details.split("```json")[1].split("```")[0].strip()
            elif "```" in operation_details and "```" in operation_details.split("```")[1]:
                json_str = operation_details.split("```")[1].strip()
            else:
                # Just try to parse the whole string
                json_str = operation_details

            return json.loads(json_str)
        except Exception as e:
            # Fallback to basic parsing if JSON extraction fails
            result = {}
            operation_lower = operation_details.lower()

            # Determine operation type
            if "select" in operation_lower:
                result["operation_type"] = "select"
            elif "insert" in operation_lower:
                result["operation_type"] = "insert"
            elif "update" in operation_lower:
                result["operation_type"] = "update"
            elif "delete" in operation_lower:
                result["operation_type"] = "delete"
            else:
                result["operation_type"] = "unknown"

            # Extract table name (simplified approach)
            tables = []
            for table_name in self.database_schema.keys():
                if table_name.lower() in operation_lower:
                    tables.append(table_name)

            if tables:
                result["table"] = tables[0]

            return result

    def execute_natural_language_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a natural language query using the LangGraph agent.

        Args:
            query (str): The natural language query from the user

        Returns:
            Dict[str, Any]: The results and response
        """
        try:
            # Get agent results based on natural language query
            agent_result = query_database(query, self.database_schema)

            # If no operation details, return the agent's response
            if "operation_details" not in agent_result.get("context", {}):
                return {
                    "success": True,
                    "agent_response": agent_result["response"],
                    "data": None
                }

            # Parse operation details
            operation_details = self._parse_operation_details(
                agent_result["context"]["operation_details"]
            )

            # Execute actual database operation
            if operation_details.get("operation_type") == "select":
                db_result = self.db_connector.execute_operation(
                    "select",
                    {
                        "table": operation_details.get("table", ""),
                        "columns": operation_details.get("columns", ["*"]),
                        "where": operation_details.get("where", ""),
                        "limit": operation_details.get("limit", ""),
                        "order_by": operation_details.get("order_by", "")
                    }
                )
            elif operation_details.get("operation_type") == "insert":
                db_result = self.db_connector.execute_operation(
                    "insert",
                    {
                        "table": operation_details.get("table", ""),
                        "values": operation_details.get("values", {})
                    }
                )
            elif operation_details.get("operation_type") == "update":
                db_result = self.db_connector.execute_operation(
                    "update",
                    {
                        "table": operation_details.get("table", ""),
                        "values": operation_details.get("values", {}),
                        "where": operation_details.get("where", "")
                    }
                )
            elif operation_details.get("operation_type") == "delete":
                db_result = self.db_connector.execute_operation(
                    "delete",
                    {
                        "table": operation_details.get("table", ""),
                        "where": operation_details.get("where", "")
                    }
                )
            else:
                return {
                    "success": False,
                    "error": "Unsupported operation type",
                    "agent_response": agent_result["response"]
                }

            return {
                "success": db_result.get("success", False),
                "agent_response": agent_result["response"],
                "data": db_result.get("data"),
                "affected_rows": db_result.get("affected_rows"),
                "error": db_result.get("error")
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to execute query: {str(e)}",
                "agent_response": "I encountered an error while processing your request."
            }
