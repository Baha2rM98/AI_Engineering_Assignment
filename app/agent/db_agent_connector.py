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

    def _format_schema_for_llm(self) -> Dict[str, Any]:
        """Format the database schema in a way that's more digestible for the LLM."""
        if not self.database_schema:
            return {"error": "No schema available"}

        formatted_schema = {
            "database_name": "sakila",  # Hard-coded for now, could be retrieved from connection
            "tables": []
        }

        for table_name, table_info in self.database_schema.items():
            table_data = {
                "name": table_name,
                "columns": [
                    {
                        "name": col["name"],
                        "type": col["type"],
                        "is_primary_key": col["name"] in table_info.get("primary_keys", [])
                    }
                    for col in table_info.get("columns", [])
                ]
            }
            formatted_schema["tables"].append(table_data)

        # Add a summary for quick reference
        table_summary = ", ".join([t["name"] for t in formatted_schema["tables"]])
        formatted_schema["summary"] = f"Database contains {len(formatted_schema['tables'])} tables: {table_summary}"

        return formatted_schema

    def execute_natural_language_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a natural language query using the LangGraph agent.
        """
        try:
            print("Starting query processing...")

            # Ensure we have fresh schema data
            self._refresh_schema()

            # Format schema for better LLM understanding
            formatted_schema = self._format_schema_for_llm()
            print(f"Database has {len(self.database_schema)} tables")

            # Special case for "Show tables" type queries
            if "show" in query.lower() and "table" in query.lower():
                tables = self.db_connector.get_table_names()
                return {
                    "success": True,
                    "agent_response": f"I found {len(tables)} tables in the database: {', '.join(tables)}",
                    "data": [{"table_name": table} for table in tables]
                }

            # Get agent results based on natural language query
            try:
                print("Calling query_database...")
                agent_result = query_database(query, formatted_schema)
                print(f"Agent result type: {type(agent_result)}")
                print(f"Agent result keys: {agent_result.keys() if isinstance(agent_result, dict) else 'Not a dict'}")
            except Exception as agent_error:
                print(f"Error in query_database: {agent_error}")
                import traceback
                print(traceback.format_exc())
                return {
                    "success": False,
                    "agent_response": f"Error communicating with the AI model: {str(agent_error)}",
                    "error": str(agent_error)
                }

            # Extract response from result
            agent_response = ""
            if isinstance(agent_result, dict):
                agent_response = agent_result.get("response", "")
                agent_context = agent_result.get("context", {})
            else:
                try:
                    agent_response = agent_result.response if hasattr(agent_result, "response") else str(agent_result)
                    agent_context = agent_result.context if hasattr(agent_result, "context") else {}
                except Exception as attr_error:
                    print(f"Error accessing agent result attributes: {attr_error}")
                    agent_response = str(agent_result)
                    agent_context = {}

            print(f"Agent response: {agent_response}")

            # Extract SQL from agent response
            sql_query = None
            if "```sql" in agent_response:
                sql_parts = agent_response.split("```sql")
                if len(sql_parts) > 1:
                    sql_query = sql_parts[1].split("```")[0].strip()
            elif "```" in agent_response:
                code_blocks = agent_response.split("```")
                for i in range(1, len(code_blocks), 2):
                    if i < len(code_blocks):
                        potential_sql = code_blocks[i].strip()
                        if potential_sql.lower().startswith(("select", "insert", "update", "delete")):
                            sql_query = potential_sql
                            break

            # Try to execute SQL if found
            if sql_query:
                print(f"Executing extracted SQL: {sql_query}")
                db_result = self.db_connector.execute_query(sql_query)

                # Generate a better response based on the actual results
                if db_result.get("success"):
                    data = db_result.get("data", [])
                    row_count = len(data)

                    # Determine table name from SQL query
                    table_name = self._extract_table_name_from_sql(sql_query)

                    # Generate a better response based on the actual data
                    if row_count > 0:
                        response = f"I found {row_count} records in the {table_name} table. "
                        # Show sample data for the first few records if available
                        if row_count > 5:
                            response += f"Here are the first 5 entries."
                        else:
                            response += f"Here are all the entries."
                    else:
                        response = f"I couldn't find any records in the {table_name} table matching your criteria."

                    return {
                        "success": True,
                        "agent_response": response,
                        "data": data,
                        "affected_rows": row_count
                    }
                else:
                    return {
                        "success": False,
                        "agent_response": f"Error executing the query: {db_result.get('error')}",
                        "error": db_result.get("error")
                    }

            # If we can't extract SQL, try to determine which table the query is about
            target_table = self._identify_table_from_query(query)

            if target_table:
                # Generate a basic SQL query for the identified table
                limit_clause = "LIMIT 5" if "all" not in query.lower() else ""
                sql = f"SELECT * FROM {target_table} {limit_clause}"
                print(f"Executing fallback SQL for table {target_table}: {sql}")

                db_result = self.db_connector.execute_query(sql)

                if db_result.get("success"):
                    data = db_result.get("data", [])
                    return {
                        "success": True,
                        "agent_response": f"Here are the results from the {target_table} table ({len(data)} records found).",
                        "data": data,
                        "affected_rows": len(data)
                    }

            # If all attempts failed, return the agent's response with an error
            return {
                "success": False,
                "agent_response": agent_response or "I couldn't execute this query successfully.",
                "error": "Could not generate executable SQL from the query"
            }

        except Exception as e:
            import traceback
            print(f"Unexpected error: {e}")
            print(traceback.format_exc())
            return {
                "success": False,
                "error": f"Failed to execute query: {str(e)}",
                "agent_response": "I encountered an error while processing your request."
            }

    def _extract_table_name_from_sql(self, sql: str) -> str:
        """Extract table name from a SQL query."""
        sql_lower = sql.lower()
        # Look for FROM clause
        if "from" in sql_lower:
            from_parts = sql_lower.split("from")[1].strip().split()
            if from_parts:
                # Get table name and remove any trailing characters like commas, parentheses, etc.
                table = from_parts[0].rstrip(',;()')
                return table

        # Default to generic if we can't determine
        return "database"

    def _identify_table_from_query(self, query: str) -> str:
        """Identify which table the natural language query is referring to."""
        query_lower = query.lower()

        # Create a mapping of common terms to tables
        table_mappings = {}

        # Build mappings from actual database tables
        # This makes it dynamic based on the actual database schema
        for table in self.database_schema.keys():
            # Add the table name itself
            table_mappings[table.lower()] = table
            # Add singular/plural forms
            if table.lower().endswith('s'):
                table_mappings[table.lower()[:-1]] = table  # singular form
            else:
                table_mappings[table.lower() + 's'] = table  # plural form

        # Add some common synonyms
        if 'film' in table_mappings:
            table_mappings['movie'] = 'film'
            table_mappings['movies'] = 'film'

        # Check for table mentions in the query
        for term, table in table_mappings.items():
            if term in query_lower:
                return table

        # Default to a common table if we can't determine
        if 'actor' in self.database_schema:
            return 'actor'

        # Last resort: return the first table in the database
        if self.database_schema:
            return list(self.database_schema.keys())[0]

        return None
