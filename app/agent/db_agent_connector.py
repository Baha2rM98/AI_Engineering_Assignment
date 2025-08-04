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

            # Extract response and context from result
            agent_response = ""
            agent_context = {}
            execution_details = {}

            if isinstance(agent_result, dict):
                agent_response = agent_result.get("response", "")
                agent_context = agent_result.get("context", {})
                execution_details = agent_result.get("execution_details", {})
            else:
                try:
                    agent_response = agent_result.response if hasattr(agent_result, "response") else str(agent_result)
                    agent_context = agent_result.context if hasattr(agent_result, "context") else {}
                    execution_details = agent_result.execution_result if hasattr(agent_result,
                                                                                 "execution_result") else {}
                except Exception as attr_error:
                    print(f"Error accessing agent result attributes: {attr_error}")
                    agent_response = str(agent_result)

            print(f"Agent response: {agent_response}")

            # Method 1: Try to extract SQL directly from the response
            sql_query = None

            # Look for SQL code blocks in the response
            if "```sql" in agent_response and "```" in agent_response.split("```sql")[1]:
                sql_query = agent_response.split("```sql")[1].split("```")[0].strip()
            elif "```" in agent_response:
                # Try to find any code block that might contain SQL
                code_blocks = agent_response.split("```")
                for i in range(1, len(code_blocks), 2):
                    if i < len(code_blocks):
                        potential_sql = code_blocks[i].strip()
                        if potential_sql.lower().startswith(("select", "insert", "update", "delete")):
                            sql_query = potential_sql
                            break

            # If SQL found, execute it directly
            if sql_query:
                print(f"Executing SQL: {sql_query}")
                try:
                    result = self.db_connector.execute_query(sql_query)
                    return {
                        "success": result.get("success", False),
                        "agent_response": agent_response,
                        "data": result.get("data"),
                        "affected_rows": result.get("affected_rows")
                    }
                except Exception as sql_error:
                    print(f"SQL execution error: {sql_error}")
                    # Continue to try other methods

            # Method 2: Try to parse operation details from context or execution details
            operation_details = agent_context.get("operation_details", "")

            if not operation_details and "execution_plan" in agent_context:
                operation_details = agent_context["execution_plan"]

            if operation_details:
                try:
                    # Parse the operation details
                    operation_params = self._parse_operation_details(operation_details)

                    # Extract operation type and table
                    operation_type = operation_params.get("operation_type", "").lower()
                    table = operation_params.get("table", "")

                    if operation_type and table:
                        print(f"Executing operation: {operation_type} on {table}")
                        result = self.db_connector.execute_operation(operation_type, operation_params)
                        return {
                            "success": result.get("success", False),
                            "agent_response": agent_response,
                            "data": result.get("data"),
                            "affected_rows": result.get("affected_rows")
                        }
                except Exception as op_error:
                    print(f"Operation execution error: {op_error}")
                    # Continue to try other methods

            # Method 3: Generate SQL using the agent response
            try:
                # Create a prompt to generate executable SQL
                sql_generation_prompt = f"""
                Based on the database schema and the query, generate a valid SQL query.
                The user query was: "{query}"

                Generate ONLY the SQL code with no explanation.
                """

                # Using your existing LLM setup to generate SQL
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    temperature=0,
                    convert_system_message_to_human=True
                )

                sql_result = llm.invoke(sql_generation_prompt)
                generated_sql = sql_result.content if hasattr(sql_result, "content") else str(sql_result)

                # Extract SQL if wrapped in code blocks
                if "```sql" in generated_sql:
                    generated_sql = generated_sql.split("```sql")[1].split("```")[0].strip()
                elif "```" in generated_sql:
                    generated_sql = generated_sql.split("```")[1].split("```")[0].strip()

                if generated_sql:
                    print(f"Executing generated SQL: {generated_sql}")
                    result = self.db_connector.execute_query(generated_sql)
                    return {
                        "success": result.get("success", False),
                        "agent_response": agent_response + f"\n\nExecuted SQL: {generated_sql}",
                        "data": result.get("data"),
                        "affected_rows": result.get("affected_rows")
                    }
            except Exception as gen_error:
                print(f"SQL generation error: {gen_error}")
                # Fall through to final return

            # If we reached here, none of the methods worked
            return {
                "success": False,
                "agent_response": agent_response if agent_response else "I couldn't translate your query into a database operation.",
                "error": "Failed to execute database operation"
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
