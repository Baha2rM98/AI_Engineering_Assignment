from typing import Dict, Any, List, Optional
import json
import traceback
import logging
from app.database.db_connector import DatabaseConnector
from app.agent.langraph_agent import query_database

logger = logging.getLogger(__name__)


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
            logger.error(f"Failed to fetch schema: {e}")
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
            logger.debug(f"JSON parsing failed: {e}, falling back to basic parsing")
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

    def _extract_sql_from_response(self, response: str, context: Dict[str, Any]) -> Optional[str]:
        """Extract SQL from agent response or context."""
        # Try to get SQL from context first
        if "sql_query" in context:
            return context["sql_query"]

        # Try to extract from code blocks
        if "```sql" in response:
            parts = response.split("```sql")
            if len(parts) > 1:
                sql = parts[1].split("```")[0].strip()
                if sql:
                    return sql

        # Check for code blocks without language specification
        if "```" in response:
            parts = response.split("```")
            for i in range(1, len(parts), 2):
                if i < len(parts):
                    code = parts[i].strip()
                    # Check if it looks like SQL
                    if code.lower().startswith(("select", "insert", "update", "delete")):
                        return code

        # Try to extract from operation details in context
        if "operation_details" in context and isinstance(context["operation_details"], str):
            details = context["operation_details"]
            # Look for SQL in the details
            if "```sql" in details:
                sql = details.split("```sql")[1].split("```")[0].strip()
                if sql:
                    return sql

        return None

    def _generate_sql_from_llm(self, query: str, schema_info: Dict[str, Any]) -> Optional[str]:
        """Generate SQL directly using the LLM."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0,
                convert_system_message_to_human=True
            )

            # Create a focused prompt for SQL generation
            prompt = f"""
            Generate SQL for PostgreSQL to answer this query: "{query}"

            Database schema summary:
            {schema_info.get("summary", "No schema information available")}

            Important:
            1. Return ONLY the SQL query with no explanations or markdown formatting
            2. Use appropriate filtering, sorting, and limits based on the query
            3. Make sure all table and column names are correct
            4. For numeric limits mentioned in the query, use those exact numbers
            """

            result = llm.invoke(prompt)
            sql = result.content if hasattr(result, "content") else str(result)

            # Clean up the SQL to ensure it's executable
            sql = sql.strip()
            if sql.startswith("```sql"):
                sql = sql.replace("```sql", "", 1)
            if sql.endswith("```"):
                sql = sql[:-3]

            sql = sql.strip()

            # Validate that it looks like SQL
            if sql.lower().startswith(("select", "insert", "update", "delete")):
                return sql

            return None
        except Exception as e:
            logger.error(f"Error generating SQL from LLM: {e}")
            return None

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

    def _generate_result_message(self, query: str, sql: str, data: List[Dict[str, Any]]) -> str:
        """Generate appropriate response message based on the query and results."""
        row_count = len(data)

        # Extract operation type
        operation = "query"
        if sql.lower().startswith("select"):
            operation = "SELECT"
        elif sql.lower().startswith("insert"):
            operation = "INSERT"
        elif sql.lower().startswith("update"):
            operation = "UPDATE"
        elif sql.lower().startswith("delete"):
            operation = "DELETE"

        # Extract table name
        table_name = self._extract_table_name_from_sql(sql) or "the database"

        # Generate appropriate message
        if operation == "SELECT":
            if row_count == 0:
                return f"I couldn't find any matching records in {table_name} for your query."
            elif row_count == 1:
                return f"I found 1 record in {table_name} that matches your query."
            else:
                # Check if this is a limited result
                if "limit" in sql.lower() and row_count < 20:  # Assuming small limits are intentional
                    return f"Here are the {row_count} records you requested from {table_name}."
                else:
                    return f"I found {row_count} records in {table_name} that match your query."
        elif operation == "INSERT":
            return f"Successfully inserted {row_count} record(s) into {table_name}."
        elif operation == "UPDATE":
            return f"Successfully updated {row_count} record(s) in {table_name}."
        elif operation == "DELETE":
            return f"Successfully deleted {row_count} record(s) from {table_name}."
        else:
            return f"Query executed successfully. Affected {row_count} record(s)."

    def execute_natural_language_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a natural language query using the LangGraph agent.

        This method processes a natural language query through these steps:
        1. Calls the LangGraph agent to understand the query
        2. Extracts SQL or operation details from the agent's response
        3. Executes the SQL against the database
        4. Formats the results with a natural language response

        Args:
            query: A natural language query string

        Returns:
            Dictionary containing:
            - success: Boolean indicating if the query was successful
            - agent_response: Natural language response to the query
            - data: Query results (if applicable)
            - affected_rows: Number of rows affected (if applicable)
            - error: Error message (if unsuccessful)
        """
        try:
            logger.info(f"Processing natural language query: {query}")

            # Ensure we have fresh schema data
            self._refresh_schema()

            # Format schema for better LLM understanding
            formatted_schema = self._format_schema_for_llm()
            logger.info(f"Database has {len(self.database_schema)} tables")

            # Special case for "Show tables" type queries
            if "show" in query.lower() and "table" in query.lower():
                tables = self.db_connector.get_table_names()
                return {
                    "success": True,
                    "agent_response": f"I found {len(tables)} tables in the database: {', '.join(tables)}",
                    "data": [{"table_name": table} for table in tables],
                    "affected_rows": len(tables)
                }

            # Get agent results based on natural language query
            try:
                logger.info("Calling LangGraph agent...")
                agent_result = query_database(query, formatted_schema)
                logger.info(f"Agent result type: {type(agent_result)}")
            except Exception as agent_error:
                logger.error(f"Error in query_database: {agent_error}")
                logger.error(traceback.format_exc())
                return {
                    "success": False,
                    "agent_response": f"Error communicating with the AI model: {str(agent_error)}",
                    "error": str(agent_error)
                }

            # Extract response and context from agent result
            agent_response = ""
            agent_context = {}
            if isinstance(agent_result, dict):
                agent_response = agent_result.get("response", "")
                agent_context = agent_result.get("context", {})
            else:
                try:
                    agent_response = agent_result.response if hasattr(agent_result, "response") else str(agent_result)
                    agent_context = agent_result.context if hasattr(agent_result, "context") else {}
                except Exception as attr_error:
                    logger.error(f"Error accessing agent result attributes: {attr_error}")
                    agent_response = str(agent_result)

            logger.info(f"Agent response: {agent_response[:100]}...")  # Log first 100 chars

            # STAGE 1: Try to directly extract SQL from the response or context
            sql_query = self._extract_sql_from_response(agent_response, agent_context)

            if sql_query:
                logger.info(f"Executing extracted SQL: {sql_query}")
                db_result = self.db_connector.execute_query(sql_query)

                if db_result.get("success"):
                    data = db_result.get("data", [])
                    table_name = self._extract_table_name_from_sql(sql_query) or "the database"
                    return {
                        "success": True,
                        "agent_response": self._generate_result_message(query, sql_query, data),
                        "data": data,
                        "affected_rows": len(data)
                    }
                else:
                    logger.error(f"SQL execution failed: {db_result.get('error')}")

            # STAGE 2: If direct SQL extraction failed, try to generate SQL using a dedicated LLM call
            sql_query = self._generate_sql_from_llm(query, formatted_schema)

            if sql_query:
                logger.info(f"Executing LLM-generated SQL: {sql_query}")
                db_result = self.db_connector.execute_query(sql_query)

                if db_result.get("success"):
                    data = db_result.get("data", [])
                    return {
                        "success": True,
                        "agent_response": self._generate_result_message(query, sql_query, data),
                        "data": data,
                        "affected_rows": len(data)
                    }

            # STAGE 3: Last resort - try to find mentioned tables and do a basic query
            target_table = self._identify_table_from_query(query)
            if target_table:
                # Extract numeric values for potential limits
                limit = 5  # Default
                for word in query.lower().split():
                    if word.isdigit() and 1 <= int(word) <= 1000:  # Reasonable limit range
                        limit = int(word)
                        break

                sql = f"SELECT * FROM {target_table} LIMIT {limit}"
                logger.info(f"Executing last-resort SQL: {sql}")

                db_result = self.db_connector.execute_query(sql)
                if db_result.get("success"):
                    data = db_result.get("data", [])
                    return {
                        "success": True,
                        "agent_response": f"Found {len(data)} records in the {target_table} table.",
                        "data": data,
                        "affected_rows": len(data)
                    }

            # If all attempts failed, return the agent's response
            return {
                "success": False,
                "agent_response": agent_response or "I couldn't execute your query successfully.",
                "error": "Failed to generate executable SQL from query"
            }

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": f"Failed to execute query: {str(e)}",
                "agent_response": "I encountered an error while processing your request."
            }
