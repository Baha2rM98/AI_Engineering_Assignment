from typing import Dict, Any, List, Optional
import json
import traceback
import logging
from datetime import datetime, timedelta
from threading import Lock
from app.database.db_connector import DatabaseConnector
from app.agent.langraph_agent import query_database
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)


class ConversationSession:
    """Class to manage individual conversation sessions."""

    def __init__(self, session_id: str, max_history: int = 10):
        self.session_id = session_id
        self.history = []
        self.context = {}
        self.last_table = None
        self.last_operation = None
        self.last_sql = None
        self.last_filter = None  # ADD THIS
        self.last_result_count = None  # ADD THIS
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.max_history = max_history

    def add_query(self, query: str, result: Dict[str, Any]):
        """Add a query and its result to the session history."""
        self.last_activity = datetime.now()

        # Extract context information from result
        if result.get("success"):
            # Try to extract table name from the result
            if "sql_query" in result:
                self._extract_table_from_sql(result["sql_query"])
                self._extract_filter_from_sql(result["sql_query"])  # ADD THIS

            # Store result count for context
            if "data" in result and isinstance(result["data"], list):
                self.last_result_count = len(result["data"])

            # Remember the operation type
            if "operation_type" in result:
                self.last_operation = result["operation_type"]

        # Add to history
        history_entry = {
            "timestamp": self.last_activity.isoformat(),
            "query": query,
            "success": result.get("success", False),
            "table": self.last_table,
            "operation": self.last_operation,
            "row_count": result.get("affected_rows", 0)
        }

        self.history.append(history_entry)

        # Maintain history limit
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def _extract_filter_from_sql(self, sql: str):
        """Extract and remember the filter conditions from SQL."""
        if sql and "where" in sql.lower():
            try:
                where_part = sql.lower().split("where")[1].split("order by")[0].split("limit")[0].strip()
                self.last_filter = where_part
            except:
                self.last_filter = None
        else:
            self.last_filter = None

    def _extract_table_from_sql(self, sql: str):
        """Extract and remember the table name from SQL."""
        if sql:
            sql_lower = sql.lower().strip()

            # Handle SELECT statements
            if "from" in sql_lower:
                parts = sql_lower.split("from")[1].strip().split()
                if parts:
                    self.last_table = parts[0].rstrip(',;()')

            # Handle INSERT statements
            elif sql_lower.startswith("insert into"):
                parts = sql_lower.split("insert into")[1].strip().split()
                if parts:
                    self.last_table = parts[0].rstrip(',;()')

            # Handle UPDATE statements
            elif sql_lower.startswith("update"):
                parts = sql_lower.split("update")[1].strip().split()
                if parts:
                    self.last_table = parts[0].rstrip(',;()')

            # Handle DELETE statements
            elif sql_lower.startswith("delete from"):
                parts = sql_lower.split("delete from")[1].strip().split()
                if parts:
                    self.last_table = parts[0].rstrip(',;()')

    def get_context_summary(self) -> str:
        """Get a summary of recent conversation context."""
        if not self.history:
            return "No previous conversation history."

        recent = self.history[-3:]  # Last 3 interactions
        summary_parts = []

        if self.last_table:
            summary_parts.append(f"Recently working with table: {self.last_table}")

        if self.last_operation:
            summary_parts.append(f"Last operation: {self.last_operation}")

        summary_parts.append(f"Recent queries: {[h['query'] for h in recent]}")

        return " | ".join(summary_parts)

    def is_expired(self, timeout_minutes: int = 60) -> bool:
        """Check if the session has expired."""
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)


class DBAgentConnector:
    """
    Enhanced connector class with memory management and context awareness.
    """

    def __init__(self, connection_string=None):
        """Initialize the connector with database connection and session management."""
        self.db_connector = DatabaseConnector(connection_string)
        self.database_schema = None

        # Session management
        self.conversation_sessions: Dict[str, ConversationSession] = {}
        self.session_lock = Lock()
        self.session_config = {
            "max_history": 10,
            "session_timeout": 60,  # minutes
            "max_sessions": 100
        }

        self._refresh_schema()

    def _get_or_create_session(self, session_id: str) -> ConversationSession:
        """Get existing session or create a new one."""
        with self.session_lock:
            # Clean up expired sessions first
            self._cleanup_expired_sessions()

            if session_id not in self.conversation_sessions:
                # Limit total sessions
                if len(self.conversation_sessions) >= self.session_config["max_sessions"]:
                    # Remove oldest session
                    oldest_session = min(
                        self.conversation_sessions.values(),
                        key=lambda s: s.last_activity
                    )
                    del self.conversation_sessions[oldest_session.session_id]

                # Create new session
                self.conversation_sessions[session_id] = ConversationSession(
                    session_id,
                    self.session_config["max_history"]
                )

            return self.conversation_sessions[session_id]

    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        timeout = self.session_config["session_timeout"]
        expired_sessions = [
            sid for sid, session in self.conversation_sessions.items()
            if session.is_expired(timeout)
        ]

        for sid in expired_sessions:
            del self.conversation_sessions[sid]
            logger.debug(f"Cleaned up expired session: {sid}")

    def _is_table_listing_query(self, query: str) -> bool:
        """
        Improved detection for table listing queries.
        Only returns True for explicit table listing requests.
        """
        query_lower = query.lower().strip()

        # Explicit table listing patterns
        table_listing_patterns = [
            "show tables",
            "list tables",
            "show all tables",
            "list all tables",
            "what tables",
            "which tables",
            "get tables",
            "display tables"
        ]

        # Check for exact matches or patterns at start of query
        for pattern in table_listing_patterns:
            if query_lower == pattern or query_lower.startswith(pattern):
                return True

        # FIXED: Check for contextual references FIRST, then explicit table listing
        contextual_references = [
            "that table",
            "the same table",
            "this table",
            "from that table",
            "from the same table",
            "from this table"
        ]

        # If it's a contextual reference, it's NOT a table listing
        for ref in contextual_references:
            if ref in query_lower:
                return False

        # Only allow very specific "show table" patterns (not "show ... table")
        if query_lower in ["show table", "show the table", "list table", "list the table"]:
            return True

        return False

    def _resolve_contextual_references(self, query: str, session: ConversationSession) -> str:
        """Enhanced contextual reference resolution with filter context."""
        query_lower = query.lower()
        resolved_query = query

        # Handle count/quantity references with filter context
        quantity_queries = [
            "how many?",
            "count them",
            "how many are there?",
            "what's the count?",
            "give me the count"
        ]

        if query_lower.strip() in quantity_queries and session.last_table:
            if session.last_filter:
                # Count with previous filter
                resolved_query = f"Count records in the {session.last_table} table where {session.last_filter}"
            else:
                # Count all records
                resolved_query = f"Count all records in the {session.last_table} table"
            logger.info(f"Resolved count query with filter context: {session.last_filter}")

        # Handle "show me more/first N" with filter context
        limit_patterns = ["show me the first", "just show me the first", "show first", "first"]

        for pattern in limit_patterns:
            if pattern in query_lower:
                # Extract number
                words = query.split()
                limit_num = 5  # default
                for i, word in enumerate(words):
                    if word.isdigit():
                        limit_num = int(word)
                        break

                if session.last_table:
                    if session.last_filter:
                        # Apply previous filter
                        resolved_query = f"Show me the first {limit_num} records from the {session.last_table} table where {session.last_filter}"
                    else:
                        resolved_query = f"Show me the first {limit_num} records from the {session.last_table} table"
                    logger.info(f"Resolved limit query with filter context: {session.last_filter}")
                    break

        # Handle table references (existing logic)
        table_references = [
            "that table",
            "the same table",
            "this table",
            "from that table",
            "from the same table",
            "from this table"
        ]

        if session.last_table and any(ref in query_lower for ref in table_references):
            for ref in table_references:
                if ref in query_lower:
                    if "from" in ref:
                        replacement = f"from the {session.last_table} table"
                    else:
                        replacement = f"the {session.last_table} table"
                    resolved_query = resolved_query.replace(ref, replacement)
            logger.info(f"Resolved table reference: '{session.last_table}'")

        return resolved_query

    def _create_context_aware_prompt(self, query: str, session: ConversationSession, schema_info: Dict[str, Any]) -> \
            Dict[str, Any]:
        """Create an enhanced prompt with conversation context."""
        context_summary = session.get_context_summary()

        enhanced_schema = schema_info.copy()
        enhanced_schema["conversation_context"] = {
            "summary": context_summary,
            "last_table": session.last_table,
            "last_operation": session.last_operation,
            "last_filter": session.last_filter,  # ADD THIS
            "last_result_count": session.last_result_count,  # ADD THIS
            "recent_queries": [h["query"] for h in session.history[-3:]]
        }

        return enhanced_schema

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
            if "```json" in operation_details and "```" in operation_details.split("```json")[1]:
                json_str = operation_details.split("```json")[1].split("```")[0].strip()
            elif "```" in operation_details and "```" in operation_details.split("```")[1]:
                json_str = operation_details.split("```")[1].strip()
            else:
                json_str = operation_details

            return json.loads(json_str)
        except Exception as e:
            logger.debug(f"JSON parsing failed: {e}, falling back to basic parsing")
            result = {}
            operation_lower = operation_details.lower()

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

            # Extract table name
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
            "database_name": "sakila",
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

        table_summary = ", ".join([t["name"] for t in formatted_schema["tables"]])
        formatted_schema["summary"] = f"Database contains {len(formatted_schema['tables'])} tables: {table_summary}"

        return formatted_schema

    def _extract_sql_from_response(self, response: str, context: Dict[str, Any]) -> Optional[str]:
        """Extract SQL from agent response or context."""
        # Check context first
        if "sql_query" in context:
            return context["sql_query"]

        # Look for SQL code blocks
        if "```sql" in response.lower():
            parts = response.split("```sql")
            if len(parts) > 1:
                sql = parts[1].split("```")[0].strip()
                if sql:
                    return sql

        # Look for any code blocks that might contain SQL
        if "```" in response:
            parts = response.split("```")
            for i in range(1, len(parts), 2):
                if i < len(parts):
                    code = parts[i].strip()
                    # Check if it starts with common SQL keywords
                    if code.lower().startswith(("select", "insert", "update", "delete")):
                        return code

        # Look in operation_details
        if "operation_details" in context and isinstance(context["operation_details"], str):
            details = context["operation_details"]
            if "```sql" in details.lower():
                sql = details.split("```sql")[1].split("```")[0].strip()
                if sql:
                    return sql
            # Also check for INSERT statements without code blocks
            if "insert into" in details.lower():
                lines = details.split('\n')
                for line in lines:
                    if line.strip().lower().startswith("insert into"):
                        return line.strip()

        return None

    # Add this method to db_agent_connector.py:
    def _generate_insert_sql_from_llm(self, query: str, schema_info: Dict[str, Any]) -> Optional[str]:
        """Generate INSERT SQL specifically for insert operations."""
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0,
                convert_system_message_to_human=True
            )

            # Get actor table schema for reference
            actor_schema = "actor (actor_id SERIAL PRIMARY KEY, first_name VARCHAR(45) NOT NULL, last_name VARCHAR(45) NOT NULL, last_update TIMESTAMP DEFAULT NOW())"

            prompt = f"""
            Generate a PostgreSQL INSERT statement for this request: "{query}"

            Important table schemas:
            - {actor_schema}
            - customer (customer_id SERIAL PRIMARY KEY, store_id SMALLINT NOT NULL DEFAULT 1, first_name VARCHAR(45) NOT NULL, last_name VARCHAR(45) NOT NULL, email VARCHAR(50), address_id SMALLINT NOT NULL DEFAULT 1, activebool BOOLEAN DEFAULT true, create_date DATE DEFAULT CURRENT_DATE, last_update TIMESTAMP DEFAULT NOW(), active INTEGER DEFAULT 1)

            Rules:
            1. Generate ONLY the INSERT SQL statement
            2. Include all required fields (NOT NULL fields without defaults)
            3. Use appropriate default values for optional fields
            4. Do not include auto-increment primary key fields
            5. End with semicolon
            6. No explanations or markdown formatting

            Example for "Insert actor John Doe":
            INSERT INTO actor (first_name, last_name) VALUES ('John', 'Doe');
            """

            result = llm.invoke(prompt)
            sql = result.content if hasattr(result, "content") else str(result)

            # Clean up the SQL
            sql = sql.strip()
            if sql.startswith("```sql"):
                sql = sql.replace("```sql", "", 1)
            if sql.endswith("```"):
                sql = sql[:-3]
            sql = sql.strip()

            if sql.lower().startswith("insert"):
                return sql

            return None
        except Exception as e:
            logger.error(f"Error generating INSERT SQL: {e}")
            return None

    def _generate_sql_from_llm(self, query: str, schema_info: Dict[str, Any]) -> Optional[str]:
        """Generate SQL directly using the LLM with context awareness."""
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0,
                convert_system_message_to_human=True
            )

            # Enhanced prompt with conversation context
            context_info = ""
            if "conversation_context" in schema_info:
                ctx = schema_info["conversation_context"]
                if ctx["last_table"]:
                    context_info += f"Previously working with table: {ctx['last_table']}\n"
                if ctx["recent_queries"]:
                    context_info += f"Recent queries: {', '.join(ctx['recent_queries'])}\n"

            prompt = f"""
            Generate SQL for PostgreSQL to answer this query: "{query}"

            Database schema summary:
            {schema_info.get("summary", "No schema information available")}

            Conversation context:
            {context_info}

            Important:
            1. Return ONLY the SQL query with no explanations or markdown formatting
            2. Use conversation context to resolve references like "that table"
            3. Use appropriate filtering, sorting, and limits based on the query
            4. Make sure all table and column names are correct
            5. For numeric limits mentioned in the query, use those exact numbers
            """

            result = llm.invoke(prompt)
            sql = result.content if hasattr(result, "content") else str(result)

            sql = sql.strip()
            if sql.startswith("```sql"):
                sql = sql.replace("```sql", "", 1)
            if sql.endswith("```"):
                sql = sql[:-3]

            sql = sql.strip()

            if sql.lower().startswith(("select", "insert", "update", "delete")):
                return sql

            return None
        except Exception as e:
            logger.error(f"Error generating SQL from LLM: {e}")
            return None

    def _extract_table_name_from_sql(self, sql: str) -> str:
        """Extract table name from a SQL query."""
        sql_lower = sql.lower()
        if "from" in sql_lower:
            from_parts = sql_lower.split("from")[1].strip().split()
            if from_parts:
                table = from_parts[0].rstrip(',;()')
                return table
        return "database"

    def _identify_table_from_query(self, query: str) -> str | None:
        """Identify which table the natural language query is referring to."""
        query_lower = query.lower()
        table_mappings = {}

        for table in self.database_schema.keys():
            table_mappings[table.lower()] = table
            if table.lower().endswith('s'):
                table_mappings[table.lower()[:-1]] = table
            else:
                table_mappings[table.lower() + 's'] = table

        if 'film' in table_mappings:
            table_mappings['movie'] = 'film'
            table_mappings['movies'] = 'film'

        for term, table in table_mappings.items():
            if term in query_lower:
                return table

        if 'actor' in self.database_schema:
            return 'actor'

        if self.database_schema:
            return list(self.database_schema.keys())[0]

        return None

    def _generate_result_message(self, query: str, sql: str, data: List[Dict[str, Any]],
                                 affected_rows: int = None) -> str:
        """Generate appropriate response message based on the query and results."""
        # Use affected_rows if provided, otherwise fall back to len(data)
        if affected_rows is not None:
            row_count = affected_rows
        else:
            row_count = len(data)

        operation = "SELECT"
        if sql.lower().startswith("insert"):
            operation = "INSERT"
        elif sql.lower().startswith("update"):
            operation = "UPDATE"
        elif sql.lower().startswith("delete"):
            operation = "DELETE"

        table_name = self._extract_table_name_from_sql(sql) or "the database"

        if operation == "SELECT":
            if row_count == 0:
                return f"I couldn't find any matching records in {table_name} for your query."
            elif row_count == 1:
                return f"I found 1 record in {table_name} that matches your query."
            else:
                if "limit" in sql.lower() and row_count < 20:
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

    def execute_natural_language_query(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Execute a natural language query with session-based memory management.
        FIXED: Contextual reference resolution now happens before special case handling.

        Args:
            query: A natural language query string
            session_id: Unique identifier for the conversation session

        Returns:
            Dictionary containing success status, response, data, and context information
        """
        try:
            logger.info(f"Processing query for session {session_id}: {query}")

            # ULTRA-QUICK GARBAGE DETECTION - Add this ONE block:
            if len(query.strip()) > 15 and not any(c in 'aeiou ' for c in query.lower()[:20]):
                return {
                    "success": False,
                    "agent_response": "Please provide a valid database query.",
                    "error": "Invalid query format"
                }

            # STEP 1: Get or create session FIRST
            session = self._get_or_create_session(session_id)

            # STEP 2: Resolve contextual references using session history BEFORE anything else
            resolved_query = self._resolve_contextual_references(query, session)
            if resolved_query != query:
                logger.info(f"Resolved query: '{query}' -> '{resolved_query}'")

            # STEP 3: Refresh schema and format for LLM
            self._refresh_schema()
            formatted_schema = self._format_schema_for_llm()

            # STEP 4: NOW check for special cases (after context resolution)
            if self._is_table_listing_query(resolved_query):
                tables = self.db_connector.get_table_names()
                result = {
                    "success": True,
                    "agent_response": f"I found {len(tables)} tables in the database: {', '.join(tables)}",
                    "data": [{"table_name": table} for table in tables],
                    "affected_rows": len(tables)
                }
                session.add_query(query, result)
                return result

            # STEP 5: Create context-aware prompt
            context_aware_schema = self._create_context_aware_prompt(
                resolved_query, session, formatted_schema
            )

            # STEP 6: Process with LangGraph agent
            try:
                logger.info("Calling LangGraph agent with context...")
                agent_result = query_database(resolved_query, context_aware_schema)
                logger.info(f"Agent result type: {type(agent_result)}")
            except Exception as agent_error:
                logger.error(f"Error in query_database: {agent_error}")
                result = {
                    "success": False,
                    "agent_response": f"Error communicating with the AI model: {str(agent_error)}",
                    "error": str(agent_error)
                }
                session.add_query(query, result)
                return result

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

            logger.info(f"Agent response: {agent_response[:100]}...")

            # STAGE 1: Try to directly extract SQL from the response or context
            sql_query = self._extract_sql_from_response(agent_response, agent_context)

            if sql_query:
                logger.info(f"Executing extracted SQL: {sql_query}")
                db_result = self.db_connector.execute_query(sql_query)

                if db_result.get("success"):
                    data = db_result.get("data", [])
                    actual_affected_rows = db_result.get("affected_rows", len(data))
                    result = {
                        "success": True,
                        "agent_response": self._generate_result_message(resolved_query, sql_query, data,
                                                                        actual_affected_rows),
                        "data": data,
                        "affected_rows": actual_affected_rows,
                        "sql_query": sql_query
                    }
                    session.add_query(query, result)
                    return result
                else:
                    logger.error(f"SQL execution failed: {db_result.get('error')}")

            # STAGE 1.5: Special handling for INSERT operations
            if "insert" in resolved_query.lower() and not sql_query:
                logger.info("Detected INSERT operation, trying specialized SQL generation...")
                sql_query = self._generate_insert_sql_from_llm(resolved_query, context_aware_schema)

                if sql_query:
                    logger.info(f"Generated INSERT SQL: {sql_query}")
                    db_result = self.db_connector.execute_query(sql_query)

                    if db_result.get("success"):
                        affected_rows = db_result.get("affected_rows", 0)
                        table_name = self._extract_table_name_from_sql(sql_query) or "the database"
                        result = {
                            "success": True,
                            "agent_response": f"Successfully inserted {affected_rows} record(s) into {table_name}.",
                            "data": [],
                            "affected_rows": affected_rows,
                            "sql_query": sql_query
                        }
                        session.add_query(query, result)
                        return result
                    else:
                        logger.error(f"INSERT execution failed: {db_result.get('error')}")

            # STAGE 2: If direct SQL extraction failed, try to generate SQL using LLM with context
            sql_query = self._generate_sql_from_llm(resolved_query, context_aware_schema)

            if sql_query:
                logger.info(f"Executing context-aware LLM-generated SQL: {sql_query}")
                db_result = self.db_connector.execute_query(sql_query)

                if db_result.get("success"):
                    data = db_result.get("data", [])
                    actual_affected_rows = db_result.get("affected_rows", len(data))
                    result = {
                        "success": True,
                        "agent_response": self._generate_result_message(resolved_query, sql_query, data,
                                                                        actual_affected_rows),
                        "data": data,
                        "affected_rows": actual_affected_rows,
                        "sql_query": sql_query
                    }
                    session.add_query(query, result)
                    return result

            # STAGE 3: Last resort - try to find mentioned tables and do a basic query with session context
            target_table = session.last_table or self._identify_table_from_query(resolved_query)
            if target_table:
                # Extract numeric values for potential limits
                limit = 5  # Default
                for word in resolved_query.lower().split():
                    if word.isdigit() and 1 <= int(word) <= 1000:
                        limit = int(word)
                        break

                sql = f"SELECT * FROM {target_table} LIMIT {limit}"
                logger.info(f"Executing context-aware last-resort SQL: {sql}")

                db_result = self.db_connector.execute_query(sql)
                if db_result.get("success"):
                    data = db_result.get("data", [])
                    result = {
                        "success": True,
                        "agent_response": f"Found {len(data)} records in the {target_table} table.",
                        "data": data,
                        "affected_rows": len(data),
                        "sql_query": sql
                    }
                    session.add_query(query, result)
                    return result

            # If all attempts failed, return the agent's response
            result = {
                "success": False,
                "agent_response": agent_response or "I couldn't execute your query successfully.",
                "error": "Failed to generate executable SQL from query"
            }
            session.add_query(query, result)
            return result

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.error(traceback.format_exc())
            result = {
                "success": False,
                "error": f"Failed to execute query: {str(e)}",
                "agent_response": "I encountered an error while processing your request."
            }
            # Try to add to session even on error
            try:
                session = self._get_or_create_session(session_id)
                session.add_query(query, result)
            except:
                pass
            return result

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a specific session."""
        if session_id in self.conversation_sessions:
            session = self.conversation_sessions[session_id]
            return {
                "session_id": session_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "query_count": len(session.history),
                "last_table": session.last_table,
                "last_operation": session.last_operation,
                "context_summary": session.get_context_summary()
            }
        return {"error": f"Session {session_id} not found"}

    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session."""
        with self.session_lock:
            if session_id in self.conversation_sessions:
                del self.conversation_sessions[session_id]
                return True
            return False

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        self._cleanup_expired_sessions()
        return list(self.conversation_sessions.keys())
