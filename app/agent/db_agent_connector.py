from typing import Dict, Any, List, Optional
import json
import traceback
import logging
from datetime import datetime, timedelta
from threading import Lock
from app.database.db_connector import DatabaseConnector
from app.agent.langgraph_agent import query_database
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
            sql_lower = sql.lower()

            # Handle different SQL statement types
            if "from" in sql_lower:
                # For SELECT statements
                parts = sql_lower.split("from")[1].strip().split()
                if parts:
                    self.last_table = parts[0].rstrip(',;()')
            elif sql_lower.strip().startswith("insert into"):
                # For INSERT statements
                parts = sql_lower.split("insert into")[1].strip().split()
                if parts:
                    self.last_table = parts[0].rstrip(',;()')
            elif sql_lower.strip().startswith("update"):
                # For UPDATE statements
                parts = sql_lower.split("update")[1].strip().split()
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

    @staticmethod
    def _resolve_contextual_references(query: str, session: ConversationSession) -> str:
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

    @staticmethod
    def _create_context_aware_prompt(session: ConversationSession, schema_info: Dict[str, Any]) -> \
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

    # def _parse_operation_details(self, operation_details: str) -> Dict[str, Any]:
    #     """Parse the operation details from the agent into executable parameters."""
    #     try:
    #         # Try to extract JSON from the text
    #         if "```json" in operation_details and "```" in operation_details.split("```json")[1]:
    #             json_str = operation_details.split("```json")[1].split("```")[0].strip()
    #         elif "```" in operation_details and "```" in operation_details.split("```")[1]:
    #             json_str = operation_details.split("```")[1].strip()
    #         else:
    #             json_str = operation_details
    #
    #         return json.loads(json_str)
    #     except Exception as e:
    #         logger.debug(f"JSON parsing failed: {e}, falling back to basic parsing")
    #         result = {}
    #         operation_lower = operation_details.lower()
    #
    #         if "select" in operation_lower:
    #             result["operation_type"] = "select"
    #         elif "insert" in operation_lower:
    #             result["operation_type"] = "insert"
    #         elif "update" in operation_lower:
    #             result["operation_type"] = "update"
    #         elif "delete" in operation_lower:
    #             result["operation_type"] = "delete"
    #         else:
    #             result["operation_type"] = "unknown"
    #
    #         # Extract table name
    #         tables = []
    #         for table_name in self.database_schema.keys():
    #             if table_name.lower() in operation_lower:
    #                 tables.append(table_name)
    #
    #         if tables:
    #             result["table"] = tables[0]
    #
    #         return result

    def _format_schema_for_llm(self) -> Dict[str, Any]:
        """Format the full database schema for LLM consumption with rich metadata."""
        if not self.database_schema:
            return {"error": "No schema available"}

        dbname = self.database_schema.get("database_name", "unknown")
        raw_tables = self.database_schema.get("tables", {})

        # Normalize tables into an iterable of (qualified_key, table_info)
        # Supports either dict {"schema.table": {...}} or list [{"name": ...}, ...]
        tables_iter = []
        if isinstance(raw_tables, dict):
            tables_iter = list(raw_tables.items())
        elif isinstance(raw_tables, list):
            # Fallback to list shape like [{"name": "film", "columns": [...]}, ...]
            tables_iter = [(t.get("name"), t) for t in raw_tables]
        else:
            tables_iter = []

        formatted_tables = []
        schema_counts: Dict[str, int] = {}
        total_columns = 0
        total_pks = 0
        total_fks = 0

        for key, tinfo in tables_iter:
            # Try to extract schema and table name robustly
            schema_name = tinfo.get("schema")
            table_name = tinfo.get("table_name") or tinfo.get("name") or key

            if not schema_name and isinstance(key, str) and "." in key:
                schema_name, table_name = key.split(".", 1)

            schema_name = schema_name or "public"

            cols = tinfo.get("columns", [])
            pks = tinfo.get("primary_keys", []) or []
            fks_raw = tinfo.get("foreign_keys", []) or []
            idxs = tinfo.get("indices", []) or []

            # Column formatting with richer metadata
            columns_fmt = []
            for c in cols:
                col_name = c.get("name")
                col_type = str(c.get("type"))
                is_pk = col_name in pks
                # Detect if this column participates in any FK
                fk_refs = []
                for fk in fks_raw:
                    if col_name in (fk.get("constrained_columns") or []):
                        fk_refs.append({
                            "referred_schema": fk.get("referred_schema") or "public",
                            "referred_table": fk.get("referred_table"),
                            "referred_columns": fk.get("referred_columns")
                        })
                columns_fmt.append({
                    "name": col_name,
                    "type": col_type,
                    "nullable": c.get("nullable", True),
                    "default": c.get("default"),
                    "is_primary_key": is_pk,
                    "foreign_key_refs": fk_refs  # empty list if none
                })

            # FK formatting at table level
            foreign_keys_fmt = [{
                "constrained_columns": fk.get("constrained_columns"),
                "referred_schema": fk.get("referred_schema") or "public",
                "referred_table": fk.get("referred_table"),
                "referred_columns": fk.get("referred_columns")
            } for fk in fks_raw]

            # Index formatting (pass through common fields if present)
            indices_fmt = []
            for ix in idxs:
                indices_fmt.append({
                    "name": ix.get("name"),
                    "column_names": ix.get("column_names") or ix.get("column_names".upper()) or ix.get("columns"),
                    "unique": ix.get("unique", False)
                })

            formatted_tables.append({
                "schema": schema_name,
                "name": table_name,
                "qualified_name": f"{schema_name}.{table_name}",
                "columns": columns_fmt,
                "primary_keys": pks,
                "foreign_keys": foreign_keys_fmt,
                "indices": indices_fmt
            })

            # Stats
            schema_counts[schema_name] = schema_counts.get(schema_name, 0) + 1
            total_columns += len(cols)
            total_pks += len(pks)
            total_fks += len(fks_raw)

        # Deterministic ordering
        formatted_tables.sort(key=lambda t: (t["schema"], t["name"]))

        # Build a terse summary
        table_names = [t["qualified_name"] for t in formatted_tables]
        summary = {
            "database_name": dbname,
            "table_count": len(formatted_tables),
            "column_count": total_columns,
            "primary_key_count": total_pks,
            "foreign_key_count": total_fks,
            "tables_by_schema": {sch: schema_counts[sch] for sch in sorted(schema_counts)},
            "table_list": table_names  # keep last for easy truncation upstream if needed
        }

        return {
            "database_name": dbname,
            "tables": formatted_tables,
            "summary": summary
        }

    @staticmethod
    def _extract_sql_from_response(response: str, context: Dict[str, Any]) -> Optional[str]:
        """Extract SQL from agent response or context."""
        if "sql_query" in context:
            return context["sql_query"]

        if "```sql" in response:
            parts = response.split("```sql")
            if len(parts) > 1:
                sql = parts[1].split("```")[0].strip()
                if sql:
                    return sql

        if "```" in response:
            parts = response.split("```")
            for i in range(1, len(parts), 2):
                if i < len(parts):
                    code = parts[i].strip()
                    if code.lower().startswith(("select", "insert", "update", "delete")):
                        return code

        if "operation_details" in context and isinstance(context["operation_details"], str):
            details = context["operation_details"]
            if "```sql" in details:
                sql = details.split("```sql")[1].split("```")[0].strip()
                if sql:
                    return sql

        return None

    @staticmethod
    def _generate_sql_from_llm(query: str, schema_info: Dict[str, Any]) -> Optional[str]:
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

    @staticmethod
    def _extract_table_name_from_sql(sql: str) -> str:
        """Extract table name from a SQL query."""
        sql_lower = sql.lower()
        if "from" in sql_lower:
            from_parts = sql_lower.split("from")[1].strip().split()
            if from_parts:
                table = from_parts[0].rstrip(',;()')
                return table
        return "database"

    def _generate_result_message(self, sql: str, data: List[Dict[str, Any]],
                                 affected_rows: int = None) -> str:
        """Generate appropriate response message based on the query and results."""
        if affected_rows is None:
            affected_rows = len(data)

        operation = "SELECT"
        if sql.lower().startswith("insert"):
            operation = "INSERT"
        elif sql.lower().startswith("update"):
            operation = "UPDATE"
        elif sql.lower().startswith("delete"):
            operation = "DELETE"

        table_name = self._extract_table_name_from_sql(sql) or "the database"

        if operation == "SELECT":
            if affected_rows == 0:
                return f"I couldn't find any matching records in {table_name} for your query."
            elif affected_rows == 1:
                return f"I found 1 record in {table_name} that matches your query."
            else:
                if "limit" in sql.lower() and affected_rows < 20:
                    return f"Here are the {affected_rows} records you requested from {table_name}."
                else:
                    return f"I found {affected_rows} records in {table_name} that match your query."
        elif operation == "INSERT":
            return f"Successfully inserted {affected_rows} record(s) into {table_name}."
        elif operation == "UPDATE":
            return f"Successfully updated {affected_rows} record(s) in {table_name}."
        elif operation == "DELETE":
            return f"Successfully deleted {affected_rows} record(s) from {table_name}."
        else:
            return f"Query executed successfully. Affected {affected_rows} record(s)."

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

            # STEP 5: Create context-aware prompt
            context_aware_schema = self._create_context_aware_prompt(session, formatted_schema)

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
            agent_response = agent_result.get("response", "")
            agent_context = agent_result.get("context", {})

            logger.info(f"Agent response: {agent_response}")

            # STAGE 1: Try to directly extract SQL from the response or context
            sql_query = self._extract_sql_from_response(agent_response, agent_context)

            if sql_query:
                logger.info(f"Executing extracted SQL: {sql_query}")
                db_result = self.db_connector.execute_query(sql_query)

                if db_result.get("success"):
                    data = db_result.get("data", [])
                    affected_rows = db_result.get("affected_rows", 0)  # Get from DB result
                    result = {
                        "success": True,
                        "message": self._generate_result_message(sql_query, data, affected_rows),
                        "agent_response": agent_response,
                        "data": data,
                        "affected_rows": affected_rows,  # Use DB result
                        "sql_query": sql_query
                    }
                    session.add_query(query, result)
                    return result
                else:
                    logger.error(f"SQL execution failed: {db_result.get('error')}")

            # STAGE 2: If direct SQL extraction failed, try to generate SQL using LLM with context
            sql_query = self._generate_sql_from_llm(resolved_query, context_aware_schema)

            if sql_query:
                logger.info(f"Executing context-aware Agent-generated SQL: {sql_query}")
                db_result = self.db_connector.execute_query(sql_query)

                if db_result.get("success"):
                    data = db_result.get("data", [])
                    affected_rows = db_result.get("affected_rows", 0)  # Get from DB result
                    result = {
                        "success": True,
                        "message": self._generate_result_message(sql_query, data, affected_rows),
                        "agent_response": "Agent-generated SQL executed successfully.",
                        "data": data,
                        "affected_rows": affected_rows,  # Use DB result
                        "sql_query": sql_query
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
