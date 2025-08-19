import os
import re
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
import logging

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Class to handle database connections and operations."""

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the database connector with connection details."""
        if connection_string:
            self.connection_string = connection_string
        else:
            # Default to environment variables
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_name = os.getenv("DB_NAME")

            self.connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        self.engine = create_engine(self.connection_string)
        # self._metadata_cache = {}

    def test_connection(self) -> bool:
        """Test if the database connection is working."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except SQLAlchemyError:
            return False

    def _detect_operation_type(self, query: str) -> str:
        """Detect the type of SQL operation."""
        query_lower = query.strip().lower()
        if query_lower.startswith('select'):
            return 'select'
        elif query_lower.startswith('insert'):
            return 'insert'
        elif query_lower.startswith('update'):
            return 'update'
        elif query_lower.startswith('delete'):
            return 'delete'
        else:
            return 'other'

    def _add_returning_clause(self, query: str, operation_type: str) -> str:
        """Add RETURNING clause to INSERT/UPDATE queries if not present."""
        if operation_type not in ['insert', 'update']:
            return query

        query_lower = query.lower()

        # Check if RETURNING clause already exists
        if 'returning' in query_lower:
            return query

        # Extract table name to get primary key
        table_name = None
        if operation_type == 'insert':
            match = re.search(r'insert\s+into\s+(\w+)', query_lower)
            if match:
                table_name = match.group(1)
        elif operation_type == 'update':
            match = re.search(r'update\s+(\w+)', query_lower)
            if match:
                table_name = match.group(1)

        if table_name:
            try:
                # Get primary key for the table
                inspector = inspect(self.engine)
                pk_constraint = inspector.get_pk_constraint(table_name)
                primary_keys = pk_constraint.get('constrained_columns', [])

                if primary_keys:
                    # Add RETURNING clause with primary key
                    pk_columns = ', '.join(primary_keys)
                    query = f"{query.rstrip(';')} RETURNING {pk_columns}"
                else:
                    # Fallback: return all columns
                    query = f"{query.rstrip(';')} RETURNING *"

            except Exception as e:
                logger.warning(f"Could not add RETURNING clause for table {table_name}: {e}")
                # Try generic RETURNING *
                query = f"{query.rstrip(';')} RETURNING *"

        return query

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a raw SQL query and return the results with improved INSERT/UPDATE/DELETE handling."""
        try:
            operation_type = self._detect_operation_type(query)

            # For INSERT/UPDATE operations, add RETURNING clause if needed
            if operation_type in ['insert', 'update']:
                original_query = query
                query = self._add_returning_clause(query, operation_type)
                if query != original_query:
                    logger.info(f"Added RETURNING clause to {operation_type} query")

            with self.engine.connect() as conn:
                # Start transaction for modification operations
                if operation_type in ['insert', 'update', 'delete']:
                    trans = conn.begin()
                    try:
                        if params:
                            result = conn.execute(text(query), params)
                        else:
                            result = conn.execute(text(query))
                        trans.commit()  # COMMIT THE TRANSACTION
                    except Exception as e:
                        trans.rollback()
                        raise e
                else:
                    # For SELECT, no transaction needed
                    if params:
                        result = conn.execute(text(query), params)
                    else:
                        result = conn.execute(text(query))

                # Handle different operation types
                if operation_type == 'select' or result.returns_rows:
                    columns = result.keys()
                    rows = [dict(zip(columns, row)) for row in result.fetchall()]

                    # For INSERT/UPDATE with RETURNING, count the returned rows
                    if operation_type in ['insert', 'update'] and rows:
                        affected_rows = len(rows)
                    else:
                        affected_rows = result.rowcount if result.rowcount >= 0 else len(rows)

                    return {
                        "success": True,
                        "data": rows,
                        "affected_rows": affected_rows,
                        "operation_type": operation_type
                    }
                else:
                    # For DELETE or operations without RETURNING
                    affected_rows = result.rowcount if result.rowcount >= 0 else 0

                    # Special handling for operations that might not report rowcount correctly
                    if operation_type in ['insert', 'update', 'delete'] and result.rowcount == -1:
                        # Try to estimate based on the operation
                        if operation_type == 'insert':
                            # For INSERT without RETURNING, assume 1 row if no error
                            affected_rows = 1
                        else:
                            affected_rows = 0

                    return {
                        "success": True,
                        "data": [],
                        "affected_rows": affected_rows,
                        "operation_type": operation_type
                    }

        except SQLAlchemyError as e:
            error_msg = str(e)
            logger.error(f"SQL execution failed: {error_msg}")

            # Check for specific error types
            if "duplicate key" in error_msg.lower():
                error_msg = "Cannot insert duplicate record - this entry already exists."
            elif "foreign key" in error_msg.lower():
                error_msg = "Cannot complete operation - referenced record does not exist."
            elif "not null" in error_msg.lower():
                error_msg = "Missing required field(s) - please provide all necessary information."

            return {
                "success": False,
                "error": error_msg,
                "operation_type": self._detect_operation_type(query)
            }

    def get_table_names(self, schema: Optional[str] = None) -> List[str]:
        """Get all table names in the database."""
        inspector = inspect(self.engine)
        return sorted(inspector.get_table_names(schema=schema))

    def get_table_schema(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Get the schema for a specific table."""
        inspector = inspect(self.engine)
        cols = inspector.get_columns(table_name, schema=schema)
        pks = inspector.get_pk_constraint(table_name, schema=schema)
        fks = inspector.get_foreign_keys(table_name, schema=schema)
        idx = inspector.get_indexes(table_name, schema=schema)
        return {
            "schema": schema or inspector.default_schema_name,
            "table_name": table_name,
            "columns": [
                {"name": c["name"], "type": str(c["type"]), "nullable": c.get("nullable", True),
                 "default": str(c.get("default", ""))}
                for c in cols
            ],
            "primary_keys": pks.get("constrained_columns", []),
            "foreign_keys": [
                {
                    "constrained_columns": fk["constrained_columns"],
                    "referred_schema": fk.get("referred_schema"),
                    "referred_table": fk["referred_table"],
                    "referred_columns": fk["referred_columns"],
                }
                for fk in fks
            ],
            "indices": idx,
        }

    def get_database_schema(self) -> Dict[str, Any]:
        """Get schema for all tables in the database."""
        inspector = inspect(self.engine)
        # Skip system schemas; include everything else
        skip = {"information_schema", "pg_catalog"}
        schemas = [s for s in inspector.get_schema_names() if s not in skip]
        db = {"database_name": self.engine.url.database, "tables": {}}

        for sch in sorted(schemas):
            for tbl in self.get_table_names(schema=sch):
                db["tables"][f"{sch}.{tbl}"] = self.get_table_schema(tbl, schema=sch)
        return db

    def execute_operation(self, operation_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute common database operations based on operation type."""
        if operation_type == "select":
            table = params["table"]
            columns = params.get("columns", ["*"])
            where = params.get("where", "")
            limit = params.get("limit", "")
            order_by = params.get("order_by", "")

            query = f"SELECT {', '.join(columns)} FROM {table}"
            if where:
                query += f" WHERE {where}"
            if order_by:
                query += f" ORDER BY {order_by}"
            if limit:
                query += f" LIMIT {limit}"

            return self.execute_query(query)

        elif operation_type == "insert":
            table = params["table"]
            values = params["values"]

            columns = ", ".join(values.keys())
            placeholders = ", ".join([f":{k}" for k in values.keys()])

            # The execute_query method will automatically add RETURNING clause
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            return self.execute_query(query, values)

        elif operation_type == "update":
            table = params["table"]
            values = params["values"]
            where = params.get("where", "")

            set_clause = ", ".join([f"{k} = :{k}" for k in values.keys()])
            query = f"UPDATE {table} SET {set_clause}"
            if where:
                query += f" WHERE {where}"

            # The execute_query method will automatically add RETURNING clause
            return self.execute_query(query, values)

        elif operation_type == "delete":
            table = params["table"]
            where = params.get("where", "")

            query = f"DELETE FROM {table}"
            if where:
                query += f" WHERE {where}"

            return self.execute_query(query)

        else:
            return {
                "success": False,
                "error": f"Unsupported operation type: {operation_type}"
            }
