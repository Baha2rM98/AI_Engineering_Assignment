import os
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError


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
        self._metadata_cache = {}

    def test_connection(self) -> bool:
        """Test if the database connection is working."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except SQLAlchemyError:
            return False

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a raw SQL query and return the results."""
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))

                if result.returns_rows:
                    columns = result.keys()
                    rows = [dict(zip(columns, row)) for row in result.fetchall()]
                    return {
                        "success": True,
                        "data": rows,
                        "affected_rows": result.rowcount
                    }
                else:
                    return {
                        "success": True,
                        "affected_rows": result.rowcount
                    }
        except SQLAlchemyError as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_table_names(self) -> List[str]:
        """Get all table names in the database."""
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get the schema for a specific table."""
        if table_name in self._metadata_cache:
            return self._metadata_cache[table_name]

        inspector = inspect(self.engine)

        columns = []
        for column in inspector.get_columns(table_name):
            columns.append({
                "name": column["name"],
                "type": str(column["type"]),
                "nullable": column.get("nullable", True),
                "default": str(column.get("default", ""))
            })

        primary_keys = inspector.get_pk_constraint(table_name)
        foreign_keys = []
        for fk in inspector.get_foreign_keys(table_name):
            foreign_keys.append({
                "constrained_columns": fk["constrained_columns"],
                "referred_table": fk["referred_table"],
                "referred_columns": fk["referred_columns"]
            })

        indices = inspector.get_indexes(table_name)

        table_schema = {
            "table_name": table_name,
            "columns": columns,
            "primary_keys": primary_keys.get("constrained_columns", []),
            "foreign_keys": foreign_keys,
            "indices": indices
        }

        # Cache the schema
        self._metadata_cache[table_name] = table_schema

        return table_schema

    def get_database_schema(self) -> Dict[str, Any]:
        """Get schema for all tables in the database."""
        tables = self.get_table_names()
        schema = {}

        for table in tables:
            schema[table] = self.get_table_schema(table)

        return schema

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
