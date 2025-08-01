from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from app.agent.db_agent_connector import DBAgentConnector
import os

app = FastAPI(
    title="LangGraph Database Agent API",
    description="API for natural language database interactions using LangGraph",
    version="1.0.0"
)

# Global instance of the DB agent connector
db_agent = None


# Dependency to get the DB agent connector
def get_db_agent():
    global db_agent
    if db_agent is None:
        # Build connection string from environment variables
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "postgres")
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "langagent")

        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        db_agent = DBAgentConnector(connection_string)

    return db_agent


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    success: bool
    message: str
    data: Optional[List[Dict[str, Any]]] = None
    affected_rows: Optional[int] = None


@app.get("/")
def read_root():
    """Root endpoint that returns API information."""
    return {
        "name": "LangGraph Database Agent API",
        "version": "1.0.0",
        "description": "Natural language interface for database operations"
    }


@app.post("/query", response_model=QueryResponse)
def process_query(request: QueryRequest, db_agent: DBAgentConnector = Depends(get_db_agent)):
    """
    Process a natural language query and execute it against the database.

    Args:
        request (QueryRequest): The query request

    Returns:
        QueryResponse: The query response
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = db_agent.execute_natural_language_query(request.query)

    return QueryResponse(
        success=result.get("success", False),
        message=result.get("agent_response", ""),
        data=result.get("data"),
        affected_rows=result.get("affected_rows")
    )


@app.get("/tables")
def get_tables(db_agent: DBAgentConnector = Depends(get_db_agent)):
    """Get all tables in the database."""
    try:
        tables = db_agent.db_connector.get_table_names()
        return {
            "success": True,
            "tables": tables
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tables: {str(e)}")


@app.get("/schema/{table_name}")
def get_table_schema(table_name: str, db_agent: DBAgentConnector = Depends(get_db_agent)):
    """Get schema for a specific table."""
    try:
        schema = db_agent.db_connector.get_table_schema(table_name)
        return {
            "success": True,
            "schema": schema
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")


@app.get("/health")
def health_check(db_agent: DBAgentConnector = Depends(get_db_agent)):
    """Health check endpoint that tests the database connection."""
    connection_ok = db_agent.db_connector.test_connection()

    if not connection_ok:
        raise HTTPException(status_code=503, detail="Database connection failed")

    return {
        "status": "healthy",
        "database_connection": "ok"
    }
