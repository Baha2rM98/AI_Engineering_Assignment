from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from app.agent.db_agent_connector import DBAgentConnector
import os
import traceback
import logging

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
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")

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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/query", response_model=QueryResponse)
def process_query(request: QueryRequest, db_agent: DBAgentConnector = Depends(get_db_agent)):
    """
    Process a natural language query and execute it against the database.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Log the request for debugging
        logger.info(f"Processing query: {request.query}")

        result = db_agent.execute_natural_language_query(request.query)

        # Add null check
        if result is None:
            logger.error("execute_natural_language_query returned None")
            return QueryResponse(
                success=False,
                message="Error processing query: No result returned from the database agent",
                data=None,
                affected_rows=None
            )

        # Log the result for debugging
        logger.info(f"Query result: {result}")

        return QueryResponse(
            success=result.get("success", False),
            message=result.get("agent_response", ""),
            data=result.get("data"),
            affected_rows=result.get("affected_rows")
        )
    except Exception as e:
        # Log the full exception with stack trace
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
