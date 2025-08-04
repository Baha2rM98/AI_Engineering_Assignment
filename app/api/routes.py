from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from app.agent.db_agent_connector import DBAgentConnector
import os
import traceback
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LangGraph Database Agent API",
    description="API for natural language database interactions using LangGraph with Memory Management",
    version="2.0.0"
)

# Global instance of the DB agent connector
db_agent = None


def get_db_agent():
    global db_agent
    if db_agent is None:
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
    session_id: Optional[str] = "default"


class QueryResponse(BaseModel):
    success: bool
    message: str
    data: Optional[List[Dict[str, Any]]] = None
    affected_rows: Optional[int] = None
    session_id: Optional[str] = None
    context_info: Optional[Dict[str, Any]] = None


class SessionInfoResponse(BaseModel):
    session_id: str
    created_at: str
    last_activity: str
    query_count: int
    last_table: Optional[str] = None
    last_operation: Optional[str] = None
    context_summary: str


@app.get("/")
def read_root():
    return {
        "name": "LangGraph Database Agent API with Memory Management",
        "version": "2.0.0",
        "description": "Natural language interface for database operations with conversation memory",
        "features": [
            "Session-based conversation memory",
            "Contextual reference resolution",
            "Multi-user session isolation",
            "Conversation history tracking"
        ]
    }


@app.get("/health")
def health_check(db_agent: DBAgentConnector = Depends(get_db_agent)):
    connection_ok = db_agent.db_connector.test_connection()
    if not connection_ok:
        raise HTTPException(status_code=503, detail="Database connection failed")
    return {
        "status": "healthy",
        "database_connection": "ok",
        "active_sessions": len(db_agent.conversation_sessions)
    }


@app.post("/query", response_model=QueryResponse)
def process_query(request: QueryRequest, db_agent: DBAgentConnector = Depends(get_db_agent)):
    """Process a natural language query with session-based memory management."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        logger.info(f"Processing query for session {request.session_id}: {request.query}")

        result = db_agent.execute_natural_language_query(
            request.query,
            session_id=request.session_id
        )

        # Get session info for context
        session_info = db_agent.get_session_info(request.session_id)

        return QueryResponse(
            success=result.get("success", False),
            message=result.get("agent_response", ""),
            data=result.get("data"),
            affected_rows=result.get("affected_rows"),
            session_id=request.session_id,
            context_info=session_info if session_info.get("session_id") else None
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/sessions", response_model=List[str])
def get_active_sessions(db_agent: DBAgentConnector = Depends(get_db_agent)):
    """Get list of active session IDs."""
    return db_agent.get_active_sessions()


@app.get("/sessions/{session_id}", response_model=SessionInfoResponse)
def get_session_info(session_id: str, db_agent: DBAgentConnector = Depends(get_db_agent)):
    """Get detailed information about a specific session."""
    session_info = db_agent.get_session_info(session_id)

    if "error" in session_info:
        raise HTTPException(status_code=404, detail=session_info["error"])

    return SessionInfoResponse(**session_info)


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str, db_agent: DBAgentConnector = Depends(get_db_agent)):
    """Clear a specific session and its conversation history."""
    success = db_agent.clear_session(session_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return {"message": f"Session {session_id} cleared successfully"}
