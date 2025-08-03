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
        db_name = os.getenv("DB_NAME", "sakila")

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


# @app.get("/tables")
# def get_tables(db_agent: DBAgentConnector = Depends(get_db_agent)):
#     """Get all tables in the database."""
#     try:
#         tables = db_agent.db_connector.get_table_names()
#         return {
#             "success": True,
#             "tables": tables
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get tables: {str(e)}")


# @app.get("/schema/{table_name}")
# def get_table_schema(table_name: str, db_agent: DBAgentConnector = Depends(get_db_agent)):
#     """Get schema for a specific table."""
#     try:
#         schema = db_agent.db_connector.get_table_schema(table_name)
#         return {
#             "success": True,
#             "schema": schema
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")


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


import traceback
import logging

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


# @app.post("/debug-query")
# def debug_query(request: QueryRequest, db_agent: DBAgentConnector = Depends(get_db_agent)):
#     """Debug version of the query endpoint with more verbose output."""
#     if not request.query:
#         raise HTTPException(status_code=400, detail="Query cannot be empty")
#
#     try:
#         # Test direct LLM connection first
#         from langchain_google_genai import ChatGoogleGenerativeAI
#         from langchain.prompts import ChatPromptTemplate
#
#         # Simple test of Gemini connection
#         llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, convert_system_message_to_human=True)
#         test_prompt = ChatPromptTemplate.from_messages([
#             ("human", "You are a helpful assistant. Say hello!")
#         ])
#
#         chain = test_prompt | llm
#         llm_test_result = chain.invoke({})
#
#         # Get the database schema
#         schema = db_agent.database_schema
#         schema_sample = {k: {"columns": [c["name"] for c in v["columns"]]}
#                          for k, v in list(schema.items())[:3]} if schema else {}
#
#         # Now try the actual query but catch any errors
#         result = None
#         error = None
#         traceback_info = None
#
#         try:
#             result = db_agent.execute_natural_language_query(request.query)
#         except Exception as e:
#             import traceback
#             error = str(e)
#             traceback_info = traceback.format_exc()
#
#         # Return comprehensive debug info
#         return {
#             "query": request.query,
#             "gemini_test": {
#                 "success": True,
#                 "response": llm_test_result.content if hasattr(llm_test_result, "content") else str(llm_test_result)
#             },
#             "database_schema_sample": schema_sample,
#             "schema_tables_count": len(schema) if schema else 0,
#             "result": result,
#             "error": error,
#             "traceback": traceback_info
#         }
#     except Exception as e:
#         import traceback
#         return {
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }

@app.post("/direct-sql")
def execute_direct_sql(request: dict):
    """Execute SQL directly for debugging purposes."""
    if "sql" not in request:
        raise HTTPException(status_code=400, detail="SQL query is required")

    try:
        # Get the DB connector from the agent
        db_agent = get_db_agent()

        # Execute the SQL directly
        result = db_agent.db_connector.execute_query(request["sql"])

        return {
            "success": result.get("success", False),
            "data": result.get("data"),
            "affected_rows": result.get("affected_rows", 0),
            "error": result.get("error")
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/direct-sql")
def execute_direct_sql(request: dict):
    """Execute SQL directly for debugging purposes."""
    if "sql" not in request:
        raise HTTPException(status_code=400, detail="SQL query is required")

    try:
        # Get the DB connector from the agent
        db_agent = get_db_agent()

        # Execute the SQL directly
        result = db_agent.db_connector.execute_query(request["sql"])

        return {
            "success": result.get("success", False),
            "data": result.get("data"),
            "affected_rows": result.get("affected_rows", 0),
            "error": result.get("error")
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
