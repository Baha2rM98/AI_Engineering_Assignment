# LangGraph Database Agent
*An AI-powered natural language interface for database operations*

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/Baha2rM98/AI_Engineering_Assignment/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-orange.svg)](https://langchain-ai.github.io/langgraph/)

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)

## 🎯 Overview

The LangGraph Database Agent is a sophisticated AI-powered system that enables natural language interactions with PostgreSQL databases. Built using the LangGraph framework and Google's Gemini model, it transforms conversational queries into precise SQL operations while maintaining context across sessions.

**What makes this special:**
- 🧠 **Conversational Memory**: Remembers context within sessions
- 🔄 **Multi-step Reasoning**: Uses LangGraph's state-driven workflow
- 🎯 **Context Resolution**: Understands references like "that table" or "show me more"
- 🛡️ **Error Recovery**: Intelligent fallback mechanisms
- 📊 **Real-time Processing**: Immediate query execution and results

## ✨ Key Features

### 🤖 AI Agent Capabilities
- **Natural Language Processing**: Convert human language to SQL
- **Context Awareness**: Maintains conversation history and references
- **Multi-table Operations**: Handles complex joins and relationships
- **Error Handling**: Graceful handling of ambiguous or impossible requests

### 🔧 Technical Features
- **Session Management**: Isolated conversation contexts per user
- **Schema Introspection**: Automatic database structure discovery
- **Query Optimization**: Intelligent SQL generation with proper indexing
- **Memory Management**: Configurable conversation history limits
- **API Security**: Request validation and error handling

### 🏗️ Infrastructure
- **Containerized Deployment**: Docker-ready with CI/CD pipeline
- **Scalable Architecture**: Modular design for easy extension
- **Health Monitoring**: Built-in health checks and logging
- **Environment Management**: Flexible configuration system

## 🏛️ Architecture

[//]: # ()
[//]: # (The system follows a multi-stage LangGraph workflow:)

[//]: # ()
[//]: # (```mermaid)

[//]: # (graph TD)

[//]: # (    A[Natural Language Query] --> B[Query Understanding])

[//]: # (    B --> C[Execution Planning])

[//]: # (    C --> D[SQL Generation])

[//]: # (    D --> E[Database Execution])

[//]: # (    E --> F[Response Formulation])

[//]: # (    F --> G[Natural Language Response])

[//]: # (    )
[//]: # (    H[Session Memory] --> B)

[//]: # (    H --> C)

[//]: # (    H --> F)

[//]: # (    )
[//]: # (    I[Error Handler] --> B)

[//]: # (    I --> C)

[//]: # (    I --> D)

[//]: # (    I --> E)

[//]: # (```)

### Core Components

1. **LangGraph Agent** (`app/agent/langraph_agent.py`)
   - Multi-node workflow processing
   - State management between nodes
   - Error handling and recovery

2. **DB Agent Connector** (`app/agent/db_agent_connector.py`)
   - Session management and memory
   - Context resolution and reference handling
   - SQL generation and execution coordination

3. **Database Connector** (`app/database/db_connector.py`)
   - Low-level database operations
   - Schema introspection and caching
   - Connection pooling and error handling

4. **API Layer** (`app/api/routes.py`)
   - RESTful endpoints
   - Request/response validation
   - Session routing and management

## 🛠️ Technology Stack

### AI & Language Models
- **LangGraph**: State-driven AI agent framework
- **Google Gemini 1.5 Pro**: Advanced language model for query understanding
- **LangChain**: LLM integration and prompt management

### Backend & Database
- **FastAPI**: Modern, high-performance Python web framework
- **PostgreSQL**: Primary database (tested with Sakila sample database)
- **SQLAlchemy**: Database ORM and query building
- **Pydantic**: Data validation and serialization

### DevOps & Deployment
- **Docker**: Containerization and deployment
- **GitHub Actions**: CI/CD pipeline automation
- **pytest**: Testing framework
- **uvicorn**: ASGI server for production

## 🚀 Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Google API Key (for Gemini access)
- Docker (optional, for containerized deployment)

### Local Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/Baha2rM98/AI_Engineering_Assignment.git
cd AI_Engineering_Assignment
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Initialize database**
```bash
# Set up PostgreSQL with Sakila sample database
```

6. **Run the application**
```bash
python app/main.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or use the provided Dockerfile
docker build -f docker/Dockerfile -t langraph-db-agent .
docker run -p 8000:8000 --env-file .env langraph-db-agent
```

## ⚙️ Configuration

### Environment Variables

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=sakila

# AI Model Configuration
GOOGLE_API_KEY=your_gemini_api_key

# Application Settings
PORT=8000
LOG_LEVEL=INFO

# Session Management
MAX_SESSIONS=100
SESSION_TIMEOUT=60
MAX_HISTORY_PER_SESSION=10
```

### Application Settings

The system supports various configuration options:
- Session timeout and cleanup intervals
- Maximum conversation history per session
- Database connection pooling settings
- Logging levels and output formats

## 📖 Usage

### Basic Natural Language Queries

```python
# Example queries the system can handle:

# Data retrieval
"Show me all actors"
"Find films with rating PG-13"
"How many customers are there?"

# Contextual follow-ups
"Show me the first 5"
"How many are there?"
"What about that table?"

# Complex operations
"Find all films by actors named John"
"Show customer payment history"
"List top-grossing film categories"
```

### API Usage

```python
import requests

# Basic query
response = requests.post("http://localhost:8000/query", json={
    "query": "Show me all actors named John",
    "session_id": "user123"
})

# Follow-up query in same session
response = requests.post("http://localhost:8000/query", json={
    "query": "How many are there?",
    "session_id": "user123"  # Same session maintains context
})
```

### Session Management

```python
# Get session information
response = requests.get("http://localhost:8000/sessions/user123")

# Clear session history
response = requests.delete("http://localhost:8000/sessions/user123")

# List active sessions
response = requests.get("http://localhost:8000/sessions")
```

## 📚 API Documentation

### Core Endpoints

#### `POST /query`
Execute a natural language database query.

**Request:**
```json
{
    "query": "Show me all films with rating R",
    "session_id": "optional-session-id"
}
```

**Response:**
```json
{
    "success": true,
    "message": "I found 195 records in film that match your query.",
    "data": [
        {
            "film_id": 1,
            "title": "Academy Dinosaur",
            "rating": "PG"
        },
       {
          etc...
       },
    ],
    "affected_rows": 195,
    "session_id": "user123"
}
```

#### `GET /health`
Check system health and database connectivity.

**Response:**
```json
{
    "status": "connected",
    "database_connection": "ok",
    "active_sessions": 5
}
```

#### `GET /sessions/{session_id}`
Get detailed session information.

**Response:**
```json
{
    "session_id": "user123",
    "created_at": "2024-01-15T10:30:00",
    "last_activity": "2024-01-15T11:45:00",
    "query_count": 12,
    "last_table": "film",
    "last_operation": "select",
    "context_summary": "Recently working with table: film | Last operation: select"
}
```


## 🗄️ Database Schema

The system is tested with the **Sakila DVD Rental Database**, which includes:

### Main Tables
- **film**: Movie catalog with ratings, descriptions, rental rates
- **actor**: Actor information and filmography
- **customer**: Customer data and rental history
- **rental**: Rental transactions and dates
- **payment**: Payment records and amounts
- **inventory**: Store inventory and availability

### Relationship Tables
- **film_actor**: Links films to their cast
- **film_category**: Categorizes films by genre

### Reference Tables
- **category**: Film genres and classifications
- **language**: Available languages
- **address/city/country**: Geographic data

The agent automatically discovers and understands these relationships, enabling complex queries across multiple tables.

## 🔧 Development

### Project Structure

```
AI_Engineering_Assignment/
├── app/
│   ├── agent/
│   │   ├── langraph_agent.py      # Core LangGraph workflow
│   │   └── db_agent_connector.py  # Session management & context
│   ├── api/
│   │   └── routes.py              # FastAPI endpoints
│   ├── database/
│   │   └── db_connector.py        # Database operations
│   └── main.py                    # Application entry point
├── tests/
│   ├── test_agent.py             # Agent functionality tests
│   ├── test_api.py               # API endpoint tests
│   └── test_database.py          # Database operation tests
├── docker/
│   └── Dockerfile                # Container configuration
├── .github/workflows/
│   └── ci-cd.yml                 # CI/CD pipeline
└── requirements.txt              # Python dependencies
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test categories
pytest tests/test_agent.py -v
pytest tests/test_api.py -v
```

### Code Quality

The project follows these standards:
- **PEP 8**: Python code style guidelines
- **Type Hints**: Full type annotation coverage
- **Error Handling**: Comprehensive exception management
- **Logging**: Structured logging for debugging and monitoring
- **Documentation**: Docstrings for all public methods

### Adding New Features

1. **Database Operations**: Extend `DatabaseConnector` for new operation types
2. **Agent Capabilities**: Add nodes to the LangGraph workflow
3. **API Endpoints**: Create new routes in `routes.py`
4. **Session Features**: Enhance `ConversationSession` class

## 🚢 Deployment

### Production Deployment

1. **Environment Setup**
```bash
# Production environment variables
export ENV=production
export LOG_LEVEL=WARNING
export DB_POOL_SIZE=20
```

2. **Docker Deployment**
```bash
# Build production image
docker build -f docker/Dockerfile -t langraph-agent:prod .

# Run with production settings
docker run -d \
  --name langraph-agent \
  -p 8000:8000 \
  --env-file .env.prod \
  langraph-agent:prod
```

3. **Kubernetes Deployment**
```yaml
# Example k8s deployment configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langraph-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langraph-agent
  template:
    metadata:
      labels:
        app: langraph-agent
    spec:
      containers:
      - name: langraph-agent
        image: langraph-agent:prod
        ports:
        - containerPort: 8000
```

### CI/CD Pipeline

The project includes automated GitHub Actions workflows:

- **Testing**: Runs on every push and PR
- **Security Scanning**: Dependency vulnerability checks
- **Docker Build**: Automated image building and pushing
- **Deployment**: Automatic deployment to staging/production

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Process

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Run tests and checks**
   ```bash
   pytest
   black app/ tests/  # Code formatting
   mypy app/         # Type checking
   ```

5. **Submit a pull request**
   - Provide clear description of changes
   - Reference any related issues
   - Ensure all CI checks pass

### Code Guidelines

- **Write Tests**: All new features must include tests
- **Document Changes**: Update README and docstrings
- **Handle Errors**: Implement proper error handling
- **Type Safety**: Use type hints throughout
- **Performance**: Consider the impact on session memory and database operations

### Areas for Contribution

- **New Database Support**: Add connectors for MySQL, MongoDB, etc.
- **Enhanced NLP**: Improve query understanding and context resolution
- **UI Interface**: Build a web frontend for the API
- **Performance Optimization**: Database query optimization and caching
- **Security Features**: Authentication, authorization, and audit logging

---

## 🙏 Acknowledgments

- **LangGraph Team**: For the powerful agent framework
- **Google**: For the Gemini language model
- **FastAPI**: For the excellent web framework
- **PostgreSQL**: For reliable database foundations
- **Sakila Database**: For comprehensive testing data

## 📞 Contact

**Baha2r** - [GitHub](https://github.com/Baha2rM98)

Project Link: [https://github.com/Baha2rM98/AI_Engineering_Assignment](https://github.com/Baha2rM98/AI_Engineering_Assignment)

---

*Built with ❤️ for the AI Engineering Assignment - showcasing the power of LangGraph, natural language processing, and intelligent database interactions.*