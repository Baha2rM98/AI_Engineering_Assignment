from typing import Dict, List, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel


class AgentState(BaseModel):
    query: str
    context: Dict[str, Any] = {}
    database_info: Dict[str, Any] = {}
    current_plan: List[str] = []
    execution_result: Dict[str, Any] = {}
    response: str = ""
    error: str = ""


def initialize_agent():
    """Initialize the LangGraph agent with tools and memory management."""

    # Initialize the language model with Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        convert_system_message_to_human=True  # Important for Gemini compatibility
    )

    # Define the system message for the agent
    system_message = """You are a database assistant that helps users interact with a PostgreSQL database.
    You can interpret user queries, develop plans to retrieve or modify data, and execute those plans.

    DATABASE INFORMATION:
        SAKILA DVD RENTAL DATABASE SCHEMA:

        MAIN TABLES:
        - actor (actor_id, first_name, last_name, last_update)
        - film (film_id, title, description, release_year, language_id, rental_duration, rental_rate, length, replacement_cost, rating, special_features, last_update)
        - customer (customer_id, store_id, first_name, last_name, email, address_id, activebool, create_date, last_update, active)
        - rental (rental_id, rental_date, inventory_id, customer_id, return_date, staff_id, last_update)
        - payment (payment_id, customer_id, staff_id, rental_id, amount, payment_date)
        - inventory (inventory_id, film_id, store_id, last_update)
        - store (store_id, manager_staff_id, address_id, last_update)
        - staff (staff_id, first_name, last_name, address_id, email, store_id, active, username, password, last_update)

        RELATIONSHIP TABLES:
        - film_actor (actor_id, film_id, last_update) - Links films to actors
        - film_category (film_id, category_id, last_update) - Links films to categories

        REFERENCE TABLES:
        - category (category_id, name, last_update)
        - language (language_id, name, last_update)
        - address (address_id, address, address2, district, city_id, postal_code, phone, last_update)
        - city (city_id, city, country_id, last_update)
        - country (country_id, country, last_update)

        VIEWS:
        - actor_info, customer_list, film_list, staff_list, sales_by_film_category, sales_by_store

        RATING ENUM: 'G', 'PG', 'PG-13', 'R', 'NC-17'

        COMMON QUERY PATTERNS:
        - Films by rating: WHERE rating = 'PG-13'
        - Actor search: WHERE first_name LIKE 'John%' 
        - Active customers: WHERE activebool = true
        - Recent rentals: WHERE rental_date > '2005-01-01'
        - Customer payments: JOIN customer ON payment.customer_id = customer.customer_id

    Important instructions:
    1. When asked about tables, make sure to reference ALL tables from the database
    2. For data retrieval, use the appropriate SQL operations (SELECT)
    3. Format your responses clearly and concisely
    4. If the database schema information is incomplete, ask for it

    Your goal is to understand what the user wants to do with the database and help them accomplish it.
    """

    workflow = StateGraph(AgentState)

    # Define the nodes
    # 1. Query understanding node
    def understand_query(state: AgentState) -> AgentState:
        """Parse and understand the user's natural language query."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", "{query}")
        ])

        chain = prompt | llm

        result = chain.invoke({
            "query": state.query,
            "database_info": state.database_info
        })

        # Update state with query understanding
        state.context["understood_intent"] = result.content
        return state

    # 2. Query planning node
    def plan_execution(state: AgentState) -> AgentState:
        """Develop a plan for database operations based on the understood query."""
        if "understood_intent" not in state.context:
            state.error = "Query understanding failed. Cannot create execution plan."
            return state

        # Create a prompt for the planning step
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a database operation planner.
            Your job is to convert natural language database requests into a sequence of specific operations.

            For the database with the following structure:
                SAKILA DVD RENTAL DATABASE SCHEMA:

                MAIN TABLES:
                - actor (actor_id, first_name, last_name, last_update)
                - film (film_id, title, description, release_year, language_id, rental_duration, rental_rate, length, replacement_cost, rating, special_features, last_update)
                - customer (customer_id, store_id, first_name, last_name, email, address_id, activebool, create_date, last_update, active)
                - rental (rental_id, rental_date, inventory_id, customer_id, return_date, staff_id, last_update)
                - payment (payment_id, customer_id, staff_id, rental_id, amount, payment_date)
                - inventory (inventory_id, film_id, store_id, last_update)
                - store (store_id, manager_staff_id, address_id, last_update)
                - staff (staff_id, first_name, last_name, address_id, email, store_id, active, username, password, last_update)

                RELATIONSHIP TABLES:
                - film_actor (actor_id, film_id, last_update) - Links films to actors
                - film_category (film_id, category_id, last_update) - Links films to categories

                REFERENCE TABLES:
                - category (category_id, name, last_update)
                - language (language_id, name, last_update)
                - address (address_id, address, address2, district, city_id, postal_code, phone, last_update)
                - city (city_id, city, country_id, last_update)
                - country (country_id, country, last_update)

                VIEWS:
                - actor_info, customer_list, film_list, staff_list, sales_by_film_category, sales_by_store

                RATING ENUM: 'G', 'PG', 'PG-13', 'R', 'NC-17'

                COMMON QUERY PATTERNS:
                - Films by rating: WHERE rating = 'PG-13'
                - Actor search: WHERE first_name LIKE 'John%' 
                - Active customers: WHERE activebool = true
                - Recent rentals: WHERE rental_date > '2005-01-01'
                - Customer payments: JOIN customer ON payment.customer_id = customer.customer_id

            Create a detailed plan that includes:
            1. The type of operation (select, insert, update, delete)
            2. The specific tables involved
            3. The columns to be affected
            4. Any conditions or filters
            5. Any sorting or grouping requirements

            Format your response as a JSON-like structure that can be parsed and executed.
            """),
            ("user", "User query: {query}\nUnderstood intent: {understood_intent}")
        ])

        chain = planning_prompt | llm

        try:
            result = chain.invoke({
                "query": state.query,
                "understood_intent": state.context["understood_intent"],
                "database_info": state.database_info
            })

            # Parse the plan from the LLM response
            state.context["execution_plan"] = result.content

            # Extract operation type and parameters for later execution
            # This simple extraction will be refined based on the LLM's output format
            if "select" in result.content.lower():
                state.current_plan.append("select")
            elif "insert" in result.content.lower():
                state.current_plan.append("insert")
            elif "update" in result.content.lower():
                state.current_plan.append("update")
            elif "delete" in result.content.lower():
                state.current_plan.append("delete")
            else:
                state.current_plan.append("unknown")

            return state
        except Exception as e:
            state.error = f"Failed to create execution plan: {str(e)}"
            return state

    # 3. Query execution node
    def execute_plan(state: AgentState) -> AgentState:
        """Execute the plan against the database."""
        if "execution_plan" not in state.context:
            state.error = "Execution plan missing. Cannot proceed with execution."
            return state

        # Create a prompt to parse the execution plan into actual database operations
        execution_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a database operation executor.
            Your job is to convert a database operation plan into specific parameters 
            for the database connector.

            The database has the following structure:
                SAKILA DVD RENTAL DATABASE SCHEMA:

                MAIN TABLES:
                - actor (actor_id, first_name, last_name, last_update)
                - film (film_id, title, description, release_year, language_id, rental_duration, rental_rate, length, replacement_cost, rating, special_features, last_update)
                - customer (customer_id, store_id, first_name, last_name, email, address_id, activebool, create_date, last_update, active)
                - rental (rental_id, rental_date, inventory_id, customer_id, return_date, staff_id, last_update)
                - payment (payment_id, customer_id, staff_id, rental_id, amount, payment_date)
                - inventory (inventory_id, film_id, store_id, last_update)
                - store (store_id, manager_staff_id, address_id, last_update)
                - staff (staff_id, first_name, last_name, address_id, email, store_id, active, username, password, last_update)

                RELATIONSHIP TABLES:
                - film_actor (actor_id, film_id, last_update) - Links films to actors
                - film_category (film_id, category_id, last_update) - Links films to categories

                REFERENCE TABLES:
                - category (category_id, name, last_update)
                - language (language_id, name, last_update)
                - address (address_id, address, address2, district, city_id, postal_code, phone, last_update)
                - city (city_id, city, country_id, last_update)
                - country (country_id, country, last_update)

                VIEWS:
                - actor_info, customer_list, film_list, staff_list, sales_by_film_category, sales_by_store

                RATING ENUM: 'G', 'PG', 'PG-13', 'R', 'NC-17'

                COMMON QUERY PATTERNS:
                - Films by rating: WHERE rating = 'PG-13'
                - Actor search: WHERE first_name LIKE 'John%' 
                - Active customers: WHERE activebool = true
                - Recent rentals: WHERE rental_date > '2005-01-01'
                - Customer payments: JOIN customer ON payment.customer_id = customer.customer_id

            Parse the execution plan and extract the exact parameters needed for:
            - operation_type (select, insert, update, delete)
            - table name
            - columns/fields
            - conditions
            - values (for insert/update)

            Format your response as a valid JSON object or SQL query.
            If using JSON, include at least operation_type and table fields.
            If using SQL, provide the complete SQL statement that can be executed directly.
            """),
            ("user", "Execution plan: {execution_plan}")
        ])

        chain = execution_prompt | llm

        try:
            # Extract execution parameters from the plan
            result = chain.invoke({
                "execution_plan": state.context["execution_plan"],
                "database_info": state.database_info
            })

            # Store the operation details for actual execution
            state.context["operation_details"] = result.content

            # Check if the response includes SQL
            if "```sql" in result.content.lower():
                # Extract SQL from code block
                sql = result.content.split("```sql")[1].split("```")[0].strip()
                state.context["sql_query"] = sql
                state.current_plan.append("sql")
            else:
                # Extract operation type from the response
                if "select" in result.content.lower():
                    state.current_plan.append("select")
                elif "insert" in result.content.lower():
                    state.current_plan.append("insert")
                elif "update" in result.content.lower():
                    state.current_plan.append("update")
                elif "delete" in result.content.lower():
                    state.current_plan.append("delete")
                else:
                    state.current_plan.append("unknown")

            # Set placeholder execution result
            # The actual execution happens in DBAgentConnector
            state.execution_result = {
                "success": True,
                "operation": state.current_plan[0] if state.current_plan else "unknown",
                "details": "Database operation plan created successfully.",
                "data": []  # Will be populated by the connector
            }

            return state
        except Exception as e:
            state.error = f"Failed to execute plan: {str(e)}"
            return state

    # 4. Response formulation node
    def formulate_response(state: AgentState) -> AgentState:
        """Generate a natural language response based on execution results."""
        # If there was an error, create an error response
        if state.error:
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful database assistant.
                When operations fail, you should provide a clear explanation of what went wrong
                and suggest possible fixes or alternative approaches.

                Be conversational and helpful in your error messages.
                """),
                ("user", "Error: {error}\nOriginal query: {query}")
            ])

            chain = response_prompt | llm

            result = chain.invoke({
                "error": state.error,
                "query": state.query
            })

            state.response = result.content
            return state

        # Create a prompt for generating a response based on successful execution
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful database assistant.
            Your job is to generate a natural language response explaining the results of 
            database operations in a clear, conversational manner.

            For SELECT operations, summarize what data was retrieved.
            For INSERT, UPDATE, DELETE operations, explain what changes were made.

            Be concise but informative.
            """),
            ("user", "Original query: {query}\nOperation result: {result}")
        ])

        chain = response_prompt | llm

        try:
            result = chain.invoke({
                "query": state.query,
                "result": state.execution_result
            })

            state.response = result.content
            return state
        except Exception as e:
            state.error = f"Failed to formulate response: {str(e)}"
            state.response = f"I encountered an issue while processing your request: {str(e)}"
            return state

    # 5. Error handling node
    def handle_error(state: AgentState) -> AgentState:
        """Handle any errors that occurred during processing."""
        error_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a database troubleshooter.
            Your job is to diagnose database query issues and provide helpful explanations.

            When analyzing errors:
            1. Identify the likely cause of the error
            2. Suggest possible solutions or workarounds
            3. Use a conversational, helpful tone
            4. Avoid technical jargon when possible
            """),
            ("user", "Query: {query}\nError: {error}\nContext: {context}")
        ])

        chain = error_prompt | llm

        try:
            # Generate helpful error message
            context_str = str({k: v for k, v in state.context.items()
                               if k not in ["database_info"]})

            result = chain.invoke({
                "query": state.query,
                "error": state.error,
                "context": context_str
            })

            state.response = result.content
            return state
        except Exception as e:
            # Fallback error handling
            state.response = (f"I'm sorry, but I encountered an error processing your request. "
                              f"Error details: {state.error}")
            return state

    # Add nodes to the graph
    workflow.add_node("understand_query", understand_query)
    workflow.add_node("plan_execution", plan_execution)
    workflow.add_node("execute_plan", execute_plan)
    workflow.add_node("formulate_response", formulate_response)
    workflow.add_node("handle_error", handle_error)

    # Define the edges of the graph
    workflow.add_edge("understand_query", "plan_execution")
    workflow.add_edge("plan_execution", "execute_plan")
    workflow.add_edge("execute_plan", "formulate_response")
    workflow.add_edge("formulate_response", END)

    # Add error handling edges
    workflow.add_conditional_edges(
        "understand_query",
        lambda state: "handle_error" if state.error else "plan_execution"
    )
    workflow.add_conditional_edges(
        "plan_execution",
        lambda state: "handle_error" if state.error else "execute_plan"
    )
    workflow.add_conditional_edges(
        "execute_plan",
        lambda state: "handle_error" if state.error else "formulate_response"
    )
    workflow.add_edge("handle_error", "formulate_response")

    # Set the entrypoint
    workflow.set_entry_point("understand_query")

    return workflow.compile()


def query_database(query: str, database_info: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a query against the database using the LangGraph agent."""
    try:
        print("Initializing LangGraph agent...")
        agent = initialize_agent()

        print(f"Query: {query}")
        print(f"Database info summary: {len(database_info)} tables available")

        # Prepare a concise version of schema for logging
        schema_summary = {}

        if isinstance(database_info, dict):
            # Case 1: formatted schema from _format_schema_for_llm()
            if isinstance(database_info.get("tables"), list):
                for t in database_info["tables"][:3]:
                    if isinstance(t, dict):
                        cols = t.get("columns", []) or []
                        name = t.get("name", "<unknown>")
                        schema_summary[name] = {
                            "columns_count": len(cols),
                            "sample_columns": [
                                c.get("name") for c in cols[:3]
                                if isinstance(c, dict) and "name" in c
                            ]
                        }
            else:
                # Case 2: raw schema like { "actor": {"columns":[...]}, ... }
                for table_name, table_info in list(database_info.items())[:3]:
                    if isinstance(table_info, dict) and "columns" in table_info:
                        cols = table_info.get("columns") or []
                        schema_summary[table_name] = {
                            "columns_count": len(cols),
                            "sample_columns": [
                                c.get("name") for c in cols[:3]
                                if isinstance(c, dict) and "name" in c
                            ]
                        }

        print(f"Schema summary: {schema_summary}")

        initial_state = AgentState(
            query=query,
            database_info=database_info
        )

        print("Invoking agent...")
        result = agent.invoke(initial_state)
        print(f"Agent result type: {type(result)}")

        if hasattr(result, 'response'):
            print(f"Agent response: {result.response[:100]}...")
        else:
            print(f"Agent result has no 'response' attribute: {result}")

        # Ensure we always return a dictionary
        if isinstance(result, dict):
            return result
        elif hasattr(result, 'response'):
            # Convert AgentState to dictionary
            return {
                "response": result.response,
                "context": result.context if hasattr(result, "context") else {},
                "execution_details": result.execution_result if hasattr(result, "execution_result") else {}
            }
        else:
            print(f"WARNING: Unexpected result type from agent: {type(result)}")
            return {
                "response": "The agent could not process your query properly.",
                "context": {},
                "execution_details": {}
            }

    except Exception as e:
        import traceback
        print(f"Error in query_database: {e}")
        print(traceback.format_exc())

        # Return error result instead of None
        return {
            "response": f"Error processing query: {str(e)}",
            "context": {"error": str(e)},
            "execution_details": {"error": traceback.format_exc()}
        }
