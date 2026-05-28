import re
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from database import get_user_role
from agents.rbac_helper import check_permission_by_role
from agents.router_agent import run_router_agent
from agents.rag_agent import run_rag_agent
from agents.quiz_agent import generate_quiz
from agents.student_info_agent import (
    get_student_info,
    get_student_by_name,
    get_all_students,
    get_all_student_names,
)


# ─── State definition ─────────────────────────────────────────
# Every field here is shared across all nodes in the graph.
# 'messages' uses add_messages so LangGraph APPENDS new messages
# instead of replacing the whole list.

class State(TypedDict):
    user_id:  int
    role:     str
    query:    str
    intent:   str
    response: dict
    error:    str
    messages: Annotated[list, add_messages]


# ─── Helper functions ─────────────────────────────────────────

def extract_student_id(query: str):
    """
    Extract a student ID only when the query clearly mentions one.
    Matches patterns like: 'student 2', 'id 2', 'id=2', 'student id 2'
    This avoids false matches on queries like 'top 5 students'.
    """
    # Look for 'id' or 'student' followed by a number
    match = re.search(r'\b(?:student\s+id|student|id)\s*[=:]?\s*(\d+)\b', query, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_student_name(query: str, known_names: list):
    """Check if any known student name appears in the query."""
    query_lower = query.lower()
    for name in known_names:
        if name.lower() in query_lower:
            return name
    return None


def extract_requested_field(query: str) -> str:
    """
    Figure out which data field the user is asking about.
    Returns: 'attendance', 'quiz_status', 'quiz_marks', or 'all'
    """
    query_lower = query.lower()
    if any(w in query_lower for w in ["attendance", "present", "absent"]):
        return "attendance"
    elif "quiz status" in query_lower or "status" in query_lower:
        return "quiz_status"
    elif any(w in query_lower for w in ["quiz mark", "quiz score", "marks", "score", "result"]):
        return "quiz_marks"
    else:
        return "all"


# ─── Graph nodes ──────────────────────────────────────────────
# Each node receives the current state, does its work, and
# returns the updated state (only the fields it changed).

def node_rbac(state: State) -> State:
    """Step 1: Look up the user's role from the database."""
    print(f"\n{'─'*50}", file=sys.stderr)
    print(f"  User     : {state['user_id']}", file=sys.stderr)
    print(f"  Query    : {state['query']}", file=sys.stderr)
    try:
        state["role"] = get_user_role(state["user_id"])
        print(f"  Role     : {state['role'].upper()}", file=sys.stderr)
    except ValueError as e:
        state["error"] = str(e)
        print(f"  Error    : {state['error']}", file=sys.stderr)
    print(f"{'─'*50}", file=sys.stderr)
    return state


def node_router(state: State) -> State:
    """Step 2: Decide which agent should handle this query."""
    if state.get("error"):
        return state
    state["intent"] = run_router_agent(state["query"])
    print(f"  Routing to → {state['intent'].replace('_', ' ').upper()}", file=sys.stderr)
    return state


def node_rag(state: State) -> State:
    """Step 3a: Answer a question from the uploaded PDF notes."""
    print(f"  Agent    : RAG AGENT", file=sys.stderr)

    if not check_permission_by_role(state["role"], "rag", "read"):
        state["error"] = "You don't have permission to query study materials."
        print(f"  Status   : DENIED", file=sys.stderr)
        return state

    print(f"  Status   : ALLOWED", file=sys.stderr)
    print(f"  Action   : Searching PDF notes...", file=sys.stderr)
    answer = run_rag_agent(state["query"])
    state["response"] = {"agent": "rag", "answer": answer}

    # FIX: Return new messages as a list — LangGraph's add_messages
    # reducer will APPEND these to the existing history automatically.
    state["messages"] = [
        HumanMessage(content=state["query"]),
        AIMessage(content=answer),
    ]

    print(f"  Result   :", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    print(answer, file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    return state


def node_student(state: State) -> State:
    """Step 3b: Fetch student information (attendance, marks, etc.)."""
    print(f"  Agent    : STUDENT AGENT", file=sys.stderr)

    if not check_permission_by_role(state["role"], "student", "read"):
        state["error"] = "You don't have permission to view student information."
        print(f"  Status   : DENIED", file=sys.stderr)
        return state

    field = extract_requested_field(state["query"])

    # ── Student accessing their own data ──
    if state["role"] == "student":
        known_names = get_all_student_names()
        own_info    = get_student_info(state["user_id"], "all")
        own_name    = own_info["name"].lower() if own_info["success"] else ""
        asked_name  = extract_student_name(state["query"], known_names)
        asked_id    = extract_student_id(state["query"])

        # Block students from viewing other students' data
        if asked_name and asked_name.lower() != own_name:
            state["error"] = f"Access denied. You are not allowed to view {asked_name}'s data."
            print(f"  Status   : DENIED — tried to access {asked_name}'s data", file=sys.stderr)
            return state

        if asked_id and asked_id != state["user_id"]:
            state["error"] = f"Access denied. You are not allowed to view data of student id {asked_id}."
            print(f"  Status   : DENIED — tried to access student id {asked_id}", file=sys.stderr)
            return state

        print(f"  Status   : ALLOWED", file=sys.stderr)
        print(f"  Action   : Fetching own data | field: {field}", file=sys.stderr)
        result = get_student_info(state["user_id"], field)
        state["response"] = {"agent": "student", "data": result}

        summary = f"Student data fetched: {result}"
        state["messages"] = [
            HumanMessage(content=state["query"]),
            AIMessage(content=summary),
        ]

        print(f"  Result   :", file=sys.stderr)
        print(f"  {'─'*30}", file=sys.stderr)
        for key, val in result.items():
            if key != "success":
                print(f"  {key:<15}: {val}", file=sys.stderr)
        print(f"  {'─'*30}", file=sys.stderr)
        return state

    # ── Teacher accessing any student's data ──
    known_names = get_all_student_names()
    name        = extract_student_name(state["query"], known_names)
    student_id  = extract_student_id(state["query"])

    print(f"  Status   : ALLOWED", file=sys.stderr)

    if name:
        print(f"  Action   : Fetching data for '{name}' | field: {field}", file=sys.stderr)
        result = get_student_by_name(name, field)
    elif student_id:
        print(f"  Action   : Fetching data for student id {student_id} | field: {field}", file=sys.stderr)
        result = get_student_info(student_id, field)
    else:
        print(f"  Action   : Fetching all students", file=sys.stderr)
        result = get_all_students()

    state["response"] = {"agent": "student", "data": result}

    summary = f"Student data fetched: {result}"
    state["messages"] = [
        HumanMessage(content=state["query"]),
        AIMessage(content=summary),
    ]

    print(f"  Result   :", file=sys.stderr)
    print(f"  {'─'*30}", file=sys.stderr)
    if "students" in result:
        for s in result["students"]:
            for key, val in s.items():
                print(f"  {key:<15}: {val}", file=sys.stderr)
            print(f"  {'─'*30}", file=sys.stderr)
    else:
        for key, val in result.items():
            if key != "success":
                print(f"  {key:<15}: {val}", file=sys.stderr)
        print(f"  {'─'*30}", file=sys.stderr)
    return state


def node_quiz(state: State) -> State:
    """Step 3c: Generate a quiz on a topic (teachers only)."""
    print(f"  Agent    : QUIZ AGENT", file=sys.stderr)

    if not check_permission_by_role(state["role"], "quiz", "generate"):
        state["error"] = "Only teachers can generate quizzes."
        print(f"  Status   : DENIED — only teachers can generate quizzes", file=sys.stderr)
        return state

    print(f"  Status   : ALLOWED", file=sys.stderr)
    print(f"  Action   : Generating quiz...", file=sys.stderr)
    result = generate_quiz(state["query"])
    state["response"] = {"agent": "quiz", "quiz": result}

    state["messages"] = [
        HumanMessage(content=state["query"]),
        AIMessage(content=result),
    ]

    print(f"  Result   :", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    print(result, file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    return state


# ─── Routing logic ────────────────────────────────────────────

def decide_next_node(state: State) -> str:
    """After routing, pick which agent node to run."""
    if state.get("error"):
        return END
    intent = state.get("intent", "rag_query")
    if intent == "student_query":
        return "student"
    elif intent == "quiz_query":
        return "quiz"
    else:
        return "rag"


# ─── Build the graph ──────────────────────────────────────────

# InMemorySaver stores conversation history while the server is running.
# History is lost when the server restarts (that's fine for development).
_checkpointer = InMemorySaver()


def _build_graph():
    graph = StateGraph(State)

    # Add all nodes
    graph.add_node("rbac",    node_rbac)
    graph.add_node("router",  node_router)
    graph.add_node("rag",     node_rag)
    graph.add_node("student", node_student)
    graph.add_node("quiz",    node_quiz)

    # Define the flow
    graph.set_entry_point("rbac")
    graph.add_edge("rbac", "router")
    graph.add_conditional_edges(
        "router",
        decide_next_node,
        {"rag": "rag", "student": "student", "quiz": "quiz", END: END}
    )
    graph.add_edge("rag",     END)
    graph.add_edge("student", END)
    graph.add_edge("quiz",    END)

    return graph.compile(checkpointer=_checkpointer)


_graph = _build_graph()


def get_graph():
    return _graph