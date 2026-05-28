from fastapi import APIRouter, HTTPException
from schemas.request_schemas import QueryRequest
from graph import get_graph

router = APIRouter(tags=["Query"])


@router.post("/query")
def query(request: QueryRequest):
    # Each user gets their own conversation thread.
    # This is the same pattern used in mcp_server.py.
    config = {"configurable": {"thread_id": str(request.user_id)}}

    initial_state = {
        "user_id":  request.user_id,
        "role":     "",
        "query":    request.query,
        "intent":   "",
        "response": {},
        "error":    "",
        "messages": [],
    }

    final_state = get_graph().invoke(initial_state, config=config)

    if final_state.get("error"):
        raise HTTPException(status_code=403, detail=final_state["error"])

    return {
        "user_id": request.user_id,
        "query":   request.query,
        "intent":  final_state["intent"],
        "result":  final_state["response"]
    }