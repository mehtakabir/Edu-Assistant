import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from database import setup_database
from graph import get_graph

# setup DB when server starts
setup_database()

print("Server ready", file=sys.stderr)

# create FastMCP server 
mcp = FastMCP("edu-assistant")


@mcp.tool()
async def ask_chatbot(user_id: int, query: str) -> str:
    """
    Ask the AI Education Assistant anything in natural language.
    The chatbot automatically routes to the correct agent.

    user_id:
      1 = Nakul  (student)
      2 = Aslam  (student)
      3 = Kabir  (teacher)

    Examples:
      user_id=1, query="what is machine learning"
      user_id=1, query="show my attendance"
      user_id=3, query="show attendance of Nakul"
      user_id=3, query="generate a quiz on loops"
      user_id=1, query="generate a quiz on loops"  → denied by RBAC
    """

    # call your existing LangGraph pipeline
    # exact same thing your /query route does
    initial_state = {
        "user_id":  user_id,
        "role":     "",
        "query":    query,
        "intent":   "",
        "response": {},
        "error":    "",
    }

    try:
        final_state = get_graph().invoke(initial_state)
    except Exception as e:
        return f"Something went wrong: {str(e)}"

    # RBAC denied or error
    if final_state.get("error"):
        return f"Access Denied: {final_state['error']}"

    # format response
    response = final_state.get("response", {})
    agent    = response.get("agent", "")

    if agent == "rag":
        return response.get("answer", "No answer found.")

    elif agent == "student":
        data = response.get("data", {})
        if not data.get("success"):
            return data.get("message", "Student not found.")
        elif "students" in data:
            lines = ["All Students:\n"]
            for s in data["students"]:
                lines.append(
                    f"Name       : {s['name']}\n"
                    f"Attendance : {s['attendance']}%\n"
                    f"Quiz Marks : {s['quiz_marks']}\n"
                    f"Quiz Status: {s['quiz_status']}\n"
                    f"{'─'*30}"
                )
            return "\n".join(lines)
        else:
            return (
                f"Name       : {data.get('name', '')}\n"
                f"Attendance : {data.get('attendance', '')}%\n"
                f"Quiz Marks : {data.get('quiz_marks', '')}\n"
                f"Quiz Status: {data.get('quiz_status', '')}"
            )

    elif agent == "quiz":
        return response.get("quiz", "Quiz generation failed.")

    return "No response from chatbot."


if __name__ == "__main__":
    mcp.run(transport="stdio")