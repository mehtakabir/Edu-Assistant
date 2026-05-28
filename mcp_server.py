import sys
import os
import io
import shutil

# ── CRITICAL: silence stdout during ALL imports ──────────────
# MCP uses stdout as its JSON-RPC transport.
# LangChain / LangGraph / HuggingFace print to stdout on import
# which corrupts the handshake and prevents the Inspector connecting.
_real_stdout = sys.stdout
sys.stdout   = io.StringIO()          # swallow any import-time stdout

sys.stderr.reconfigure(encoding="utf-8")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from database import setup_database
from graph import get_graph
from agents.rbac_helper import check_permission
from agents.rag_agent import upload_pdf
from agents.pdf_manager import list_pdfs, delete_pdf
from config import PDF_DIR

sys.stdout = _real_stdout             # restore stdout — MCP can now use it
sys.stdout.reconfigure(encoding="utf-8")

setup_database()
print("Server ready", file=sys.stderr)

mcp = FastMCP("edu-assistant")


# ══════════════════════════════════════════════════════════════
#  RESPONSE FORMATTER
# ══════════════════════════════════════════════════════════════

def _format_response(final_state: dict) -> str:
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


# ══════════════════════════════════════════════════════════════
#  MCP TOOLS
# ══════════════════════════════════════════════════════════════

@mcp.tool()
async def ask_chatbot(user_id: int, query: str) -> str:
    """
    Ask the AI Education Assistant in natural language.
    Remembers recent conversation — follow-up questions work.

    user_id:
      1 = Nakul  (student)
      2 = Aslam  (student)
      3 = Kabir  (teacher)

    Examples:
      user_id=1, query="what is machine learning"
      user_id=1, query="give me an example of that"
      user_id=1, query="show my attendance"
      user_id=3, query="show attendance of Nakul"
      user_id=3, query="generate a quiz on neural networks"
      user_id=1, query="generate a quiz on loops"  → denied by RBAC
    """
    # Each user gets their own isolated conversation thread.
    # LangGraph automatically loads the last checkpoint for this
    # thread_id before running, and saves a new checkpoint after.
    config = {"configurable": {"thread_id": str(user_id)}}

    initial_state = {
        "user_id":  user_id,
        "role":     "",
        "query":    query,      # plain query — no manual context building needed
        "intent":   "",
        "response": {},
        "error":    "",
        "messages": [],         # LangGraph will merge this with existing history
    }

    try:
        final_state = get_graph().invoke(initial_state, config=config)
    except Exception as e:
        return f"Something went wrong: {str(e)}"

    if final_state.get("error"):
        return f"Access Denied: {final_state['error']}"

    return _format_response(final_state)


@mcp.tool()
async def clear_memory(user_id: int) -> str:
    """
    Clear conversation history for a user.
    Use when starting a new topic so old context doesn't interfere.

    user_id: 1=Nakul, 2=Aslam, 3=Kabir
    """
    # With LangGraph MemorySaver, clearing memory means wiping the
    # checkpoint for this user's thread. We do this by invoking the
    # graph with an empty messages list and a fresh state — the
    # checkpointer will overwrite the old snapshot.
    config = {"configurable": {"thread_id": str(user_id)}}

    reset_state = {
        "user_id":  user_id,
        "role":     "",
        "query":    "__reset__",
        "intent":   "",
        "response": {},
        "error":    "reset",    # short-circuits all nodes via decide_next_node → END
        "messages": [],
    }

    try:
        get_graph().invoke(reset_state, config=config)
    except Exception:
        pass  # ignore — the checkpoint is overwritten regardless

    return f"Conversation history cleared for user {user_id}."


@mcp.tool()
async def list_uploaded_pdfs(user_id: int) -> str:
    """
    List all PDFs currently in the knowledge base.
    Available to both students and teachers.

    user_id: 1=Nakul, 2=Aslam, 3=Kabir
    """
    result = list_pdfs()

    if not result["success"]:
        return f"Error: {result['message']}"

    if result["total"] == 0:
        return "No PDFs uploaded yet. Ask a teacher to upload study materials."

    lines = [f"Uploaded PDFs ({result['total']} total):\n"]
    for pdf in result["pdfs"]:
        lines.append(f"  • {pdf['filename']:<35} ({pdf['chunks']} chunks)")
    return "\n".join(lines)


@mcp.tool()
async def delete_uploaded_pdf(user_id: int, filename: str) -> str:
    """
    Delete a PDF from the knowledge base. Teachers only.
    Removes all chunks from ChromaDB and the file from disk.

    user_id: must be a teacher — 3=Kabir
    filename: exact filename e.g. "old_notes.pdf"
    """
    if not check_permission(user_id, "rag", "upload"):
        return "Access Denied: Only teachers can delete PDFs."

    print(f"  PDF DELETE by user_id={user_id}: '{filename}'", file=sys.stderr)
    result = delete_pdf(filename)
    return result["message"]


@mcp.tool()
async def upload_pdf_from_path(user_id: int, file_path: str) -> str:
    """
    Upload a PDF by providing its path on the server. Teachers only.

    user_id:   must be a teacher — 3=Kabir
    file_path: absolute path e.g. "/Users/kabir/notes/ml_lecture.pdf"
    """
    if not check_permission(user_id, "rag", "upload"):
        return "Access Denied: Only teachers can upload PDFs."

    if not os.path.exists(file_path):
        return f"File not found: '{file_path}'"

    if not file_path.lower().endswith(".pdf"):
        return "Only PDF files are accepted."

    filename  = os.path.basename(file_path)
    dest_path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(dest_path):
        shutil.copy2(file_path, dest_path)

    print(f"  PDF UPLOAD by user_id={user_id}: '{filename}'", file=sys.stderr)
    result = upload_pdf(dest_path)

    if not result["success"]:
        return f"Upload failed: {result['message']}"

    return (
        f"'{result['filename']}' uploaded successfully.\n"
        f"  Pages read   : {result['pages_read']}\n"
        f"  Chunks added : {result['chunks_added']}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
