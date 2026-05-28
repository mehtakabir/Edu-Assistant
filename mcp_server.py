import sys
import os
import io
import shutil
import asyncio
import threading

sys.stderr.reconfigure(encoding="utf-8")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Silence stdout immediately ────────────────────────────────────────────────
# MCP uses stdout as its JSON-RPC channel. Anything printed to stdout
# corrupts the connection. We redirect it now and restore after server init.
_real_stdout = sys.stdout
sys.stdout   = io.StringIO()

from mcp.server.fastmcp import FastMCP
from database import setup_database
from graph import get_graph
from agents.rbac_helper import check_permission
from agents.rag_agent import upload_pdf, get_embeddings, get_vectorstore
from agents.pdf_manager import list_pdfs, delete_pdf
from config import PDF_DIR

# ── DB setup (fast, does not block) ──────────────────────────────────────────
setup_database()

# ── Restore stdout so MCP can start and respond to initialize immediately ─────
# The embedding model loads BELOW in a background thread AFTER MCP has
# already started. This is what prevents the 60-second startup timeout.
sys.stdout = _real_stdout
sys.stdout.reconfigure(encoding="utf-8")

mcp = FastMCP("edu-assistant")

# ── Background warm-up ────────────────────────────────────────────────────────
# We load the embedding model and graph in a background thread so that:
#   1. MCP can respond to initialize() immediately (no startup timeout)
#   2. The model is ready before any real query arrives
#
# _warmup_done is an Event — tool handlers wait on it before doing any work.
_warmup_done = threading.Event()

def _warmup():
    try:
        print("Warming up embedding model...", file=sys.stderr)
        get_embeddings()
        get_vectorstore()
        print("Embedding model ready.", file=sys.stderr)

        print("Warming up graph...", file=sys.stderr)
        get_graph()
        print("Graph ready. Server fully operational.", file=sys.stderr)
    except Exception as e:
        print(f"Warm-up error: {e}", file=sys.stderr)
    finally:
        _warmup_done.set()   # always unblock tool handlers, even on error

threading.Thread(target=_warmup, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
#  RESPONSE FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

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
            lines = []
            if "name"        in data: lines.append(f"Name       : {data['name']}")
            if "attendance"  in data: lines.append(f"Attendance : {data['attendance']}%")
            if "quiz_marks"  in data: lines.append(f"Quiz Marks : {data['quiz_marks']}")
            if "quiz_status" in data: lines.append(f"Quiz Status: {data['quiz_status']}")
            return "\n".join(lines) if lines else "No data found."

    elif agent == "quiz":
        return response.get("quiz", "Quiz generation failed.")

    return "No response from the assistant."


# ══════════════════════════════════════════════════════════════════════════════
#  MCP TOOLS
# ══════════════════════════════════════════════════════════════════════════════

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
      user_id=1, query="generate a quiz on loops"  -> denied by RBAC
    """
    # Wait for warm-up to finish (non-blocking — yields control to event loop)
    await asyncio.to_thread(_warmup_done.wait)

    config = {"configurable": {"thread_id": str(user_id)}}

    initial_state = {
        "user_id":  user_id,
        "role":     "",
        "query":    query,
        "intent":   "",
        "response": {},
        "error":    "",
        "messages": [],
    }

    try:
        # Run the entire graph in a worker thread.
        # This keeps the asyncio event loop free to handle MCP heartbeats
        # while embeddings / LLM calls run — preventing the 4-min timeout.
        final_state = await asyncio.to_thread(
            get_graph().invoke, initial_state, config
        )
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
    checkpointer = get_graph().checkpointer
    thread_id    = str(user_id)

    try:
        keys_to_delete = [
            key for key in checkpointer.storage
            if isinstance(key, tuple) and thread_id in str(key)
        ]
        for key in keys_to_delete:
            del checkpointer.storage[key]
        return f"Conversation history cleared for user {user_id}."
    except Exception as e:
        print(f"  clear_memory warning: {e}", file=sys.stderr)
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
        lines.append(f"  * {pdf['filename']:<35} ({pdf['chunks']} chunks)")
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
    Upload a PDF by providing its full path on your computer. Teachers only.

    user_id:   must be a teacher — 3=Kabir
    file_path: full path to the PDF, e.g. "C:/Users/kabir/notes/ml.pdf"
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

    # Wait for warm-up (embedding model must be ready before we can embed)
    await asyncio.to_thread(_warmup_done.wait)

    print(f"  PDF UPLOAD by user_id={user_id}: '{filename}'", file=sys.stderr)
    result = await asyncio.to_thread(upload_pdf, dest_path)

    if not result["success"]:
        return f"Upload failed: {result['message']}"

    return (
        f"'{result['filename']}' uploaded successfully.\n"
        f"  Pages read   : {result['pages_read']}\n"
        f"  Chunks added : {result['chunks_added']}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")