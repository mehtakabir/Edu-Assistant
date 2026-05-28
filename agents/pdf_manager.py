"""
agents/pdf_manager.py
─────────────────────
PDF management agent — list and delete uploaded PDFs.
Complements the existing upload_pdf() in agents/rag_agent.py.

Drop this file into your agents/ folder.
No changes needed to any existing file.
"""

import os
import sys
from agents.rag_agent import get_raw_collection
from config import PDF_DIR


# ─── List ─────────────────────────────────────────────────────

def list_pdfs() -> dict:
    """
    Returns all PDFs currently indexed in ChromaDB,
    along with how many chunks each one contributed.

    Returns:
        {
            "success": True,
            "total":   2,
            "pdfs": [
                {"filename": "chapter1.pdf", "chunks": 42},
                {"filename": "lecture2.pdf", "chunks": 31}
            ]
        }
    """
    try:
        collection = get_raw_collection()
        results    = collection.get(include=["metadatas"])

        if not results["ids"]:
            return {"success": True, "pdfs": [], "total": 0}

        # Count how many chunks belong to each source file
        chunk_counts: dict[str, int] = {}
        for meta in results["metadatas"]:
            fname = meta.get("source_file", "unknown")
            chunk_counts[fname] = chunk_counts.get(fname, 0) + 1

        pdfs = [
            {"filename": fname, "chunks": count}
            for fname, count in sorted(chunk_counts.items())
        ]

        return {"success": True, "pdfs": pdfs, "total": len(pdfs)}

    except Exception as e:
        print(f"  Error in list_pdfs: {e}", file=sys.stderr)
        return {"success": False, "message": f"Failed to list PDFs: {str(e)}"}


# ─── Delete ───────────────────────────────────────────────────

def delete_pdf(filename: str) -> dict:
    """
    Deletes a PDF from ChromaDB (all its chunks) and from disk.

    Args:
        filename: the bare filename, e.g. "chapter1.pdf"
                  (NOT a full path)

    Returns on success:
        {
            "success":       True,
            "filename":      "chapter1.pdf",
            "chunks_removed": 42,
            "disk_deleted":   True,
            "message":       "..."
        }

    Returns on failure:
        {
            "success": False,
            "reason":  "not_found" | "delete_error",
            "message": "..."
        }
    """
    try:
        collection = get_raw_collection()

        # ── 1. Confirm the file exists in the vector store
        existing = collection.get(where={"source_file": filename})
        if not existing["ids"]:
            return {
                "success": False,
                "reason":  "not_found",
                "message": f"'{filename}' was not found in the knowledge base."
            }

        chunk_count = len(existing["ids"])
        print(f"  Deleting '{filename}' — {chunk_count} chunks", file=sys.stderr)

        # ── 2. Remove all chunks from ChromaDB
        collection.delete(where={"source_file": filename})
        print(f"  ChromaDB: chunks removed", file=sys.stderr)

        # ── 3. Remove the file from disk (best-effort)
        disk_deleted = False
        file_path    = os.path.join(PDF_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            disk_deleted = True
            print(f"  Disk: file removed", file=sys.stderr)
        else:
            print(f"  Disk: file not found at {file_path} (skipped)", file=sys.stderr)

        return {
            "success":        True,
            "filename":       filename,
            "chunks_removed": chunk_count,
            "disk_deleted":   disk_deleted,
            "message":        (
                f"'{filename}' deleted successfully. "
                f"{chunk_count} chunks removed from knowledge base."
                + (" File also removed from disk." if disk_deleted else "")
            )
        }

    except Exception as e:
        print(f"  Error in delete_pdf: {e}", file=sys.stderr)
        return {
            "success": False,
            "reason":  "delete_error",
            "message": f"Failed to delete '{filename}': {str(e)}"
        }
