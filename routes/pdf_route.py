import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from agents.rbac_helper import check_permission
from agents.rag_agent import upload_pdf
from config import PDF_DIR

router = APIRouter(tags=["PDF"])


@router.post("/upload-pdf")
async def upload_pdf_route(user_id: int, file: UploadFile = File(...)):
    # Only teachers can upload
    if not check_permission(user_id, "rag", "upload"):
        raise HTTPException(status_code=403, detail="Only teachers can upload PDF notes.")

    # Only accept PDF files
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    # Save to a temp file first (so we don't leave orphan files on failure)
    temp_path  = os.path.join(PDF_DIR, f"_tmp_{file.filename}")
    final_path = os.path.join(PDF_DIR, file.filename)

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Try to process the PDF
        result = upload_pdf(temp_path)

        if not result["success"]:
            # Clean up the temp file — we don't want it lying around
            os.remove(temp_path)

            reason_to_status = {
                "duplicate":   409,
                "read_error":  422,
                "empty_pdf":   422,
                "no_chunks":   422,
                "store_error": 500,
            }
            status_code = reason_to_status.get(result["reason"], 400)
            raise HTTPException(status_code=status_code, detail=result["message"])

        # Success — rename temp file to final name
        os.rename(temp_path, final_path)
        return result

    except HTTPException:
        raise   # re-raise HTTP errors as-is

    except Exception as e:
        # Clean up temp file if something unexpected happened
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")