import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from agents.rbac_helper import check_permission
from agents.rag_agent import upload_pdf
from config import PDF_DIR

router = APIRouter(tags=["PDF"])


@router.post("/upload-pdf")
async def upload_pdf_route(user_id: int, file: UploadFile = File(...)):

    if not check_permission(user_id, "rag", "upload"):
        raise HTTPException(status_code=403, detail="Only teachers can upload PDF notes.")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    save_path = os.path.join(PDF_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = upload_pdf(save_path)

    if not result["success"]:
        reason_to_status = {
            "duplicate":   409,
            "read_error":  422,
            "empty_pdf":   422,
            "no_chunks":   422,
            "store_error": 500,
        }
        status_code = reason_to_status.get(result["reason"], 400)
        raise HTTPException(status_code=status_code, detail=result["message"])

    return result