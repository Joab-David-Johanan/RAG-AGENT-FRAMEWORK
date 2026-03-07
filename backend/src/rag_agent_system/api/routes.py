from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import shutil

from rag_agent_system.retrieval.vector_store import build_vector_store

router = APIRouter()

DATA_DIR = Path("backend/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_path = DATA_DIR / file.filename

    # save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # process file into vector store
    build_vector_store(str(file_path))

    return {"message": "Processed file in vector store"}
