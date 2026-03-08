# backend/src/rag_agent_system/api/routes.py

from fastapi import APIRouter, UploadFile, File, Form
from pathlib import Path
import shutil

from rag_agent_system.retrieval.vector_store import build_vector_store

# from rag_agent_system.agents.multi_agent_rag import run_multi_agent_rag
from rag_agent_system.agents.mar import run_multi_agent_rag


router = APIRouter()

# point to existing project folder backend/data
DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# only create folder if it does not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...), query: str = Form("Summarize the uploaded document")
):

    file_path = DATA_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # process file into vector store
    build_vector_store(str(file_path))

    # run multi-agent RAG on the uploaded file
    rag_response = run_multi_agent_rag(query)

    # response sent to Next.js
    return {
        "message": f"{file.filename} processed in vector store",
        "rag_response": rag_response,
    }
