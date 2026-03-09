# backend/src/rag_agent_system/api/routes.py

from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from pathlib import Path
import shutil

from rag_agent_system.retrieval.vector_store import build_vector_store

# from rag_agent_system.agents.multi_agent_rag import run_multi_agent_rag
from rag_agent_system.agents.mar import run_multi_agent_rag
from rag_agent_system.agents.multi_modal_rag import run_multimodal_rag
from rag_agent_system.agents.cache_rag import run_cache_rag

router = APIRouter()

# point to existing project folder backend/data
DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# only create folder if it does not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# store latest uploaded text path
LATEST_TXT_PATH = None

# store latest uploaded pdf path
LATEST_PDF_PATH = None

# ------------------------------
# CHANGE: file upload endpoint
# ------------------------------


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    global LATEST_PDF_PATH, LATEST_TXT_PATH

    file_path = DATA_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # detect file type
    suffix = file_path.suffix.lower()

    # txt files -> build normal rag vector store
    if suffix == ".txt":

        build_vector_store(str(file_path))

        LATEST_TXT_PATH = str(file_path)

        return {"message": f"{file.filename} processed in vector store"}

    # pdf files -> store path for multimodal rag
    elif suffix == ".pdf":

        print("[DEBUG] detected pdf upload")

        LATEST_PDF_PATH = str(file_path)

        return {"message": f"{file.filename} uploaded for multimodal rag"}

    else:

        return {"message": "unsupported file type"}


# ----------------------------------------
# query request schema
# ----------------------------------------


class QueryRequest(BaseModel):
    query: str


# ----------------------------------------
# multi agent rag endpoint
# ----------------------------------------


@router.post("/chat")
async def chat_query(data: QueryRequest):

    # run multi-agent RAG on the uploaded file
    response = run_multi_agent_rag(data.query)

    # rag response sent to Next.js
    return {"response": response}


# ----------------------------------------
# multimodal rag endpoint
# ----------------------------------------


@router.post("/multimodal")
async def multimodal_query(data: QueryRequest):

    global LATEST_PDF_PATH

    if not LATEST_PDF_PATH:
        return {"response": "please upload a pdf first"}

    response = run_multimodal_rag(query=data.query, pdf_path=LATEST_PDF_PATH)

    return {"response": response}


# ----------------------------------------
# cache rag endpoint
# ----------------------------------------


@router.post("/cache")
async def cache_query(data: QueryRequest):

    global LATEST_TXT_PATH

    if not LATEST_TXT_PATH:
        return {"response": "please upload a text file first"}

    result = run_cache_rag(query=data.query)

    return result
