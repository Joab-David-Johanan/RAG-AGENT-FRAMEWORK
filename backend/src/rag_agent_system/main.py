from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_agent_system.api.routes import router

app = FastAPI(title="RAG Agent Framework")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# register API routes
app.include_router(router)


@app.get("/")
def root():
    return {"message": "RAG Agent Framework API running"}
