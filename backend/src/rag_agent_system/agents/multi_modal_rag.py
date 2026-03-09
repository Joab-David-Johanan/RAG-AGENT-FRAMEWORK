# multimodal rag pipeline
# retrieves both text and images from pdf using clip embeddings
# then sends text + images to gpt-4 vision model for reasoning

import fitz
from PIL import Image
import io
import base64
import os
import torch
import numpy as np

from dotenv import load_dotenv


from langchain.chat_models import init_chat_model
from langchain.schema.messages import HumanMessage
from rag_agent_system.retrieval.multi_modal_vectorstore import (
    retrieve_multimodal,
    build_multimodal_vector_store,
)

load_dotenv()

# openai api key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

print("\n[DEBUG] multi_modal_rag.py loaded")


# -----------------------------------
# build gpt vision message
# -----------------------------------


def create_multimodal_message(query, retrieved_docs, image_data_store):

    content = []

    content.append({"type": "text", "text": f"Question: {query}\n\nContext:\n"})

    text_docs = [d for d in retrieved_docs if d.metadata.get("type") == "text"]

    image_docs = [d for d in retrieved_docs if d.metadata.get("type") == "image"]

    # add text context
    if text_docs:

        text_context = "\n\n".join(
            [f"[Page {d.metadata['page']}]: {d.page_content}" for d in text_docs]
        )

        content.append({"type": "text", "text": f"Text excerpts:\n{text_context}\n"})

    # add images
    for doc in image_docs:

        image_id = doc.metadata.get("image_id")

        if image_id in image_data_store:

            content.append(
                {"type": "text", "text": f"\nImage from page {doc.metadata['page']}:\n"}
            )

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data_store[image_id]}"
                    },
                }
            )

    content.append(
        {"type": "text", "text": "\nPlease answer using the text and images."}
    )

    return HumanMessage(content=content)


# -----------------------------------
# entry function used by routes.py
# -----------------------------------


def run_multimodal_rag(query: str, pdf_path: str):

    print("[DEBUG] run_multimodal_rag called")

    vector_store, image_data_store = build_multimodal_vector_store(pdf_path)

    docs = retrieve_multimodal(query, vector_store)

    llm = init_chat_model("openai:gpt-4.1")

    message = create_multimodal_message(query, docs, image_data_store)

    response = llm.invoke([message])

    print(f"[DEBUG] retrieved {len(docs)} multimodal docs")

    return response.content
