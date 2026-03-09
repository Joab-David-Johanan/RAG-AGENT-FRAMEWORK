import fitz
from PIL import Image
import io
import base64
import os
import torch
import numpy as np

from transformers import CLIPProcessor, CLIPModel

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# -----------------------------------
# load clip model
# -----------------------------------

# clip produces embeddings for both text and images in same space
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_model.eval()

# -----------------------------------
# embedding functions
# -----------------------------------


# embed image using clip
def embed_image(image_data):

    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data

    inputs = clip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze().numpy()


# embed text using clip
def embed_text(text):

    inputs = clip_processor(
        text=text, return_tensors="pt", padding=True, truncation=True, max_length=77
    )

    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze().numpy()


# -----------------------------------
# pdf processing + vector store
# -----------------------------------


def build_multimodal_vector_store(pdf_path):

    print("[DEBUG] building multimodal vector store")

    doc = fitz.open(pdf_path)

    all_docs = []
    all_embeddings = []

    # store images for gpt vision
    image_data_store = {}

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for page_index, page in enumerate(doc):

        # extract text
        text = page.get_text()

        if text.strip():

            temp_doc = Document(
                page_content=text, metadata={"page": page_index, "type": "text"}
            )

            chunks = splitter.split_documents([temp_doc])

            for chunk in chunks:

                emb = embed_text(chunk.page_content)

                all_docs.append(chunk)
                all_embeddings.append(emb)

        # extract images
        for img_index, img in enumerate(page.get_images(full=True)):

            try:

                xref = img[0]

                base_image = doc.extract_image(xref)

                image_bytes = base_image["image"]

                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                image_id = f"page_{page_index}_img_{img_index}"

                # store base64 image for llm
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")

                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                image_data_store[image_id] = img_base64

                emb = embed_image(pil_image)

                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={
                        "page": page_index,
                        "type": "image",
                        "image_id": image_id,
                    },
                )

                all_docs.append(image_doc)
                all_embeddings.append(emb)

            except Exception as e:
                print("[DEBUG] image processing error:", e)

    doc.close()

    embeddings_array = np.array(all_embeddings)

    # create faiss store from precomputed embeddings
    vector_store = FAISS.from_embeddings(
        text_embeddings=[
            (doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)
        ],
        embedding=None,
        metadatas=[doc.metadata for doc in all_docs],
    )

    return vector_store, image_data_store


# -----------------------------------
# multimodal retrieval
# -----------------------------------


def retrieve_multimodal(query, vector_store, k=5):

    query_embedding = embed_text(query)

    results = vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)

    return results
