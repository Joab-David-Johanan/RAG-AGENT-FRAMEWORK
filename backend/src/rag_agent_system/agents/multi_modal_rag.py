from __future__ import annotations

import os
import json
import base64
from pathlib import Path
from typing import Literal, List, Dict, Any, Optional

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

from rag_agent_system.retrieval.vector_store import build_vector_store

load_dotenv()


# ============================================================
# Streaming Handler
# ============================================================


class StreamingHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
        self.tokens.append(token)


stream_handler = StreamingHandler()


# ============================================================
# LLM Setup
# ============================================================
# IMPORTANT:
# Use a multimodal-capable model here.
# For example:
#   - gpt-4o-mini
#   - gpt-4.1-mini
#   - a Gemini multimodal model
#
# The FINAL answer step may include retrieved images/graphs,
# so the model must be able to consume image content.
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

stream_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
    callbacks=[stream_handler],
)


# ============================================================
# Helpers
# ============================================================


def image_file_to_data_url(file_path: str) -> str:
    """
    Convert a local image file into a base64 data URL so it can be passed
    into a multimodal chat model.
    """
    ext = Path(file_path).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/png")

    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def safe_read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def list_files(folder: str, suffixes: tuple[str, ...]) -> List[str]:
    if not folder or not os.path.exists(folder):
        return []

    paths = []
    for root, _, files in os.walk(folder):
        for file_name in files:
            if file_name.lower().endswith(suffixes):
                paths.append(os.path.join(root, file_name))
    return sorted(paths)


# ============================================================
# Tools
# ============================================================
# We keep retrieval modality-specific.
#
# The routing agent decides:
# - DIRECT       -> no retrieval
# - TEXT         -> query text vector store
# - IMAGE        -> retrieve image references
# - TABLE        -> retrieve table summaries
# - GRAPH        -> retrieve graph/chart references
# - MULTI        -> use several tools
#
# Then the final answer node composes all retrieved evidence.
# ============================================================


def make_text_retriever_tool(text_file_path: str) -> Tool:
    retriever = build_vector_store(text_file_path)

    def tool_fn(query: str) -> str:
        docs = retriever.invoke(query)

        print("\n--- RETRIEVED TEXT DOCUMENTS ---\n")
        for d in docs:
            print(d.page_content[:250])
            print("----")

        return "\n\n".join(doc.page_content for doc in docs)

    return Tool(
        name="TextRetriever",
        description=(
            "Use this for natural-language questions answered by text documents, "
            "notes, reports, or article-like content."
        ),
        func=tool_fn,
    )


def make_table_retriever_tool(table_summary_file: str) -> Tool:
    """
    Assumes you have a text/markdown/json summary file for tables.
    Best practice:
      - preprocess tables into natural-language summaries
      - embed those summaries into a vector store
    Here we keep it simple and use the same build_vector_store helper.
    """
    retriever = build_vector_store(table_summary_file)

    def tool_fn(query: str) -> str:
        docs = retriever.invoke(query)

        print("\n--- RETRIEVED TABLE SUMMARIES ---\n")
        for d in docs:
            print(d.page_content[:250])
            print("----")

        return "\n\n".join(doc.page_content for doc in docs)

    return Tool(
        name="TableRetriever",
        description=(
            "Use this for questions about structured tables, rows, columns, metrics, "
            "aggregates, tabular comparisons, or numerical summaries."
        ),
        func=tool_fn,
    )


def make_image_retriever_tool(image_manifest_file: str) -> Tool:
    """
    Assumes image_manifest_file is a JSON file like:
    [
      {
        "path": "data/images/figure1.png",
        "summary": "Microscope image showing cell clustering after treatment A."
      },
      ...
    ]

    We retrieve via simple keyword scoring over summaries.
    In production, you would likely use:
      - CLIP embeddings
      - multimodal embeddings
      - vector search over image captions
    """
    manifest = json.loads(safe_read_text(image_manifest_file))

    def tool_fn(query: str) -> str:
        query_words = set(query.lower().split())

        scored = []
        for item in manifest:
            summary = item.get("summary", "")
            summary_words = set(summary.lower().split())
            score = len(query_words & summary_words)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_items = [item for score, item in scored[:3] if score > 0]

        # fallback if nothing matched
        if not top_items:
            top_items = manifest[:2]

        print("\n--- RETRIEVED IMAGES ---\n")
        for item in top_items:
            print(item["path"])
            print(item.get("summary", ""))
            print("----")

        # Return JSON string so downstream code can parse image paths + summaries
        return json.dumps(top_items)

    return Tool(
        name="ImageRetriever",
        description=(
            "Use this for questions requiring visual inspection of images, figures, "
            "photos, diagrams, screenshots, or non-tabular visual content."
        ),
        func=tool_fn,
    )


def make_graph_retriever_tool(graph_manifest_file: str) -> Tool:
    """
    Assumes graph_manifest_file is JSON like:
    [
      {
        "path": "data/graphs/revenue_q4.png",
        "summary": "Line chart of revenue growth from Q1 to Q4, peaking in Q4."
      },
      ...
    ]

    Graphs are separate from generic images because:
      - they often need trend/comparison reasoning
      - they can be routed differently from photos/diagrams
    """
    manifest = json.loads(safe_read_text(graph_manifest_file))

    def tool_fn(query: str) -> str:
        query_words = set(query.lower().split())

        scored = []
        for item in manifest:
            summary = item.get("summary", "")
            summary_words = set(summary.lower().split())
            score = len(query_words & summary_words)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_items = [item for score, item in scored[:3] if score > 0]

        if not top_items:
            top_items = manifest[:2]

        print("\n--- RETRIEVED GRAPHS ---\n")
        for item in top_items:
            print(item["path"])
            print(item.get("summary", ""))
            print("----")

        return json.dumps(top_items)

    return Tool(
        name="GraphRetriever",
        description=(
            "Use this for questions about charts, plots, graphs, trends, axes, "
            "distribution shapes, or visualized numeric change over time."
        ),
        func=tool_fn,
    )


# ============================================================
# Prompts
# ============================================================


def router_prompt() -> str:
    return (
        "You are a routing agent for a multimodal RAG system.\n\n"
        "Decide whether the user query should be answered:\n"
        "- DIRECT  -> no retrieval needed\n"
        "- TEXT    -> retrieve text only\n"
        "- IMAGE   -> retrieve images only\n"
        "- TABLE   -> retrieve tables only\n"
        "- GRAPH   -> retrieve graphs/charts only\n"
        "- MULTI   -> retrieve from more than one modality\n\n"
        "Use these rules:\n"
        "- TEXT for notes, explanations, paragraphs, policies, reports.\n"
        "- IMAGE for photos, diagrams, screenshots, visual inspection.\n"
        "- TABLE for rows, columns, tabular metrics, exact structured values.\n"
        "- GRAPH for trends, plots, charts, curves, comparisons over axes.\n"
        "- MULTI if the query clearly needs combined evidence.\n"
        "- DIRECT if common knowledge or reasoning is enough.\n\n"
        "Respond with ONLY one word:\n"
        "DIRECT\nTEXT\nIMAGE\nTABLE\nGRAPH\nMULTI"
    )


def retrieval_prompt() -> str:
    return (
        "You are a retrieval agent.\n"
        "Use the available tool(s) when needed.\n"
        "Retrieve the best evidence for the user question.\n"
        "Do not answer the user finally. Focus on gathering evidence."
    )


def answer_prompt() -> str:
    return (
        "You are the final multimodal RAG answer agent.\n\n"
        "You may receive:\n"
        "- text evidence\n"
        "- table summaries\n"
        "- image references\n"
        "- graph/chart references\n\n"
        "Rules:\n"
        "1. Use retrieved evidence when available.\n"
        "2. If images/graphs are included, inspect them carefully.\n"
        "3. Prefer grounded answers over guesswork.\n"
        "4. If evidence is insufficient, say what is missing.\n"
        "5. Give a direct, complete answer."
    )


# ============================================================
# Graph Builder
# ============================================================


def build_graph(
    text_file_path: str,
    table_summary_file: str,
    image_manifest_file: str,
    graph_manifest_file: str,
    llm_model,
):
    text_tool = make_text_retriever_tool(text_file_path)
    table_tool = make_table_retriever_tool(table_summary_file)
    image_tool = make_image_retriever_tool(image_manifest_file)
    graph_tool = make_graph_retriever_tool(graph_manifest_file)

    # Router decides whether retrieval is needed at all,
    # and if yes, which modality/modalities are appropriate.
    router_agent = create_agent(
        model=llm_model,
        tools=[],
        system_prompt=router_prompt(),
    )

    # Modality-specific retrieval agents.
    text_agent = create_agent(
        model=llm_model,
        tools=[text_tool],
        system_prompt=retrieval_prompt(),
    )

    table_agent = create_agent(
        model=llm_model,
        tools=[table_tool],
        system_prompt=retrieval_prompt(),
    )

    image_agent = create_agent(
        model=llm_model,
        tools=[image_tool],
        system_prompt=retrieval_prompt(),
    )

    graph_agent = create_agent(
        model=llm_model,
        tools=[graph_tool],
        system_prompt=retrieval_prompt(),
    )

    # Multi agent can call all retrieval tools.
    multi_agent = create_agent(
        model=llm_model,
        tools=[text_tool, table_tool, image_tool, graph_tool],
        system_prompt=retrieval_prompt(),
    )

    # Final answer agent does not need tool calls.
    # We will construct the final multimodal prompt manually.
    answer_agent = create_agent(
        model=llm_model,
        tools=[],
        system_prompt=answer_prompt(),
    )

    # --------------------------------------------------------
    # Router Node
    # --------------------------------------------------------
    def router_node(
        state: MessagesState,
    ) -> Command[
        Literal[
            "direct_answer",
            "text_retrieve",
            "table_retrieve",
            "image_retrieve",
            "graph_retrieve",
            "multi_retrieve",
        ]
    ]:
        result = router_agent.invoke(state)
        decision = result["messages"][-1].content.strip().upper()

        print(f"\n--- ROUTER DECISION: {decision} ---\n")

        # This is where the system decides:
        # - retrieval NOT necessary -> DIRECT
        # - retrieval necessary     -> choose modality
        if "TEXT" in decision:
            goto = "text_retrieve"
        elif "IMAGE" in decision:
            goto = "image_retrieve"
        elif "TABLE" in decision:
            goto = "table_retrieve"
        elif "GRAPH" in decision:
            goto = "graph_retrieve"
        elif "MULTI" in decision:
            goto = "multi_retrieve"
        else:
            goto = "direct_answer"

        return Command(
            update={"messages": state["messages"]},
            goto=goto,
        )

    # --------------------------------------------------------
    # Retrieval Nodes
    # --------------------------------------------------------
    # Each retrieval node gives the agent only the tools relevant
    # to that modality. That is how tool selection is constrained.
    # --------------------------------------------------------

    def text_node(state: MessagesState) -> Command[Literal["generate_answer"]]:
        result = text_agent.invoke(state)
        return Command(update={"messages": result["messages"]}, goto="generate_answer")

    def table_node(state: MessagesState) -> Command[Literal["generate_answer"]]:
        result = table_agent.invoke(state)
        return Command(update={"messages": result["messages"]}, goto="generate_answer")

    def image_node(state: MessagesState) -> Command[Literal["generate_answer"]]:
        result = image_agent.invoke(state)
        return Command(update={"messages": result["messages"]}, goto="generate_answer")

    def graph_node(state: MessagesState) -> Command[Literal["generate_answer"]]:
        result = graph_agent.invoke(state)
        return Command(update={"messages": result["messages"]}, goto="generate_answer")

    def multi_node(state: MessagesState) -> Command[Literal["generate_answer"]]:
        result = multi_agent.invoke(state)
        return Command(update={"messages": result["messages"]}, goto="generate_answer")

    # --------------------------------------------------------
    # Direct Answer Node
    # --------------------------------------------------------
    def direct_node(state: MessagesState) -> Command[Literal[END]]:
        response = llm_model.invoke(state["messages"])
        return Command(
            update={"messages": state["messages"] + [response]},
            goto=END,
        )

    # --------------------------------------------------------
    # Final Answer Node
    # --------------------------------------------------------
    # This node looks through the message history for tool outputs.
    # - text/table tool outputs are inserted as text context
    # - image/graph tool outputs are parsed and attached as actual images
    #
    # This is what makes the system truly multimodal:
    # the final model sees both textual evidence and visual evidence.
    # --------------------------------------------------------
    def answer_node(state: MessagesState) -> Command[Literal[END]]:
        user_query = state["messages"][0].content if state["messages"] else ""

        text_contexts: List[str] = []
        table_contexts: List[str] = []
        image_items: List[Dict[str, Any]] = []
        graph_items: List[Dict[str, Any]] = []

        for msg in state["messages"]:
            content = getattr(msg, "content", "")

            if not isinstance(content, str):
                continue

            # Heuristic parsing:
            # tool outputs for image/graph retrievers are JSON arrays
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list) and parsed:
                    first = parsed[0]
                    if (
                        isinstance(first, dict)
                        and "path" in first
                        and "summary" in first
                    ):
                        # crude split between image and graph using filename/path
                        for item in parsed:
                            lower_path = item["path"].lower()
                            if any(k in lower_path for k in ["graph", "chart", "plot"]):
                                graph_items.append(item)
                            else:
                                image_items.append(item)
                        continue
            except Exception:
                pass

            # crude tagging based on agent/tool names in content trail
            # since create_agent abstracts tool calls, the simplest pattern
            # is to treat plain text retrieval outputs as textual evidence.
            if content and len(content) > 20:
                # If a retrieval node returned plain text, we store it.
                # We don't strictly know whether it came from text or table
                # retrieval, so we keep both buckets simple.
                if "|" in content or "\t" in content or "column" in content.lower():
                    table_contexts.append(content)
                else:
                    text_contexts.append(content)

        multimodal_content: List[Dict[str, Any]] = []

        # Main instruction block
        context_header = (
            f"User question:\n{user_query}\n\n"
            f"TEXT CONTEXT:\n{chr(10).join(text_contexts[:3]) if text_contexts else 'None'}\n\n"
            f"TABLE CONTEXT:\n{chr(10).join(table_contexts[:3]) if table_contexts else 'None'}\n\n"
            "You may also receive retrieved images and graphs below. "
            "Use all available evidence and answer carefully."
        )

        multimodal_content.append({"type": "text", "text": context_header})

        # Attach retrieved images
        for item in image_items[:3]:
            file_path = item["path"]
            if os.path.exists(file_path):
                multimodal_content.append(
                    {
                        "type": "text",
                        "text": f"Retrieved image summary: {item.get('summary', '')}",
                    }
                )
                multimodal_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_file_to_data_url(file_path)},
                    }
                )

        # Attach retrieved graphs/charts
        for item in graph_items[:3]:
            file_path = item["path"]
            if os.path.exists(file_path):
                multimodal_content.append(
                    {
                        "type": "text",
                        "text": f"Retrieved graph/chart summary: {item.get('summary', '')}",
                    }
                )
                multimodal_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_file_to_data_url(file_path)},
                    }
                )

        final_input = [HumanMessage(content=multimodal_content)]

        result = answer_agent.invoke({"messages": final_input})

        return Command(
            update={"messages": state["messages"] + [result["messages"][-1]]},
            goto=END,
        )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------
    workflow = StateGraph(MessagesState)

    workflow.add_node("router", router_node)
    workflow.add_node("text_retrieve", text_node)
    workflow.add_node("table_retrieve", table_node)
    workflow.add_node("image_retrieve", image_node)
    workflow.add_node("graph_retrieve", graph_node)
    workflow.add_node("multi_retrieve", multi_node)
    workflow.add_node("direct_answer", direct_node)
    workflow.add_node("generate_answer", answer_node)

    workflow.add_edge(START, "router")

    return workflow.compile()


# ============================================================
# Public API - Non Streaming
# ============================================================


def run_multimodal_rag(
    query: str,
    text_file_path: str,
    table_summary_file: str,
    image_manifest_file: str,
    graph_manifest_file: str,
) -> str:
    graph = build_graph(
        text_file_path=text_file_path,
        table_summary_file=table_summary_file,
        image_manifest_file=image_manifest_file,
        graph_manifest_file=graph_manifest_file,
        llm_model=llm,
    )

    response = graph.invoke({"messages": [HumanMessage(content=query)]})
    return response["messages"][-1].content


# ============================================================
# Public API - Streaming
# ============================================================


def run_multimodal_rag_stream(
    query: str,
    text_file_path: str,
    table_summary_file: str,
    image_manifest_file: str,
    graph_manifest_file: str,
) -> str:
    graph = build_graph(
        text_file_path=text_file_path,
        table_summary_file=table_summary_file,
        image_manifest_file=image_manifest_file,
        graph_manifest_file=graph_manifest_file,
        llm_model=stream_llm,
    )

    inputs = {"messages": [HumanMessage(content=query)]}

    final_response = ""

    print("\n===== STREAMING MULTIMODAL RAG =====\n")

    for step in graph.stream(inputs, {"recursion_limit": 20}):
        for node, state in step.items():
            last_msg = state["messages"][-1]
            print(f"\n--- Node executed: {node} ---\n")
            final_response = getattr(last_msg, "content", "")

    return final_response
