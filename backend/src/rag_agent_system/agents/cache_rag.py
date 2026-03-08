from typing import Literal
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq import ChatGroq

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

from rag_agent_system.retrieval.vector_store import build_vector_store

import json
import os

load_dotenv()


# =====================================================
# Simple Query Cache
# =====================================================
# Stores previous queries and answers
# In production you would likely use:
# - Redis
# - vector cache
# - semantic cache
# =====================================================

CACHE_FILE = "rag_cache.json"


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


query_cache = load_cache()


# =====================================================
# Streaming Handler
# =====================================================


class StreamingHandler(BaseCallbackHandler):

    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
        self.tokens.append(token)


stream_handler = StreamingHandler()


# =====================================================
# LLM Setup
# =====================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)

stream_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    streaming=True,
    callbacks=[stream_handler],
)


# =====================================================
# Retriever Tool
# =====================================================


def make_retriever_tool(file_path):

    retriever = build_vector_store(file_path)

    def tool_fn(query: str):

        docs = retriever.invoke(query)

        print("\n--- RETRIEVED DOCUMENTS ---\n")

        for d in docs:
            print(d.page_content[:200])
            print("----")

        return "\n\n".join(doc.page_content for doc in docs)

    return Tool(
        name="InternalDocs",
        description="Search internal research documents.",
        func=tool_fn,
    )


# =====================================================
# Build Graph
# =====================================================


def build_graph(text_file_path: str, llm_model):

    retriever_tool = make_retriever_tool(text_file_path)

    rag_agent = create_agent(
        model=llm_model,
        tools=[retriever_tool],
        system_prompt="Use retrieved documents to answer the question.",
    )

    # -------------------------------------------------
    # Cache Check Node
    # -------------------------------------------------

    def cache_check_node(
        state: MessagesState,
    ) -> Command[Literal["retrieve", END]]:

        query = state["messages"][-1].content

        print("\n--- CHECKING CACHE ---\n")

        # ----------------------------------------------
        # If query already exists in cache
        # we SKIP the LLM and retrieval completely
        # ----------------------------------------------

        if query in query_cache:

            cached_answer = query_cache[query]

            print("Cache HIT")

            return Command(
                update={
                    "messages": state["messages"] + [AIMessage(content=cached_answer)]
                },
                goto=END,
            )

        print("Cache MISS")

        return Command(
            update={"messages": state["messages"]},
            goto="retrieve",
        )

    # -------------------------------------------------
    # Retrieval Node
    # -------------------------------------------------

    def retrieve_node(
        state: MessagesState,
    ) -> Command[Literal["generate_answer"]]:

        result = rag_agent.invoke(state)

        return Command(
            update={"messages": result["messages"]},
            goto="generate_answer",
        )

    # -------------------------------------------------
    # Answer Node
    # -------------------------------------------------

    def answer_node(state: MessagesState) -> Command[Literal[END]]:

        final_answer = state["messages"][-1].content
        query = state["messages"][0].content

        print("\n--- SAVING TO CACHE ---\n")

        # ----------------------------------------------
        # Store query + answer for future reuse
        # ----------------------------------------------

        query_cache[query] = final_answer
        save_cache(query_cache)

        return Command(
            update={"messages": state["messages"]},
            goto=END,
        )

    # -------------------------------------------------
    # Graph
    # -------------------------------------------------

    workflow = StateGraph(MessagesState)

    workflow.add_node("cache_check", cache_check_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate_answer", answer_node)

    workflow.add_edge(START, "cache_check")

    return workflow.compile()


# =====================================================
# Non-Streaming Runner
# =====================================================


def run_cache_rag(query: str, text_file_path: str):

    graph = build_graph(text_file_path, llm)

    response = graph.invoke({"messages": [HumanMessage(content=query)]})

    return response["messages"][-1].content


# =====================================================
# Streaming Runner
# =====================================================


def run_cache_rag_stream(query: str, text_file_path: str):

    graph = build_graph(text_file_path, stream_llm)

    inputs = {"messages": [HumanMessage(content=query)]}

    final_response = ""

    print("\n===== STREAMING CACHE RAG =====\n")

    for step in graph.stream(inputs, {"recursion_limit": 20}):

        for node, state in step.items():

            last_msg = state["messages"][-1]

            print(f"\n--- Node executed: {node} ---\n")

            final_response = last_msg.content

    return final_response
