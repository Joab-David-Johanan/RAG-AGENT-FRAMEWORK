from typing import Literal
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

from rag_agent_system.retrieval.vector_store import get_retriever

import json
from pathlib import Path

load_dotenv()


# =====================================================
# cache storage
# =====================================================

# store cache in backend/data/rag_cache.json
CACHE_FILE = Path(__file__).resolve().parents[3] / "data" / "rag_cache.json"


def load_cache():
    # load cache if file exists
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    # write cache to disk
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


query_cache = load_cache()


# =====================================================
# setup llm
# =====================================================

llm = init_chat_model("openai:gpt-4o-mini")


# =====================================================
# retriever tool
# =====================================================


def make_retriever_tool():

    retriever = get_retriever()

    def tool_fn(query: str):

        docs = retriever.invoke(query)

        print("\n--- RETRIEVED DOCUMENTS ---\n")

        for d in docs:
            print(d.page_content[:200])
            print("----")

        return "\n\n".join(doc.page_content for doc in docs)

    return Tool(
        name="InternalDocs",
        description="Search internal documents",
        func=tool_fn,
    )


# =====================================================
# build graph
# =====================================================


def build_graph():

    retriever_tool = make_retriever_tool()

    rag_agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt="Use retrieved documents to answer the question.",
    )

    # node that runs retrieval + generation
    def retrieve_node(
        state: MessagesState,
    ) -> Command[Literal["generate_answer"]]:

        result = rag_agent.invoke(state)

        return Command(
            update={"messages": result["messages"]},
            goto="generate_answer",
        )

    # node that saves answer to cache
    def answer_node(
        state: MessagesState,
    ) -> Command[Literal[END]]:

        final_answer = state["messages"][-1].content
        query = state["messages"][0].content

        print("\n--- SAVING TO CACHE ---\n")

        query_cache[query] = final_answer
        save_cache(query_cache)

        return Command(
            update={"messages": state["messages"]},
            goto=END,
        )

    workflow = StateGraph(MessagesState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate_answer", answer_node)

    workflow.add_edge(START, "retrieve")

    return workflow.compile()


# =====================================================
# public runner
# =====================================================


def run_cache_rag(query: str):

    print("\n===== CACHE RAG EXECUTION =====\n")

    # -----------------------------------
    # check cache BEFORE running graph
    # -----------------------------------

    if query in query_cache:

        print("\n--- CACHE HIT ---\n")

        return {
            "response": query_cache[query],
            "cache_hit": True,
        }

    print("\n--- CACHE MISS ---\n")

    # -----------------------------------
    # retrieve context for graph input
    # -----------------------------------

    retriever = get_retriever()
    docs = retriever.invoke(query)

    context = "\n\n".join([d.page_content for d in docs])

    graph = build_graph()

    inputs = {
        "messages": [
            HumanMessage(
                content=f"""
        User question:
        {query}

        Retrieved document:
        {context}

        Discuss and produce a final explanation.
        """
            )
        ]
    }

    final_response = ""

    # run graph execution
    for step in graph.stream(inputs, {"recursion_limit": 20}):

        for node, state in step.items():

            last_msg = state["messages"][-1]

            print(f"\n--- Node executed: {node} ---\n")

            final_response = last_msg.content

    # -----------------------------------
    # save to cache
    # -----------------------------------

    query_cache[query] = final_response
    save_cache(query_cache)

    return {
        "response": final_response,
        "cache_hit": False,
    }
