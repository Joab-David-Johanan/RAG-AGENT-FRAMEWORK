from typing import Literal
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq import ChatGroq

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

from rag_agent_system.retrieval.vector_store import build_vector_store

load_dotenv()


# ---------------------------
# Streaming Handler
# ---------------------------


class StreamingHandler(BaseCallbackHandler):

    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
        self.tokens.append(token)


stream_handler = StreamingHandler()


# ---------------------------
# LLM Setup
# ---------------------------

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


# ---------------------------
# Web Search Tool
# ---------------------------

tavily = TavilySearch(max_results=3)


# ---------------------------
# Retriever Tool
# ---------------------------


def make_retriever_tool(file_path):

    retriever = build_vector_store(file_path)

    def tool_fn(query: str) -> str:

        docs = retriever.invoke(query)

        print("\n--- RETRIEVED INTERNAL DOCUMENTS ---\n")

        for d in docs:
            print(d.page_content[:200])
            print("----")

        return "\n\n".join(doc.page_content for doc in docs)

    return Tool(
        name="InternalKnowledgeBase",
        description="Search internal company documents and notes.",
        func=tool_fn,
    )


# ---------------------------
# Prompts
# ---------------------------


def grader_prompt():

    return (
        "You are a document relevance grader.\n\n"
        "Given the user question and retrieved documents, decide if the "
        "documents are relevant enough to answer the question.\n\n"
        "Respond with ONLY:\n"
        "GOOD\n"
        "BAD"
    )


def answer_prompt():

    return (
        "You are an expert assistant.\n"
        "Use the context provided to answer the question clearly.\n"
        "If no context is provided rely on external search results."
    )


# ---------------------------
# Build Graph
# ---------------------------


def build_graph(text_file_path: str, llm_model):

    retriever_tool = make_retriever_tool(text_file_path)

    # ---------------------------
    # Retrieval Agent
    # ---------------------------

    retrieval_agent = create_agent(
        model=llm_model,
        tools=[retriever_tool],
        system_prompt="Retrieve relevant documents using the tool.",
    )

    # ---------------------------
    # Web Search Agent
    # ---------------------------

    web_agent = create_agent(
        model=llm_model,
        tools=[tavily],
        system_prompt="Use web search if internal knowledge fails.",
    )

    # ---------------------------
    # Answer Agent
    # ---------------------------

    answer_agent = create_agent(
        model=llm_model,
        tools=[],
        system_prompt=answer_prompt(),
    )

    # ---------------------------
    # Document Grader Agent
    # ---------------------------

    grader_agent = create_agent(
        model=llm_model,
        tools=[],
        system_prompt=grader_prompt(),
    )

    # ---------------------------
    # Retrieval Node
    # ---------------------------

    def retrieve_node(state: MessagesState) -> Command[Literal["grade_docs"]]:

        # Agent retrieves documents via vector DB

        result = retrieval_agent.invoke(state)

        return Command(
            update={"messages": result["messages"]},
            goto="grade_docs",
        )

    # ---------------------------
    # Document Grading Node
    # ---------------------------

    def grade_node(
        state: MessagesState,
    ) -> Command[Literal["generate_answer", "web_search"]]:

        # The grader checks if retrieved docs are useful

        result = grader_agent.invoke(state)

        decision = result["messages"][-1].content.strip().upper()

        print(f"\n--- DOCUMENT GRADE: {decision} ---\n")

        # -----------------------------
        # Corrective RAG Logic
        #
        # GOOD → proceed with internal docs
        # BAD  → fallback to web search
        # -----------------------------

        if "GOOD" in decision:
            goto = "generate_answer"
        else:
            goto = "web_search"

        return Command(
            update={"messages": state["messages"]},
            goto=goto,
        )

    # ---------------------------
    # Web Search Node
    # ---------------------------

    def web_node(state: MessagesState) -> Command[Literal["generate_answer"]]:

        # If retrieval failed we call web search

        result = web_agent.invoke(state)

        return Command(
            update={"messages": result["messages"]},
            goto="generate_answer",
        )

    # ---------------------------
    # Final Answer Node
    # ---------------------------

    def answer_node(state: MessagesState) -> Command[Literal[END]]:

        result = answer_agent.invoke(state)

        return Command(
            update={"messages": result["messages"]},
            goto=END,
        )

    # ---------------------------
    # Graph
    # ---------------------------

    workflow = StateGraph(MessagesState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_docs", grade_node)
    workflow.add_node("web_search", web_node)
    workflow.add_node("generate_answer", answer_node)

    workflow.add_edge(START, "retrieve")

    return workflow.compile()


# =====================================================
# NON-STREAMING VERSION
# =====================================================


def run_corrective_rag(query: str, text_file_path: str):

    graph = build_graph(text_file_path, llm)

    response = graph.invoke({"messages": [HumanMessage(content=query)]})

    return response["messages"][-1].content


# =====================================================
# STREAMING VERSION
# =====================================================


def run_corrective_rag_stream(query: str, text_file_path: str):

    graph = build_graph(text_file_path, stream_llm)

    inputs = {"messages": [HumanMessage(content=query)]}

    final_response = ""

    print("\n===== STREAMING CORRECTIVE RAG =====\n")

    for step in graph.stream(inputs, {"recursion_limit": 20}):

        for node, state in step.items():

            last_msg = state["messages"][-1]

            print(f"\n--- Node executed: {node} ---\n")

            final_response = last_msg.content

    return final_response
