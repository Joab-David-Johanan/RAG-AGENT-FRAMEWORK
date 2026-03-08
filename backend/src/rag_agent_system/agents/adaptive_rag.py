from typing import Literal
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.callbacks import BaseCallbackHandler

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


# ---------------------------
# LLM Setup
# ---------------------------

stream_handler = StreamingHandler()

llm = ChatGroq(
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

        # When this tool is called it means:
        # the LLM already decided that internal knowledge retrieval is needed

        docs = retriever.invoke(query)

        print("\n--- INTERNAL DOCUMENTS RETRIEVED ---\n")

        for d in docs:
            print(d.page_content[:200])
            print("----")

        return "\n\n".join(doc.page_content for doc in docs)

    return Tool(
        name="InternalKnowledgeBase",
        description="Search internal company documents and research notes.",
        func=tool_fn,
    )


# ---------------------------
# System Prompts
# ---------------------------


def router_prompt():

    return (
        "You are a routing agent.\n\n"
        "Your job is to decide how the system should answer the question.\n\n"
        "Possible strategies:\n"
        "1. INTERNAL → if the answer likely exists in internal documents\n"
        "2. WEB → if the question requires external knowledge or current events\n"
        "3. DIRECT → if the LLM can answer without retrieval\n\n"
        "Respond with ONLY one word:\n"
        "INTERNAL\nWEB\nDIRECT"
    )


def answer_prompt():

    return (
        "You are an expert AI assistant.\n"
        "Use the provided context if available.\n"
        "If tools are available you may call them.\n\n"
        "Always produce a clear final answer."
    )


# ---------------------------
# Build Graph
# ---------------------------


def build_graph(text_file_path: str):

    retriever_tool = make_retriever_tool(text_file_path)

    # ---------------------------
    # Router Agent
    # ---------------------------

    router_agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=router_prompt(),
    )

    # ---------------------------
    # Internal RAG Agent
    # ---------------------------

    internal_rag_agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt=answer_prompt(),
    )

    # ---------------------------
    # Web RAG Agent
    # ---------------------------

    web_rag_agent = create_agent(
        model=llm,
        tools=[tavily],
        system_prompt=answer_prompt(),
    )

    # ---------------------------
    # Router Node
    # ---------------------------

    def router_node(
        state: MessagesState,
    ) -> Command[Literal["internal_rag", "web_rag", "direct_answer"]]:

        result = router_agent.invoke(state)

        decision = result["messages"][-1].content.strip().upper()

        print(f"\n--- ROUTER DECISION: {decision} ---\n")

        # --------------------------------------
        # This is the core of Adaptive RAG
        #
        # The LLM decides whether retrieval
        # is required and which source to use
        # --------------------------------------

        if "INTERNAL" in decision:
            goto = "internal_rag"

        elif "WEB" in decision:
            goto = "web_rag"

        else:
            goto = "direct_answer"

        return Command(
            update={"messages": state["messages"]},
            goto=goto,
        )

    # ---------------------------
    # Internal RAG Node
    # ---------------------------

    def internal_node(state: MessagesState) -> Command[Literal[END]]:

        # The agent has access to the retrieval tool.
        # It decides itself whether to call the tool.

        result = internal_rag_agent.invoke(state)

        return Command(
            update={"messages": result["messages"]},
            goto=END,
        )

    # ---------------------------
    # Web RAG Node
    # ---------------------------

    def web_node(state: MessagesState) -> Command[Literal[END]]:

        # The agent has access to Tavily search.
        # It will decide whether to call the search tool.

        result = web_rag_agent.invoke(state)

        return Command(
            update={"messages": result["messages"]},
            goto=END,
        )

    # ---------------------------
    # Direct Answer Node
    # ---------------------------

    def direct_node(state: MessagesState) -> Command[Literal[END]]:

        # No tools are used here.
        # The LLM answers purely from its own knowledge.

        response = llm.invoke(state["messages"])

        return Command(
            update={"messages": state["messages"] + [response]},
            goto=END,
        )

    # ---------------------------
    # Graph
    # ---------------------------

    workflow = StateGraph(MessagesState)

    workflow.add_node("router", router_node)
    workflow.add_node("internal_rag", internal_node)
    workflow.add_node("web_rag", web_node)
    workflow.add_node("direct_answer", direct_node)

    workflow.add_edge(START, "router")

    return workflow.compile()


# ---------------------------
# Public Runner
# ---------------------------


def run_adaptive_rag(query: str, text_file_path: str):

    graph = build_graph(text_file_path)

    inputs = {"messages": [HumanMessage(content=query)]}

    final_response = ""

    print("\n===== ADAPTIVE RAG EXECUTION =====\n")

    for step in graph.stream(inputs, {"recursion_limit": 20}):

        for node, state in step.items():

            last_msg = state["messages"][-1]

            print(f"\n--- Node executed: {node} ---\n")

            final_response = last_msg.content

    return final_response
