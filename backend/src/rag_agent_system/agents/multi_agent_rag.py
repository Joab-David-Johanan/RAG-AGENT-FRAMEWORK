from typing import Literal
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.callbacks import BaseCallbackHandler

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

from rag_agent_system.retrieval.vector_store import build_vector_store

load_dotenv()


class StreamingHandler(BaseCallbackHandler):

    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
        self.tokens.append(token)


# ---------------------------
# Setup LLM
# ---------------------------

llm = init_chat_model("openai:gpt-4o-mini")

stream_handler = StreamingHandler()

# llm = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=0,
#     streaming=True,
#     callbacks=[stream_handler],
# )


# ---------------------------
# Tool: Tavily Search
# ---------------------------

tavily = TavilySearch(max_results=2)


# ---------------------------
# Retriever Tool
# ---------------------------


def make_retriever_tool(file_path):

    retriever = build_vector_store(file_path)

    def tool_fn(query: str) -> str:
        docs = retriever.invoke(query)
        print("\n--- RETRIEVED DOCUMENTS ---\n")

        for d in docs:
            print(d.page_content[:200])
            print("----")
        return "\n\n".join(doc.page_content for doc in docs)

    return Tool(
        name="InternalResearchNotes",
        description="Search internal research notes for experiments",
        func=tool_fn,
    )


# ---------------------------
# System Prompt
# ---------------------------


def system_prompt(role: str):

    return (
        "You are one member of a multi-agent AI team.\n"
        "Cooperate with other agents and use tools responsibly.\n"
        "If you produce the final answer, prefix it with: FINAL ANSWER:\n\n"
        f"Your role: {role}"
    )


# ---------------------------
# Build Graph
# ---------------------------


def build_graph(text_file_path: str):

    retriever_tool = make_retriever_tool(text_file_path)

    # Research Agent
    research_agent = create_agent(
        model=llm,
        tools=[retriever_tool, tavily],
        system_prompt=system_prompt("Researcher – gather facts for the writer."),
    )

    # Blog Agent
    blog_agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=system_prompt("Writer – produce a detailed blog."),
    )

    # Research Node
    def research_node(state: MessagesState) -> Command[Literal["blog_writer", END]]:

        result = research_agent.invoke(state)

        last = result["messages"][-1]

        goto = END if "FINAL ANSWER" in last.content else "blog_writer"

        result["messages"][-1] = HumanMessage(
            content=last.content,
            name="researcher",
        )

        return Command(update={"messages": result["messages"]}, goto=goto)

    # Blog Node
    def blog_node(state: MessagesState) -> Command[Literal["researcher", END]]:

        result = blog_agent.invoke(state)

        last = result["messages"][-1]

        goto = END if "FINAL ANSWER" in last.content else "researcher"

        result["messages"][-1] = HumanMessage(
            content=last.content,
            name="blog_writer",
        )

        return Command(update={"messages": result["messages"]}, goto=goto)

    # Graph
    workflow = StateGraph(MessagesState)

    workflow.add_node("researcher", research_node)
    workflow.add_node("blog_writer", blog_node)

    workflow.add_edge(START, "researcher")

    return workflow.compile()


# ---------------------------
# Public Function (Router will call this)
# ---------------------------


# def run_multi_agent_rag(query: str, text_file_path: str):

#     graph = build_graph(text_file_path)

#     response = graph.invoke({"messages": [HumanMessage(content=query)]})

#     print("\n===== FULL MESSAGE TRACE =====\n")

#     for i, msg in enumerate(response["messages"]):

#         agent_name = getattr(msg, "name", "user")

#         print(f"\n--- Step {i+1} ---")
#         print(f"Type : {msg.type}")
#         print(f"Agent: {agent_name}")
#         print(f"Content:\n{msg.content}")

#         # Show tool calls if any
#         if hasattr(msg, "tool_calls") and msg.tool_calls:
#             print("Tool calls:", msg.tool_calls)

#     # Count only agent exchanges
#     agent_msgs = [
#         m
#         for m in response["messages"]
#         if getattr(m, "name", None) in ["researcher", "blog_writer"]
#     ]

#     print("\n===== AGENT EXCHANGE COUNT =====")
#     print(f"Back-and-forth iterations: {len(agent_msgs)}")

#     return response["messages"][-1].content


def run_multi_agent_rag(query: str, text_file_path: str):

    graph = build_graph(text_file_path)

    inputs = {"messages": [HumanMessage(content=query)]}

    final_response = ""

    print("\n===== STREAMING EXECUTION =====\n")

    for step in graph.stream(inputs, {"recursion_limit": 20}):

        for node, state in step.items():

            last_msg = state["messages"][-1]
            agent_name = getattr(last_msg, "name", node)

            print(f"\n--- Agent step: {agent_name} ---\n")

            final_response = last_msg.content

    return final_response
