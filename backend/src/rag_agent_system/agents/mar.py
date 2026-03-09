# backend/src/rag_agent_system/agents/multi_agent_rag.py
# FULL FILE
# CHANGE: hard stop after 5 agent turns and force FINAL ANSWER

from typing import Literal
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_tavily import TavilySearch

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

from rag_agent_system.retrieval.vector_store import get_retriever

load_dotenv()

print("\n[DEBUG] multi_agent_rag.py loaded")


class StreamingHandler(BaseCallbackHandler):

    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
        self.tokens.append(token)


stream_handler = StreamingHandler()

llm = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0,
    streaming=True,
    callbacks=[stream_handler],
)

tavily = TavilySearch(max_results=2)


# ----------------------------------------
# GRAPH
# ----------------------------------------


def build_graph():

    research_agent = create_agent(
        model=llm,
        tools=[tavily],
        system_prompt="You are a research agent gathering facts.",
    )

    blog_agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="You are a writer producing the final explanation.",
    )

    # --------------------------
    # Research node
    # --------------------------

    def research_node(state: MessagesState) -> Command[Literal["blog_writer", END]]:

        turn_count = len(state["messages"])

        print(f"[DEBUG] RESEARCH TURN {turn_count}")

        # CHANGE: stop after 5 turns
        if turn_count >= 10:
            print("[DEBUG] Turn limit reached → forcing END")
            return Command(goto=END)

        result = research_agent.invoke(state)

        last = result["messages"][-1]

        result["messages"][-1] = HumanMessage(
            content=last.content,
            name="researcher",
        )

        return Command(update={"messages": result["messages"]}, goto="blog_writer")

    # --------------------------
    # Blog node
    # --------------------------

    def blog_node(state: MessagesState) -> Command[Literal["researcher", END]]:

        turn_count = len(state["messages"])

        print(f"[DEBUG] BLOG TURN {turn_count}")

        result = blog_agent.invoke(state)

        last = result["messages"][-1]

        # CHANGE: if turn limit reached → summarize and finish
        if turn_count >= 2:

            summary_prompt = f"""
Summarize the discussion and produce the FINAL ANSWER.

Conversation:
{last.content}

Respond with:
FINAL ANSWER: <summary>
"""

            summary = llm.invoke(summary_prompt)

            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=summary.content,
                            name="blog_writer",
                        )
                    ]
                },
                goto=END,
            )

        result["messages"][-1] = HumanMessage(
            content=last.content,
            name="blog_writer",
        )

        return Command(update={"messages": result["messages"]}, goto="researcher")

    workflow = StateGraph(MessagesState)

    workflow.add_node("researcher", research_node)
    workflow.add_node("blog_writer", blog_node)

    workflow.add_edge(START, "researcher")

    return workflow.compile()


# ----------------------------------------
# ENTRY FUNCTION
# ----------------------------------------


def run_multi_agent_rag(query: str):

    print("[DEBUG] run_multi_agent_rag called")

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

    for step in graph.stream(inputs, {"recursion_limit": 50}):

        for node, state in step.items():

            last_msg = state["messages"][-1]

            print(f"\n--- Agent step: {node} ---")

            print(last_msg.content[:200])

            final_response = last_msg.content

    return final_response
