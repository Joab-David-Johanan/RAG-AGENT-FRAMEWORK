from rag_agent_system.agents.multi_agent_rag import run_multi_agent_rag

response = run_multi_agent_rag(
    query="Write a detailed explanation of transformer models.",
    text_file_path="data/sample.txt",
)

print("\n===== RESPONSE =====\n")
print(response)
