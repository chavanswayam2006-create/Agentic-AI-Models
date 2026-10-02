from langchain_groq import ChatGroq

#Create Groq LLM
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)

#Send questions
response = llm.invoke(
    "Explain Docker in simple terms"
)

print(response.content)