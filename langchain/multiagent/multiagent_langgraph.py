import os
import smtplib
from email.message import EmailMessage
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from reportlab.pdfgen import canvas

# Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
    api_key=os.environ.get("GROQ_API_KEY", "dummy"),
)


# Shared state for all agents
class AgentState(TypedDict):
    question: str
    research: str
    technical: str
    report: str
    pdf_file: str
    email: str
    email_password: str


# Coordinator node
def coordinator(state: AgentState):
    print("\n[Coordinator] Starting workflow")
    return {
        "research": "",
        "technical": "",
        "report": "",
        "pdf_file": "",
    }


# Agent 1: Research Agent
def research_agent(state: AgentState):
    print("\n[Research Agent] Working...")

    prompt = f"""
    Analyze the following question as a Research Agent.

    Identify:
    - Important concepts
    - Requirements
    - Benefits
    - Challenges

    Question: {state['question']}
    """

    response = llm.invoke([
        SystemMessage(content="You are an IT Research Agent."),
        HumanMessage(content=prompt)
    ])

    return {"research": response.content}


# Agent 2: Technical Agent
def technical_agent(state: AgentState):
    print("\n[Technical Agent] Working...")

    prompt = f"""
    You are a Kubernetes Technical Architect.

    Based on the research, propose a technical solution.

    Include:
    - Architecture
    - Kubernetes components
    - Deployment approach
    - Security
    - Monitoring

    User question:
    {state['question']}

    Research findings:
    {state['research']}
    """

    response = llm.invoke([
        SystemMessage(content="You are a Kubernetes expert."),
        HumanMessage(content=prompt)
    ])

    return {"technical": response.content}


# Agent 3: Report Agent
def report_agent(state: AgentState):
    print("\n[Report Agent] Working...")

    prompt = f"""
    Prepare a structured technical report.

    Include:
    1. Executive summary
    2. Research findings
    3. Technical architecture
    4. Implementation steps
    5. Conclusion

    User question:
    {state['question']}

    Research:
    {state['research']}

    Technical solution:
    {state['technical']}

    Do not invent unsupported facts.
    """

    response = llm.invoke([
        SystemMessage(content="You are a Technical Report Agent."),
        HumanMessage(content=prompt)
    ])

    return {"report": response.content}


# Agent 4: PDF Agent
def pdf_agent(state: AgentState):
    print("\n[PDF Agent] Creating PDF...")

    pdf_path = "report.pdf"
    pdf = canvas.Canvas(pdf_path)

    pdf.drawString(50, 800, "Technical Report")
    pdf.drawString(50, 770, "Question: " + state["question"])

    y = 740
    for line in state["report"].split("\n"):
        pdf.drawString(50, y, line[:100])
        y -= 20
        if y < 50:
            pdf.showPage()
            y = 800

    pdf.save()
    return {"pdf_file": pdf_path}


# Agent 5: Email Agent
def email_agent(state: AgentState):
    print("\n[Email Agent] Sending email...")

    recipient = state.get("email") or os.environ.get("EMAIL_TO")
    sender = os.environ.get("EMAIL_FROM", "chavan.swayam2006mail@gmail.com")
    password = state.get("email_password") or os.environ.get("EMAIL_PASSWORD")

    if not recipient or not password:
        print("[Email Agent] No recipient or password provided. Skipping email send.")
        return state

    email = EmailMessage()
    email["Subject"] = "Technical Report"
    email["From"] = sender
    email["To"] = recipient
    email.set_content(state["report"])

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(email)
        print("[Email Agent] Email sent successfully.")
    except Exception as exc:
        print(f"[Email Agent] Email failed: {exc}")

    return state


# Build LangGraph
graph = StateGraph(AgentState)

# Register nodes
graph.add_node("coordinator", coordinator)
graph.add_node("research", research_agent)
graph.add_node("technical", technical_agent)
graph.add_node("report", report_agent)
graph.add_node("pdf", pdf_agent)
graph.add_node("email", email_agent)

# Define workflow
graph.add_edge(START, "coordinator")
graph.add_edge("coordinator", "research")
graph.add_edge("research", "technical")
graph.add_edge("technical", "report")
graph.add_edge("report", "pdf")
graph.add_edge("pdf", "email")
graph.add_edge("email", END)

# Compile graph
app = graph.compile()


# Execute application
if __name__ == "__main__":
    question = input("Enter your question: ").strip()
    recipient_email = input("Enter recipient email: ").strip()

    gmail_password = os.environ.get("EMAIL_PASSWORD")
    if not gmail_password:
        gmail_password = input("Enter your Gmail app password: ").strip()

    result = app.invoke({
        "question": question,
        "research": "",
        "technical": "",
        "report": "",
        "pdf_file": "",
        "email": recipient_email,
        "email_password": gmail_password,
    })

    print("\n========== FINAL REPORT ==========")
    print(result["report"])
    print(f"\nPDF created: {result.get('pdf_file', 'report.pdf')}")