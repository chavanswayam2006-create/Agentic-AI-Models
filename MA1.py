import json
import os
import subprocess
import sys

import psutil
from groq import Groq

api_key = os.environ.get("GROQ_API_KEY") or (sys.argv[1] if len(sys.argv) > 1 else None)
if not api_key:
    raise RuntimeError(
        "Set the GROQ_API_KEY environment variable first, e.g.:\n"
        "  $env:GROQ_API_KEY='your_key_here'\n"
        "  python MA1.py\n"
        "or:\n"
        "  setx GROQ_API_KEY \"your_key_here\""
    )

client = Groq(api_key=api_key)
MODEL_NAME = "qwen/qwen-3-32b"


# ==========================================
# 1. DEFINE AGENT TOOLS (Python Functions)
# ==========================================

def get_system_metrics() -> str:
    """Returns local host system resource utilization (CPU, RAM, Disk)."""
    disk_root = os.path.abspath(os.sep)
    metrics = {
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage(disk_root).percent,
    }
    return json.dumps(metrics)


def ping_host(hostname: str) -> str:
    """
    Pings a network host to evaluate connectivity.
    Args:
        hostname: Domain or IP to test (e.g. '1.1.1.1' or 'google.com')
    """
    try:
        if os.name == "nt":
            command = ["ping", "-n", "2", hostname]
        else:
            command = ["ping", "-c", "2", hostname]

        result = subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return json.dumps({"status": "success", "raw_output": result.strip()})
    except Exception as err:
        return json.dumps({"status": "error", "message": str(err)})


# Map function names to executable Python code
SYSTEM_TOOLS_MAP = {"get_system_metrics": get_system_metrics}
NETWORK_TOOLS_MAP = {"ping_host": ping_host}

# ==========================================
# 2. DEFINE JSON SCHEMAS
# ==========================================

SYSTEM_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_system_metrics",
            "description": "Get current CPU, RAM, and Disk metrics of the machine.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]

NETWORK_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "ping_host",
            "description": "Ping a specified hostname or IP address to verify connectivity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hostname": {"type": "string", "description": "Target hostname or IP address."}
                },
                "required": ["hostname"],
            },
        },
    }
]

# ==========================================
# 3. GENERIC AGENT RUNNER
# ==========================================

def run_agent(agent_name: str, system_prompt: str, user_query: str, tools_schema: list, tools_map: dict):
    print(f"\n--- [{agent_name}] Activated ---")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",
    )
    assistant_message = response.choices[0].message
    messages.append(assistant_message.model_dump(exclude_none=True))

    tool_calls = assistant_message.tool_calls or []

    if not tool_calls:
        print(f"[{agent_name} Response]:")
        print(assistant_message.content or "")
        return

    for call in tool_calls:
        fn_name = call.function.name
        fn_args = json.loads(call.function.arguments)

        print(f"  └─ Executing Tool: `{fn_name}` with parameters: {fn_args}")
        if fn_name in tools_map:
            result = tools_map[fn_name](**fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": fn_name,
                "content": result,
            })

    final_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    print(f"\n[{agent_name} Final Answer]:")
    print(final_response.choices[0].message.content or "")


# ==========================================
# 4. ORCHESTRATOR ROUTER
# ==========================================

def orchestrate_query(user_query: str):
    print(f"\n==========================================")
    print(f"USER QUERY: \"{user_query}\"")
    print(f"==========================================")

    router_prompt = f"""You are a query router. Analyze the user prompt and respond with ONLY ONE word:
    - 'SYSTEM' if the query asks about local CPU, disk, memory, or system hardware status.
    - 'NETWORK' if the query asks about pinging, network latency, or internet connectivity.
    - 'UNKNOWN' if it fits neither.
    Query: {user_query}"""

    route_res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": router_prompt}],
    )

    decision = (route_res.choices[0].message.content or "").strip().upper()

    if "SYSTEM" in decision:
        run_agent(
            agent_name="System Health Agent",
            system_prompt="You are a system administration agent. Use tools to check CPU, RAM, or Disk metrics.",
            user_query=user_query,
            tools_schema=SYSTEM_TOOLS_SCHEMA,
            tools_map=SYSTEM_TOOLS_MAP,
        )
    elif "NETWORK" in decision:
        run_agent(
            agent_name="Network Agent",
            system_prompt="You are a network diagnostic agent. Use tools to check network status and ping target hosts.",
            user_query=user_query,
            tools_schema=NETWORK_TOOLS_SCHEMA,
            tools_map=NETWORK_TOOLS_MAP,
        )
    else:
        print("\n[Router]: Request does not match active agent domains.")


# ==========================================
# 5. TEST RUNS
# ==========================================

if __name__ == "__main__":
    orchestrate_query("Can you check if my CPU or RAM are overloading right now?")
    orchestrate_query("Ping 8.8.8.8 to see if our network connection to DNS is stable.")
