import json
import os

from groq import Groq

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Set the GROQ_API_KEY environment variable before running this script.")

client = Groq(api_key=api_key)
MODEL_NAME = "qwen/qwen-3-32b"


def echo_fun(content: str):
    """Echo the user's message."""
    print("User:", content)
    return content


def function1(content: str):
    """Return the model-provided content in upper case."""
    if content is None:
        return content
    try:
        return content.upper()
    except Exception:
        return str(content).upper()


def function2():
    return "Function 2 executed"


def function3():
    return "Function 3 executed"


TOOLS = [echo_fun, function1, function2, function3]


SYSTEM_PROMPT = """You are a model which likes to eat vadapav and dosa as well.
Your job is to read tools and print the messages.
"""


def agent_run():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Groq Echo Agent")
    print("Type 'exit' to quit.")

    while True:
        prompt = input("\nMessage me for echoing .... ").strip()

        if prompt.lower() == "exit":
            break

        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )

        assistant_message = response.choices[0].message.content or ""
        upper_message = function1(assistant_message)

        print("\nModel:", upper_message)
        messages.append({"role": "assistant", "content": upper_message})


if __name__ == "__main__":
    agent_run()