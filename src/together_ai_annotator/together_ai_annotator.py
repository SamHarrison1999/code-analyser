# together_ai_annotator.py

import os
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

def annotate_code_with_together_ai(
    code: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    if not TOGETHER_API_KEY:
        raise ValueError("TOGETHER_API_KEY is not set in the environment.")

    system_prompt = (
        "You are a secure code reviewer and machine learning expert.\n"
        "Your task is to insert the following types of inline comments into Python source code:\n\n"
        "# ⚠️ SAST Risk: <reason>\n"
        "# 🧠 ML Signal: <reason>\n"
        "# ✅ Best Practice: <reason>\n\n"
        "Place each comment directly above the relevant line. Do not change the code. Keep spacing intact.\n"
        "Always return the full annotated code."
    )

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": code}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
    resp.raise_for_status()
    output = resp.json()["choices"][0]["message"]["content"]
    # Strip Markdown-style code blocks if present
    if output.startswith("```"):
        output = output.strip("`").split("\n", 1)[1]  # remove first line
        if output.endswith("```"):
            output = output.rsplit("\n", 1)[0]
    return output

