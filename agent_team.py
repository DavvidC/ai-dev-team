"""
agent_team.py
=============
Four-agent loop that works on any codebase.

  Product Owner → Developer → Reviewer → QA → Product Owner
  └──────────────── repeat until approved or MAX_ITERS ──────┘

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY=sk-...
    export APP_DIR=/path/to/your/project
    export PROVIDER=anthropic             # or openai (default: anthropic)

    /opt/homebrew/bin/python3.11 agent_team.py "Add a /health endpoint"
    /opt/homebrew/bin/python3.11 agent_team.py        # PO picks the task

    # Optional: tell QA exactly how to start the app
    export START_CMD="npm run dev"

Models (override via env vars):
    ANTHROPIC:  PO_MODEL=claude-opus-4-7   AGENT_MODEL=claude-sonnet-4-6
    OPENAI:     PO_MODEL=o3                AGENT_MODEL=o4-mini

Safety: agents run shell commands scoped to APP_DIR only.
"""

from __future__ import annotations
import os
import sys
import json
import subprocess
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_DIR   = Path(os.environ.get("APP_DIR", Path(__file__).parent)).resolve()
START_CMD = os.environ.get("START_CMD", "")
MAX_ITERS = int(os.environ.get("MAX_ITERS", "3"))
PROVIDER  = os.environ.get("PROVIDER", "anthropic").lower()

if PROVIDER == "openai":
    PO_MODEL    = os.environ.get("PO_MODEL", "o3")
    AGENT_MODEL = os.environ.get("AGENT_MODEL", "o4-mini")
else:
    PO_MODEL    = os.environ.get("PO_MODEL", "claude-opus-4-7")
    AGENT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Tool definitions (Anthropic format — converted for OpenAI on the fly)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command inside the project directory. Working dir is APP_DIR. Timeout: 60 s.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file at a path relative to APP_DIR.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Overwrite (or create) a file at a path relative to APP_DIR.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "Relative path"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
    },
]


def _safe_path(rel: str) -> Path:
    resolved = (APP_DIR / rel).resolve()
    if not str(resolved).startswith(str(APP_DIR)):
        raise PermissionError(f"Path escape blocked: {rel!r}")
    return resolved


def _run_tool(name: str, inp: dict) -> str:
    if name == "bash":
        result = subprocess.run(
            inp["command"], shell=True, capture_output=True,
            text=True, cwd=str(APP_DIR), timeout=60,
        )
        out = (result.stdout + result.stderr).strip()
        return out[:8000] or "(no output)"

    if name == "read_file":
        p = _safe_path(inp["path"])
        return p.read_text()[:12000] if p.exists() else f"Not found: {inp['path']}"

    if name == "write_file":
        p = _safe_path(inp["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(inp["content"])
        return f"Wrote {len(inp['content'])} bytes → {inp['path']}"

    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Provider abstraction
# Each complete() call returns (text: str, tool_calls: list[dict])
# where tool_calls = [{"id": ..., "name": ..., "input": {...}}, ...]
# ---------------------------------------------------------------------------

def _complete_anthropic(model: str, system: str, messages: list[dict]):
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        tools=TOOLS,
        messages=messages,
    )
    text = "\n".join(b.text for b in response.content if b.type == "text")
    tool_calls = [
        {"id": b.id, "name": b.name, "input": b.input}
        for b in response.content if b.type == "tool_use"
    ]
    # Return raw content blocks so we can append them to messages correctly
    return text, tool_calls, response.content, response.stop_reason


def _anthropic_tool_result_message(tool_calls_raw, results: list[str]) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tc["id"], "content": res}
            for tc, res in zip(tool_calls_raw, results)
        ],
    }


def _complete_openai(model: str, system: str, messages: list[dict]):
    import openai
    client = openai.OpenAI()

    # Convert tools to OpenAI format
    oai_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in TOOLS
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=oai_tools,
        tool_choice="auto",
    )

    msg = response.choices[0].message
    text = msg.content or ""
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "input": json.loads(tc.function.arguments),
            })

    stop_reason = response.choices[0].finish_reason
    return text, tool_calls, msg, stop_reason


def _openai_tool_result_message(tool_calls_raw, results: list[str]) -> list[dict]:
    return [
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": res,
        }
        for tc, res in zip(tool_calls_raw.tool_calls, results)
    ]


# ---------------------------------------------------------------------------
# Generic agent runner
# ---------------------------------------------------------------------------
def run_agent(role: str, model: str, system: str, user_message: str) -> str:
    print(f"\n{'='*64}")
    print(f"  [{PROVIDER.upper()}] {role}  ({model})")
    print(f"{'='*64}")

    if PROVIDER == "openai":
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ]
    else:
        messages = [{"role": "user", "content": user_message}]

    while True:
        if PROVIDER == "openai":
            text, tool_calls, raw_msg, stop_reason = _complete_openai(model, system, messages)
        else:
            text, tool_calls, raw_content, stop_reason = _complete_anthropic(model, system, messages)

        if text:
            print(textwrap.fill(text, width=76, subsequent_indent="  "))

        # Append assistant turn
        if PROVIDER == "openai":
            messages.append(raw_msg)
        else:
            messages.append({"role": "assistant", "content": raw_content})

        done = (stop_reason in ("end_turn", "stop")) or not tool_calls
        if done:
            return text

        # Execute tools
        results = []
        for tc in tool_calls:
            print(f"\n  [tool] {tc['name']}({json.dumps(tc['input'])[:100]})")
            result = _run_tool(tc["name"], tc["input"])
            print(f"  → {result[:400]}")
            results.append(result)

        # Append tool results
        if PROVIDER == "openai":
            for tc, res in zip(raw_msg.tool_calls, results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": res,
                })
        else:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tc["id"], "content": res}
                    for tc, res in zip(tool_calls, results)
                ],
            })


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
_start_cmd_hint = (
    f"The command to start the app for testing is: {START_CMD}"
    if START_CMD else
    "Inspect the project to figure out how to start it (check package.json, "
    "Makefile, requirements.txt, Procfile, README, etc.) and include the start "
    "command in the spec so Developer and QA know what to use. Use port 8001 to "
    "avoid conflicts with the production server."
)

PO_SYSTEM = f"""\
You are the Product Owner. You work on whatever codebase is at: {APP_DIR}

You have TWO modes:

MODE 1 — PLANNING (when given a goal, or no goal)
  • Inspect the codebase with read_file and bash (ls, cat, find).
  • Understand the project type, structure, and how it runs.
  • Choose or refine ONE concrete, deliverable task.
  • {_start_cmd_hint}
  • Output a spec with:
      - Project type and how to start it (command + port)
      - What to build (2–3 sentences)
      - Numbered acceptance criteria (testable, specific)
      - Files that will likely change

MODE 2 — JUDGING (when given spec + QA report)
  • Read the actual changed files to verify the claims.
  • Return ONLY valid JSON — no prose, no markdown fences:
    {{"approved": true, "reasons": ["..."], "next_steps": []}}
    {{"approved": false, "reasons": ["..."], "next_steps": ["fix X", "add Y"]}}
"""

DEV_SYSTEM = f"""\
You are the Developer. Implement exactly what the spec says — nothing more.

Project root: {APP_DIR}

The spec will tell you how to start the app if you need to test locally.

Rules:
  • Always read a file before editing it.
  • Make the minimal change that satisfies the spec.
  • Prefer editing existing files over creating new ones.
  • Do not add comments explaining what you did — the code should speak.

End your response with a short summary: files changed + why.
"""

REVIEWER_SYSTEM = f"""\
You are the Code Reviewer.

Project root: {APP_DIR}

Read the files that were changed, then review them against the spec.
Check: correctness, edge cases, consistency with existing code style, security.
Output:
  • What looks good
  • What MUST change (if anything)
  • Overall verdict: LGTM  or  NEEDS REWORK
"""

QA_SYSTEM = f"""\
You are the QA Engineer. Test the app end-to-end using curl and shell commands.

Project root: {APP_DIR}

The spec includes how to start the app. Follow those instructions exactly.
After starting it, wait 2-3 seconds for it to be ready before testing.

Steps:
  1. Start the app using the command from the spec (background it with &).
  2. Wait for it to be ready.
  3. Test EVERY acceptance criterion using curl or shell commands.
  4. Report each criterion: ✅ PASS or ❌ FAIL — with actual output as evidence.
  5. Note any unexpected behaviour.
  6. Kill the test server when done (use pkill or kill on the PID).
"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_team(goal: str) -> None:
    banner = f"  GOAL: {goal}" if goal else "  GOAL: (PO will choose)"
    print(f"\n{'#'*64}\n{banner}\n{'#'*64}")
    print(f"  PROJECT:  {APP_DIR}")
    print(f"  PROVIDER: {PROVIDER.upper()}  |  PO: {PO_MODEL}  |  Agents: {AGENT_MODEL}")
    if START_CMD:
        print(f"  START:    {START_CMD}")
    print(f"{'#'*64}")

    po_prompt = (
        f"Goal: {goal}\n\nInspect the codebase and produce the spec."
        if goal else
        "No explicit goal. Inspect the codebase and choose the highest-value next task. Produce the spec."
    )
    spec = run_agent("PRODUCT OWNER — Planning", PO_MODEL, PO_SYSTEM, po_prompt)

    for iteration in range(1, MAX_ITERS + 1):
        print(f"\n{'─'*64}")
        print(f"  ITERATION {iteration} / {MAX_ITERS}")
        print(f"{'─'*64}")

        dev_summary = run_agent("DEVELOPER",  AGENT_MODEL, DEV_SYSTEM,
                                f"Spec:\n{spec}\n\nImplement it.")

        review      = run_agent("REVIEWER",   AGENT_MODEL, REVIEWER_SYSTEM,
                                f"Spec:\n{spec}\n\nDeveloper summary:\n{dev_summary}\n\nReview the actual files.")

        qa_report   = run_agent("QA",         AGENT_MODEL, QA_SYSTEM,
                                f"Spec (acceptance criteria):\n{spec}\n\nRun the tests now.")

        verdict_raw = run_agent("PRODUCT OWNER — Judging", PO_MODEL, PO_SYSTEM,
                                f"Spec:\n{spec}\n\nDeveloper summary:\n{dev_summary}\n\n"
                                f"Code review:\n{review}\n\nQA report:\n{qa_report}\n\n"
                                "Is the work complete? Return only the JSON verdict.")

        try:
            raw = verdict_raw.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            verdict: dict = json.loads(raw)
        except Exception as e:
            print(f"\n[warn] Could not parse verdict JSON: {e}")
            verdict = {"approved": False, "reasons": ["parse error"], "next_steps": []}

        approved = verdict.get("approved", False)
        print(f"\n{'='*64}")
        print(f"  PO VERDICT: {'✅  APPROVED' if approved else '❌  NEEDS WORK'}")
        for r in verdict.get("reasons", []):
            print(f"    • {r}")
        print(f"{'='*64}")

        if approved:
            print("\n🎉  Done! Delivering to you.\n")
            print("── DELIVERY ────────────────────────────────────────────────")
            print(f"Goal : {goal or '(PO-chosen)'}")
            print(f"\nSpec :\n{spec}")
            print(f"\nWhat was done :\n{dev_summary}")
            print(f"\nQA result :\n{qa_report}")
            return

        if iteration < MAX_ITERS:
            spec += (
                f"\n\n--- Round {iteration} feedback ---\n"
                f"Review: {review}\n\nQA: {qa_report}\n\n"
                f"PO next steps: {'; '.join(verdict.get('next_steps', []))}"
            )
            print("\n↩  Cycling back with feedback.\n")
        else:
            print(f"\n⚠️  Reached max iterations ({MAX_ITERS}) without approval.")
            print("Escalating to you — here's where things stand:")
            print(f"\nLast QA report:\n{qa_report}")
            print(f"\nPO next steps: {verdict.get('next_steps', [])}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip()
    if not goal and sys.stdin.isatty():
        goal = input("What should the team work on? (leave blank for PO to decide) > ").strip()
    run_team(goal)
