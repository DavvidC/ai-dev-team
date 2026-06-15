"""
agent_team.py
=============
Four-agent loop that works on the customer-health dashboard.

  Product Owner → Developer → Reviewer → QA → Product Owner
  └──────────────── repeat until approved or MAX_ITERS ──────┘

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python agent_team.py "Add a filter by health status to the customer list"
    python agent_team.py          # PO picks the next best task itself

Safety: agents run shell commands scoped to APP_DIR only.
"""

from __future__ import annotations
import os
import sys
import json
import subprocess
import textwrap
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_DIR = Path(os.environ.get(
    "APP_DIR",
    Path(__file__).parent / "customer-health-dashboard"
)).resolve()

MAX_ITERS = int(os.environ.get("MAX_ITERS", "3"))

OPUS   = "claude-opus-4-7"
SONNET = "claude-sonnet-4-6"

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Tools — every agent shares these
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "bash",
        "description": (
            "Run a shell command inside the project directory. "
            "Working dir is APP_DIR. Timeout: 60 s."
        ),
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
# Generic agent runner
# ---------------------------------------------------------------------------
def run_agent(role: str, model: str, system: str, user_message: str) -> str:
    """Agentic loop: keep calling tools until stop_reason == end_turn."""
    messages: list[dict] = [{"role": "user", "content": user_message}]

    print(f"\n{'='*64}")
    print(f"  {role}")
    print(f"{'='*64}")

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        text_blocks = [b.text for b in response.content if b.type == "text"]
        tool_uses   = [b      for b in response.content if b.type == "tool_use"]

        for t in text_blocks:
            print(textwrap.fill(t, width=76, subsequent_indent="  "))

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn" or not tool_uses:
            return "\n".join(text_blocks)

        tool_results = []
        for tu in tool_uses:
            args_preview = json.dumps(tu.input)[:100]
            print(f"\n  [tool] {tu.name}({args_preview})")
            result = _run_tool(tu.name, tu.input)
            print(f"  → {result[:400]}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Agent system prompts
# ---------------------------------------------------------------------------
_BACKEND  = f"{APP_DIR}/backend"
_FRONTEND = f"{APP_DIR}/frontend"

PO_SYSTEM = f"""\
You are the Product Owner for a FastAPI + static-HTML customer-health dashboard.

App layout:
  {_BACKEND}/app.py          — FastAPI routes
  {_BACKEND}/connectors.py   — data sources (mock data, no credentials needed)
  {_BACKEND}/scoring.py      — pure scoring functions
  {_FRONTEND}/customer-health-dashboard.html  — list view
  {_FRONTEND}/customer-detail.html            — detail page

You have TWO modes:

MODE 1 — PLANNING (when given a goal or no goal)
  • Inspect the codebase with read_file / bash.
  • Choose or refine ONE concrete, deliverable task.
  • Output a spec with:
      - What to build (2–3 sentences)
      - Numbered acceptance criteria (testable, specific)
      - Files that will likely change

MODE 2 — JUDGING (when given spec + QA report)
  • Read the actual changed files to verify claims.
  • Return ONLY valid JSON — no prose, no markdown fences:
    {{"approved": true, "reasons": ["..."], "next_steps": []}}
    {{"approved": false, "reasons": ["..."], "next_steps": ["fix X", "add Y"]}}
"""

DEV_SYSTEM = f"""\
You are the Developer. Implement exactly what the spec says — nothing more.

App root: {APP_DIR}
Run the backend (for local testing) with:
  cd {_BACKEND} && /opt/homebrew/bin/python3.11 -m uvicorn app:app --port 8001 &

Rules:
  • Always read a file before editing it.
  • Make the minimal change that satisfies the spec.
  • Prefer editing existing files over creating new ones.
  • Do not add comments explaining what you did — the code should speak.

End your response with a short summary: files changed + why.
"""

REVIEWER_SYSTEM = f"""\
You are the Code Reviewer.

App root: {APP_DIR}

Read the files that were changed, then review them against the spec.
Check: correctness, edge cases, consistency with existing code style, security.
Output:
  • What looks good
  • What MUST change (if anything)
  • Overall verdict: LGTM  or  NEEDS REWORK
"""

QA_SYSTEM = f"""\
You are the QA Engineer. Test the app end-to-end using curl and shell commands.

App root: {APP_DIR}

Steps:
  1. Start the backend if not already running:
       cd {_BACKEND} && /opt/homebrew/bin/python3.11 -m uvicorn app:app --port 8001 &
       sleep 2
  2. Test EVERY acceptance criterion from the spec using curl against http://localhost:8001
  3. Report each criterion: ✅ PASS or ❌ FAIL — with the actual curl output as evidence.
  4. Note any unexpected behaviour.
  5. Kill the test server when done:
       pkill -f "uvicorn.*8001" 2>/dev/null || true
"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_team(goal: str) -> None:
    banner = f"  GOAL: {goal}" if goal else "  GOAL: (PO will choose)"
    print(f"\n{'#'*64}\n{banner}\n{'#'*64}")

    # ── Step 1: PO writes the spec ──────────────────────────────────────────
    po_prompt = (
        f"Goal: {goal}\n\nInspect the codebase and produce the spec."
        if goal else
        "No explicit goal. Inspect the codebase and choose the highest-value next task. Produce the spec."
    )
    spec = run_agent("PRODUCT OWNER — Planning", OPUS, PO_SYSTEM, po_prompt)

    # ── Main loop ───────────────────────────────────────────────────────────
    for iteration in range(1, MAX_ITERS + 1):
        print(f"\n{'─'*64}")
        print(f"  ITERATION {iteration} / {MAX_ITERS}")
        print(f"{'─'*64}")

        dev_summary = run_agent(
            "DEVELOPER", SONNET, DEV_SYSTEM,
            f"Spec:\n{spec}\n\nImplement it.",
        )

        review = run_agent(
            "REVIEWER", SONNET, REVIEWER_SYSTEM,
            f"Spec:\n{spec}\n\nDeveloper summary:\n{dev_summary}\n\nReview the actual files.",
        )

        qa_report = run_agent(
            "QA", SONNET, QA_SYSTEM,
            f"Spec (acceptance criteria):\n{spec}\n\nRun the tests now.",
        )

        verdict_raw = run_agent(
            "PRODUCT OWNER — Judging", OPUS, PO_SYSTEM,
            (
                f"Spec:\n{spec}\n\n"
                f"Developer summary:\n{dev_summary}\n\n"
                f"Code review:\n{review}\n\n"
                f"QA report:\n{qa_report}\n\n"
                "Is the work complete? Return only the JSON verdict."
            ),
        )

        # Parse verdict (handle occasional ```json fences)
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

        # Feed feedback into next iteration
        if iteration < MAX_ITERS:
            spec += (
                f"\n\n--- Round {iteration} feedback ---\n"
                f"Review: {review}\n\n"
                f"QA: {qa_report}\n\n"
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
