"""
agent_team_cc.py
================
Same four-agent loop as agent_team.py but powered by the Claude Code CLI
instead of the Anthropic API. Uses your existing Claude Code subscription —
no separate API key needed.

  Product Owner → Developer → Reviewer → QA → Product Owner
  └──────────────── repeat until approved or MAX_ITERS ──────┘

Usage:
    export APP_DIR=/path/to/your/project
    /opt/homebrew/bin/python3.11 agent_team_cc.py "Add a /health endpoint"
    /opt/homebrew/bin/python3.11 agent_team_cc.py   # PO picks the task

    # Optional: tell QA exactly how to start the app
    export START_CMD="npm run dev"

Requirements:
    claude CLI must be on PATH and logged in (run `claude` once to verify).

Note:
    Each agent spawns a fresh `claude --print` process. Claude Code has full
    file + shell access to APP_DIR. --dangerously-skip-permissions is used
    so the loop runs without interactive prompts — point APP_DIR at a git
    branch, not production.
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
CLAUDE    = os.environ.get("CLAUDE_BIN", "claude")   # path to claude CLI


# ---------------------------------------------------------------------------
# Run one agent via claude --print
# ---------------------------------------------------------------------------
def run_agent(role: str, prompt: str) -> str:
    print(f"\n{'='*64}")
    print(f"  [CC] {role}")
    print(f"{'='*64}")

    result = subprocess.run(
        [CLAUDE, "--print", "--dangerously-skip-permissions", prompt],
        capture_output=True,
        text=True,
        cwd=str(APP_DIR),
        timeout=300,
    )

    output = result.stdout.strip()
    if result.returncode != 0 and result.stderr:
        print(f"  [stderr] {result.stderr[:400]}")

    if output:
        print(textwrap.fill(output, width=76, subsequent_indent="  "))

    return output


# ---------------------------------------------------------------------------
# System prompts — baked into each agent's prompt since CC has no system param
# ---------------------------------------------------------------------------
_start_cmd_hint = (
    f"The command to start the app for testing is: {START_CMD}"
    if START_CMD else
    "Inspect the project to figure out how to start it (check package.json, "
    "Makefile, requirements.txt, Procfile, README, etc.) and include the start "
    "command in the spec so Developer and QA know what to use. Use port 8001."
)


def _po_plan_prompt(goal: str) -> str:
    task = f"Goal: {goal}" if goal else (
        "No explicit goal. Choose the single highest-value next task for this project."
    )
    return f"""\
You are the Product Owner for the project at: {APP_DIR}

{task}

Inspect the codebase (read files, run ls/find/cat as needed), then produce a spec:

1. Project type and how to start it (command + port)
   {_start_cmd_hint}
2. What to build — 2-3 sentences
3. Numbered acceptance criteria — each must be testable with curl or a shell command
4. Files that will likely change

Be concrete and specific. Do not implement anything.
"""


def _dev_prompt(spec: str) -> str:
    return f"""\
You are the Developer working on the project at: {APP_DIR}

Here is the spec to implement:

{spec}

Instructions:
- Read every file before editing it.
- Make the minimal change that satisfies the spec — nothing more.
- Prefer editing existing files over creating new ones.
- Do not add comments explaining what you did.
- Actually make the changes now — edit the files.

When done, output a short summary: which files you changed and why.
"""


def _reviewer_prompt(spec: str, dev_summary: str) -> str:
    return f"""\
You are the Code Reviewer for the project at: {APP_DIR}

Spec:
{spec}

Developer's summary:
{dev_summary}

Read the files that were changed, then review them against the spec.
Check: correctness, edge cases, consistency with existing code style, security.

Output:
- What looks good
- What MUST change (if anything)
- Overall verdict: LGTM  or  NEEDS REWORK
"""


def _qa_prompt(spec: str) -> str:
    return f"""\
You are the QA Engineer for the project at: {APP_DIR}

Spec (acceptance criteria):
{spec}

The spec tells you how to start the app. Follow those instructions.

Steps:
1. Start the app as specified (background it with &), then wait 2-3 seconds.
2. Test EVERY acceptance criterion using curl or shell commands.
3. Report each criterion: PASS or FAIL — with the actual command output as evidence.
4. Note any unexpected behaviour.
5. Kill the test server when done.

Run the tests now.
"""


def _po_judge_prompt(spec: str, dev_summary: str, review: str, qa_report: str) -> str:
    return f"""\
You are the Product Owner for the project at: {APP_DIR}

You must judge whether the implementation is complete and correct.

Original spec:
{spec}

Developer summary:
{dev_summary}

Code review:
{review}

QA report:
{qa_report}

Read the actual changed files to verify the claims, then return ONLY valid JSON
with no prose or markdown around it:

{{"approved": true, "reasons": ["..."], "next_steps": []}}
{{"approved": false, "reasons": ["..."], "next_steps": ["fix X", "add Y"]}}
"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_team(goal: str) -> None:
    banner = f"  GOAL: {goal}" if goal else "  GOAL: (PO will choose)"
    print(f"\n{'#'*64}\n{banner}\n{'#'*64}")
    print(f"  PROJECT: {APP_DIR}")
    print(f"  ENGINE:  Claude Code CLI  ({CLAUDE})")
    if START_CMD:
        print(f"  START:   {START_CMD}")
    print(f"{'#'*64}")

    # ── Step 1: PO writes the spec ──────────────────────────────────────────
    spec = run_agent("PRODUCT OWNER — Planning", _po_plan_prompt(goal))

    # ── Main loop ───────────────────────────────────────────────────────────
    for iteration in range(1, MAX_ITERS + 1):
        print(f"\n{'─'*64}")
        print(f"  ITERATION {iteration} / {MAX_ITERS}")
        print(f"{'─'*64}")

        dev_summary = run_agent("DEVELOPER",  _dev_prompt(spec))
        review      = run_agent("REVIEWER",   _reviewer_prompt(spec, dev_summary))
        qa_report   = run_agent("QA",         _qa_prompt(spec))
        verdict_raw = run_agent("PRODUCT OWNER — Judging",
                                _po_judge_prompt(spec, dev_summary, review, qa_report))

        # Parse verdict
        try:
            raw = verdict_raw.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            # Find the JSON object even if there's surrounding text
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            verdict: dict = json.loads(raw[start:end])
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
    # Verify claude CLI is available
    check = subprocess.run([CLAUDE, "--version"], capture_output=True, text=True)
    if check.returncode != 0:
        print(f"Error: '{CLAUDE}' not found or not working. Make sure claude CLI is installed and logged in.")
        sys.exit(1)

    goal = " ".join(sys.argv[1:]).strip()
    if not goal and sys.stdin.isatty():
        goal = input("What should the team work on? (leave blank for PO to decide) > ").strip()
    run_team(goal)
