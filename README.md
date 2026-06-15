# ai-dev-team

A four-agent loop that works on any codebase: Product Owner → Developer → Reviewer → QA → Product Owner.

## How it works

1. **Product Owner** (Opus) — inspects the codebase, writes a spec with testable acceptance criteria
2. **Developer** (Sonnet) — implements the spec, reads/writes real files
3. **Reviewer** (Sonnet) — reads the actual changed code, flags issues
4. **QA** (Sonnet) — starts the app, runs end-to-end tests against every acceptance criterion
5. **PO** (Opus) — judges the result against the spec, returns a JSON verdict
6. If approved → delivered to you. If not → feedback is fed back and the cycle repeats (up to `MAX_ITERS` times, then escalates to you).

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Point at your project and give it a task
export APP_DIR=/path/to/your/project
/opt/homebrew/bin/python3.11 agent_team.py "Add a /health endpoint"

# Leave the goal blank to let the PO decide what to work on next
/opt/homebrew/bin/python3.11 agent_team.py
```

## Config

Edit the top of `agent_team.py`:

| Variable | Default | Effect |
|----------|---------|--------|
| `APP_DIR` | `$APP_DIR` env var | Root of the project the agents work on |
| `MAX_ITERS` | `3` | Max cycles before escalating to you |
| `OPUS` | `claude-opus-4-7` | Model for the Product Owner |
| `SONNET` | `claude-sonnet-4-6` | Model for Dev, Reviewer, QA |

## Safety

Agents can run arbitrary shell commands scoped to `APP_DIR`. Point it at a git branch, not production.
