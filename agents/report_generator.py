# agents/justifier.py
import json
from typing import List, Dict, Any, Optional

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

from tools.fastf1 import fastf1_session_summary, fastf1_driver_laps, fastf1_telemetry
from tools.db import run_sql_query as db_run_sql_query
from tools.strat import read_strategy_yaml


def _require_fastf1_sources(j: Dict[str, Any]) -> Optional[str]:
    """Return error string if any change lacks a FastF1 source."""
    misses = []
    for ch in j.get("justifications", []):
        has_f1 = any(src.get("type") == "FastF1" for src in ch.get("sources", []))
        if not has_f1:
            misses.append(ch.get("change_id", "?"))
    if misses:
        return f"Every change must have at least one FastF1 citation; missing: {', '.join(misses)}"
    return None


@tool("structured_justification_response", return_direct=True)
def structured_justification_response(payload: str = None, **kwargs) -> str:
    """
    Accepts either:
      - payload: JSON string with the full response object, OR
      - direct keyword args: strategy_id, justifications, db_used
    Returns pretty JSON or 'ERROR: ...'
    """
    import json

    # 1) If caller passed a dict via kwargs, build payload from it.
    if payload is None and kwargs:
        try:
            payload = json.dumps(kwargs)
        except Exception as e:
            return f"ERROR: failed to build JSON from kwargs: {e}"

    if payload is None:
        return "ERROR: missing 'payload' (or equivalent keyword fields)."

    # 2) Validate structure
    try:
        obj = json.loads(payload)
    except Exception as e:
        return f"ERROR: invalid JSON: {e}"

    # require >=1 FastF1 source per change
    misses = []
    for ch in obj.get("justifications", []):
        has_f1 = any(src.get("type") == "FastF1" for src in ch.get("sources", []))
        if not has_f1:
            misses.append(ch.get("change_id", "?"))
    if misses:
        return f"ERROR: Every change must have at least one FastF1 citation; missing: {', '.join(misses)}"

    if "strategy_id" not in obj or "justifications" not in obj:
        return "ERROR: payload must include 'strategy_id' and 'justifications'"

    return json.dumps(obj, ensure_ascii=False, indent=2)


def get_justification_agent(model):
    """
    Agent that:
      - Reads the baseline strategy (for context)
      - Takes the analysis agent's change plan (plain text)
      - Calls FastF1/DB tools to gather concrete evidence
      - Emits a strict JSON justification via structured_justification_response
    """
    prompt = (
        "You are a JUSTIFICATION agent. You receive a proposed strategy change plan produced by the analysis agent.\n"
        "Your job: For EACH change, produce a compact justification that CITES EXACT DATA SOURCES.\n\n"
        "REQUIRED SOURCES PER CHANGE:\n"
        "- >=1 FastF1 citation using the provided tools (fastf1_driver_laps, fastf1_session_summary, or fastf1_telemetry).\n"
        "- DB citation ONLY IF you actually call the SQL tool (run_sql_query). If you do not call it, set db_used=false and omit DB sources.\n\n"
        "WHAT TO INCLUDE AS CITATIONS:\n"
        "- For FastF1 tools, include the exact tool name and the params you used (query/year/GP/session, driver, lap indices, channels).\n"
        "- For DB queries, include the exact SQL text you executed.\n"
        "- Add a 1–2 line 'evidence' text per source summarizing the relevant fact you used from the result.\n\n"
        "OUTPUT FORMAT:\n"
        "Call structured_justification_response(payload=JSON) with this shape:\n"
        "{ 'strategy_id': str, 'justifications': [ { 'change_id': str, 'summary': str, 'reasoning': str, 'sources': [ ... ] } ], 'db_used': bool }\n"
        "Make it terse and factual; no extra prose outside the JSON.\n"
        "If a change lacks sufficient grounds from FastF1, DO NOT invent data; still include a FastF1 tool call that proves it is insufficient and state that in 'evidence'.\n"
    )

    return create_react_agent(
        model=model,
        tools=[
            read_strategy_yaml,
            fastf1_session_summary,
            fastf1_driver_laps,
            fastf1_telemetry,
            db_run_sql_query,
            structured_justification_response,
        ],
        prompt=prompt,
        name="justification_agent",
    )
