import os
import json
import difflib
from pathlib import Path
from dotenv import load_dotenv
import yaml
from jsonschema import Draft202012Validator, ValidationError
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from tools.db import run_sql_query as db_run_sql_query
from tools.fastf1 import fastf1_session_summary, fastf1_driver_laps, fastf1_telemetry

from db import load_strategy, save_strategy
from tools.strat import (
    _load_schema_obj,
    _validate_yaml_against_schema,
    read_strategy_yaml,
    read_strategy_schema,
    structured_strategy_response,
    validate_strategy_yaml,
    save_updated_strategy,
    diff_strategies,
    domain_validate_strategy,
    check_yaml_completeness,
)


## ------- FILE PATHS -------
UPDATED_STRATEGY = None
SCHEMA_PATH = Path("strategy.schema.json")


def check_yaml(yaml_text: str) -> str:
    """Check if YAML has all required top-level sections."""
    try:
        data = yaml.safe_load(yaml_text)
        required = [
            "version",
            "metadata",
            "assumptions",
            "user_view",
            "stints",
            "pit_ops",
            "costs",
        ]
        missing = [s for s in required if s not in data]
        if missing:
            return f"INCOMPLETE: Missing sections: {', '.join(missing)}"

        # Also check if stints is properly populated
        if "stints" in data and (not data["stints"] or len(data["stints"]) == 0):
            return "INCOMPLETE: stints section is empty"

        return "COMPLETE"
    except Exception as e:
        return f"PARSE ERROR: {e}"


def get_analysis_agent(model):
    """Step 1 Agent: Analyzes the change request and current strategy"""
    analysis_agent = create_react_agent(
        model=model,
        tools=[
            read_strategy_yaml,
            read_strategy_schema,
            db_run_sql_query,
            fastf1_session_summary,
            fastf1_driver_laps,
            fastf1_telemetry,
        ],
        prompt=(
            "You are an F1 strategy ANALYST. Your job is to analyze a change request and the current strategy.\n"
            "\n"
            "AVAILABLE TOOLS:\n"
            "- read_strategy_yaml: get the current baseline strategy\n"
            "- read_strategy_schema: get the JSON schema for validation\n"
            "- db_run_sql_query: query team/car/event data\n"
            "- fastf1_session_summary: get F1 session summary data\n"
            "- fastf1_driver_laps: get driver lap data\n"
            "- fastf1_telemetry: get detailed telemetry data\n"
            "\n"
            "You must begin by invoking read_strategy_yaml to retrieve the current strategy.\n"
            "Do NOT describe the strategy before reading it.\n"
            "Invoke read_strategy_schema next to see what fields can be changed.\n"
            "Only after reading those may you output a change plan."
            "YOUR TASK:\n"
            "1. Read the current strategy YAML\n"
            "2. Analyze the change request\n"
            "3. Use FastF1 tools if needed to get real race data for better decisions\n"
            "4. Create a DETAILED change plan that specifies:\n"
            "   - Which stints need to be modified (by stint_id)\n"
            "   - What specific fields need to change\n"
            "   - New values for those fields\n"
            "   - Any cascading changes needed for consistency\n"
            "\n"
            "OUTPUT FORMAT:\n"
            "Provide a structured change plan like:\n"
            "CHANGES NEEDED:\n"
            "- Stint 1: change compound from X to Y, adjust planned_inlap from A to B\n"
            "- Stint 2: change start_lap to C (due to stint 1 change)\n"
            "- user_view.planned_pit_laps: update to [new values]\n"
            "- user_view.plan_summary: update description\n"
            "\n"
            "Be VERY specific with exact values."
            "Do not provide additional explanations beyond the change plan."
        ),
        name="analysis_agent",
    )
    return analysis_agent


def get_implementation_agent(model):
    """Step 2 Agent: Implements the changes to create complete YAML"""
    implementation_agent = create_react_agent(
        model=model,
        tools=[
            read_strategy_yaml,
            read_strategy_schema,
            validate_strategy_yaml,
            save_updated_strategy,
            diff_strategies,
            domain_validate_strategy,
            check_yaml_completeness,
            structured_strategy_response,
        ],
        prompt=(
            "You are an F1 strategy IMPLEMENTER. You receive a change plan and create the complete YAML.\n"
            "\n"
            "AVAILABLE TOOLS:\n"
            "- read_strategy_yaml: get the current baseline strategy\n"
            "- read_strategy_schema: get the JSON schema\n"
            "- validate_strategy_yaml: validate your YAML against schema\n"
            "- domain_validate_strategy: check domain logic\n"
            "- check_yaml_completeness: verify all sections present\n"
            "- diff_strategies: compare with baseline\n"
            "- save_updated_strategy: save the final YAML\n"
            "\n"
            "YOUR TASK:\n"
            "1. Read the baseline strategy YAML\n"
            "2. Apply the changes from the change plan\n"
            "3. Create a COMPLETE YAML with ALL sections:\n"
            "   - version\n"
            "   - metadata (complete with track details)\n"
            "   - assumptions (fuel, tire_availability, constraints)\n"
            "   - user_view (plan_summary, planned_pit_laps)\n"
            "   - stints (complete array with all stint details)\n"
            "   - pit_ops (all timing parameters)\n"
            "   - costs (all weight parameters)\n"
            "4. Come up with a meaningful strategy name and include it in metadata.strategy_name\n"
            "\n"
            "CRITICAL REQUIREMENTS:\n"
            "- The YAML must be COMPLETE - include EVERY section\n"
            "- Use validate_strategy_yaml to check schema compliance\n"
            "- Use domain_validate_strategy to check logic\n"
            "- Use check_yaml_completeness to ensure nothing is missing\n"
            "- Keep iterating until all validations pass\n"
            "- The final strategy should have a meaningful name different from default_strategy\n"
            "- Save the final YAML using save_updated_strategy\n"
            "\n"
            "OUTPUT:\n"
            "After saving, output ONLY the strategy name."
        ),
        name="implementation_agent",
    )
    return implementation_agent


if __name__ == "__main__":

    ## -------- Input --------
    MODEL_NAME = "openai/gpt-oss-120b"  # Using model with larger context
    SUGGESTION_TEXT = "Adjust pit stops: Adjust the pit stop strategy to account for the changed conditions, potentially delaying or accelerating pit stops to take advantage of the rain."

    ## Validation
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in environment variables")
        raise SystemExit(1)

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"strategy.schema.json not found at {SCHEMA_PATH}")

    # Initialize model with explicit parameters
    model = ChatGroq(model=MODEL_NAME)

    if not SUGGESTION_TEXT.strip():
        print("Enter your strategy change request:")
        SUGGESTION_TEXT = input("> ").strip()
        if not SUGGESTION_TEXT:
            raise SystemExit("No suggestion provided")

    print(f"\n=== Processing change request: {SUGGESTION_TEXT} ===\n")

    ## STEP 1: Analyze the change request
    print("Step 1: Analyzing change request...")
    analysis_agent = get_analysis_agent(model)

    analysis_task = (
        f"Analyze this change request and create a detailed change plan:\n"
        f"CHANGE REQUEST: {SUGGESTION_TEXT}\n\n"
        "First read the current strategy, then analyze what specific changes are needed.\n"
        "Consider using FastF1 tools to get real data if it would help make better decisions.\n"
        "Be very specific about what needs to change."
    )

    try:
        analysis_result = analysis_agent.invoke(
            {"messages": [HumanMessage(content=analysis_task)]},
            config={"recursion_limit": 50},
        )

        # Extract the change plan from the last AI message
        change_plan = None
        if isinstance(analysis_result, dict) and "messages" in analysis_result:
            for msg in reversed(analysis_result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    change_plan = msg.content
                    break

        if not change_plan:
            raise RuntimeError("Analysis agent did not produce a change plan")

        print(f"\nChange Plan:\n{change_plan}\n")

    except Exception as e:
        import traceback

        print("\n--- Full error trace ---")
        traceback.print_exc()
        if hasattr(e, "body"):
            print("\n--- Groq error body ---")
            print(json.dumps(str(e.body), indent=2))  # type: ignore
        else:
            print("\n(No error body available)")
        raise

    from report_generator import get_justification_agent

    justifier = get_justification_agent(model)

    justif_task = (
        "Produce source-backed justifications for the change plan below.\n"
        "Return ONLY via structured_justification_response.\n\n"
        f"{change_plan}"
    )

    justif_result = justifier.invoke(
        {"messages": [HumanMessage(content=justif_task)]},
        config={"recursion_limit": 50},
    )

    # Extract the JSON payload from the tool call if you want to persist/log it
    justification_json = None
    if isinstance(justif_result, dict) and "messages" in justif_result:
        for msg in reversed(justif_result["messages"]):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == "structured_justification_response":
                        justification_json = tc.get("args", {}).get("payload")
                        break
    print(f"justification_json: {justification_json}")

    ## STEP 2: Implement the changes
    print("\nStep 2: Implementing changes to create complete YAML...")
    implementation_agent = get_implementation_agent(model)

    implementation_task = (
        f"Implement the following changes to create a COMPLETE strategy YAML:\n\n"
        f"ORIGINAL REQUEST: {SUGGESTION_TEXT}\n\n"
        f"CHANGE PLAN:\n{change_plan}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Read the baseline strategy YAML first\n"
        "2. Apply the changes from the plan above\n"
        "3. Generate a COMPLETE YAML with ALL sections present\n"
        "4. Validate using validate_strategy_yaml\n"
        "5. Check completeness using check_yaml_completeness\n"
        "6. Validate domain logic using domain_validate_strategy\n"
        "7. Keep fixing until all validations pass\n"
        "8. Save using save_updated_strategy\n"
        "9. Output ONLY the final complete YAML\n\n"
        "The YAML MUST include these sections in order:\n"
        "- version\n"
        "- metadata (with track subdocument)\n"
        "- assumptions (with fuel, tire_availability, constraints)\n"
        "- user_view\n"
        "- stints (complete array)\n"
        "- pit_ops\n"
        "- costs"
    )

    max_retries = 3
    final_yaml_text = None

    for attempt in range(max_retries):
        try:
            print(f"Implementation attempt {attempt + 1}...")
            result = implementation_agent.invoke(
                {"messages": [HumanMessage(content=implementation_task)]},
                config={
                    "recursion_limit": 100
                },  # Increase recursion for multiple validation attempts
            )

            # Extract YAML from the result
            if isinstance(result, dict) and "messages" in result:
                for msg in reversed(result["messages"]):
                    # Check tool calls first
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if tool_call.get("name") == "save_updated_strategy":
                                yaml_text = tool_call.get("args", {}).get(
                                    "yaml_text", ""
                                )
                                if yaml_text:
                                    final_yaml_text = yaml_text
                                    break

                    # Then check message content
                    if (
                        not final_yaml_text
                        and isinstance(msg, AIMessage)
                        and msg.content
                    ):
                        content = msg.content.strip()  # type: ignore
                        # Look for YAML-like content
                        if "version:" in content or "metadata:" in content:
                            # Clean up the content
                            lines = content.split("\n")
                            yaml_lines = []
                            in_yaml = False
                            for line in lines:
                                if "version:" in line or (in_yaml and line.strip()):
                                    in_yaml = True
                                    yaml_lines.append(line)
                                elif in_yaml and not line.strip():
                                    yaml_lines.append(line)

                            if yaml_lines:
                                final_yaml_text = "\n".join(yaml_lines)
                                break

            if final_yaml_text:
                # Validate completeness
                completeness = check_yaml(final_yaml_text)
                if "INCOMPLETE" in completeness:
                    print(
                        f"Attempt {attempt + 1} produced incomplete YAML: {completeness}"
                    )
                    if attempt < max_retries - 1:
                        # Retry with more specific instructions
                        implementation_task = (
                            f"The previous YAML was incomplete: {completeness}\n\n"
                            "Please generate the COMPLETE YAML again with ALL sections.\n"
                            "Do not stop until you have included:\n"
                            "- version\n- metadata\n- assumptions\n- user_view\n"
                            "- stints (with actual stint data)\n- pit_ops\n- costs\n\n"
                            f"Original change request: {SUGGESTION_TEXT}"
                        )
                        final_yaml_text = None
                        continue
                else:
                    print(
                        f"Successfully generated complete YAML on attempt {attempt + 1}"
                    )
                    break

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                raise

    if not final_yaml_text:
        raise RuntimeError("Failed to generate complete YAML after all attempts")

    print(f"\nExtracted YAML (first 500 chars):\n{final_yaml_text[:500]}...\n")

    ## Final validation and save
    schema_obj = _load_schema_obj(SCHEMA_PATH)
    schema_err = _validate_yaml_against_schema(final_yaml_text, schema_obj)
    # domain_err = domain_validate_strategy(final_yaml_text)

    if schema_err:
        print(f"\nSchema validation FAILED:\n{schema_err}")
        # UPDATED_STRATEGY_PATH.with_suffix(".invalid.yaml").write_text(final_yaml_text, encoding="utf-8")
        raise SystemExit(1)

    # if "ERROR" in domain_err:
    #     print(f"\nDomain validation FAILED:\n{domain_err}")
    #     # UPDATED_STRATEGY_PATH.with_suffix(".invalid.yaml").write_text(final_yaml_text, encoding="utf-8")
    #     raise SystemExit(1)

    # UPDATED_STRATEGY_PATH.write_text(final_yaml_text, encoding="utf-8")

    ## Print success report
    # old_text = _load_yaml_text(ORIG_STRATEGY_PATH)
    # diff = _unified_diff(old_text, final_yaml_text)

    print("\n=== SUCCESS ===")
    print(f"Validation: PASSED")
    # print(f"Saved to: {UPDATED_STRATEGY_PATH}")
    # print(f"\nChanges made:\n{diff[:1000]}..." if len(diff) > 1000 else f"\nChanges made:\n{diff}")
