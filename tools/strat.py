import yaml
import json
import difflib
from jsonschema import Draft202012Validator, ValidationError
from pathlib import Path
from langchain_core.tools import tool

from db import load_strategy, save_strategy

SCHEMA_PATH = Path("strategy.schema.json")


def _load_yaml_text(p):
    return p.read_text(encoding="utf-8")


def _load_schema_obj(p):
    return json.loads(p.read_text(encoding="utf-8"))


def _validate_yaml_against_schema(yaml_text, schema_obj):
    """Return None if valid, else a multi-line error string."""
    try:
        data = yaml.safe_load(yaml_text)
    except Exception as e:
        return f"YAML parse error: {e}"

    try:
        Draft202012Validator(schema_obj).validate(data)
        return None
    except ValidationError as e:
        path = "$" + "".join(
            f"[{repr(p)}]" if isinstance(p, int) else f".{p}" for p in e.path
        )
        return f"Schema validation error at {path}:\n{e.message}"


def _unified_diff(a, b):
    return "".join(
        difflib.unified_diff(
            a.splitlines(keepends=True),
            b.splitlines(keepends=True),
            fromfile="strategy.yaml",
            tofile="strategy.updated.yaml",
            n=3,
        )
    )


## -------- Tools ----------


@tool("read_strategy_yaml", return_direct=False)
def read_strategy_yaml(strategy: str = "default_strategy") -> str:
    """Return the current baseline strategy.yaml as text. The default is 'default_strategy'."""
    return yaml.dump(load_strategy(strategy))


@tool("read_strategy_schema", return_direct=False)
def read_strategy_schema() -> str:
    """Return the JSON schema (as compact JSON string) used to validate strategy YAML."""
    if not SCHEMA_PATH.exists():
        return "ERROR: strategy.schema.json not found."
    return json.dumps(_load_schema_obj(SCHEMA_PATH))


@tool("validate_strategy_yaml", return_direct=False)
def validate_strategy_yaml(yaml_text: str) -> str:
    """Validate provided YAML text against the loaded JSON schema. Returns 'OK' or an error report."""
    if not SCHEMA_PATH.exists():
        return "ERROR: strategy.schema.json not found."
    schema = _load_schema_obj(SCHEMA_PATH)
    err = _validate_yaml_against_schema(yaml_text, schema)
    return "OK" if err is None else err


@tool(return_direct=False)
def save_updated_strategy(yaml_text: str) -> str:
    """Write the new strategy YAML to the database."""
    try:
        data = yaml.safe_load(yaml_text)
        strategy_name = data.get("metadata", {}).get(
            "strategy_name", "default_strategy"
        )
        save_strategy(strategy_name, data)
    except Exception as e:
        return f"ERROR saving strategy: {e}"
    return f"SAVED strategy {strategy_name} to database."


@tool("diff_strategies", return_direct=False)
def diff_strategies(new_yaml_text: str) -> str:
    """Return unified diff between baseline strategy.yaml and the new YAML text."""
    # load old db strategy

    old_yaml = load_strategy("default_strategy")
    old_text = yaml.dump(old_yaml)
    return _unified_diff(old_text, new_yaml_text)


@tool("check_yaml_completeness", return_direct=False)
def check_yaml_completeness(yaml_text: str) -> str:
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


@tool("domain_validate_strategy", return_direct=False)
def domain_validate_strategy(yaml_text: str) -> str:
    """
    Domain-level validation (beyond schema) for race logic.
    Returns 'OK' or a multi-line report of errors/warnings.
    """
    try:
        data = yaml.safe_load(yaml_text)
    except Exception as e:
        return f"YAML parse error: {e}"

    errors = []
    warnings = []

    try:
        track = data["metadata"]["track"]
        total_laps = int(track["laps"])
        constraints = data.get("assumptions", {}).get("constraints", {}) or {}
        inv = data.get("assumptions", {}).get("tire_availability", {}) or {}
        allow_used = bool(constraints.get("allow_used_tyre", False))
        min_compounds = int(constraints.get("min_compounds_required", 0))
        max_pitstops = int(constraints.get("max_pitstops", 99))
        stints = data.get("stints", []) or []
        user_view = data.get("user_view", {}) or {}
        uv_pit_laps = list(user_view.get("planned_pit_laps", []) or [])
    except Exception as e:
        return f"Domain parse error: {e}"

    # 1) ≥ 2 stints unless explicit wet exemption
    compounds = [s.get("compound", "").strip() for s in stints]
    only_wet = (
        all(c in {"Wet", "Intermediate"} for c in compounds) and len(compounds) > 0
    )
    if len(stints) < 2 and not only_wet:
        errors.append("Must schedule ≥ 2 stints for dry/normal conditions; found < 2.")

    # 2) Min dry compounds if dry compounds appear
    dry_compounds_used = {c for c in compounds if c in {"Soft", "Medium", "Hard"}}
    if dry_compounds_used and min_compounds > 0:
        if len(dry_compounds_used) < min_compounds:
            errors.append(
                f"Uses {len(dry_compounds_used)} dry compound(s) but min_compounds_required={min_compounds}."
            )

    # 3) Max pitstops
    planned_inlaps = [int(s.get("planned_inlap", 0) or 0) for s in stints]
    actual_stops = len([x for x in planned_inlaps if x and x > 0])
    if actual_stops > max_pitstops:
        errors.append(
            f"Planned stops={actual_stops} exceed max_pitstops={max_pitstops}."
        )

    # 4) Continuity
    for i in range(1, len(stints)):
        prev = stints[i - 1]
        cur = stints[i]
        prev_in = int(prev.get("planned_inlap", 0) or 0)
        cur_start = int(cur.get("start_lap", 0) or 0)
        if prev_in == 0 and i != len(stints) - 1:
            errors.append(
                f"Only final stint may have planned_inlap=0; found at stint_id={prev.get('stint_id')}."
            )
        if prev_in != 0 and cur_start != prev_in + 1:
            errors.append(
                f"Stint continuity violated between stint_id={prev.get('stint_id')} and {cur.get('stint_id')} (expected start_lap={prev_in+1}, got {cur_start})."
            )

    # 5) user_view planned pit laps match stints
    expected_uv = sorted([x for x in planned_inlaps if x > 0])
    if uv_pit_laps != expected_uv:
        errors.append(
            f"user_view.planned_pit_laps {uv_pit_laps} must equal non-zero planned_inlap {expected_uv}."
        )

    # 6) Inventory + allow_used
    usage = {}
    for s in stints:
        comp = s.get("compound", "")
        cond = s.get("set_condition", "new")
        key = f"{comp}_{cond}"
        usage[key] = usage.get(key, 0) + 1
        if cond == "used" and not allow_used:
            errors.append(
                f"allow_used_tyre=false but stint_id={s.get('stint_id')} uses 'used' set."
            )

    for key, count in usage.items():
        parts = key.split("_")
        if len(parts) == 2:
            ck = f"{parts[0]}_{parts[1]}"
            if ck in inv:
                if count > int(inv.get(ck, 0)):
                    errors.append(
                        f"Stints require {count} of {ck} but inventory has {inv.get(ck)}."
                    )
            else:
                errors.append(
                    f"Inventory missing {ck}; cannot allocate {count} set(s)."
                )

    # 7) No marathon dry stint
    for s in stints:
        comp = s.get("compound", "")
        if comp in {"Soft", "Medium", "Hard"}:
            tlen = int(s.get("target_len_laps", 0) or 0)
            if tlen > 0.7 * total_laps:
                warnings.append(
                    f"Dry stint {comp} length {tlen} laps > 70% of race; likely unrealistic."
                )

    if errors:
        return (
            "ERRORS:\n- "
            + "\n- ".join(errors)
            + ("\n\nWARNINGS:\n- " + "\n- ".join(warnings) if warnings else "")
        )
    if warnings:
        return "WARNINGS:\n- " + "\n- ".join(warnings)
    return "OK"
