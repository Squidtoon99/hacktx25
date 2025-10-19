from typing import Optional
import psycopg2
from langchain_core.tools import tool
from psycopg2.extras import Json
from psycopg2.extras import RealDictCursor
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

db_conn = None
cur = None


def initialize_db_connection():
    """Initialize the database connection and cursor."""
    global db_conn, cur
    if db_conn is None or cur is None:
        db_conn = psycopg2.connect(os.getenv("PG_CONNECTION_STRING"))
        cur = db_conn.cursor(cursor_factory=RealDictCursor)


# infer the format of the database
if os.getenv("PG_CONNECTION_STRING"):
    # Only initialize at import time when a connection string is provided.
    # This avoids raising errors during local imports when no DB is configured.
    initialize_db_connection()

# CREATE TABLE strategies (
#     id SERIAL PRIMARY KEY,
#     name TEXT NOT NULL,
#     details JSONB
# );


def load_strategy(strategy_name: str) -> dict:
    """Load a strategy from the database by name."""
    if not cur:
        return {}
    cur.execute("SELECT details FROM strategies WHERE name = %s;", (strategy_name,))
    strategy = cur.fetchone()

    if strategy:
        return dict(strategy["details"])
    else:
        raise ValueError(f"Strategy '{strategy_name}' not found.")


def save_strategy(strategy_name: str, strategy_details: dict) -> None:
    """Save a strategy to the database."""
    if not cur or not db_conn:
        return

    cur.execute(
        """
        INSERT INTO strategies (name, details)
        VALUES (%s, %s)
        ON CONFLICT (name) DO UPDATE SET details = EXCLUDED.details;
        """,
        (strategy_name, Json(strategy_details)),
    )
    db_conn.commit()


def remove_strategy(strategy_name: str) -> None:
    """Remove a strategy from the database by name."""
    if not cur or not db_conn:
        return

    cur.execute(
        "DELETE FROM strategies WHERE name = %s;",
        (strategy_name,),
    )
    db_conn.commit()


def list_strategies() -> list:
    """Return a list of strategy names stored in the database."""
    if not cur:
        return []
    cur.execute("SELECT name FROM strategies;")
    rows = cur.fetchall()
    # rows are RealDict rows like {'name': 'default_strategy'}
    return [r["name"] for r in rows]


if __name__ == "__main__":
    # Example usage
    strategy = {
        "version": 1,
        "metadata": {
            "strategy_id": "STRAT-2025-BHR-LEC-A",
            "created_utc": "2025-10-19T05:00:00Z",
            "team": "RedBull",
            "driver": "VER",
            "event": "2025 Bahrain Grand Prix",
            "track": {
                "name": "Bahrain International Circuit",
                "laps": 57,
                "pit_lane_time_loss_s": 22.5,
                "degradation_model": {
                    "units": "s_per_lap",
                    "Soft": 0.08,
                    "Medium": 0.06,
                    "Hard": 0.045,
                    "Intermediate": 0.09,
                    "Wet": 0.11,
                },
                "warmup_loss_first_lap_s": {
                    "Soft": 0.2,
                    "Medium": 0.5,
                    "Hard": 0.8,
                    "Intermediate": 0.6,
                    "Wet": 0.7,
                },
            },
        },
        "assumptions": {
            "fuel": {"start_mass_kg": 110.0, "burn_rate_kg_per_lap": 1.65},
            "tire_availability": {
                "Soft_new": 1,
                "Medium_new": 2,
                "Hard_new": 1,
                "Soft_used": 0,
                "Medium_used": 0,
                "Hard_used": 0,
                "Intermediate_new": 1,
                "Wet_new": 1,
            },
            "constraints": {
                "min_compounds_required": 2,
                "allow_used_tyre": False,
                "max_pitstops": 3,
                "avoid_compounds": [],
                "keep_track_position_bias": 0.6,
                "undercut_aggressiveness": 0.7,
                "tyre_age_tolerance_laps": 2,
            },
        },
        "user_view": {
            "plan_summary": "2-stop: Start I → W → I",
            "planned_pit_laps": [17, 38],
        },
        "stints": [
            {
                "stint_id": 1,
                "start_lap": 1,
                "planned_inlap": 17,
                "compound": "Intermediate",
                "set_condition": "new",
                "target_len_laps": 17,
                "expected_age_at_box_laps": 16,
                "target_pace_adjust_s": 0.0,
                "push_profile": "normal",
                "pit_window_laps": [15, 20],
                "notes": "Cover early undercut if leaders stop <= lap 15 (+/- 1 lap).",
            },
            {
                "stint_id": 2,
                "start_lap": 18,
                "planned_inlap": 38,
                "compound": "Wet",
                "set_condition": "new",
                "target_len_laps": 21,
                "expected_age_at_box_laps": 21,
                "target_pace_adjust_s": 0.1,
                "push_profile": "conserve",
                "pit_window_laps": [34, 40],
                "notes": "Prefer extending to lap 38–40 if clear air.",
            },
            {
                "stint_id": 3,
                "start_lap": 39,
                "planned_inlap": 0,
                "compound": "Intermediate",
                "set_condition": "new",
                "target_len_laps": 19,
                "expected_age_at_flag_laps": 19,
                "target_pace_adjust_s": -0.05,
                "push_profile": "push",
                "pit_window_laps": [],
            },
        ],
        "pit_ops": {
            "nominal_box_time_s": 2.6,
            "unsafe_release_risk_weight": 0.15,
            "traffic_merge_penalty_s": 0.7,
        },
        "costs": {
            "weight_total_time": 1.0,
            "weight_track_position": 0.6,
            "weight_tire_life_buffer": 0.4,
            "weight_compound_preference": 0.2,
            "weight_risk": 0.5,
        },
    }

    save_strategy("default_strategy", strategy)
    loaded_strategy = load_strategy("default_strategy")
    print("Loaded Strategy:", loaded_strategy)
