import os
import traceback

from langchain_core.tools import tool

import fastf1
import pandas as pd


def ensure_fastf1_available():
    if fastf1 is None:
        raise RuntimeError(
            "fastf1 (and/or pandas) is not installed. Add 'fastf1' and"
            " 'pandas' to requirements.txt and install them."
        )


def _set_cache_dir():
    # Allow overriding via env var FASTF1_CACHE
    cache_dir = os.getenv("FASTF1_CACHE_DIR") or os.path.expanduser("~/.cache/fastf1")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


@tool
def fastf1_session_summary(session_query: str = "2023 Spanish GP Race") -> str:
    """Return a short summary for a FastF1 session.

    Params:
    * session_query: string describing the session to load.
        Options include year, grand_prix (or event), and session (or type).
    Returns:
    * A short summary of the session including top lap times.

    Example session_query formats:
    - "2023 Spanish GP Race"
    - "2022 Monaco GP Race"
    - "2021 British GP Race"
    """
    ensure_fastf1_available()
    _set_cache_dir()

    # Parse session_query
    parts = session_query.split()
    year = None
    session_name = None
    gp = None
    if parts and parts[0].isdigit():
        year = int(parts[0])
        # assume next tokens until 'GP' is the grand prix
        if "GP" in parts:
            gp_idx = parts.index("GP")
            gp = " ".join(parts[1 : gp_idx + 1])
            if len(parts) > gp_idx + 1:
                session_name = parts[gp_idx + 1]
        elif len(parts) >= 3:
            gp = parts[1]
            session_name = parts[2]

    # Use fastf1 API to load session
    session, session_name = None, "Race"  # Hardcoded default
    if year and gp and session_name:
        # fastf1 expects event name without 'GP' sometimes; try both
        try:
            session = fastf1.get_session(year, gp.replace(" GP", ""), session_name)
        except Exception:
            try:
                session = fastf1.get_session(year, gp, session_name)
            except Exception as e:
                return f"Failed to load session: {str(e)}"
    else:
        return "Could not parse session from query. Provide 'year grand_prix session' or 'year=YYYY;grand_prix=Name;session=FP1'."

    session.load(laps=False, telemetry=False)

    # Build a short summary
    summary_lines = [f"Session: {session.event['EventName']} ({session.name})"]
    summary_lines.append(f"Date: {session.event['EventDate']}")
    summary_lines.append(f"Session type: {session.name}")

    # Top lap times (laps not loaded, try to load minimal laps)
    session.load(telemetry=False)

    if not session.laps.empty:
        best = session.laps.nsmallest(5, "LapTime")["LapTime"].dt.total_seconds()
        summary_lines.append(
            "Top lap times (s): " + ", ".join([f"{x:.3f}" for x in best.tolist()])
        )

    return "\n".join(summary_lines)


@tool
def fastf1_driver_laps(query: str = "2023 Spanish GP Race", driver: str = "VER"):
    """Fetch laps for a driver in a session.

    Params:
    * query: session query in format '2023 Spanish GP Race'. Must match format
      exactly (e.g., GP instead of Grand Prix).
    * driver: driver code (e.g., 'VER')

    Example: fastf1_driver_laps("2023 Spanish GP Race", driver="VER")
    """
    ensure_fastf1_available()
    _set_cache_dir()

    # Parse session params like in session_summary function
    parts = query.split()
    year = int(parts[0]) if parts and parts[0].isdigit() else None
    gp = None
    session_name = None
    if "GP" in parts:
        gp_idx = parts.index("GP")
        gp = " ".join(parts[1 : gp_idx + 1])
        if len(parts) > gp_idx + 1:
            session_name = parts[gp_idx + 1]

    if not (year and gp and session_name):
        return "Could not parse session from query."

    session_name = "Race"
    session = fastf1.get_session(year, gp.replace(" GP", ""), session_name)
    session.load(telemetry=False)
    driver_laps = session.laps.pick_driver(driver)
    if driver_laps.empty:
        return f"No laps found for driver {driver} in session {query}"

    return driver_laps


@tool()
def fastf1_telemetry(
    lap_idx: int, driver: str = "VER", channels: list[str] = ["LapNumber"]
):
    """Fetch raw telemetry for a driver's lap.

    Available Telemetry:
    •	SessionTime - The elapsed time since the session began at the moment of the sample.
        •	Driver — Three-letter driver identifier (e.g., VER, HAM).
        •	DriverNumber — Driver’s racing number as a string (e.g., “1”, “16”).
        •	LapTime — Recorded lap time for this lap.
        •	LapNumber — Sequential lap counter within the session.
        •	Stint — Stint number (increments when tyres change).
        •	PitOutTime — Session time the car exited the pit lane (marks an out-lap if present).
        •	PitInTime — Session time the car entered the pit lane (marks an in-lap if present).
        •	Sector1Time — Sector 1 time for this lap.
        •	Sector2Time — Sector 2 time for this lap.
        •	Sector3Time — Sector 3 time for this lap.
        •	Sector1SessionTime — Session time when the S1 time was set.
        •	Sector2SessionTime — Session time when the S2 time was set.
        •	Sector3SessionTime — Session time when the S3 time (lap end) was set.
        •	SpeedI1 — Speed trap in sector 1 (km/h).
        •	SpeedI2 — Speed trap in sector 2 (km/h).
        •	SpeedFL — Speed at the finish line (km/h).
        •	SpeedST — “Longest straight” speed trap (km/h, as provided by timing feed).
        •	IsPersonalBest — True if this is the driver’s official personal-best lap.
        •	Compound — Tyre compound name (SOFT/MEDIUM/HARD/INTERMEDIATE/WET/TEST_UNKNOWN/UNKNOWN).
        •	TyreLife — Laps driven on this tyre set (may include laps from other sessions if used).
        •	FreshTyre — True if tyre life was 0 at stint start (i.e., a new set).
        •	Team — Team name.
        •	LapStartTime — Session time at the start of the lap.
        •	LapStartDate — Absolute timestamp (UTC) at the start of the lap.
        •	TrackStatus — Concatenated track-status codes that occurred during this lap.
        •	Position — Driver’s classification at the end of the lap (NaN for practice/quali and crash laps).
        •	Deleted — True if lap time was deleted by the stewards (requires race-control messages loaded).
        •	DeletedReason — Reason for lap deletion (requires race-control messages).
        •	FastF1Generated — Indicates FastF1 added this lap heuristically (limited/interpolated info).
        •	IsAccurate — True if lap start/end timestamps passed FastF1’s accuracy checks.
        •	AirTemp — Air temperature in °C.
        •	Humidity — Relative humidity in %.
        •	Pressure — Air pressure in mbar.
        •	Rainfall — Boolean indicating whether rain is occurring.
        •	TrackTemp — Track surface temperature in °C.
        •	WindDirection — Wind direction in degrees (0–359).
        •	WindSpeed — Wind speed in m/s.
        •	Date — Absolute timestamp (UTC) for the sample.
        •	Speed — Car speed in km/h.
        •	RPM — Engine speed in revolutions per minute.
        •	nGear — Current gear number (exposed as “Gear” in docs but the data field is nGear).
        •	Throttle — Throttle pedal position 0–100%.
        •	Brake — Boolean brake flag.
        •	DRS — DRS state code from the stream (even values are “enabled” states).
        •	Source — Origin flag for the sample (“car” here; helps track interpolation/merging).
        •	Date — Absolute timestamp (UTC) for the position sample.
        •	Status — “OnTrack” or “OffTrack” state.
        •	X, Y, Z — Cartesian position coordinates (0.1 m increments since 2020).
        •	DriverAhead — Racing number of the car immediately ahead over time.
        •	DistanceToDriverAhead — Gap in meters to the car immediately ahead.

    Params:
    * lap_number: the lap index (0-based) or the actual lap number as per
                  the LapNumber column.
    * driver: the driver code (e.g., 'VER')
    * channels: list of telemetry channels to include (default: ['LapNumber'])

    Example: fastf1_telemetry(5, driver='VER', channels=['Speed', 'RPM', 'Throttle'])
    """
    ensure_fastf1_available()
    _set_cache_dir()

    # Assume session is hardcoded here
    tokens = "2023 Spanish GP Race".split()
    year = int(tokens[0]) if tokens and tokens[0].isdigit() else None
    gp = None
    session_name = None
    if "GP" in tokens:
        gp_idx = tokens.index("GP")
        gp = " ".join(tokens[1 : gp_idx + 1])
        if len(tokens) > gp_idx + 1:
            session_name = tokens[gp_idx + 1]

    if not (year and gp and session_name):
        return "Could not parse session from query."

    # load everything we might need
    session = fastf1.get_session(year, gp.replace(" GP", ""), session_name)
    session.load(laps=True, telemetry=True, weather=True, messages=True)

    # pick driver laps
    laps = session.laps.pick_driver(driver)
    if laps.empty:
        return f"No laps found for driver {driver} in session {tokens}"

    # Fetch telemetry for the specified lap and filter channels
    telemetry_data = laps.iloc[lap_idx].get_telemetry().add_distance()
    filtered_telemetry = telemetry_data[
        [c for c in telemetry_data.columns if c in channels]
    ]

    # Fetch weather data for the specified lap and filter channels
    weather_data = laps.iloc[lap_idx].get_weather_data()
    filtered_weather = weather_data[
        [c for c in session.weather_data.columns if c in channels]  # type: ignore
    ]

    # Fetch lap data and filter channels
    filtered_laps = laps.iloc[lap_idx][[c for c in laps.columns if c in channels]]

    # Return as tuple of DataFrames/Series
    return (filtered_telemetry, filtered_weather, filtered_laps)


# Export the tools list for other modules to import
available_tools = [fastf1_session_summary, fastf1_driver_laps, fastf1_telemetry]
