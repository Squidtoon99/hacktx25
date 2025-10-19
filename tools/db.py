import os

import psycopg2
from langchain_core.tools import tool
from psycopg2.extras import RealDictCursor
from db import cur
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass


def get_table_names(conn):
    """Return a list of table names."""
    table_names = []
    # Use PostgreSQL information_schema to list user tables
    # conn is expected to be a cursor (psycopg2 cursor). Use execute() then fetchall().
    conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type='BASE TABLE';"
    )
    rows = conn.fetchall()
    for row in rows:
        # RealDictCursor returns dict-like rows
        if isinstance(row, dict):
            table_names.append(row.get("table_name"))
        else:
            table_names.append(row[0])
    return table_names

def get_column_names(conn, table_name):
    """Return a list of column names."""
    column_names = []
    # Query information_schema for column names
    conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name = %s ORDER BY ordinal_position;",
        (table_name,)
    )
    columns = conn.fetchall()
    for col in columns:
        if isinstance(col, dict):
            column_names.append(col.get("column_name"))
        else:
            column_names.append(col[0])
    return column_names

def get_metric_sensors(conn):
    """Return a list of sensors that are currently in the database"""
    sensors = []
    conn.execute(
        "SELECT DISTINCT(sensor_name) FROM metrics;"
    )
    
    for row in  conn.fetchall():
        sensors.append(row['sensor_name'])
    return list(set(sensors))

def get_database_info(conn):
    """Return a list of dicts containing the table name and columns for each table in the database."""
    table_dicts = []
    for table_name in get_table_names(conn):
        columns_names = get_column_names(conn, table_name)
        table_dicts.append({"table_name": table_name, "column_names": columns_names})
    return table_dicts

database_schema_dict = get_database_info(cur)
database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)

sensor_names = get_metric_sensors(cur)

# print("Inferred db schema: ", database_schema_string, "\nsensor names: ", sensor_names)

@tool(description=f"""
    Runs an SQL query in the Neon database.
    
    SCHEMA: {database_schema_string}
    
    metric.sensor_names: {sensor_names}
    There is A LOT of metrics data, provide limits of 500 datapoints or aggregations. Only query specific sensors you will need.
    Args:
        query: The SQL query to execute
    Returns:
        the result of the SQL query
    """)
def run_sql_query(query: str) -> str:
    # Use a new cursor per call to avoid closing module-level cursor/connection
    conn = None
    try:
        conn = psycopg2.connect(os.getenv("PG_CONNECTION_STRING"))
        with conn.cursor(cursor_factory=RealDictCursor) as local_cur:
            local_cur.execute(query)
            try:
                records = local_cur.fetchall()
                conn.commit()
                # Convert records to CSV
                csv_result = records_to_csv(records)
                return csv_result
            except psycopg2.ProgrammingError:
                # No results to fetch (e.g., INSERT/UPDATE)
                conn.commit()
                return "Query executed successfully"
    except Exception as e:
        if conn:
            conn.rollback()
        return f"Failed to execute SQL query: {str(e)}"
    finally:
        if conn:
            conn.close()
        
available_tools = [run_sql_query]


def records_to_csv(records: list) -> str:
    """Convert a list of records (RealDict rows or tuples) into a CSV string.

    Handles:
    - list of dicts (RealDictCursor)
    - list of tuples/lists
    - empty list -> returns an empty string
    """
    import csv
    import io

    if not records:
        return ""

    output = io.StringIO()

    # Determine headers and rows
    first = records[0]
    if isinstance(first, dict):
        headers = list(first.keys())
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        for r in records:
            # r might already be a dict-like
            writer.writerow({k: r.get(k) for k in headers})
    else:
        # assume sequence of sequences
        # create numeric headers like col0, col1, ...
        num_cols = len(first)
        headers = [f"col{i}" for i in range(num_cols)]
        writer = csv.writer(output)
        writer.writerow(headers)
        for r in records:
            writer.writerow(r)

    return output.getvalue()[:10000].rsplit('\n', 1)[0]