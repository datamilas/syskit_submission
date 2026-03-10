
from __future__ import annotations

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd



TELEMETRY_END_DATE = pd.Timestamp("2024-06-29")
SNAPSHOT_DATE = pd.Timestamp("2024-06-30")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_tables(db_path: str | Path | None = None) -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    try:
        tenants = pd.read_sql("SELECT * FROM tenants", conn)
        subscriptions = pd.read_sql(
            "SELECT * FROM subscriptions",
            conn,
            parse_dates=["contract_start_date", "renewal_date", "churn_date"],
        )
        users = pd.read_sql(
            "SELECT * FROM users",
            conn,
            parse_dates=["registered_at", "last_seen_at"],
        )
        events = pd.read_sql("SELECT * FROM events", conn, parse_dates=["event_time"])
        crm_companies = pd.read_sql(
            "SELECT * FROM crm_companies", conn, parse_dates=["created_at"]
        )
        crm_activities = pd.read_sql(
            "SELECT * FROM crm_activities", conn, parse_dates=["activity_date"]
        )
    finally:
        conn.close()

    events["event_date"] = events["event_time"].dt.normalize()
    return {
        "tenants": tenants,
        "subscriptions": subscriptions,
        "users": users,
        "events": events,
        "crm_companies": crm_companies,
        "crm_activities": crm_activities,
    }
















