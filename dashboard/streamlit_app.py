import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import re
from pathlib import Path


APP_TITLE = "Syskit Customer Base Dashboard"
APP_SUBTITLE = "Snapshot: 2024-06-29"
SNAPSHOT_DATE = pd.Timestamp("2024-06-29")
VALUE_EVENTS = [
    "sensitivity_label_applied",
    "license_recommendation_applied",
    "pp_sync_completed",
    "risky_workspace_resolved",
    "policy_created",
]

HEALTH_SEGMENT_ORDER = ["critical", "watchlist", "stable", "strong"]
HEALTH_SEGMENT_COLORS = {
    "critical": "#dc2626",
    "watchlist": "#f59e0b",
    "stable": "#2563eb",
    "strong": "#16a34a",
    "unknown": "#9ca3af",
}
FLAG_TOOLTIPS = {
    "at_risk_flag": "Customers that show weak health (critical/watchlist) and are close to renewal, so they need immediate retention attention.",
    "quiet_account_flag": "Customers with very low recent product activity, declining usage and no recent customer-success contact.",
    "expansion_candidate_flag": "Customers on starter/business plans showing rising usage and strong account health, with good potential for upsell.",
}
ACTION_FLAG_OPTIONS = {
    "At Risk": "at_risk_flag",
    "Quiet Accounts": "quiet_account_flag",
    "Expansion Candidates": "expansion_candidate_flag",
}

@st.cache_data(show_spinner=False)
def load_data():
    base_dir = Path(__file__).resolve().parents[1]
    results_dir = base_dir / "results"
    events_path = base_dir / "data" / "events.csv"
    tenant_scores = pd.read_csv(results_dir / "tenant_scores_snapshot_2024_06_30.csv")
    events = pd.read_csv(events_path, parse_dates=["event_time"])


    date_cols = [
        "contract_start_date",
        "renewal_date",
        "next_renewal_date",
        "churn_date",
        "last_crm_touch_date",
        "created_at",
        "event_time"
    ]
    for col in date_cols:
        if col in tenant_scores.columns:
            tenant_scores[col] = pd.to_datetime(tenant_scores[col], errors="coerce")
    events["event_date"] = pd.to_datetime(events["event_time"]).dt.normalize()

    return tenant_scores, events


def format_kpi(value, fmt=",.0f"):
    if pd.isna(value):
        return "—"
    return format(value, fmt)


def renewal_bucket_order(values):
    def bucket_key(label):
        text = str(label).strip().lower()
        match = re.search(r"\d+", text)
        if "+" in text and match:
            return (int(match.group()), 1)
        if match:
            return (int(match.group()), 0)
        return (10**9, 0)

    return sorted(values, key=bucket_key)

def usage_trend_12w(events: pd.DataFrame, snapshot_date: pd.Timestamp = SNAPSHOT_DATE, value_events: list = VALUE_EVENTS) -> pd.DataFrame:
    snapshot_date = pd.Timestamp(snapshot_date).normalize()
    current_week_start = snapshot_date - pd.Timedelta(days=snapshot_date.weekday())
    last_week_start = (
        current_week_start
        if snapshot_date.weekday() == 6
        else current_week_start - pd.Timedelta(weeks=1)
    )
    first_week_start = last_week_start - pd.Timedelta(weeks=11)
    week_starts = pd.date_range(first_week_start, periods=12, freq="W-MON")

    tmp = events.loc[
        (events["event_date"] >= first_week_start)
        & (events["event_date"] < last_week_start + pd.Timedelta(weeks=1))
    ].copy()
    if "health_segment" not in tmp.columns:
        tmp["health_segment"] = "unknown"
    tmp["health_segment"] = tmp["health_segment"].fillna("unknown").str.lower()
    tmp["week"] = tmp["event_date"].dt.to_period("W-SUN").dt.start_time
    tmp["value_event_count"] = np.where(tmp["event_name"].isin(value_events), tmp["event_count"], 0)
    weekly = (
        tmp.groupby(["week", "tenant_id", "health_segment"])
        .agg(
            weekly_events=("event_count", "sum"),
            weekly_value_events=("value_event_count", "sum"),
            weekly_active_users=("user_id", "nunique"),
        )
        .reset_index()
    )
    roll = (
        weekly.groupby(["week", "health_segment"])
        .agg(
            avg_events_per_tenant=("weekly_events", "mean"),
            avg_value_events_per_tenant=("weekly_value_events", "mean"),
        )
        .reset_index()
        .sort_values(["week", "health_segment"])
    )
    health_segments = sorted(roll["health_segment"].dropna().unique().tolist())
    if not health_segments:
        health_segments = ["unknown"]

    # Keep the chart anchored to 12 complete weeks even if some segment-week pairs are missing.
    complete_index = pd.MultiIndex.from_product(
        [week_starts, health_segments],
        names=["week", "health_segment"],
    )
    roll = (
        roll.set_index(["week", "health_segment"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )
    return roll

tenant_scores, events = load_data()

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

st.sidebar.header("Filters")

def filter_options(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())

health_filter = st.sidebar.multiselect(
    "Health Segment",
    options=filter_options(tenant_scores, "health_segment"),
)
renewal_filter = st.sidebar.multiselect(
    "Renewal Bucket",
    options=filter_options(tenant_scores, "renewal_bucket"),
)
action_flag_filter = st.sidebar.multiselect(
    "Action Flags",
    options=list(ACTION_FLAG_OPTIONS.keys()),
)
plan_filter = st.sidebar.multiselect(
    "Plan",
    options=filter_options(tenant_scores, "plan"),
)
acquisition_filter = st.sidebar.multiselect(
    "Acquisition Source",
    options=filter_options(tenant_scores, "acquisition_source"),
)
region_filter = st.sidebar.multiselect(
    "Region",
    options=filter_options(tenant_scores, "region"),
)
industry_filter = st.sidebar.multiselect(
    "Industry",
    options=filter_options(tenant_scores, "industry"),
)
csm_filter = st.sidebar.multiselect(
    "CSM Assigned",
    options=filter_options(tenant_scores, "csm_assigned"),
)

filtered = tenant_scores.copy()
if health_filter:
    filtered = filtered[filtered["health_segment"].astype(str).isin(health_filter)]
if plan_filter:
    filtered = filtered[filtered["plan"].astype(str).isin(plan_filter)]
if acquisition_filter:
    filtered = filtered[filtered["acquisition_source"].astype(str).isin(acquisition_filter)]
if region_filter:
    filtered = filtered[filtered["region"].astype(str).isin(region_filter)]
if industry_filter:
    filtered = filtered[filtered["industry"].astype(str).isin(industry_filter)]
if renewal_filter:
    filtered = filtered[filtered["renewal_bucket"].astype(str).isin(renewal_filter)]
if csm_filter:
    filtered = filtered[filtered["csm_assigned"].astype(str).isin(csm_filter)]
if action_flag_filter:
    selected_flag_columns = [ACTION_FLAG_OPTIONS[label] for label in action_flag_filter]
    selected_flags_mask = filtered[selected_flag_columns].fillna(0).eq(1).any(axis=1)
    filtered = filtered[selected_flags_mask]

overview_tab, cs_tab, sales_tab = st.tabs(
    ["Overview", "Customer Success", "Sales & Expansion"]
)

with overview_tab:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Tenants", format_kpi(filtered["tenant_id"].nunique()))
    col2.metric("Avg Health", format_kpi(filtered["health_score_0_100"].mean(), ".1f"))
    col3.metric("At Risk", format_kpi(filtered["at_risk_flag"].sum()), help=FLAG_TOOLTIPS["at_risk_flag"])
    col4.metric("Quiet Accounts", format_kpi(filtered["quiet_account_flag"].sum()), help=FLAG_TOOLTIPS["quiet_account_flag"])
    col5.metric(
        "Expansion Candidates",
        format_kpi(filtered["expansion_candidate_flag"].sum()),
        help=FLAG_TOOLTIPS["expansion_candidate_flag"],
    )

    st.divider()

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Health Distribution")
        health_counts = (
            filtered["health_segment"]
            .fillna("unknown")
            .str.lower()
            .value_counts()
            .rename_axis("health_segment")
            .reset_index(name="tenants")
        )
        health_counts["pct_total_base"] = np.where(
            health_counts["tenants"].sum() > 0,
            health_counts["tenants"] / health_counts["tenants"].sum(),
            0,
        )
        health_order = [x for x in HEALTH_SEGMENT_ORDER if x in set(health_counts["health_segment"])]
        if "unknown" in set(health_counts["health_segment"]):
            health_order.append("unknown")
        health_color_range = [HEALTH_SEGMENT_COLORS.get(s, "#9ca3af") for s in health_order]
        health_chart = (
            alt.Chart(health_counts)
            .mark_bar()
            .encode(
                x=alt.X("health_segment:N", title="Health Segment", sort=health_order),
                y=alt.Y("tenants:Q", title="Tenants"),
                color=alt.Color(
                    "health_segment:N",
                    legend=None,
                    sort=health_order,
                    scale=alt.Scale(domain=health_order, range=health_color_range),
                ),
                tooltip=[
                    "health_segment",
                    alt.Tooltip("tenants:Q", title="Tenants"),
                    alt.Tooltip("pct_total_base:Q", title="% of Total Base", format=".1%"),
                ],
            )
            .properties(height=300)
        )
        health_labels = (
            alt.Chart(health_counts)
            .mark_text(dy=-8, fontSize=11, color="#111827")
            .encode(
                x=alt.X("health_segment:N", sort=health_order),
                y=alt.Y("tenants:Q"),
                text=alt.Text("pct_total_base:Q", format=".1%"),
            )
        )
        st.altair_chart(health_chart + health_labels, use_container_width=True)

    with right:
        st.subheader("Renewal Window Distribution")
        renewal_counts = (
            filtered["renewal_bucket"]
            .fillna("unknown")
            .value_counts()
            .rename_axis("renewal_bucket")
            .reset_index(name="tenants")
        )
        renewal_counts["pct_total_base"] = np.where(
            renewal_counts["tenants"].sum() > 0,
            renewal_counts["tenants"] / renewal_counts["tenants"].sum(),
            0,
        )
        renewal_order = renewal_bucket_order(renewal_counts["renewal_bucket"].tolist())
        renewal_chart = (
            alt.Chart(renewal_counts)
            .mark_bar()
            .encode(
                x=alt.X("renewal_bucket:N", title="Renewal Window", sort=renewal_order),
                y=alt.Y("tenants:Q", title="Tenants"),
                color=alt.Color(
                    "renewal_bucket:N",
                    legend=None,
                    sort=renewal_order,
                    scale=alt.Scale(scheme="purples"),
                ),
                tooltip=[
                    "renewal_bucket",
                    alt.Tooltip("tenants:Q", title="Tenants"),
                    alt.Tooltip("pct_total_base:Q", title="% of Total Base", format=".1%"),
                ],
            )
            .properties(height=300)
        )
        renewal_labels = (
            alt.Chart(renewal_counts)
            .mark_text(dy=-8, fontSize=11, color="#111827")
            .encode(
                x=alt.X("renewal_bucket:N", sort=renewal_order),
                y=alt.Y("tenants:Q"),
                text=alt.Text("pct_total_base:Q", format=".1%"),
            )
        )
        st.altair_chart(renewal_chart + renewal_labels, use_container_width=True)

    st.divider()

    st.subheader("Usage Trend Last 12 Weeks")
    usage_metric = st.selectbox(
        "Usage Metric",
        options=["avg_events_per_tenant", "avg_value_events_per_tenant"],
        index=0,
        format_func=lambda x: "Avg Events per Tenant" if x == "avg_events_per_tenant" else "Avg Value Events per Tenant",
    )
    usage_title = "Avg Events per Tenant" if usage_metric == "avg_events_per_tenant" else "Avg Value Events per Tenant"

    usage_trend_filtered = usage_trend_12w(
        events.merge(filtered[["tenant_id", "health_segment"]], on="tenant_id", how="inner")
    )
    usage_health_order = [x for x in HEALTH_SEGMENT_ORDER if x in set(usage_trend_filtered["health_segment"])]
    if "unknown" in set(usage_trend_filtered["health_segment"]):
        usage_health_order.append("unknown")
    usage_color_range = [HEALTH_SEGMENT_COLORS.get(s, "#9ca3af") for s in usage_health_order]

    usage_chart = (
        alt.Chart(usage_trend_filtered)
        .mark_line(point=True)
        .encode(
            x=alt.X("week:T", title="Week"),
            y=alt.Y(f"{usage_metric}:Q", title=usage_title),
            color=alt.Color(
                "health_segment:N",
                title="Health Segment",
                sort=usage_health_order,
                scale=alt.Scale(domain=usage_health_order, range=usage_color_range),
            ),
            tooltip=[
                "week:T",
                "health_segment:N",
                alt.Tooltip("avg_events_per_tenant:Q", title="Avg Events per Tenant", format=".2f"),
                alt.Tooltip("avg_value_events_per_tenant:Q", title="Avg Value Events per Tenant", format=".2f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(usage_chart, use_container_width=True)

with cs_tab:
    st.subheader("At-Risk Customers")
    st.caption(f"{FLAG_TOOLTIPS['at_risk_flag']}")
    at_risk = filtered[filtered["at_risk_flag"] == 1].copy()
    at_risk = at_risk.sort_values(["days_to_next_renewal", "health_score_0_100"])
    at_risk_display = at_risk[
        [
            "company_name",
            "plan",
            "region",
            "arr",
            "health_score_0_100",
            "health_segment",
            "days_to_next_renewal",
            "csm_assigned",
        ]
    ]
    st.dataframe(
        at_risk_display,
        use_container_width=True,
        height=300,
        hide_index=True,
    )
    st.divider()

    st.subheader("Quiet Customers")
    st.caption(f"{FLAG_TOOLTIPS['quiet_account_flag']}")
    quiet_accounts = filtered[filtered["quiet_account_flag"] == 1].copy()
    quiet_accounts = quiet_accounts.sort_values(["days_to_next_renewal", "health_score_0_100"])
    quiet_accounts_display = quiet_accounts[
        [
            "company_name",
            "plan",
            "region",
            "arr",
            "health_score_0_100",
            "health_segment",
            "days_to_next_renewal",
            "csm_assigned",
        ]
    ]
    st.dataframe(
        quiet_accounts_display,
        use_container_width=True,
        height=300,
        hide_index=True,
    )

with sales_tab:
    st.subheader("Expansion Candidate Tenants")
    st.caption(f"{FLAG_TOOLTIPS['expansion_candidate_flag']}")
    expansion_candidates = filtered[filtered["expansion_candidate_flag"] == 1].copy()
    expansion_candidates = expansion_candidates.sort_values(
        ["health_score_0_100", "arr"], ascending=[False, False]
    )
    expansion_candidates_display = expansion_candidates[
        [
            "company_name",
            "plan",
            "region",
            "industry",
            "arr",
            "health_score_0_100",
            "health_segment",
            "days_to_next_renewal",
            "csm_assigned",
        ]
    ]
    st.dataframe(
        expansion_candidates_display,
        use_container_width=True,
        height=420,
        hide_index=True,
    )
