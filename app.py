
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.dashboard-title {
    font-size: 2.2rem;
    font-weight: 750;
    margin-bottom: 0.25rem;
}
.dashboard-subtitle {
    color: #718096;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_segments.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(
        DATA_PATH,
        dtype={"CustomerID": str}
    )

    required = [
        "CustomerID", "Recency", "Frequency",
        "Monetary", "Cluster", "Segment"
    ]

    missing = [
        c for c in required
        if c not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return data

try:
    df = load_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Unable to load customer data: {exc}")
    st.stop()

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="dashboard-title">'
    'Customer Segmentation Analytics'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'RFM Analysis | K-Means Clustering | '
    'Oasis Infobyte — Level 1 Task 2'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("Dashboard Controls")

all_segments = sorted(
    df["Segment"].dropna().unique().tolist()
)

selected_segments = st.sidebar.multiselect(
    "Customer Segments",
    options=all_segments,
    default=all_segments
)

all_clusters = sorted(
    df["Cluster"].dropna().unique().tolist()
)

selected_clusters = st.sidebar.multiselect(
    "K-Means Clusters",
    options=all_clusters,
    default=all_clusters
)

minimum_recency = int(df["Recency"].min())
maximum_recency = int(df["Recency"].max())

if minimum_recency < maximum_recency:
    recency_range = st.sidebar.slider(
        "Recency (Days)",
        min_value=minimum_recency,
        max_value=maximum_recency,
        value=(minimum_recency, maximum_recency)
    )
else:
    recency_range = (
        minimum_recency,
        maximum_recency
    )

filtered = df[
    df["Segment"].isin(selected_segments)
    & df["Cluster"].isin(selected_clusters)
    & df["Recency"].between(
        recency_range[0],
        recency_range[1]
    )
].copy()

st.sidebar.divider()

st.sidebar.caption(
    "Data: UCI Online Retail | "
    "Historical customer transactions"
)

if filtered.empty:
    st.warning(
        "No customers match the selected filters. "
        "Please change your filter selection."
    )
    st.stop()

# ============================================================
# KPI CARDS
# ============================================================

total_customers = filtered["CustomerID"].nunique()
total_revenue = filtered["Monetary"].sum()
avg_recency = filtered["Recency"].mean()
avg_frequency = filtered["Frequency"].mean()
avg_monetary = filtered["Monetary"].mean()

st.subheader("Executive Overview")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

k2.metric(
    "Observed Revenue",
    f"£{total_revenue:,.0f}"
)

k3.metric(
    "Avg Recency",
    f"{avg_recency:.1f} days"
)

k4.metric(
    "Avg Frequency",
    f"{avg_frequency:.2f}"
)

k5.metric(
    "Avg Monetary",
    f"£{avg_monetary:,.0f}"
)

st.divider()

# ============================================================
# SEGMENT DISTRIBUTION
# ============================================================

st.subheader("Customer Segment Overview")

left, right = st.columns(2)

segment_count = (
    filtered.groupby("Segment")
    .size()
    .reset_index(name="Customers")
    .sort_values("Customers", ascending=False)
)

with left:
    fig_count = px.bar(
        segment_count,
        x="Segment",
        y="Customers",
        color="Segment",
        text="Customers",
        title="Customers by Segment"
    )

    fig_count.update_layout(
        showlegend=False,
        xaxis_title="Customer Segment",
        yaxis_title="Customers"
    )

    st.plotly_chart(
        fig_count,
        use_container_width=True
    )

with right:
    fig_pie = px.pie(
        segment_count,
        names="Segment",
        values="Customers",
        hole=0.55,
        title="Customer Segment Share"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

# ============================================================
# REVENUE ANALYSIS
# ============================================================

st.subheader("Revenue Contribution")

revenue_by_segment = (
    filtered.groupby(
        "Segment", as_index=False
    )["Monetary"]
    .sum()
    .sort_values(
        "Monetary",
        ascending=False
    )
)

fig_revenue = px.bar(
    revenue_by_segment,
    x="Segment",
    y="Monetary",
    color="Segment",
    text_auto=".3s",
    title="Observed Revenue by Segment"
)

fig_revenue.update_layout(
    showlegend=False,
    yaxis_title="Revenue (£)"
)

st.plotly_chart(
    fig_revenue,
    use_container_width=True
)

st.divider()

# ============================================================
# RFM SCATTER PLOTS
# ============================================================

st.subheader("Customer Behaviour & Clustering")

c1, c2 = st.columns(2)

with c1:
    fig_rf = px.scatter(
        filtered,
        x="Recency",
        y="Frequency",
        color="Segment",
        hover_data=["CustomerID", "Monetary"],
        title="Recency vs Frequency",
        opacity=0.65,
        log_y=True
    )

    st.plotly_chart(
        fig_rf,
        use_container_width=True
    )

with c2:
    fig_fm = px.scatter(
        filtered[
            (filtered["Frequency"] > 0)
            & (filtered["Monetary"] > 0)
        ],
        x="Frequency",
        y="Monetary",
        color="Segment",
        hover_data=["CustomerID", "Recency"],
        title="Frequency vs Monetary",
        log_x=True,
        log_y=True,
        opacity=0.65
    )

    st.plotly_chart(
        fig_fm,
        use_container_width=True
    )

fig_rm = px.scatter(
    filtered[filtered["Monetary"] > 0],
    x="Recency",
    y="Monetary",
    color="Segment",
    hover_data=["CustomerID", "Frequency"],
    title="Recency vs Monetary",
    log_y=True,
    opacity=0.65
)

st.plotly_chart(
    fig_rm,
    use_container_width=True
)

# ============================================================
# SEGMENT PROFILE
# ============================================================

st.divider()
st.subheader("Segment RFM Profile")

profile = (
    filtered.groupby("Segment")
    .agg(
        Customers=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum")
    )
    .round(2)
    .reset_index()
)

st.dataframe(
    profile,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# AUTOMATIC BUSINESS INSIGHTS
# ============================================================

st.subheader("Dynamic Business Insights")

largest_segment = (
    segment_count.iloc[0]["Segment"]
)

highest_revenue_segment = (
    revenue_by_segment.iloc[0]["Segment"]
)

highest_revenue_value = (
    revenue_by_segment.iloc[0]["Monetary"]
)

revenue_share = (
    highest_revenue_value
    / total_revenue * 100
)

st.markdown(
    f"""
    - **Largest Customer Segment:** {largest_segment}
    - **Highest Revenue Segment:** {highest_revenue_segment}
    - **Revenue Share of Leading Segment:** {revenue_share:.2f}%
    - **Customers in Current Selection:** {total_customers:,}
    - **Average Purchase Frequency:** {avg_frequency:.2f}
    """
)

# ============================================================
# MARKETING RECOMMENDATIONS
# ============================================================

st.divider()
st.subheader("Marketing Recommendations")

recommendations = {
    "Champions": (
        "Prioritise retention through VIP benefits, "
        "exclusive offers, and premium recommendations."
    ),
    "Loyal / Engaged": (
        "Use loyalty rewards, cross-selling, "
        "personalised offers, and referral incentives."
    ),
    "Developing Customers": (
        "Encourage repeat purchases through targeted "
        "promotions, bundles, and product recommendations."
    ),
    "Low-Engagement / At-Risk": (
        "Use reactivation campaigns, feedback collection, "
        "and carefully targeted win-back incentives."
    )
}

for segment in sorted(
    filtered["Segment"].unique()
):
    with st.expander(
        f"Marketing Strategy: {segment}",
        expanded=False
    ):
        st.write(
            recommendations.get(
                segment,
                "Review this segment's RFM profile "
                "before selecting a marketing strategy."
            )
        )

# ============================================================
# CUSTOMER EXPLORER
# ============================================================

st.divider()
st.subheader("Customer Data Explorer")

search_customer = st.text_input(
    "Search by Customer ID",
    placeholder="Enter Customer ID"
)

display_df = filtered.copy()

if search_customer:
    display_df = display_df[
        display_df["CustomerID"]
        .str.contains(
            search_customer,
            case=False,
            regex=False,
            na=False
        )
    ]

display_df = display_df.sort_values(
    "Monetary",
    ascending=False
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"Showing {len(display_df):,} customer records."
)

# ============================================================
# DOWNLOAD
# ============================================================

st.download_button(
    label="Download Filtered Customer Data",
    data=display_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_customer_segments.csv",
    mime="text/csv"
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Developed by Shubham Maity | "
    "Oasis Infobyte Data Analytics Internship | "
    "Level 1 – Task 2 | "
    "RFM + K-Means Clustering"
)
