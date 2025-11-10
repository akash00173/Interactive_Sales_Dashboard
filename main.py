# app.py
import os
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output, State
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc

# -------------------------
# Sample data generation (CSV + SQLite)
# -------------------------
DATA_DIR = "data_example"
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "products_sales.csv")
SQLITE_PATH = os.path.join(DATA_DIR, "customers.sqlite")

def create_sample_csv(csv_path, rows=1000):
    rng = np.random.default_rng(42)
    start = datetime.now() - timedelta(days=365)
    dates = [start + timedelta(days=int(x)) for x in rng.integers(0, 365, size=rows)]
    categories = ["Electronics", "Office", "Furniture", "Clothing"]
    regions = ["North", "South", "East", "West"]
    products = {
        "Electronics": ["Phone", "Laptop", "Headphones"],
        "Office": ["Pen", "Paper", "Stapler"],
        "Furniture": ["Chair", "Desk", "Shelf"],
        "Clothing": ["Shirt", "Jacket", "Hat"],
    }

    data = []
    for d in dates:
        cat = rng.choice(categories)
        prod = rng.choice(products[cat])
        qty = int(rng.integers(1, 10))
        price = round(float(rng.uniform(5, 1500)), 2)
        region = rng.choice(regions)
        order_id = f"ORD{rng.integers(100000,999999)}"
        data.append({
            "order_id": order_id,
            "date": d.date().isoformat(),
            "category": cat,
            "product": prod,
            "quantity": qty,
            "unit_price": price,
            "region": region,
            "sales": round(qty * price,2)
        })

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return df

def create_sample_sqlite(sqlite_path, rows=400):
    rng = np.random.default_rng(123)
    segments = ["Consumer", "Corporate", "Home Office"]
    cities = ["Mumbai", "Delhi", "Bengaluru", "Kolkata", "Hyderabad"]
    conn = sqlite3.connect(sqlite_path)
    df_customers = pd.DataFrame({
        "customer_id": [f"CUST{1000+i}" for i in range(rows)],
        "customer_name": [f"Customer_{i}" for i in range(rows)],
        "city": rng.choice(cities, size=rows),
        "segment": rng.choice(segments, size=rows),
    })
    df_customers.to_sql("customers", conn, if_exists="replace", index=False)
    conn.close()
    return df_customers

# Create sample data if not found
if not os.path.exists(CSV_PATH):
    print("Creating sample CSV...")
    create_sample_csv(CSV_PATH, rows=1200)
if not os.path.exists(SQLITE_PATH):
    print("Creating sample SQLite DB...")
    create_sample_sqlite(SQLITE_PATH, rows=600)

# -------------------------
# Load data
# -------------------------
sales_df = pd.read_csv(CSV_PATH, parse_dates=["date"])
engine = create_engine(f"sqlite:///{SQLITE_PATH}")
customers_df = pd.read_sql("customers", engine)

rng = np.random.default_rng(999)
sales_df["customer_id"] = rng.choice(customers_df["customer_id"].values, size=len(sales_df))
merged = sales_df.merge(customers_df, on="customer_id", how="left")

merged["month"] = merged["date"].dt.to_period("M").dt.to_timestamp()
merged["sales"] = merged["sales"].astype(float)

# -------------------------
# Dash App Setup
# -------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# -------------------------
# Layout
# -------------------------
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Interactive Sales Dashboard"), md=8),
        dbc.Col(html.Div([
            html.Label("Data sources: CSV (sales) + SQLite (customers)"),
            html.Br(),
            html.Small(f"Rows (sales): {len(sales_df)} — Customers: {len(customers_df)}")
        ], style={"textAlign":"right"}), md=4)
    ], align="center", className="my-2"),

    dbc.Row([
        dbc.Col([
            html.Label("Date range"),
            dcc.DatePickerRange(
                id="date-range",
                start_date=merged["date"].min().date(),
                end_date=merged["date"].max().date(),
                display_format="YYYY-MM-DD"
            )
        ], md=4),
        dbc.Col([
            html.Label("Category"),
            dcc.Dropdown(
                id="category-filter",
                options=[{"label": c, "value": c} for c in sorted(merged["category"].unique())],
                multi=True,
                placeholder="Select categories (or leave all)"
            )
        ], md=4),
        dbc.Col([
            html.Label("Region"),
            dcc.Dropdown(
                id="region-filter",
                options=[{"label": r, "value": r} for r in sorted(merged["region"].unique())],
                multi=True,
                placeholder="Select regions (or leave all)"
            )
        ], md=4)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="time-series"), md=8),
        dbc.Col(dcc.Graph(id="by-category"), md=4),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="by-segment"), md=6),
        dbc.Col(html.Div([
            html.H6("Top Products (table)"),
            DataTable(
                id="top-products-table",
                page_size=8,
                style_table={"overflowX":"auto"},
                sort_action="native",
                columns=[
                    {"name":"product","id":"product"},
                    {"name":"category","id":"category"},
                    {"name":"sales","id":"sales", "type":"numeric"},
                    {"name":"quantity","id":"quantity", "type":"numeric"},
                ]
            ),
            html.Br(),
            dbc.Button("Download filtered data (CSV)", id="download-btn"),
            dcc.Download(id="download-data")
        ]), md=6)
    ], className="mt-3"),

    html.Hr(),
    html.H4("Advanced Analytics", className="mt-4 mb-2"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="sales-map"), md=6),
        dbc.Col(dcc.Graph(id="sales-funnel"), md=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="cohort-analysis"), md=12)
    ]),
    html.Hr(),
    html.Div(id="debug", style={"fontSize":"12px", "color":"#666"})
], fluid=True)

# -------------------------
# Helpers
# -------------------------
def filter_df(df, start_date, end_date, categories, regions):
    d = df.copy()
    if start_date:
        d = d[d["date"] >= pd.to_datetime(start_date)]
    if end_date:
        d = d[d["date"] <= pd.to_datetime(end_date)]
    if categories and len(categories) > 0:
        d = d[d["category"].isin(categories)]
    if regions and len(regions) > 0:
        d = d[d["region"].isin(regions)]
    return d

# -------------------------
# Main Callback
# -------------------------
@app.callback(
    Output("time-series", "figure"),
    Output("by-category", "figure"),
    Output("by-segment", "figure"),
    Output("top-products-table", "data"),
    Output("sales-map", "figure"),
    Output("sales-funnel", "figure"),
    Output("cohort-analysis", "figure"),
    Output("debug", "children"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("category-filter", "value"),
    Input("region-filter", "value"),
)
def update_charts(start_date, end_date, category_values, region_values):
    try:
        min_date, max_date = merged["date"].min(), merged["date"].max()
        if not start_date or not end_date:
            start_date, end_date = min_date, max_date
        if pd.to_datetime(start_date) > pd.to_datetime(end_date):
            start_date, end_date = end_date, start_date

        # --- Filter ---
        d = filter_df(merged, start_date, end_date, category_values, region_values)
        debug_text = f"Filtered rows: {len(d)} | {start_date} → {end_date}"

        if d.empty:
            blank = px.scatter(title="No data available")
            return blank, blank, blank, [], blank, blank, blank, debug_text

        # --- Time series ---
        ts = d.groupby("date", as_index=False)["sales"].sum().sort_values("date")
        fig_ts = px.line(ts, x="date", y="sales", title="Sales Over Time", markers=True)

        # --- Category bar ---
        cat = d.groupby("category", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
        fig_cat = px.bar(cat, x="category", y="sales", title="Sales by Category")

        # --- Segment pie ---
        seg = d.groupby("segment", as_index=False)["sales"].sum()
        fig_seg = px.pie(seg, values="sales", names="segment", hole=0.4, title="Sales by Segment")

        # --- Top products ---
        top = (
            d.groupby(["product", "category"], as_index=False)
            .agg({"sales": "sum", "quantity": "sum"})
            .sort_values("sales", ascending=False)
            .head(15)
        )
        table_data = top.to_dict("records")

        # --- Map ---
        region_sales = d.groupby("region", as_index=False)["sales"].sum()
        fig_map = px.bar(region_sales, x="region", y="sales", title="Sales by Region")  # simpler & safe

        # --- Funnel ---
        total_orders = len(d)
        unique_cust = d["customer_id"].nunique()
        repeat_cust = d["customer_id"].value_counts().gt(1).sum()
        high_value = d.groupby("customer_id")["sales"].sum().gt(10000).sum()

        funnel = pd.DataFrame({
            "Stage": ["Total Orders", "Unique Customers", "Repeat Customers", "High-Value Customers"],
            "Count": [total_orders, unique_cust, repeat_cust, high_value],
        })
        fig_funnel = px.funnel(funnel, x="Count", y="Stage", title="Customer Funnel")

        # --- Cohort ---
        cohort_df = d.copy()
        cohort_df["order_month"] = cohort_df["date"].dt.to_period("M")
        first = cohort_df.groupby("customer_id")["order_month"].min().reset_index()
        cohort_df = cohort_df.merge(first, on="customer_id", suffixes=("", "_cohort"))
        cohort_df["months_since"] = (
            (cohort_df["order_month"].dt.year - cohort_df["order_month_cohort"].dt.year) * 12
            + (cohort_df["order_month"].dt.month - cohort_df["order_month_cohort"].dt.month)
        )
        pivot = (
            cohort_df.groupby(["order_month_cohort", "months_since"])["customer_id"]
            .nunique()
            .reset_index()
        )
        if pivot.empty:
            fig_cohort = go.Figure()
            fig_cohort.update_layout(title="Cohort Retention (No Data)")
        else:
            pivot = pivot.pivot(
                index="order_month_cohort", columns="months_since", values="customer_id"
            ).fillna(0)
            pivot = pivot.divide(pivot.iloc[:, 0], axis=0)
            fig_cohort = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index.astype(str),
                    colorscale="Blues"
                )
            )
            fig_cohort.update_layout(title="Cohort Retention Heatmap")

        return fig_ts, fig_cat, fig_seg, table_data, fig_map, fig_funnel, fig_cohort, debug_text

    except Exception as e:
        print("❌ Callback error:", e)
        empty = px.scatter(title=f"Error: {e}")
        return empty, empty, empty, [], empty, empty, empty, str(e)

# Download Data

@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("category-filter", "value"),
    State("region-filter", "value"),
    prevent_initial_call=True
)
def download_filtered(n_clicks, start_date, end_date, categories, regions):
    d = filter_df(merged, start_date, end_date, categories, regions)
    return dcc.send_data_frame(d.to_csv, f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# Run

if __name__ == "__main__":
    app.run(debug=True)
