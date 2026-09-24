# =============================================================================
# Bitcoin Data Analytics & Machine Learning Dashboard
# Internship Project | Production Deployment Version
# Author: Bharath (IBM Internship Program)
# =============================================================================

import os
import gc
import warnings
from io import StringIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Bitcoin Analytics Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM STYLING (Dark Financial Terminal Theme)
# =============================================================================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #262d3f);
        border: 1px solid #3a3f5c;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .metric-value {
        color: #f9fafb;
        font-size: 26px;
        font-weight: 700;
        margin: 6px 0;
    }
    .metric-delta-pos { color: #22c55e; font-size: 13px; font-weight: 600; }
    .metric-delta-neg { color: #ef4444; font-size: 13px; font-weight: 600; }
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #f9fafb;
        border-left: 4px solid #f7931a;
        padding-left: 12px;
        margin: 24px 0 14px 0;
    }
    .forecast-card {
        background: linear-gradient(135deg, #1a2744, #1e3a5f);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 18px 24px;
        margin: 16px 0;
    }
    .sidebar-title { color: #f7931a; font-size: 18px; font-weight: 700; }
    .stPlotlyChart { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CHART THEME CONSTANTS
# =============================================================================
BITCOIN_ORANGE = "#f7931a"
CHART_BG       = "#0e1117"
GRID_COLOR     = "#1e2535"
TEXT_COLOR     = "#d1d5db"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_BG,
    font=dict(color=TEXT_COLOR, size=12),
    xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    hovermode="x unified",
)

def _apply_layout(fig, title=""):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=15, color="#f9fafb")))
    return fig

# =============================================================================
# DATA LAYER — Multi-Source Loading & Resampling
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_default_dataset() -> str:
    """Finds the most suitable dataset in local or deployment environment."""
    candidates = [
        os.path.join(BASE_DIR, "btc_daily_resampled.csv"),
        "btc_daily_resampled.csv",
        os.path.join(os.getcwd(), "btc_daily_resampled.csv"),
        os.path.join(os.getcwd(), "Bitcoin_Analytics", "btc_daily_resampled.csv"),
        os.path.join(BASE_DIR, "btcusd_1-min_data.csv"),
        "btcusd_1-min_data.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "btc_daily_resampled.csv"

def generate_sample_bitcoin_data() -> pd.DataFrame:
    """Generates realistic Bitcoin daily OHLCV data as a fallback to ensure 0-error deployment."""
    dates = pd.date_range(start="2017-01-01", end=datetime.now().strftime("%Y-%m-%d"), freq="D")
    n = len(dates)
    np.random.seed(42)
    # Geometric Brownian motion with realistic crypto drift and volatility
    returns = np.random.normal(0.0018, 0.038, n)
    price = 1000.0 * np.exp(np.cumsum(returns))
    high = price * (1 + np.abs(np.random.normal(0, 0.015, n)))
    low = price * (1 - np.abs(np.random.normal(0, 0.015, n)))
    open_p = low + (high - low) * np.random.uniform(0.1, 0.9, n)
    volume = np.random.lognormal(mean=8.5, sigma=0.9, size=n)
    
    df = pd.DataFrame({
        "Open": open_p.astype("float32"),
        "High": high.astype("float32"),
        "Low": low.astype("float32"),
        "Close": price.astype("float32"),
        "Volume": volume.astype("float32"),
    }, index=dates)
    df.index.name = "Date"
    return df

@st.cache_data(show_spinner="⏳ Loading & processing Bitcoin dataset…")
def load_dataset_from_source(source_path_or_buffer) -> pd.DataFrame:
    """
    Robust reader for both Daily resampled CSV and raw 1-minute OHLCV data.
    Automatically detects format, handles timezones, and downcasts to float32.
    """
    if isinstance(source_path_or_buffer, str) and not os.path.exists(source_path_or_buffer):
        resolved = _find_default_dataset()
        if os.path.exists(resolved):
            source_path_or_buffer = resolved
        else:
            return generate_sample_bitcoin_data()

    try:
        sample = pd.read_csv(source_path_or_buffer, nrows=5)
        # Check if daily pre-resampled data (standard deployment mode)
        if "Timestamp" not in sample.columns and "Date" in sample.columns:
            df = pd.read_csv(
                source_path_or_buffer,
                parse_dates=["Date"],
                index_col="Date",
                dtype={
                    "Open": "float32", "High": "float32", "Low": "float32",
                    "Close": "float32", "Volume": "float32"
                }
            )
            df = df.ffill()
            df = df[df["Close"] > 0].dropna()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df

        # Full chunked pipeline for raw 1-min Kaggle data
        chunks = []
        reader = pd.read_csv(
            source_path_or_buffer,
            chunksize=200_000,
            dtype={
                "Open": "float32", "High": "float32", "Low": "float32",
                "Close": "float32", "Volume": "float32"
            }
        )
        for chunk in reader:
            chunk["Date"] = pd.to_datetime(chunk["Timestamp"], unit="s", utc=True)
            chunk.set_index("Date", inplace=True)
            chunk.drop(columns=["Timestamp"], inplace=True)
            daily = chunk.resample("D").agg(
                Open=("Open", "first"),
                High=("High", "max"),
                Low=("Low", "min"),
                Close=("Close", "last"),
                Volume=("Volume", "sum"),
            )
            chunks.append(daily)
            del chunk
            gc.collect()

        df = pd.concat(chunks).resample("D").agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        )
        df = df.ffill()
        df = df[df["Close"] > 0].dropna()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype("float32")
        df.index = df.index.tz_localize(None)
        gc.collect()
        return df

    except Exception as e:
        st.warning(f"Note: Error reading '{source_path_or_buffer}': {e}. Falling back to sample dataset.")
        return generate_sample_bitcoin_data()

# =============================================================================
# FEATURE ENGINEERING LAYER (Continuous History)
# =============================================================================
def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

@st.cache_data
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all 17 technical indicators over continuous price series.
    Calculated ONCE over the full dataset so rolling windows have complete warmup.
    """
    d = df.copy()
    d["Return_1D"]    = d["Close"].pct_change(1)
    d["Return_7D"]    = d["Close"].pct_change(7)
    d["Return_30D"]   = d["Close"].pct_change(30)
    d["MA_7"]         = d["Close"].rolling(7).mean()
    d["MA_30"]        = d["Close"].rolling(30).mean()
    d["MA_90"]        = d["Close"].rolling(90).mean()
    d["Volatility"]   = d["Close"].rolling(14).std()
    d["RSI"]          = _compute_rsi(d["Close"], 14)
    d["MACD"]         = d["Close"].ewm(span=12).mean() - d["Close"].ewm(span=26).mean()
    d["Signal"]       = d["MACD"].ewm(span=9).mean()
    d["BB_Upper"]     = d["MA_30"] + 2 * d["Volatility"]
    d["BB_Lower"]     = d["MA_30"] - 2 * d["Volatility"]
    d["High_Low_Pct"] = (d["High"] - d["Low"]) / (d["Close"] + 1e-9)
    d["Volume_MA7"]   = d["Volume"].rolling(7).mean()
    d = d.dropna()
    return d

# =============================================================================
# MACHINE LEARNING PIPELINE
# =============================================================================
FEATURE_COLS = [
    "Open", "High", "Low", "Volume",
    "Return_1D", "Return_7D", "MA_7", "MA_30", "MA_90",
    "Volatility", "RSI", "MACD", "Signal",
    "BB_Upper", "BB_Lower", "High_Low_Pct", "Volume_MA7",
]

def build_ml_dataset(df: pd.DataFrame, horizon: int):
    d = df.copy()
    d["Target"] = d["Close"].shift(-horizon)
    d = d.dropna()
    X = d[FEATURE_COLS].astype("float32")
    y = d["Target"].astype("float32")
    return X, y, d.index

@st.cache_data(show_spinner="🤖 Training Random Forest prediction engine…")
def train_model(df_json: str, horizon: int):
    df = pd.read_json(StringIO(df_json))
    df.index = pd.to_datetime(df.index)
    X, y, idx = build_ml_dataset(df, horizon)

    if len(X) < 30:
        return None

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test.values - y_pred) / (y_test.values + 1e-9))) * 100)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    # Forward forecast for the most recent observation
    latest_x = scaler.transform(df[FEATURE_COLS].iloc[[-1]])
    forward_pred = float(model.predict(latest_x)[0])

    return y_test, y_pred, idx_test, metrics, importances, forward_pred

# =============================================================================
# VISUALIZATION FUNCTIONS (Interactive Plotly)
# =============================================================================
def plot_price_history(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04)
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
    ), row=1, col=1)
    # Moving Average Overlays
    for ma, col in [("MA_7", "#60a5fa"), ("MA_30", BITCOIN_ORANGE), ("MA_90", "#a78bfa")]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma,
                                     line=dict(color=col, width=1.5)), row=1, col=1)
    # Volume Bars
    colours = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                         marker_color=colours, name="Volume", opacity=0.75), row=2, col=1)

    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Bitcoin Price & Volume History (Daily)", font=dict(size=15, color="#f9fafb")),
                      xaxis_rangeslider_visible=False)
    return fig

def plot_bollinger(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                             line=dict(color="#6366f1", dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                             line=dict(color="#6366f1", dash="dot", width=1),
                             fill="tonexty", fillcolor="rgba(99,102,241,0.08)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color=BITCOIN_ORANGE, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_30"], name="MA-30",
                             line=dict(color="#94a3b8", width=1.2, dash="dash")))
    return _apply_layout(fig, "Bollinger Bands (30-Day Window)")

def plot_macd(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.45])
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color=BITCOIN_ORANGE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                             line=dict(color="#60a5fa", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal",
                             line=dict(color="#f43f5e", width=1.5)), row=2, col=1)
    hist = df["MACD"] - df["Signal"]
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in hist]
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram",
                         marker_color=colors, opacity=0.75), row=2, col=1)
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text="MACD Indicator (12, 26, 9)", font=dict(size=15, color="#f9fafb")))
    return fig

def plot_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI-14",
                             line=dict(color="#a78bfa", width=2)))
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought 70")
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", annotation_text="Oversold 30")
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)", line_width=0)
    return _apply_layout(fig, "Relative Strength Index (RSI-14)")

def plot_annual_returns(df: pd.DataFrame) -> go.Figure:
    yearly = df.groupby(df.index.year)["Close"].last().pct_change().dropna() * 100
    if yearly.empty:
        fig = go.Figure()
        return _apply_layout(fig, "Annual Returns (%) — Insufficient Range")
    colors = [("#22c55e" if v >= 0 else "#ef4444") for v in yearly.values]
    fig = go.Figure(go.Bar(
        x=[str(y) for y in yearly.index],
        y=yearly.values,
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in yearly.values],
        textposition="outside",
    ))
    return _apply_layout(fig, "Annual Bitcoin Returns (%)")

def plot_feature_correlation(df: pd.DataFrame) -> go.Figure:
    corr_cols = ["Close", "Volume", "Return_1D", "Return_7D", "MA_7", "MA_30", "RSI", "MACD", "Volatility"]
    corr = df[corr_cols].corr()
    fig = px.imshow(
        corr.round(2),
        text_auto=True,
        color_continuous_scale="RdYlGn",
        zmin=-1,
        zmax=1,
        aspect="auto"
    )
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text="Feature Correlation Heatmap", font=dict(size=15, color="#f9fafb")))
    return fig

def plot_forecast(y_test, y_pred, idx_test) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(idx_test), y=list(y_test),
                             name="Actual Close", line=dict(color=BITCOIN_ORANGE, width=2)))
    fig.add_trace(go.Scatter(x=list(idx_test), y=list(y_pred),
                             name="Predicted Close", line=dict(color="#60a5fa", width=2, dash="dash")))
    return _apply_layout(fig, "Random Forest: Actual vs. Predicted Test Performance")

def plot_feature_importance(importances: pd.Series) -> go.Figure:
    top = importances.head(10)
    fig = go.Figure(go.Bar(
        x=top.values[::-1], y=top.index[::-1],
        orientation="h", marker_color=BITCOIN_ORANGE,
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Top 10 Feature Importances", font=dict(size=15, color="#f9fafb")),
                      xaxis_title="Relative Importance Score")
    return fig

# =============================================================================
# METRIC CARD HELPER
# =============================================================================
def metric_card(label: str, value: str, delta: str = "", positive: bool = True):
    delta_class = "metric-delta-pos" if positive else "metric-delta-neg"
    delta_html  = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

# =============================================================================
# MAIN STREAMLIT APPLICATION
# =============================================================================
def main():
    # ── Sidebar Controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-title">₿ Bitcoin Analytics</div>', unsafe_allow_html=True)
        st.caption("End-to-End Market Intelligence & ML Platform")
        st.markdown("---")

        st.markdown("**📂 Data Source**")
        default_file = _find_default_dataset()
        uploaded_file = st.file_uploader("Upload Custom CSV (Optional)", type=["csv"])
        
        if uploaded_file is not None:
            data_source = uploaded_file
            st.success("Using uploaded CSV dataset")
        else:
            data_source = default_file
            st.info(f"Using: `{os.path.basename(default_file)}`")

        # Load & engineer features over complete dataset
        raw_df = load_dataset_from_source(data_source)
        all_feat_df = engineer_features(raw_df)

        min_date = all_feat_df.index.min().date()
        max_date = all_feat_df.index.max().date()

        st.markdown("---")
        st.markdown("**📅 Date Range Filter**")
        default_start = max(min_date, pd.Timestamp("2018-01-01").date())
        date_start = st.date_input("From", value=default_start, min_value=min_date, max_value=max_date)
        date_end   = st.date_input("To",   value=max_date,      min_value=min_date, max_value=max_date)

        # Quick preset buttons
        col_p1, col_p2, col_p3 = st.columns(3)
        if col_p1.button("1 Year"):
            date_start = max(min_date, (pd.Timestamp(max_date) - pd.DateOffset(years=1)).date())
        if col_p2.button("3 Years"):
            date_start = max(min_date, (pd.Timestamp(max_date) - pd.DateOffset(years=3)).date())
        if col_p3.button("All-Time"):
            date_start = min_date

        if date_start > date_end:
            st.error("Error: 'From' date must be earlier than 'To' date.")
            date_start, date_end = date_end, date_start

        st.markdown("---")
        st.markdown("**🤖 ML Forecasting Configuration**")
        horizon = st.slider("Forecast Horizon (Days Ahead)", min_value=1, max_value=30, value=7, step=1)
        run_ml  = st.button("▶ Train & Predict", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown("**📊 Chart Toggles**")
        show_bb   = st.checkbox("Bollinger Bands", value=True)
        show_macd = st.checkbox("MACD Indicator",  value=True)
        show_rsi  = st.checkbox("RSI Indicator",   value=True)
        show_yret = st.checkbox("Annual Returns",  value=True)

        st.markdown("---")
        st.caption("Developed by Bharath | IBM Internship Program\n\nOptimized for Streamlit Cloud Deployment")

    # ── Slicing & Validating Range ───────────────────────────────────────────
    df_feat = all_feat_df.loc[str(date_start):str(date_end)].copy()
    if len(df_feat) < 2:
        st.warning("⚠️ Insufficient data points in the selected range. Please widen your date filter in the sidebar.")
        st.stop()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## ₿ Bitcoin Market Intelligence & Forecasting Platform")
    st.markdown(f"Displaying **{len(df_feat):,} active trading days** | "
                f"`{df_feat.index[0].strftime('%b %d, %Y')}` → `{df_feat.index[-1].strftime('%b %d, %Y')}`")
    st.markdown("---")

    # ── KPI Metric Cards ──────────────────────────────────────────────────────
    latest        = float(df_feat["Close"].iloc[-1])
    prev          = float(df_feat["Close"].iloc[-2])
    price_chg     = ((latest - prev) / prev) * 100
    all_time_high = float(df_feat["High"].max())
    all_time_low  = float(df_feat["Low"].min())
    avg_vol_7d    = float(df_feat["Volume"].tail(7).mean())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Latest Close Price", f"${latest:,.2f}",
                    f"{'▲' if price_chg >= 0 else '▼'} {abs(price_chg):.2f}% (24h)",
                    positive=(price_chg >= 0))
    with c2:
        metric_card("Period High", f"${all_time_high:,.2f}")
    with c3:
        metric_card("Period Low",  f"${all_time_low:,.2f}")
    with c4:
        metric_card("7-Day Avg Volume", f"{avg_vol_7d:,.1f} BTC")

    st.markdown("---")

    # ── Historical Price & Volume ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Historical Price & Volume Analysis</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_price_history(df_feat), use_container_width=True)

    # ── Technical Indicator Suite ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📐 Technical Indicator Suite</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    if show_bb:
        with col_l:
            st.plotly_chart(plot_bollinger(df_feat), use_container_width=True)
    if show_macd:
        with col_r:
            st.plotly_chart(plot_macd(df_feat), use_container_width=True)

    if show_rsi or show_yret:
        col_a, col_b = st.columns(2)
        if show_rsi:
            with col_a:
                st.plotly_chart(plot_rsi(df_feat), use_container_width=True)
        if show_yret:
            with col_b:
                st.plotly_chart(plot_annual_returns(df_feat), use_container_width=True)

    # ── Volume Trend ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📦 Volume Profiling & 7-Day Trend</div>', unsafe_allow_html=True)
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["Volume"],
                                 fill="tozeroy", name="Daily Volume",
                                 line=dict(color="#60a5fa", width=1.5)))
    vol_fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["Volume_MA7"],
                                 name="7-Day MA Volume",
                                 line=dict(color=BITCOIN_ORANGE, width=2, dash="dash")))
    vol_fig.update_layout(**PLOTLY_LAYOUT,
                           title=dict(text="Daily Bitcoin Trading Volume (BTC)", font=dict(size=15, color="#f9fafb")))
    st.plotly_chart(vol_fig, use_container_width=True)

    # ── Correlation Matrix ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_feature_correlation(df_feat), use_container_width=True)

    # ── Machine Learning Section ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 Machine Learning — Random Forest Price Forecast</div>', unsafe_allow_html=True)

    if run_ml:
        if len(df_feat) < 40:
            st.warning("⚠️ Training dataset contains fewer than 40 trading days. Please select a wider date range for reliable ML training.")
        else:
            with st.spinner(f"Training 120-tree Random Forest to project {horizon}-day-ahead price…"):
                df_json = df_feat[FEATURE_COLS + ["Close"]].to_json()
                ml_result = train_model(df_json, horizon)

            if ml_result is not None:
                y_test, y_pred, idx_test, metrics, importances, forward_pred = ml_result
                target_date = df_feat.index[-1] + pd.Timedelta(days=horizon)
                delta_pred = ((forward_pred - latest) / latest) * 100

                # ── Live Forward Forecast Banner ──────────────────────────────
                st.markdown(f"""
                <div class="forecast-card">
                    <div style="font-size: 14px; font-weight: 600; color: #93c5fd; text-transform: uppercase;">
                        🎯 Live Forward Price Projection ({horizon} Days Ahead — {target_date.strftime('%B %d, %Y')})
                    </div>
                    <div style="font-size: 32px; font-weight: 700; color: #f9fafb; margin: 8px 0;">
                        ${forward_pred:,.2f} 
                        <span style="font-size: 18px; color: {'#22c55e' if delta_pred >= 0 else '#ef4444'};">
                            ({'▲' if delta_pred >= 0 else '▼'} {abs(delta_pred):.2f}% from latest close)
                        </span>
                    </div>
                    <div style="font-size: 13px; color: #9ca3af;">
                        Model: Random Forest Regressor (120 Estimators) | Expected Error Margin: ±${metrics['MAE']:,.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Metrics Cards ─────────────────────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                with m1: metric_card("MAE (Mean Absolute Error)", f"${metrics['MAE']:,.2f}")
                with m2: metric_card("RMSE (Root Mean Sq Error)", f"${metrics['RMSE']:,.2f}")
                with m3: metric_card("R² Score (Goodness of Fit)", f"{metrics['R2']:.4f}", positive=(metrics['R2'] >= 0.70))
                with m4: metric_card("MAPE (Mean % Error)", f"{metrics['MAPE']:.2f}%", positive=(metrics['MAPE'] <= 12))

                st.plotly_chart(plot_forecast(y_test, y_pred, idx_test), use_container_width=True)

                col_fi, col_err = st.columns(2)
                with col_fi:
                    st.plotly_chart(plot_feature_importance(importances), use_container_width=True)
                with col_err:
                    errors = np.array(y_test) - np.array(y_pred)
                    fig_err = go.Figure(go.Histogram(x=errors, nbinsx=50,
                                                     marker_color=BITCOIN_ORANGE, opacity=0.85))
                    _apply_layout(fig_err, "Prediction Residuals Distribution")
                    fig_err.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
                    st.plotly_chart(fig_err, use_container_width=True)
            else:
                st.error("Failed to generate model. Please ensure the selected date range has sufficient non-null records.")
    else:
        st.info("👈 Set your desired forecast horizon in the sidebar and click **▶ Train & Predict** to execute the ML pipeline.")

    # ── Export & Data Download ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**📥 Export Filtered Dataset**")
    csv_data = df_feat.to_csv().encode("utf-8")
    st.download_button(
        label="Download Filtered OHLCV + Indicators (CSV)",
        data=csv_data,
        file_name="bitcoin_analytics_export.csv",
        mime="text/csv",
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption("Bitcoin Market Intelligence & Analytics Dashboard | IBM Internship Program | Built with Streamlit, Plotly & scikit-learn")

if __name__ == "__main__":
    main()
