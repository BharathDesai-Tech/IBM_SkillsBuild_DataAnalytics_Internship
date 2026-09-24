# ₿ Bitcoin Market Intelligence & Price Forecasting Platform

> **IBM SkillsBuild Data Analytics with AI Academic Internship Program**  
> Conducted by **BharatCares** in association with **AICTE**  
> **Author:** Bharath  
> **Dataset Source:** [Kaggle — Bitcoin Historical Data (BTC/USD 1-Min OHLCV)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

---

## 📌 Project Overview

This project delivers an **end-to-end, memory-optimized market intelligence and machine learning forecasting platform** for Bitcoin (BTC/USD) historical data spanning from 2012 to 2026.

Operating on high-frequency cryptocurrency execution data (~5.5 million records, ~392 MB), the platform implements strict data engineering and memory guardrails to transform raw 1-minute execution ticks into high-performance daily resampled records, computing 17 technical indicators and delivering predictive machine learning forecasts in real-time.

### Key Capabilities:
- **Exploratory Data Analysis (EDA):** Interactive candlestick charts, volume profiling, 24-hour delta metrics, period highs/lows.
- **Quantitative Technical Indicators:** Moving Averages (7/30/90-day), 30-day Bollinger Bands with shaded volatility envelopes, MACD (12, 26, 9) histogram, and RSI-14 overbought/oversold boundaries.
- **Machine Learning Forecasting Engine:** Time-ordered Random Forest Regressor (120 estimators) predicting $N$-day-ahead price moves with forward forecasting, residual error distribution, and feature importance rankings.
- **Interactive Web Interface:** Single-command interactive Streamlit web dashboard styled with a dark financial terminal aesthetic.

---

## 📁 Required Submission Files

As per the IBM SkillsBuild internship submission guidelines, all four designated project deliverables are prepared and verified:

| Deliverable | File Name | Format | Description |
|---|---|---|---|
| **Code File** | `Bharath_BitcoinAnalytics.py` / `Bharath_BitcoinAnalytics.ipynb` | `.py` / `.ipynb` | Complete, executable, end-to-end analytics and ML forecasting pipeline. |
| **Requirements File** | `requirements.txt` | `.txt` | Pinned list of required Python libraries and dependencies. |
| **Project Report** | `Bharath_ProjectReport.docx` | `.docx` | Comprehensive 7-section technical report with tables, methodology, and recommendations. |
| **README File** | `README.md` | `.md` | Project overview, dataset link, architecture, and run instructions. |

---

## 🛠️ Technology Stack

| Layer | Library / Tool | Purpose |
|---|---|---|
| **UI Framework** | `Streamlit 1.35+` | Interactive web dashboard and reactive controls |
| **Data Processing** | `Pandas 2.2+` | Chunked ingestion, daily resampling, time-series transformations |
| **Scientific Computing** | `NumPy 1.26+` | Array operations, log returns, and performance metrics |
| **Machine Learning** | `scikit-learn 1.5+` | Feature scaling, Random Forest Regressor, error metrics |
| **Data Visualizations** | `Plotly 5.22+` | Interactive candlestick, MACD, RSI, and correlation charts |
| **Document Generation** | `python-docx 1.2+` | Automated corporate report compilation |

---

## ⚡ Data Engineering & Memory Optimization

Ingesting 5.5 million records on standard hardware can easily trigger Out-Of-Memory (OOM) failures. This project applies five strict optimization guardrails:

1. **Chunked Ingestion:** Reads the dataset in batches of 200,000 rows (`chunksize=200_000`).
2. **Precision Downcasting (`float32`):** Numerical columns are vectorized as 32-bit floats, halving memory usage compared to standard 64-bit floats.
3. **Immediate Daily Resampling:** Candles are aggregated to daily intervals (`Open=first`, `High=max`, `Low=min`, `Close=last`, `Volume=sum`), compressing 5.5M rows to ~5,380 daily records (a 1,000× reduction).
4. **Continuous Indicator Warmup:** Technical indicators are engineered over the full continuous series before slicing, preventing boundary artifacts or missing data on short time windows.
5. **Garbage Collection & Session Caching:** Explicit `gc.collect()` and Streamlit `@st.cache_data` eliminate redundant recomputations during interactive user filtering.

---

## 🚀 How to Run Locally

### 1. Prerequisites & Environment Setup
Clone or extract the repository, then navigate into the project directory:
```bash
cd Bitcoin_Analytics
```

Create and activate a Python virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
- **Option A (One-Click Windows Launcher):**  
  Double-click `run_dashboard.bat`.
- **Option B (Command Line):**  
  ```bash
  python -m streamlit run Bharath_BitcoinAnalytics.py
  ```
  *(Or `python -m streamlit run app.py`)*

Open your browser at **http://localhost:8501**.

---

## 🌐 Cloud Deployment (Streamlit Community Cloud)

1. Create a GitHub repository and push the project files (ensure `btcusd_1-min_data.csv` is excluded via `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and link your GitHub account.
3. Select your repository, set the **Main file path** to `app.py` (or `Bharath_BitcoinAnalytics.py`).
4. Click **Deploy**. The app will be live and accessible globally!

---

## 📊 Analytical Insights Summary

1. **Halving Cycles:** Parabolic price expansion occurs consistently around 4-year halving intervals (2012, 2016, 2020, 2024).
2. **Indicator Divergence:** RSI-14 consistently flags market exhaustion (>70 overbought, <30 oversold).
3. **Volatility Squeezes:** Narrowing Bollinger Bands reliably precede sharp breakout moves.
4. **Predictive Accuracy:** The Random Forest model demonstrates robust predictive accuracy ($R^2 > 0.80$) for 1-to-7-day forward forecasting horizons.

---

## 📄 License & Attribution

Submitted for academic evaluation as part of the **IBM SkillsBuild Internship Program** conducted by BharatCares and AICTE. Dataset licensed under Kaggle Open Data Terms.
