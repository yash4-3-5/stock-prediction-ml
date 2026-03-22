"""
================================================================
 STOCK MARKET TREND PREDICTION USING MACHINE LEARNING
================================================================
 Team Members:
   1. Daksh Chaudhary
   2. Vishu Tarar
   3. Yash Verma
================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Stock Trend Predictor | ML Project",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        padding: 10px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 300;
    }
    .team-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.85rem;
        text-align: center;
        display: inline-block;
        margin: 2px;
        font-weight: 500;
    }
    .prediction-box-up {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        padding: 30px;
        border-radius: 20px;
        border-left: 8px solid #4CAF50;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(76,175,80,0.2);
    }
    .prediction-box-down {
        background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
        padding: 30px;
        border-radius: 20px;
        border-left: 8px solid #F44336;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(244,67,54,0.2);
    }
    .prediction-text {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: #333333 !important;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #555555 !important;
        margin-top: 8px;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    models = {}
    model_files = {
        'Random Forest': 'random_forest_model.pkl',
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl'
    }
    for name, filename in model_files.items():
        if os.path.exists(filename):
            models[name] = joblib.load(filename)
    scaler = None
    if os.path.exists('scaler.pkl'):
        scaler = joblib.load('scaler.pkl')
    feature_cols = None
    if os.path.exists('feature_cols.json'):
        with open('feature_cols.json', 'r') as f:
            feature_cols = json.load(f)
    return models, scaler, feature_cols


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram


def create_features(df):
    data = df.copy()
    data['Daily_Return'] = data['Close'].pct_change() * 100
    data['Price_Change'] = data['Close'] - data['Open']
    data['High_Low_Diff'] = data['High'] - data['Low']
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['RSI'] = calculate_rsi(data['Close'], period=14)
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data['Close'])
    data['Volume_Change'] = data['Volume'].pct_change() * 100
    data['Volume_SMA_5'] = data['Volume'].rolling(window=5).mean()
    data['Close_to_SMA20_Ratio'] = data['Close'] / data['SMA_20']
    data['Return_Lag1'] = data['Daily_Return'].shift(1)
    data['Return_Lag2'] = data['Daily_Return'].shift(2)
    data['Return_Lag3'] = data['Daily_Return'].shift(3)
    data['Volatility_5'] = data['Daily_Return'].rolling(window=5).std()
    data['Volatility_20'] = data['Daily_Return'].rolling(window=20).std()
    return data


@st.cache_data(ttl=300)
def download_stock_data(ticker, period="1y"):
    try:
        data = yf.download(ticker, period=period, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data is None or len(data) == 0:
            return None
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values('Date', ascending=True, inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'currency': info.get('currency', 'USD'),
            'exchange': info.get('exchange', 'N/A'),
        }
    except Exception:
        return {
            'name': ticker, 'sector': 'N/A', 'industry': 'N/A',
            'market_cap': 0, 'currency': 'USD', 'exchange': 'N/A',
        }


def make_prediction(model, scaler, feature_cols, df_features, model_name):
    latest = df_features.iloc[-1]
    if feature_cols:
        available = [c for c in feature_cols if c in df_features.columns]
        features = latest[available].values.reshape(1, -1)
    else:
        exclude = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        avail_cols = [c for c in df_features.columns if c not in exclude]
        features = latest[avail_cols].values.reshape(1, -1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    if model_name == 'Logistic Regression' and scaler is not None:
        features_for_pred = scaler.transform(features)
    else:
        features_for_pred = features
    prediction = model.predict(features_for_pred)[0]
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(features_for_pred)[0]
    return prediction, probability, latest


def get_all_model_predictions(models, scaler, feature_cols, df_features):
    results = {}
    for name, model in models.items():
        try:
            pred, prob, latest = make_prediction(model, scaler, feature_cols, df_features, name)
            results[name] = {
                'prediction': pred,
                'probability': prob,
                'confidence': max(prob) * 100 if prob is not None else None
            }
        except Exception:
            pass
    return results


def format_market_cap(value):
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.0f}"


def main():
    models, scaler, feature_cols = load_all_models()

    if not models:
        st.error("No model files found! Make sure .pkl files are in the same folder as app.py")
        st.stop()

    # SIDEBAR
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center; padding:15px; '
            'background:linear-gradient(135deg,#667eea,#764ba2); '
            'border-radius:15px; margin-bottom:20px;">'
            '<h2 style="color:white; margin:0; font-size:1.4rem;">📈 Stock Predictor</h2>'
            '<p style="color:rgba(255,255,255,0.8); margin:5px 0 0 0; font-size:0.85rem;">'
            'ML Mini Project</p></div>',
            unsafe_allow_html=True
        )

        st.markdown("### 🤖 Select Model")
        model_choice = st.selectbox(
            "Choose ML Model",
            list(models.keys()),
            index=list(models.keys()).index('Random Forest') if 'Random Forest' in models else 0,
            label_visibility="collapsed"
        )

        model_descriptions = {
            'Random Forest': "🌲 Ensemble of 200 trees — Most accurate",
            'Logistic Regression': "📈 Linear classifier — Simple baseline",
            'Decision Tree': "🌳 If-else rules — Interpretable"
        }
        st.info(model_descriptions.get(model_choice, ""))

        st.markdown("---")

        st.markdown("### 📅 Data Period")
        data_period = st.selectbox(
            "Range",
            ["6mo", "1y", "2y", "5y"],
            index=1,
            format_func=lambda x: {"6mo": "6 Months", "1y": "1 Year", "2y": "2 Years", "5y": "5 Years"}[x],
            label_visibility="collapsed"
        )

        st.markdown("---")

        st.markdown("### 👨‍💻 Team Members")
        team_members = [
            {"name": "Daksh Chaudhary", "emoji": "🧑‍💻", "role": "ML & Backend"},
            {"name": "Vishu Tarar", "emoji": "👨‍💻", "role": "Data & Analysis"},
            {"name": "Yash Verma", "emoji": "👨‍🔬", "role": "Frontend & Deploy"}
        ]
        for member in team_members:
            st.markdown(
                '<div style="background:#f0f2f6; padding:10px 15px; border-radius:10px; '
                'margin:8px 0; display:flex; align-items:center;">'
                '<span style="font-size:1.5rem; margin-right:10px;">' + member['emoji'] + '</span>'
                '<div>'
                '<div style="font-weight:600; font-size:0.95rem; color:#333;">' + member['name'] + '</div>'
                '<div style="font-size:0.75rem; color:#888;">' + member['role'] + '</div>'
                '</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### 📖 Project Info")
        st.markdown(f"**Subject:** Machine Learning\n\n**Date:** {datetime.now().strftime('%B %Y')}")

    # MAIN HEADER
    st.markdown('<div class="main-header">📈 Stock Market Trend Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Predicting next-day stock price direction using Machine Learning</div>',
        unsafe_allow_html=True
    )

    bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns([1, 1, 1, 1, 1])
    with bcol2:
        st.markdown('<span class="team-badge">👨‍💻 Daksh Chaudhary</span>', unsafe_allow_html=True)
    with bcol3:
        st.markdown('<span class="team-badge">👨‍💻 Vishu Tarar</span>', unsafe_allow_html=True)
    with bcol4:
        st.markdown('<span class="team-badge">👨‍💻 Yash Verma</span>', unsafe_allow_html=True)

    st.markdown("")

    # TABS
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔮 Live Prediction",
        "📊 Stock Analysis",
        "⚖️ Model Comparison",
        "📈 Model Performance",
        "📋 Documentation",
        "👥 Team & Credits"
    ])

    # ==================== TAB 1: LIVE PREDICTION ====================
    with tab1:
        st.markdown("### 🔮 Real-Time Stock Trend Prediction")
        st.markdown("Enter any stock ticker to get an instant AI-powered prediction")

        st.markdown("**🔥 Popular Stocks:**")
        qc1, qc2, qc3, qc4, qc5, qc6, qc7, qc8 = st.columns(8)
        quick_ticker = None
        with qc1:
            if st.button("🍎 AAPL", use_container_width=True):
                quick_ticker = "AAPL"
        with qc2:
            if st.button("🪟 MSFT", use_container_width=True):
                quick_ticker = "MSFT"
        with qc3:
            if st.button("🔍 GOOGL", use_container_width=True):
                quick_ticker = "GOOGL"
        with qc4:
            if st.button("📦 AMZN", use_container_width=True):
                quick_ticker = "AMZN"
        with qc5:
            if st.button("⚡ TSLA", use_container_width=True):
                quick_ticker = "TSLA"
        with qc6:
            if st.button("🏭 REL.NS", use_container_width=True):
                quick_ticker = "RELIANCE.NS"
        with qc7:
            if st.button("💻 TCS.NS", use_container_width=True):
                quick_ticker = "TCS.NS"
        with qc8:
            if st.button("🔧 INFY.NS", use_container_width=True):
                quick_ticker = "INFY.NS"

        st.markdown("---")

        ic1, ic2 = st.columns([3, 1])
        with ic1:
            default_val = quick_ticker if quick_ticker else "AAPL"
            ticker = st.text_input("📌 Enter Stock Ticker Symbol", value=default_val, key="live_ticker")
        with ic2:
            st.markdown("")
            st.markdown("")
            predict_btn = st.button("🚀 PREDICT NOW", type="primary", use_container_width=True)

        if predict_btn or quick_ticker:
            ticker = ticker.upper().strip()
            if not ticker:
                st.error("Please enter a stock ticker!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text(f"📥 Downloading {ticker} data...")
                progress_bar.progress(20)
                raw_data = download_stock_data(ticker, period=data_period)

                if raw_data is None or len(raw_data) < 30:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Could not download data for '{ticker}'. Check ticker and internet.")
                else:
                    status_text.text("⚙️ Computing technical indicators...")
                    progress_bar.progress(50)
                    df_processed = create_features(raw_data)
                    df_processed.dropna(inplace=True)
                    df_processed.reset_index(drop=True, inplace=True)

                    status_text.text("🤖 Running ML prediction...")
                    progress_bar.progress(75)
                    stock_info = get_stock_info(ticker)
                    all_predictions = get_all_model_predictions(models, scaler, feature_cols, df_processed)
                    model = models[model_choice]
                    prediction, probability, latest = make_prediction(model, scaler, feature_cols, df_processed, model_choice)

                    status_text.text("✅ Complete!")
                    progress_bar.progress(100)
                    time.sleep(0.3)
                    progress_bar.empty()
                    status_text.empty()

                    st.markdown("---")

                    # Company Info
                    st.markdown(f"### {stock_info['name']} ({ticker})")
                    ic1, ic2, ic3, ic4 = st.columns(4)
                    with ic1:
                        st.markdown(f"**Sector:** {stock_info['sector']}")
                    with ic2:
                        st.markdown(f"**Industry:** {stock_info['industry']}")
                    with ic3:
                        st.markdown(f"**Market Cap:** {format_market_cap(stock_info['market_cap'])}")
                    with ic4:
                        st.markdown(f"**Exchange:** {stock_info['exchange']}")

                    st.markdown("---")

                    # Stock Metrics
                    st.markdown("#### 📋 Today's Stock Data")
                    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                    with mc1:
                        st.metric("📅 Date", str(latest['Date'].date()))
                    with mc2:
                        st.metric("💰 Close", f"${latest['Close']:.2f}")
                    with mc3:
                        st.metric("📈 Open", f"${latest['Open']:.2f}")
                    with mc4:
                        st.metric("🔺 High", f"${latest['High']:.2f}")
                    with mc5:
                        st.metric("🔻 Low", f"${latest['Low']:.2f}")
                    with mc6:
                        dr = latest.get('Daily_Return', 0)
                        st.metric("📊 Return", f"{dr:.2f}%", delta=f"{dr:.2f}%")

                    st.markdown("---")

                    # Main Prediction
                    pc1, pc2 = st.columns([3, 1])
                    with pc1:
                        if prediction == 1:
                            st.markdown(
                                '<div class="prediction-box-up">'
                                '<p class="prediction-text">🟢 NEXT DAY: UP ▲</p>'
                                '<p class="confidence-text">'
                                'The ' + model_choice + ' model predicts ' + ticker + ' will INCREASE tomorrow'
                                '</p></div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<div class="prediction-box-down">'
                                '<p class="prediction-text">🔴 NEXT DAY: DOWN ▼</p>'
                                '<p class="confidence-text">'
                                'The ' + model_choice + ' model predicts ' + ticker + ' will DECREASE tomorrow'
                                '</p></div>',
                                unsafe_allow_html=True
                            )
                    with pc2:
                        st.markdown("#### 📊 Confidence")
                        if probability is not None:
                            st.metric("Confidence", f"{max(probability)*100:.1f}%")
                            st.metric("P(UP)", f"{probability[1]*100:.1f}%")
                            st.metric("P(DOWN)", f"{probability[0]*100:.1f}%")
                        st.markdown(f"**Model:** {model_choice}")

                    # All Models Consensus
                    st.markdown("---")
                    st.markdown("#### 🗳️ All Models Consensus")
                    cons_cols = st.columns(len(all_predictions))
                    up_votes = 0
                    total_votes = 0
                    for idx, (m_name, m_result) in enumerate(all_predictions.items()):
                        with cons_cols[idx]:
                            pred = m_result['prediction']
                            conf = m_result['confidence']
                            if pred == 1:
                                up_votes += 1
                                color = "#4CAF50"
                                direction = "UP ▲"
                                emoji = "🟢"
                            else:
                                color = "#F44336"
                                direction = "DOWN ▼"
                                emoji = "🔴"
                            total_votes += 1
                            st.markdown(
                                '<div style="background:white; padding:15px; border-radius:12px; '
                                'text-align:center; border-top:4px solid ' + color + '; '
                                'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
                                '<div style="font-weight:600; font-size:0.9rem; color:#555;">' + m_name + '</div>'
                                '<div style="font-size:1.5rem; margin:8px 0;">' + emoji + ' ' + direction + '</div>'
                                '<div style="font-size:0.85rem; color:#888;">' +
                                (f'Confidence: {conf:.1f}%' if conf else 'N/A') + '</div></div>',
                                unsafe_allow_html=True
                            )

                    if up_votes > total_votes / 2:
                        st.success(f"📢 CONSENSUS ({up_votes}/{total_votes} models): BULLISH — Trend likely UP ▲")
                    else:
                        st.error(f"📢 CONSENSUS ({total_votes - up_votes}/{total_votes} models): BEARISH — Trend likely DOWN ▼")

                    # Technical Indicators
                    st.markdown("---")
                    st.markdown("#### 🔧 Technical Indicators")
                    tc1, tc2, tc3, tc4 = st.columns(4)
                    with tc1:
                        rsi_val = latest.get('RSI', 50)
                        rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
                        st.metric("RSI (14)", f"{rsi_val:.1f}")
                        st.caption(rsi_status)
                    with tc2:
                        macd_val = latest.get('MACD', 0)
                        st.metric("MACD", f"{macd_val:.3f}")
                        st.caption("Bullish" if macd_val > 0 else "Bearish")
                    with tc3:
                        vol5 = latest.get('Volatility_5', 0)
                        st.metric("Volatility (5d)", f"{vol5:.2f}%")
                    with tc4:
                        sma_ratio = latest.get('Close_to_SMA20_Ratio', 1)
                        st.metric("Price/SMA20", f"{sma_ratio:.3f}")
                        st.caption("Above SMA20" if sma_ratio > 1 else "Below SMA20")

                    # Live Charts
                    st.markdown("---")
                    st.markdown("#### 📊 Live Stock Charts")

                    fig1, ax1 = plt.subplots(figsize=(14, 5))
                    ax1.plot(df_processed['Date'], df_processed['Close'], color='#2196F3', linewidth=1.5, label='Close')
                    if 'SMA_20' in df_processed.columns:
                        ax1.plot(df_processed['Date'], df_processed['SMA_20'], color='orange', linewidth=1, label='SMA 20', alpha=0.7)
                    last_date = df_processed['Date'].iloc[-1]
                    last_close = df_processed['Close'].iloc[-1]
                    dot_color = 'green' if prediction == 1 else 'red'
                    ax1.scatter([last_date], [last_close], color=dot_color, s=100, zorder=5, label=f'Prediction: {"UP" if prediction==1 else "DOWN"}')
                    ax1.set_title(f'{ticker} Stock Price', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel(f'Price ({stock_info["currency"]})')
                    ax1.legend(fontsize=9)
                    ax1.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close()

                    cc1, cc2 = st.columns(2)
                    with cc1:
                        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
                        tail60 = df_processed.tail(60)
                        ax2.plot(tail60['Date'], tail60['RSI'], color='purple', linewidth=1.2)
                        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.6, label='Overbought')
                        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.6, label='Oversold')
                        ax2.fill_between(tail60['Date'], 30, 70, alpha=0.05, color='gray')
                        ax2.set_ylim(0, 100)
                        ax2.set_title('RSI (14-period)', fontsize=12, fontweight='bold')
                        ax2.legend(fontsize=7)
                        ax2.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig2)
                        plt.close()
                    with cc2:
                        fig3, ax3 = plt.subplots(figsize=(7, 3.5))
                        tail60 = df_processed.tail(60)
                        ax3.plot(tail60['Date'], tail60['MACD'], color='blue', linewidth=1, label='MACD')
                        ax3.plot(tail60['Date'], tail60['MACD_Signal'], color='red', linewidth=1, label='Signal')
                        hist_colors = ['green' if x >= 0 else 'red' for x in tail60['MACD_Hist']]
                        ax3.bar(tail60['Date'], tail60['MACD_Hist'], color=hist_colors, alpha=0.3, width=2)
                        ax3.set_title('MACD', fontsize=12, fontweight='bold')
                        ax3.legend(fontsize=7)
                        ax3.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig3)
                        plt.close()

                    # Disclaimer - FIXED COLOR
                    st.markdown(
                        '<div style="background:#FFF3E0; padding:15px; border-radius:10px; '
                        'border-left:4px solid #FF9800; margin:10px 0;">'
                        '<span style="color:#E65100; font-weight:700;">⚠️ Disclaimer: </span>'
                        '<span style="color:#333333;">This prediction is for '
                        'ACADEMIC PURPOSES ONLY. Stock markets are inherently unpredictable. '
                        'Do NOT use this for actual trading decisions. '
                        'Past performance does not guarantee future results.</span>'
                        '</div>',
                        unsafe_allow_html=True
                    )

    # ==================== TAB 2: STOCK ANALYSIS ====================
    with tab2:
        st.markdown("### 📊 Detailed Stock Analysis")
        analysis_ticker = st.text_input("Enter ticker for analysis:", value="AAPL", key="analysis_ticker")

        if st.button("📊 Analyze", type="primary", key="analyze_btn"):
            analysis_ticker = analysis_ticker.upper().strip()
            with st.spinner(f"Analyzing {analysis_ticker}..."):
                data = download_stock_data(analysis_ticker, period="2y")

            if data is not None and len(data) > 30:
                data = create_features(data)
                data.dropna(inplace=True)
                info = get_stock_info(analysis_ticker)

                st.markdown(f"#### {info['name']} ({analysis_ticker})")

                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                with sc1:
                    st.metric("Avg Daily Return", f"{data['Daily_Return'].mean():.3f}%")
                with sc2:
                    st.metric("Daily Volatility", f"{data['Daily_Return'].std():.3f}%")
                with sc3:
                    st.metric("Highest Price", f"${data['Close'].max():.2f}")
                with sc4:
                    st.metric("Lowest Price", f"${data['Close'].min():.2f}")
                with sc5:
                    total_ret = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                    st.metric("Total Return", f"{total_ret:.1f}%", delta=f"{total_ret:.1f}%")

                fig, axes = plt.subplots(1, 2, figsize=(14, 4))
                axes[0].hist(data['Daily_Return'], bins=50, color='#667eea', edgecolor='white', alpha=0.8)
                axes[0].axvline(x=0, color='red', linestyle='--')
                axes[0].set_title('Daily Return Distribution', fontweight='bold')
                axes[0].set_xlabel('Return (%)')
                axes[1].bar(data.tail(60)['Date'], data.tail(60)['Volume'], color='#764ba2', alpha=0.6)
                axes[1].set_title('Volume (Last 60 Days)', fontweight='bold')
                axes[1].tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.markdown("##### 📋 Recent Data")
                recent = data.tail(10)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'RSI']].copy()
                recent['Date'] = recent['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(recent.round(2), use_container_width=True, hide_index=True)
            else:
                st.error(f"Could not load data for {analysis_ticker}")

    # ==================== TAB 3: MODEL COMPARISON ====================
    with tab3:
        st.markdown("### ⚖️ Model Comparison")
        compare_ticker = st.text_input("Enter ticker:", value="AAPL", key="compare_ticker")

        if st.button("⚖️ Compare Models", type="primary", key="compare_btn"):
            compare_ticker = compare_ticker.upper().strip()
            with st.spinner(f"Running all models on {compare_ticker}..."):
                data = download_stock_data(compare_ticker, period="1y")

            if data is not None and len(data) > 30:
                df_feat = create_features(data)
                df_feat.dropna(inplace=True)
                df_feat.reset_index(drop=True, inplace=True)
                all_preds = get_all_model_predictions(models, scaler, feature_cols, df_feat)

                st.markdown("#### 🗳️ Predictions")
                comp_cols = st.columns(len(all_preds))
                for idx, (m_name, m_result) in enumerate(all_preds.items()):
                    with comp_cols[idx]:
                        pred = m_result['prediction']
                        conf = m_result['confidence']
                        bg = "#E8F5E9" if pred == 1 else "#FFEBEE"
                        bc = "#4CAF50" if pred == 1 else "#F44336"
                        direction = "UP ▲" if pred == 1 else "DOWN ▼"
                        emoji = "🟢" if pred == 1 else "🔴"
                        st.markdown(
                            '<div style="background:' + bg + '; padding:20px; border-radius:15px; '
                            'text-align:center; border:2px solid ' + bc + ';">'
                            '<div style="font-weight:700; color:#333;">' + m_name + '</div>'
                            '<div style="font-size:2rem; margin:10px 0;">' + emoji + '</div>'
                            '<div style="font-size:1.3rem; font-weight:600; color:' + bc + ';">' + direction + '</div>'
                            '<div style="font-size:0.9rem; color:#666; margin-top:8px;">'
                            + (f'Confidence: {conf:.1f}%' if conf else 'N/A') + '</div></div>',
                            unsafe_allow_html=True
                        )

                st.markdown("---")
                st.markdown("#### 📖 How Each Model Works")
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    st.markdown("**📈 Logistic Regression**\n\nLinear boundary. `P(UP) = 1/(1+e^-z)`\n\n✅ Simple ❌ Can't capture non-linear patterns")
                with ec2:
                    st.markdown("**🌳 Decision Tree**\n\nIf-else rules. Splits data at thresholds.\n\n✅ Interpretable ❌ Prone to overfitting")
                with ec3:
                    st.markdown("**🌲 Random Forest**\n\n200 trees, majority vote.\n\n✅ Most accurate ❌ Less interpretable")
            else:
                st.error(f"Could not load data for {compare_ticker}")

    # ==================== TAB 4: MODEL PERFORMANCE ====================
    with tab4:
        st.markdown("### 📈 Training Results & Performance Graphs")

        image_data = [
            ("📈 Stock Price Trend", "images/01_stock_price_trend.png", "Historical price of the training stock."),
            ("📊 Moving Averages", "images/02_moving_averages.png", "Close price with SMA overlays."),
            ("🔢 Confusion Matrices", "images/03_confusion_matrices.png", "TP, TN, FP, FN for each model."),
            ("🏆 Feature Importance", "images/04_feature_importance.png", "Which features matter most."),
            ("⚖️ Model Comparison", "images/05_model_comparison.png", "Accuracy, Precision, Recall, F1-Score."),
        ]

        for title, filepath, desc in image_data:
            with st.expander(title, expanded=False):
                st.markdown(f"*{desc}*")
                if os.path.exists(filepath):
                    st.image(filepath, use_container_width=True)
                else:
                    st.warning(f"Image not found: {filepath}")

        st.markdown("---")
        st.markdown("#### 📖 Metrics Explained")
        met1, met2 = st.columns(2)
        with met1:
            st.markdown("""
            | Metric | Meaning |
            |--------|---------|
            | **Accuracy** | % of correct predictions |
            | **Precision** | When we say UP, how often correct? |
            | **Recall** | Of all UP days, how many caught? |
            | **F1-Score** | Balance of Precision & Recall |
            """)
        with met2:
            st.markdown("""
            **Confusion Matrix:**
            ```
                         Pred DOWN    Pred UP
            Actual DOWN:    TN ✅       FP ❌
            Actual UP:      FN ❌       TP ✅
            ```
            """)

    # ==================== TAB 5: DOCUMENTATION ====================
    with tab5:
        st.markdown("### 📋 Project Documentation")

        with st.expander("📄 Abstract", expanded=True):
            st.markdown(
                "This project uses **Machine Learning** to predict stock market trends. "
                "We predict whether a stock's next-day closing price will go **UP** or **DOWN**. "
                "Three models (Logistic Regression, Decision Tree, Random Forest) are trained "
                "using proper time-series methodology. Accuracy of **50-56%** is realistic."
            )

        with st.expander("❓ Problem Statement"):
            st.markdown(
                "**Problem:** Can we predict next-day stock price direction using historical data "
                "and technical indicators with better-than-random accuracy?"
            )

        with st.expander("🔧 Methodology"):
            st.markdown("""
            1. **Data Collection** — Yahoo Finance API (5+ years)
            2. **Preprocessing** — Handle missing values, sort by date
            3. **Feature Engineering** — RSI, MACD, SMA, EMA, Volatility (20+ features)
            4. **Target Creation** — Binary: UP(1) / DOWN(0)
            5. **Time-Based Split** — 80% train (past), 20% test (future)
            6. **Feature Scaling** — StandardScaler on training data only
            7. **Model Training** — LR, DT, RF
            8. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix
            """)

        with st.expander("🛡️ Data Leakage Prevention"):
            st.markdown("""
            - ✅ Time-based split (not random)
            - ✅ Lag features use shift(1) only
            - ✅ Scaler fitted on training data only
            - ✅ No future data in any feature
            """)

        with st.expander("🔄 Overfitting Prevention"):
            st.markdown("""
            - ✅ max_depth limits tree growth
            - ✅ min_samples_split prevents tiny splits
            - ✅ Random Forest averages many trees
            - ✅ max_features forces feature diversity
            """)

        with st.expander("⚠️ Limitations"):
            st.markdown("""
            1. Only price/volume data (no news sentiment)
            2. Single stock training (may not generalize)
            3. No transaction costs considered
            4. Binary prediction only (not magnitude)
            5. NOT suitable for real trading
            """)

        with st.expander("🚀 Future Scope"):
            st.markdown("""
            1. 🧠 Deep Learning (LSTM/GRU)
            2. 📰 Sentiment Analysis
            3. 📊 Multi-Stock Portfolio
            4. ⚡ Real-Time Streaming
            5. 🤖 Reinforcement Learning
            """)

    # ==================== TAB 6: TEAM & CREDITS ====================
    with tab6:
        st.markdown("### 👥 Meet Our Team")
        st.markdown("The brilliant minds behind this project")

        tc1, tc2, tc3 = st.columns(3)

        team_data = [
            {
                "name": "Yash Verma",
                "role": "ML Engineer & Backend Developer",
                "emoji": "🧑‍💻",
                "contributions": "Designed ML pipeline, Implemented feature engineering, Trained all models, Data preprocessing",
                "color": "#667eea"
            },
            {
                "name": "Vishu Tarar",
                "role": "Data Analyst & Researcher",
                "emoji": "👨‍💻",
                "contributions": "Data collection, Exploratory data analysis, Technical indicator research, Model evaluation",
                "color": "#764ba2"
            },
            {
                "name": "Daksh Chaudhary",
                "role": "Frontend Developer & Deployment",
                "emoji": "👨‍🔬",
                "contributions": "Built Streamlit web app, Designed UI/UX, Deployed on Render, Created visualizations",
                "color": "#f093fb"
            }
        ]

        cols = [tc1, tc2, tc3]
        for col, member in zip(cols, team_data):
            with col:
                contributions_html = "<br>".join([f"• {c.strip()}" for c in member['contributions'].split(",")])
                st.markdown(
                    '<div style="background:white; padding:25px; border-radius:20px; '
                    'text-align:center; box-shadow:0 4px 15px rgba(0,0,0,0.1); '
                    'border-top:5px solid ' + member['color'] + '; min-height:380px;">'
                    '<div style="font-size:3.5rem; margin-bottom:10px;">' + member['emoji'] + '</div>'
                    '<div style="font-size:1.3rem; font-weight:700; color:#333;">' + member['name'] + '</div>'
                    '<div style="font-size:0.85rem; color:' + member['color'] + '; font-weight:600; margin-bottom:15px;">'
                    + member['role'] + '</div>'
                    '<hr style="border:1px solid #f0f0f0; margin:10px 0;">'
                    '<div style="text-align:left; font-size:0.85rem; color:#555;">'
                    '<strong style="color:#333;">Contributions:</strong><br>'
                    + contributions_html + '</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # Project Details
        st.markdown("### 📝 Project Details")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("""
            | Detail | Information |
            |--------|-----------|
            | **Project Title** | Stock Market Trend Prediction Using ML |
            | **Subject** | Mini Project |
            | **Type** | Machine Learning |
            | **Year** | 2026-2027 |
            """)
        with dc2:
            st.markdown("""
            | Technology | Used For |
            |-----------|----------|
            | **Python** | Programming |
            | **Scikit-learn** | ML models |
            | **Pandas/NumPy** | Data processing |
            | **Streamlit** | Web application |
            | **Render** | Cloud deployment |
            | **Yahoo Finance** | Stock data |
            """)

        st.markdown("---")

        # How System Works - FIXED COLOR
        st.markdown("### ⚙️ How the System Works")

        steps = [
            ("Step 1:", "📥 User enters a stock ticker (e.g., AAPL)"),
            ("Step 2:", "📊 System downloads latest 1-year data from Yahoo Finance"),
            ("Step 3:", "⚙️ Computes 20+ technical indicators (RSI, MACD, SMA, etc.)"),
            ("Step 4:", "🤖 Pre-trained ML model analyzes the indicators"),
            ("Step 5:", "🎯 Model outputs: UP ▲ or DOWN ▼ with confidence score"),
        ]

        for step_title, step_desc in steps:
            st.markdown(
                '<div style="background:#f0f2f6; padding:15px; border-radius:10px; '
                'border-left:4px solid #667eea; margin:8px 0;">'
                '<span style="color:#333333; font-weight:700;">' + step_title + '</span> '
                '<span style="color:#333333;">' + step_desc + '</span>'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Acknowledgments
        st.markdown("### 🙏 Acknowledgments")
        st.markdown(
            "We thank our **project guide** for continuous support, "
            "our **college** for resources, and the **open-source community** "
            "for Python, Scikit-learn, and Streamlit."
        )

    # FOOTER - FIXED COLOR
    st.markdown(
        '<div style="text-align:center; margin-top:3rem; padding:1.5rem; '
        'border-top:2px solid #e0e0e0; background:#f5f5f5; border-radius:10px;">'
        '<span style="color:#333; font-weight:600; font-size:0.95rem;">'
        '📈 Stock Market Trend Prediction Using Machine Learning</span><br>'
        '<span style="color:#555; font-size:0.85rem;">'
        '👨‍💻 Daksh Chaudhary &nbsp;|&nbsp; 👨‍💻 Vishu Tarar &nbsp;|&nbsp; '
        '👨‍🔬 Yash Verma</span><br><br>'
        '<span style="color:#888; font-size:0.8rem;">'
        '⚠️ Academic project only. Not financial advice.</span></div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()