import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Gauri's Final DS Project", layout="wide")
st.title("Hybrid ARIMA + LSTM Stock Forecast Dashboard")

# -----------------------------------------------------------------------------
# Data loaders (cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_cleaned():
    """Load cleaned historical OHLC data."""
    return pd.read_csv(
        '/content/selected_tickers_clean.csv',
        parse_dates=['Date']
    )

@st.cache_data
def load_forecasts():
    """Load combined ARIMA+LSTM forecasts for May 1–5, 2025."""
    return pd.read_csv(
        '/content/may1_5_forecasts_v2.csv',
        parse_dates=['Date']
    )

@st.cache_data
def load_actual():
    """
    Load actual May 1–5, 2025 closes from wide-format sheet
    and melt into long form.
    """
    w = pd.read_csv('/content/Actual - Sheet1.csv')
    date_cols = [c for c in w.columns if c != 'Ticker']
    df = w.melt(
        id_vars=['Ticker'],
        value_vars=date_cols,
        var_name='Date',
        value_name='Actual'
    )
    df['Date'] = pd.to_datetime(df['Date'])
    return df[['Ticker','Date','Actual']]

clean_df     = load_cleaned()
forecasts_df = load_forecasts()
actual_df    = load_actual()

tickers = sorted(clean_df['Ticker'].unique())

# -----------------------------------------------------------------------------
# Sidebar: Ticker selection
# -----------------------------------------------------------------------------
selected_ticker = st.sidebar.selectbox("Select Ticker", tickers)

# Precompute date bounds
today      = clean_df['Date'].max()
hist_start = today - timedelta(days=365*6)
forecast_start = pd.Timestamp('2025-05-01')
forecast_end   = pd.Timestamp('2025-05-05')

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 6-Year Trend",
    "🔮 Forecast vs Actual",
    "⚙️ Model Efficiency",
    "📊 Train/Test Comparison"
])

# -----------------------------------------------------------------------------
# Tab 1: 6-Year Trendline
# -----------------------------------------------------------------------------
with tab1:
    st.subheader(f"{selected_ticker} — Last 6 Years Close Price")
    hist = clean_df[
        (clean_df['Ticker'] == selected_ticker) &
        (clean_df['Date'] >= hist_start)
    ]
    fig = px.line(
        hist,
        x='Date',
        y='Close',
        title=f"{selected_ticker} Close Price (Last 6 Years)",
        labels={'Close':'Price','Date':'Date'}
    )
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 2: Forecast vs Actual
# -----------------------------------------------------------------------------
with tab2:
    st.subheader(f"Forecast vs Actual (May 1–5, 2025) — {selected_ticker}")
    # slice forecast & actual
    fc = forecasts_df[forecasts_df['Ticker'] == selected_ticker]
    ac = actual_df   [actual_df   ['Ticker'] == selected_ticker]
    # merge on Date
    merged = pd.merge(fc, ac, on='Date', how='inner')
    # plot overlay
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=merged['Date'], y=merged['Combined_forecast'],
        mode='lines+markers', name='Forecast'
    ))
    fig2.add_trace(go.Bar(
        x=merged['Date'], y=merged['Actual'],
        name='Actual', opacity=0.6
    ))
    fig2.update_layout(
        title="Forecast vs Actual",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 3: Model Efficiency
# -----------------------------------------------------------------------------
with tab3:
    st.subheader(f"Model Efficiency — {selected_ticker}")
    # re-merge to ensure local context
    fc   = forecasts_df[forecasts_df['Ticker'] == selected_ticker]
    ac   = actual_df   [actual_df   ['Ticker'] == selected_ticker]
    merged = pd.merge(fc, ac, on='Date', how='inner')
    # compute efficiency
    merged['Efficiency'] = 1 - (merged['Combined_forecast'] - merged['Actual']).abs() / merged['Actual']
    avg_eff = merged['Efficiency'].mean()
    # gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_eff * 100,
        number={'suffix': "%"},
        gauge={'axis': {'range': [0,100]}},
        title={'text': "Avg Daily Efficiency"}
    ))
    st.plotly_chart(gauge, use_container_width=True)
    # daily line
    eff_fig = go.Figure(go.Scatter(
        x=merged['Date'], y=merged['Efficiency']*100,
        mode='lines+markers', name='Efficiency (%)'
    ))
    eff_fig.update_layout(
        title="Daily Efficiency (%)",
        yaxis={'title': 'Efficiency (%)'}
    )
    st.plotly_chart(eff_fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 4: Train/Test Comparison
# -----------------------------------------------------------------------------
with tab4:
    st.subheader(f"Train vs Test Efficiency — {selected_ticker}")
    # in-sample train
    hist_train = clean_df[
        (clean_df['Ticker'] == selected_ticker) &
        (clean_df['Date'] < forecast_start)
    ]
    arima_model = ARIMA(hist_train['Close'], order=(3,1,5)).fit()
    resid       = hist_train['Close'] - arima_model.fittedvalues
    train_eff   = (1 - resid.abs() / hist_train['Close']).mean()
    # out-of-sample test = avg from tab3
    test_eff    = avg_eff
    df_eff = pd.DataFrame({
        'Period': ['Train','Test'],
        'Efficiency (%)': [train_eff * 100, test_eff * 100]
    })
    fig3 = go.Figure(go.Bar(
        x=df_eff['Period'],
        y=df_eff['Efficiency (%)'],
        text=df_eff['Efficiency (%)'].round(2),
        textposition='auto'
    ))
    fig3.update_layout(
        title="Train vs Test Efficiency (%)",
        yaxis={'range': [0,100]}
    )
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "**App:** Gauri's Final Data Science Project  |  "
    "Hybrid ARIMA(3,1,5) + LSTM (20-day)  |  6-year history"
)
