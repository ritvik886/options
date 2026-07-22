import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# Hide the Streamlit menu and footer
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.set_page_config(
    page_title="Unusual Options Activity Scanner",
    layout="wide",
    menu_items={
        'Get help': None,
        'Report a bug': None,
        'About': None
    }
)

# --- CONFIGURATION ---
DEFAULT_TICKERS = "AAPL, NVDA, TSLA"

def safe_ratio(num, denom, cap=10.0):
    return np.where(denom > 0, num / denom, np.where(num > 0, cap, 0.0))

def get_avg_iv(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="1mo")
        if len(hist) > 1:
            return float(hist['Close'].pct_change().dropna().std() * np.sqrt(252))
    except: pass
    return 0.25

@st.cache_data(ttl=3600)
def get_historical_options_volume(ticker, weeks_back=4):
    """Get average weekly options volume for the past X weeks"""
    try:
        tkr = yf.Ticker(ticker)
        expiries = tkr.options[:5]
        
        total_calls_vol = 0
        total_puts_vol = 0
        count = 0
        
        for exp in expiries:
            try:
                chain = tkr.option_chain(exp)
                if not chain.calls.empty:
                    total_calls_vol += chain.calls['volume'].fillna(0).sum()
                    total_puts_vol += chain.puts['volume'].fillna(0).sum()
                    count += 1
            except:
                continue
        
        if count > 0:
            avg_weekly_calls = total_calls_vol / count
            avg_weekly_puts = total_puts_vol / count
            return avg_weekly_calls, avg_weekly_puts
        return 0, 0
    except:
        return 0, 0

@st.cache_data(ttl=300)
def scan_options(tickers_str, vol_oi, vol_avg, min_prem, otm_pct, iv_mult):
    tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
    all_data = []
    
    for ticker in tickers:
        try:
            tkr = yf.Ticker(ticker)
            hist = tkr.history(period="5d")
            if hist.empty: continue
            current_price = float(hist['Close'].iloc[-1])
            avg_iv = get_avg_iv(ticker)
            
            for exp in tkr.options[:3]: 
                chain = tkr.option_chain(exp)
                for opt_type, df in [('Call', chain.calls), ('Put', chain.puts)]:
                    if df.empty: continue
                    df = df.copy()
                    df['ticker'], df['expiry'], df['type'] = ticker, exp, opt_type
                    df['underlying_price'], df['ticker_avg_iv'] = current_price, avg_iv
                    df = df.rename(columns={'openInterest': 'open_interest', 'impliedVolatility': 'iv'})
                    all_data.append(df)
        except: continue

    if not all_data: return pd.DataFrame()
    df = pd.concat(all_data, ignore_index=True)
    
    df['volume'] = df['volume'].fillna(0).astype(int)
    df['open_interest'] = df['open_interest'].fillna(0).astype(int)
    df['iv'], df['lastPrice'] = df['iv'].fillna(0.0), df['lastPrice'].fillna(0.0)
    df['avg_volume_20d'] = df['volume'].apply(lambda x: max(x * 0.2, 10.0))
    df['premium'] = df['volume'] * df['lastPrice'] * 100
    
    df['is_deep_otm'] = np.where(
        (df['type'] == 'Call') & (df['strike'] >= df['underlying_price'] * (1 + otm_pct)), True,
        np.where((df['type'] == 'Put') & (df['strike'] <= df['underlying_price'] * (1 - otm_pct)), True, False)
    )
    
    today = datetime.now().date()
    df['days_to_expiry'] = (pd.to_datetime(df['expiry']) - pd.Timestamp(today)).dt.days
    df['expiry_flag'] = np.where(df['days_to_expiry'] <= 14, '⚠️ <14d', 'OK')
    
    df['vol_oi_ratio'] = safe_ratio(df['volume'], df['open_interest'])
    df['vol_avg_ratio'] = safe_ratio(df['volume'], df['avg_volume_20d'])
    df['iv_spike'] = safe_ratio(df['iv'], df['ticker_avg_iv'])
    
    conditions = [
        df['volume'] >= vol_oi * df['open_interest'],
        df['volume'] >= vol_avg * df['avg_volume_20d'],
        df['premium'] >= min_prem,
        (df['is_deep_otm']) & (df['volume'] >= 50),
        df['iv_spike'] >= iv_mult
    ]
    df['is_unusual'] = np.logical_or.reduce(conditions)
    
    score = pd.Series(0.0, index=df.index)
    mask = df['vol_oi_ratio'] >= vol_oi
    score[mask] += 2.0 * (df.loc[mask, 'vol_oi_ratio'] / vol_oi)
    mask = df['vol_avg_ratio'] >= vol_avg
    score[mask] += 2.0 * (df.loc[mask, 'vol_avg_ratio'] / vol_avg)
    mask = df['premium'] >= min_prem
    score[mask] += 1.5 * (df.loc[mask, 'premium'] / min_prem)
    mask = df['is_deep_otm'] & (df['volume'] >= 50)
    score[mask] += 1.5
    mask = df['iv_spike'] >= iv_mult
    score[mask] += 1.0 * (df.loc[mask, 'iv_spike'] / iv_mult)
    mask = df['days_to_expiry'] <= 14
    score[mask] += 1.0
    
    df['unusualness_score'] = score
    return df[df['is_unusual']].sort_values('unusualness_score', ascending=False)

# --- USER INTERFACE ---
st.title(" Unusual Options Activity (UOA) Scanner")

with st.sidebar:
    st.header("⚙️ Configuration")
    tickers_input = st.text_input("Tickers (comma separated)", value=DEFAULT_TICKERS)
    
    st.subheader("Thresholds")
    vol_oi = st.slider("Vol / OI Ratio", 1.0, 10.0, 3.0, 0.5)
    vol_avg = st.slider("Vol / Avg Vol Ratio", 1.0, 10.0, 5.0, 0.5)
    min_prem = st.number_input("Min Premium ($)", value=100000, step=50000)
    otm_pct = st.slider("OTM % Threshold", 0.05, 0.25, 0.10, 0.05)
    iv_mult = st.slider("IV Spike Multiplier", 1.0, 3.0, 1.5, 0.1)
    
    scan_btn = st.button(" Run Scan", type="primary", use_container_width=True)

if scan_btn:
    with st.spinner("Fetching live options chains and analyzing..."):
        df = scan_options(tickers_input, vol_oi, vol_avg, min_prem, otm_pct, iv_mult)
    
    if df.empty:
        st.warning("No unusual activity detected for these tickers with current thresholds. Try widening the sliders in the sidebar.")
    else:
        st.success(f"Found {len(df)} unusual contracts!")
        
        # Call vs Put Volume Chart
        st.subheader("📊 Unusual Volume: Calls vs Puts")
        skew_data = []
        for ticker in df['ticker'].unique():
            tkr_df = df[df['ticker'] == ticker]
            c_vol = int(tkr_df[tkr_df['type'] == 'Call']['volume'].sum())
            p_vol = int(tkr_df[tkr_df['type'] == 'Put']['volume'].sum())
            skew_data.append({'Ticker': ticker, 'Type': 'Calls', 'Volume': c_vol})
            skew_data.append({'Ticker': ticker, 'Type': 'Puts', 'Volume': p_vol})
        
        skew_df = pd.DataFrame(skew_data)
        fig1 = px.bar(skew_df, x='Ticker', y='Volume', color='Type', 
                      barmode='group', title="Call vs Put Volume",
                      color_discrete_map={'Calls': '#00FF00', 'Puts': '#FF0000'})
        fig1.update_layout(template='plotly_dark')
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("---")
        
        # Historical Volume Comparison Table
        st.subheader(" Historical Volume Comparison (Current vs Normal)")
        st.markdown("Shows how much more volume is trading compared to normal weekly averages")
        
        comparison_data = []
        for ticker in df['ticker'].unique():
            tkr_df = df[df['ticker'] == ticker]
            current_calls = int(tkr_df[tkr_df['type'] == 'Call']['volume'].sum())
            current_puts = int(tkr_df[tkr_df['type'] == 'Put']['volume'].sum())
            
            hist_calls, hist_puts = get_historical_options_volume(ticker)
            
            if hist_calls > 0 and current_calls > 0:
                calls_multiplier = current_calls / hist_calls
            else:
                calls_multiplier = 0
            
            if hist_puts > 0 and current_puts > 0:
                puts_multiplier = current_puts / hist_puts
            else:
                puts_multiplier = 0
            
            comparison_data.append({
                'Ticker': ticker,
                'Current Call Volume': f"{current_calls:,}",
                'Normal Call Volume': f"{int(hist_calls):,}",
                'Call Multiplier': f"{calls_multiplier:.1f}x" if calls_multiplier > 0 else "N/A",
                'Current Put Volume': f"{current_puts:,}",
                'Normal Put Volume': f"{int(hist_puts):,}",
                'Put Multiplier': f"{puts_multiplier:.1f}x" if puts_multiplier > 0 else "N/A",
            })
        
        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Tabbed Interface for Tables
        tab1, tab2 = st.tabs([" Full Data & Export", "📊 Directional Skew Table"])
        
        with tab1:
            st.subheader("Complete Unusual Activity Dataset")
            display_cols = ['ticker', 'type', 'strike', 'expiry', 'volume', 'open_interest', 'iv', 'premium', 'unusualness_score', 'expiry_flag']
            st.dataframe(df[display_cols].round({'iv': 4, 'premium': 2, 'unusualness_score': 2}), use_container_width=True, hide_index=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(" Download Full Results as CSV", data=csv, file_name="uoa_scan_results.csv", mime="text/csv")

        with tab2:
            st.subheader("Net Unusual Volume: Calls vs. Puts")
            skew_table = []
            for ticker in df['ticker'].unique():
                tkr_df = df[df['ticker'] == ticker]
                c_vol = int(tkr_df[tkr_df['type'] == 'Call']['volume'].sum())
                p_vol = int(tkr_df[tkr_df['type'] == 'Put']['volume'].sum())
                skew_table.append({'Ticker': ticker, 'Call Vol': c_vol, 'Put Vol': p_vol, 'Net Skew (C-P)': c_vol - p_vol, 'Bias': '🟢 BULLISH' if (c_vol - p_vol) > 0 else '🔴 BEARISH' if (c_vol - p_vol) < 0 else '⚪ NEUTRAL'})
            st.dataframe(pd.DataFrame(skew_table), use_container_width=True, hide_index=True)