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
   