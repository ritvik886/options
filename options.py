import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Unusual Options Activity Scanner",
    layout="wide",
    menu_items={
        'Get help': None,
        'Report a bug': None,
        'About': None
    }
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

DEFAULT_TICKERS = "AAPL, NVDA, TSLA"

def safe_ratio(num, denom, cap=100.0):
    if isinstance(num, pd.Series):
        return np.where(denom > 0, num / denom, np.where(num > 0, cap, 0.0))
    else:
        return num / denom if denom > 0 else cap

@st.cache_data(ttl=3600)
def get_baseline_volume(ticker):
    """
    Get realistic baseline volume based on stock's typical options activity.
    Uses stock volume to estimate normal options flow (typically 1-5% of stock volume)
    """
    try:
        tkr = yf.Ticker(ticker)
        hist = tkr.history(period="3mo")
        if not hist.empty:
            avg_stock_volume = hist['Volume'].mean()
            # Options volume is typically 1-5% of stock volume
            baseline_options_volume = avg_stock_volume * 0.02  # 2% baseline
            baseline_premium = baseline_options_volume * 2.0 * 100  # rough avg option price
            return baseline_options_volume, baseline_premium
        return 50000, 1000000  # Default baseline
    except:
        return 50000, 1000000

def calculate_iv_rank(ticker, current_iv):
    """
    Estimate IV rank by comparing to realized volatility.
    Note: This is an estimate since yfinance doesn't provide historical IV data.
    """
    try:
        if current_iv <= 0:
            return 50
        
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) > 30:
            hist_vol = hist['Close'].pct_change().dropna().std() * np.sqrt(252)
            
            if hist_vol <= 0:
                return 50
            
            iv_ratio = current_iv / hist_vol
            
            # Convert ratio to approximate rank
            if iv_ratio >= 2.5:
                return 95
            elif iv_ratio >= 2.0:
                return 85
            elif iv_ratio >= 1.5:
                return 70
            elif iv_ratio >= 1.0:
                return 50
            elif iv_ratio >= 0.7:
                return 30
            else:
                return 15
        return 50
    except:
        return 50

@st.cache_data(ttl=300)
def scan_options(tickers_str, vol_oi, vol_avg, min_prem, otm_pct, iv_mult):
    tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
    all_data = []
    failed_tick = []
    
    for ticker in tickers:
        try:
            tkr = yf.Ticker(ticker)
            hist = tkr.history(period="5d")
            if hist.empty:
                failed_tick.append(f"{ticker} (no price data)")
                continue
            current_price = float(hist['Close'].iloc[-1])
            
            for exp in tkr.options[:3]: 
                try:
                    chain = tkr.option_chain(exp)
                    for opt_type, df in [('Call', chain.calls), ('Put', chain.puts)]:
                        if df.empty:
                            continue
                        df = df.copy()
                        df['ticker'], df['expiry'], df['type'] = ticker, exp, opt_type
                        df['underlying_price'] = current_price
                        df = df.rename(columns={'openInterest': 'open_interest', 'impliedVolatility': 'iv'})
                        all_data.append(df)
                except Exception as e:
                    failed_tick.append(f"{ticker} {exp} ({str(e)[:50]})")
                    continue
        except Exception as e:
            failed_tick.append(f"{ticker} ({str(e)[:50]})")
            continue

    if not all_data:
        return pd.DataFrame(), failed_tick
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Fill NaN values safely
    df['volume'] = df['volume'].fillna(0).astype(int)
    df['open_interest'] = df['open_interest'].fillna(0).astype(int)
    df['iv'] = df['iv'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    df['lastPrice'] = df['lastPrice'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    # Calculate premium
    df['premium'] = df['volume'] * df['lastPrice'] * 100
    
    # FIX #1: Use open_interest as proxy for typical daily volume (5% of OI)
    # This is more realistic than using today's volume
    df['avg_volume_20d'] = df['open_interest'].apply(lambda x: max(x * 0.05, 10.0))
    
    # Block trade detection - scaled by ticker liquidity
    df['avg_oi_for_ticker'] = df.groupby('ticker')['open_interest'].transform('mean')
    df['is_block_trade'] = df.apply(
        lambda row: row['volume'] >= max(1000, row['avg_oi_for_ticker'] * 0.1), axis=1
    )
    
    # Sweep detection - medium-sized aggressive trades
    df['is_sweep'] = (df['volume'] >= 500) & (df['volume'] <= 2000)
    
    # Calculate moneyness (how far OTM/ITM)
    df['moneyness'] = (df['strike'] - df['underlying_price']) / df['underlying_price']
    
    # Calculate ratios safely
    df['vol_oi_ratio'] = safe_ratio(df['volume'], df['open_interest'])
    df['vol_avg_ratio'] = safe_ratio(df['volume'], df['avg_volume_20d'])
    
    # Time calculations
    today = datetime.now().date()
    df['days_to_expiry'] = (pd.to_datetime(df['expiry']) - pd.Timestamp(today)).dt.days
    df['days_to_expiry'] = df['days_to_expiry'].fillna(30).astype(int)
    df['is_weekly'] = df['days_to_expiry'] <= 7
    df['is_monthly'] = (df['days_to_expiry'] > 7) & (df['days_to_expiry'] <= 30)
    
    # Build conditions list
    conditions = [
        df['volume'] >= vol_oi * df['open_interest'],
        df['volume'] >= vol_avg * df['avg_volume_20d'],
        df['premium'] >= min_prem,
        df['is_block_trade'],
    ]
    
    # FIX #2: Actually use otm_pct parameter
    if otm_pct > 0:
        otm_filter = np.where(
            (df['type'] == 'Call') & (df['moneyness'] >= otm_pct), True,
            np.where((df['type'] == 'Put') & (df['moneyness'] <= -otm_pct), True, False)
        )
        conditions.append(otm_filter)
    
    # FIX #2: Actually use iv_mult parameter
    if iv_mult > 1.0:
        iv_filter = df['iv'] >= (iv_mult * 0.25)  # 0.25 is rough baseline IV
        conditions.append(iv_filter)
    
    df['is_unusual'] = np.logical_or.reduce(conditions)
    
    # FIX #8: Normalize score components to [0,1] before weighting
    score = pd.Series(0.0, index=df.index)
    
    # Volume/OI ratio component (normalized)
    mask = df['vol_oi_ratio'] >= vol_oi
    normalized_vol_oi = (df.loc[mask, 'vol_oi_ratio'] / vol_oi).clip(0, 10) / 10
    score[mask] += 3.0 * normalized_vol_oi
    
    # Premium component (normalized)
    mask = df['premium'] >= min_prem
    normalized_prem = (df.loc[mask, 'premium'] / min_prem).clip(0, 10) / 10
    score[mask] += 2.5 * normalized_prem
    
    # Block trade component
    mask = df['is_block_trade']
    score[mask] += 4.0
    
    # Sweep component (FIX #5: Actually use sweep detection)
    mask = df['is_sweep']
    score[mask] += 1.5
    
    # Weekly expiry component
    mask = df['is_weekly']
    score[mask] += 2.0
    
    # OTM component (higher score for more speculative bets)
    otm_score = np.abs(df['moneyness']).clip(0, 0.2) / 0.2  # Normalize to [0,1]
    score += 1.5 * otm_score
    
    df['unusualness_score'] = score
    
    # Return only unusual contracts, sorted by score
    result = df[df['is_unusual']].sort_values('unusualness_score', ascending=False)
    
    if result.empty:
        return pd.DataFrame(), failed_tick
    
    return result, failed_tick

st.title(" Unusual Options Activity Scanner")
st.markdown("**Detect unusual options flow with institutional-grade metrics**")
st.info("⚠️ Volume data is snapshot-based and may not reflect intraday accumulation")

with st.sidebar:
    st.header("️ Configuration")
    tickers_input = st.text_input("Tickers (comma separated)", value=DEFAULT_TICKERS)
    
    st.subheader("Detection Thresholds")
    vol_oi = st.slider("Vol / OI Ratio", 1.0, 20.0, 3.0, 0.5)
    vol_avg = st.slider("Vol / Avg Vol Ratio", 1.0, 20.0, 5.0, 0.5)
    min_prem = st.number_input("Min Premium ($)", value=100000, step=50000)
    otm_pct = st.slider("OTM % Threshold (0 = include all)", 0.0, 0.30, 0.0, 0.05)
    iv_mult = st.slider("IV Multiplier (1.0 = include all)", 1.0, 5.0, 1.0, 0.1)
    
    scan_btn = st.button(" Run Scan", type="primary", use_container_width=True)

if scan_btn:
    with st.spinner("Fetching options data and analyzing..."):
        df, failed_tickers = scan_options(tickers_input, vol_oi, vol_avg, min_prem, otm_pct, iv_mult)
    
    # FIX #6: Show failed tickers
    if failed_tickers:
        st.warning(f"⚠️ Failed to fetch: {', '.join(failed_tickers[:5])}" + 
                  (f" and {len(failed_tickers)-5} more" if len(failed_tickers) > 5 else ""))
    
    if df.empty:
        st.warning("No unusual activity detected. Try lowering thresholds.")
    else:
        st.success(f"Identified {len(df)} unusual contracts!")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            " Executive Summary", 
            " Premium Flow", 
            "📋 Contract Details",
            " Block Trades"
        ])
        
        with tab1:
            st.subheader("Executive Summary - Key Findings")
            
            total_premium = float(df['premium'].sum())
            total_volume = int(df['volume'].sum())
            block_trades = df[df['is_block_trade']]
            sweep_trades = df[df['is_sweep']]
            bullish_volume = int(df[df['type'] == 'Call']['volume'].sum())
            bearish_volume = int(df[df['type'] == 'Put']['volume'].sum())
            put_call_ratio = bearish_volume / bullish_volume if bullish_volume > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Premium Flow", f"${total_premium:,.0f}")
            col2.metric("Total Contracts", f"{total_volume:,}")
            col3.metric("Block Trades", f"{len(block_trades)}")
            col4.metric("Put/Call Ratio", f"{put_call_ratio:.2f}")
            
            # FIX: Add Put/Call Volume Chart
            st.subheader("Call vs Put Volume")
            call_put_data = []
            for ticker in df['ticker'].unique():
                tkr_df = df[df['ticker'] == ticker]
                call_vol = int(tkr_df[tkr_df['type'] == 'Call']['volume'].sum())
                put_vol = int(tkr_df[tkr_df['type'] == 'Put']['volume'].sum())
                call_put_data.append({'Ticker': ticker, 'Type': 'Calls', 'Volume': call_vol})
                call_put_data.append({'Ticker': ticker, 'Type': 'Puts', 'Volume': put_vol})
            
            call_put_df = pd.DataFrame(call_put_data)
            fig = px.bar(call_put_df, x)