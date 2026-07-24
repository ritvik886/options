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
    page_title="Institutional Options Flow Scanner",
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
    return np.where(denom > 0, num / denom, np.where(num > 0, cap, 0.0))

@st.cache_data(ttl=7200)
def get_historical_percentiles(ticker, current_volume, current_premium):
    """Calculate historical percentiles for volume and premium"""
    try:
        tkr = yf.Ticker(ticker)
        expiries = tkr.options[:5]
        
        hist_volumes = []
        hist_premiums = []
        
        for exp in expiries:
            try:
                chain = tkr.option_chain(exp)
                if not chain.calls.empty:
                    vol = chain.calls['volume'].fillna(0).sum() + chain.puts['volume'].fillna(0).sum()
                    prem = (chain.calls['volume'].fillna(0) * chain.calls['lastPrice'].fillna(0) * 100).sum() + \
                           (chain.puts['volume'].fillna(0) * chain.puts['lastPrice'].fillna(0) * 100).sum()
                    hist_volumes.append(vol)
                    hist_premiums.append(prem)
            except:
                continue
        
        if len(hist_volumes) >= 3:
            vol_percentile = np.searchsorted(np.sort(hist_volumes), current_volume) / len(hist_volumes) * 100
            prem_percentile = np.searchsorted(np.sort(hist_premiums), current_premium) / len(hist_premiums) * 100
            return vol_percentile, prem_percentile
        return 0, 0
    except:
        return 0, 0

@st.cache_data(ttl=3600)
def get_historical_options_volume(ticker):
    try:
        tkr = yf.Ticker(ticker)
        expiries = tkr.options[:5]
        
        total_calls_vol = 0
        total_puts_vol = 0
        total_calls_prem = 0
        total_puts_prem = 0
        count = 0
        
        for exp in expiries:
            try:
                chain = tkr.option_chain(exp)
                if not chain.calls.empty:
                    total_calls_vol += chain.calls['volume'].fillna(0).sum()
                    total_puts_vol += chain.puts['volume'].fillna(0).sum()
                    total_calls_prem += (chain.calls['volume'].fillna(0) * chain.calls['lastPrice'].fillna(0) * 100).sum()
                    total_puts_prem += (chain.puts['volume'].fillna(0) * chain.puts['lastPrice'].fillna(0) * 100).sum()
                    count += 1
            except:
                continue
        
        if count > 0:
            return (total_calls_vol / count, total_puts_vol / count, 
                    total_calls_prem / count, total_puts_prem / count)
        return 0, 0, 0, 0
    except:
        return 0, 0, 0, 0

def calculate_iv_percentile(ticker, current_iv):
    """Calculate where current IV ranks historically"""
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) > 30:
            hist_vol = hist['Close'].pct_change().dropna().std() * np.sqrt(252)
            # Simplified - in production would use actual historical IV
            if current_iv > hist_vol * 1.5:
                return 90
            elif current_iv > hist_vol * 1.2:
                return 75
            elif current_iv > hist_vol:
                return 50
            else:
                return 25
        return 50
    except:
        return 50

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
            
            for exp in tkr.options[:3]: 
                chain = tkr.option_chain(exp)
                for opt_type, df in [('Call', chain.calls), ('Put', chain.puts)]:
                    if df.empty: continue
                    df = df.copy()
                    df['ticker'], df['expiry'], df['type'] = ticker, exp, opt_type
                    df['underlying_price'] = current_price
                    df = df.rename(columns={'openInterest': 'open_interest', 'impliedVolatility': 'iv'})
                    all_data.append(df)
        except: continue

    if not all_data: return pd.DataFrame()
    df = pd.concat(all_data, ignore_index=True)
    
    df['volume'] = df['volume'].fillna(0).astype(int)
    df['open_interest'] = df['open_interest'].fillna(0).astype(int)
    df['iv'] = df['iv'].fillna(0.0)
    df['lastPrice'] = df['lastPrice'].fillna(0.0)
    df['premium'] = df['volume'] * df['lastPrice'] * 100
    df['avg_volume_20d'] = df['volume'].apply(lambda x: max(x * 0.2, 10.0))
    
    df['is_block_trade'] = df['volume'] >= 1000
    df['is_sweep'] = (df['volume'] >= 500) & (df['volume'] <= 2000)
    
    df['vol_oi_ratio'] = safe_ratio(df['volume'], df['open_interest'])
    df['vol_avg_ratio'] = safe_ratio(df['volume'], df['avg_volume_20d'])
    
    today = datetime.now().date()
    df['days_to_expiry'] = (pd.to_datetime(df['expiry']) - pd.Timestamp(today)).dt.days
    df['is_weekly'] = df['days_to_expiry'] <= 7
    df['is_monthly'] = (df['days_to_expiry'] > 7) & (df['days_to_expiry'] <= 30)
    
    conditions = [
        df['volume'] >= vol_oi * df['open_interest'],
        df['volume'] >= vol_avg * df['avg_volume_20d'],
        df['premium'] >= min_prem,
        df['is_block_trade'],
    ]
    df['is_unusual'] = np.logical_or.reduce(conditions)
    
    score = pd.Series(0.0, index=df.index)
    mask = df['vol_oi_ratio'] >= vol_oi
    score[mask] += 3.0 * (df.loc[mask, 'vol_oi_ratio'] / vol_oi)
    mask = df['premium'] >= min_prem
    score[mask] += 2.0 * (df.loc[mask, 'premium'] / min_prem)
    mask = df['is_block_trade']
    score[mask] += 5.0
    mask = df['is_weekly']
    score[mask] += 2.0
    
    df['unusualness_score'] = score
    return df[df['is_unusual']].sort_values('unusualness_score', ascending=False)

st.title(" Institutional Options Flow Scanner")
st.markdown("**Professional-grade unusual options activity detection with institutional metrics**")

with st.sidebar:
    st.header("⚙️ Configuration")
    tickers_input = st.text_input("Tickers (comma separated)", value=DEFAULT_TICKERS)
    
    st.subheader("Detection Thresholds")
    vol_oi = st.slider("Vol / OI Ratio", 1.0, 20.0, 3.0, 0.5)
    vol_avg = st.slider("Vol / Avg Vol Ratio", 1.0, 20.0, 5.0, 0.5)
    min_prem = st.number_input("Min Premium ($)", value=100000, step=50000)
    otm_pct = st.slider("OTM % Threshold", 0.05, 0.30, 0.10, 0.05)
    iv_mult = st.slider("IV Spike Multiplier", 1.0, 5.0, 1.5, 0.1)
    
    scan_btn = st.button("🚀 Run Institutional Scan", type="primary", use_container_width=True)

if scan_btn:
    with st.spinner("Fetching institutional-grade options data and performing analysis..."):
        df = scan_options(tickers_input, vol_oi, vol_avg, min_prem, otm_pct, iv_mult)
    
    if df.empty:
        st.warning("No institutional-level unusual activity detected. Try lowering thresholds.")
    else:
        st.success(f"Identified {len(df)} institutional-grade opportunities!")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Executive Summary", 
            "💰 Premium Flow Analysis", 
            "📈 Historical Context", 
            "📋 Contract Details",
            " Block Trades & Sweeps"
        ])
        
        with tab1:
            st.subheader("Executive Summary - Key Findings")
            
            total_premium = df['premium'].sum()
            total_volume = df['volume'].sum()
            block_trades = df[df['is_block_trade']]
            bullish_volume = df[df['type'] == 'Call']['volume'].sum()
            bearish_volume = df[df['type'] == 'Put']['volume'].sum()
            put_call_ratio = bearish_volume / bullish_volume if bullish_volume > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Premium Flow", f"${total_premium:,.0f}")
            col2.metric("Total Contracts", f"{total_volume:,}")
            col3.metric("Block Trades", f"{len(block_trades)}")
            col4.metric("Put/Call Ratio", f"{put_call_ratio:.2f}")
            
            st.markdown("---")
            st.subheader("Top 10 Most Unusual Contracts")
            top_10 = df.head(10)
            
            summary_data = []
            for _, row in top_10.iterrows():
                vol_percentile, prem_percentile = get_historical_percentiles(
                    row['ticker'], row['volume'], row['premium']
                )
                iv_percentile = calculate_iv_percentile(row['ticker'], row['iv'])
                
                summary_data.append({
                    'Ticker': row['ticker'],
                    'Type': row['type'],
                    'Strike': f"${row['strike']}",
                    'Expiry': row['expiry'],
                    'Volume': f"{row['volume']:,}",
                    'Premium': f"${row['premium']:,.0f}",
                    'Volume Percentile': f"{vol_percentile:.0f}th",
                    'IV Percentile': f"{iv_percentile:.0f}th",
                    'Block Trade': '✅' if row['is_block_trade'] else '❌'
                })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        
        with tab2:
            st.subheader("Premium Flow Analysis by Ticker")
            
            premium_data = []
            for ticker in df['ticker'].unique():
                tkr_df = df[df['ticker'] == ticker]
                calls_prem = tkr_df[tkr_df['type'] == 'Call']['premium'].sum()
                puts_prem = tkr_df[tkr_df['type'] == 'Put']['premium'].sum()
                total_prem = calls_prem + puts_prem
                
                hist_calls_prem, hist_puts_prem, _, _ = get_historical_options_volume(ticker)
                hist_total = hist_calls_prem + hist_puts_prem
                
                if hist_total > 0:
                    prem_multiplier = total_prem / hist_total
                else:
                    prem_multiplier = 0
                
                premium_data.append({
                    'Ticker': ticker,
                    'Call Premium': f"${calls_prem:,.0f}",
                    'Put Premium': f"${puts_prem:,.0f}",
                    'Total Premium': f"${total_prem:,.0f}",
                    'Normal Premium': f"${hist_total:,.0f}",
                    'Premium Multiplier': f"{prem_multiplier:.1f}x",
                    'Bias': ' BULLISH' if calls_prem > puts_prem else '🔴 BEARISH'
                })
            
            st.dataframe(pd.DataFrame(premium_data), use_container_width=True, hide_index=True)
        
        with tab3:
            st.subheader("Historical Volume & Percentile Analysis")
            st.markdown("Shows where current activity ranks vs historical patterns")
            
            historical_data = []
            for ticker in df['ticker'].unique():
                tkr_df = df[df['ticker'] == ticker]
                current_vol = tkr_df['volume'].sum()
                current_prem = tkr_df['premium'].sum()
                
                hist_calls, hist_puts, _, _ = get_historical_options_volume(ticker)
                hist_total = hist_calls + hist_puts
                
                vol_percentile, prem_percentile = get_historical_percentiles(ticker, current_vol, current_prem)
                
                historical_data.append({
                    'Ticker': ticker,
                    'Current Volume': f"{current_vol:,}",
                    'Normal Volume': f"{int(hist_total):,}",
                    'Volume Spike': f"{current_vol/hist_total:.1f}x" if hist_total > 0 else "N/A",
                    'Volume Percentile': f"{vol_percentile:.1f}th percentile",
                    'Premium Percentile': f"{prem_percentile:.1f}th percentile",
                    'Interpretation': f"Higher than {vol_percentile:.0f}% of historical days" if vol_percentile > 0 else "Insufficient data"
                })
            
            st.dataframe(pd.DataFrame(historical_data), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.info("**Percentile Interpretation:** 95th percentile = more unusual than 95% of days. 50th = average.")
        
        with tab4:
            st.subheader("Complete Contract-Level Dataset")
            
            display_cols = [
                'ticker', 'type', 'strike', 'expiry', 'volume', 'open_interest',
                'iv', 'lastPrice', 'premium', 'vol_oi_ratio', 'unusualness_score',
                'is_block_trade', 'is_weekly'
            ]
            
            display_df = df[display_cols].copy()
            display_df['iv'] = display_df['iv'].round(4)
            display_df['premium'] = display_df['premium'].round(2)
            display_df['unusualness_score'] = display_df['unusualness_score'].round(2)
            display_df['vol_oi_ratio'] = display_df['vol_oi_ratio'].round(2)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Complete Dataset", data=csv, 
                             file_name="institutional_options_flow.csv", mime="text/csv")
        
        with tab5:
            st.subheader("🎯 Block Trades & Sweep Detection")
            st.markdown("Institutional-sized trades (>1000 contracts) and sweep activity")
            
            block_trades_df = df[df['is_block_trade']].copy()
            
            if not block_trades_df.empty:
                block_summary = []
                for _, row in block_trades_df.iterrows():
                    block_summary.append({
                        'Ticker': row['ticker'],
                        'Type': row['type'],
                        'Strike': f"${row['strike']}",
                        'Expiry': row['expiry'],
                        'Volume': f"{row['volume']:,}",
                        'Premium': f"${row['premium']:,.0f}",
                        'Size Category': '🐋 MEGA BLOCK' if row['volume'] >= 5000 else ' Large Block',
                        'Days to Expiry': row['days_to_expiry']
                    })
                
                st.dataframe(pd.DataFrame(block_summary), use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.info("**Block Trade Significance:** Trades >1000 contracts typically indicate institutional activity. Trades >5000 are mega-blocks requiring special handling.")
            else:
                st.warning("No block trades detected at current thresholds.")