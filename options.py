
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import plotly.express as px
import plotly.graph_objects as go
import time

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Unusual Options Activity Scanner",
    layout="wide",
    menu_items={'Get help': None, 'Report a bug': None, 'About': None}
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# 10 Core Large Cap Tech Stocks (Reduced from 15 to prevent timeouts)
LARGE_CAP_TECH = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "AMD", "NFLX"]

def safe_ratio(num, denom, cap=100.0):
    if isinstance(num, pd.Series):
        return np.where(denom > 0, num / denom, np.where(num > 0, cap, 0.0))
    else:
        return num / denom if denom > 0 else cap

@st.cache_data(ttl=3600)
def get_baseline_volume(ticker):
    try:
        tkr = yf.Ticker(ticker)
        hist = tkr.history(period="3mo")
        if not hist.empty:
            avg_stock_volume = hist['Volume'].mean()
            return avg_stock_volume * 0.02, avg_stock_volume * 0.02 * 2.0 * 100
        return 50000, 1000000
    except:
        return 50000, 1000000

# REMOVED CACHE HERE TO PREVENT STREAMLIT FROM REMEMBERING ERRORS
def scan_single_ticker(ticker, vol_oi, vol_avg, min_prem, otm_pct, iv_mult):
    """Scans a single ticker safely"""
    try:
        tkr = yf.Ticker(ticker)
        hist = tkr.history(period="5d")
        if hist.empty: return pd.DataFrame()
        current_price = float(hist['Close'].iloc[-1])
        
        all_data = []
        for exp in tkr.options[:2]: 
            try:
                chain = tkr.option_chain(exp)
                for opt_type, opt_df in [('Call', chain.calls), ('Put', chain.puts)]:
                    if opt_df.empty: continue
                    opt_df = opt_df.copy()
                    opt_df['ticker'], opt_df['expiry'], opt_df['type'] = ticker, exp, opt_type
                    opt_df['underlying_price'] = current_price
                    opt_df = opt_df.rename(columns={'openInterest': 'open_interest', 'impliedVolatility': 'iv'})
                    all_data.append(opt_df)
            except: continue

        if not all_data: return pd.DataFrame()
        df = pd.concat(all_data, ignore_index=True)
        
        df['volume'] = df['volume'].fillna(0).astype(int)
        df['open_interest'] = df['open_interest'].fillna(0).astype(int)
        df['iv'] = df['iv'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        df['lastPrice'] = df['lastPrice'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        df['premium'] = df['volume'] * df['lastPrice'] * 100
        df['avg_volume_20d'] = df['open_interest'].apply(lambda x: max(x * 0.05, 10.0))
        
        avg_oi = df['open_interest'].mean()
        df['is_block_trade'] = df['volume'] >= max(1000, avg_oi * 0.1)
        df['is_sweep'] = (df['volume'] >= 500) & (df['volume'] <= 2000)
        df['moneyness'] = (df['strike'] - df['underlying_price']) / df['underlying_price']
        
        df['vol_oi_ratio'] = safe_ratio(df['volume'], df['open_interest'])
        df['vol_avg_ratio'] = safe_ratio(df['volume'], df['avg_volume_20d'])
        
        today = datetime.now().date()
        df['days_to_expiry'] = (pd.to_datetime(df['expiry']) - pd.Timestamp(today)).dt.days
        df['days_to_expiry'] = df['days_to_expiry'].fillna(30).astype(int)
        df['is_weekly'] = df['days_to_expiry'] <= 7
        
        conditions = [
            df['volume'] >= vol_oi * df['open_interest'],
            df['volume'] >= vol_avg * df['avg_volume_20d'],
            df['premium'] >= min_prem,
            df['is_block_trade'],
        ]
        
        if otm_pct > 0:
            otm_filter = np.where((df['type'] == 'Call') & (df['moneyness'] >= otm_pct), True,
                                  np.where((df['type'] == 'Put') & (df['moneyness'] <= -otm_pct), True, False))
            conditions.append(otm_filter)
        if iv_mult > 1.0:
            conditions.append(df['iv'] >= (iv_mult * 0.25))
        
        df['is_unusual'] = np.logical_or.reduce(conditions)
        
        score = pd.Series(0.0, index=df.index)
        mask = df['vol_oi_ratio'] >= vol_oi
        score[mask] += 3.0 * (df.loc[mask, 'vol_oi_ratio'] / vol_oi).clip(0, 10) / 10
        mask = df['premium'] >= min_prem
        score[mask] += 2.5 * (df.loc[mask, 'premium'] / min_prem).clip(0, 10) / 10
        mask = df['is_block_trade']
        score[mask] += 4.0
        mask = df['is_sweep']
        score[mask] += 1.5
        mask = df['is_weekly']
        score[mask] += 2.0
        score += 1.5 * (np.abs(df['moneyness']).clip(0, 0.2) / 0.2)
        
        df['unusualness_score'] = score
        return df[df['is_unusual']].sort_values('unusualness_score', ascending=False)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def scan_options_batch(tickers_str, vol_oi, vol_avg, min_prem, otm_pct, iv_mult):
    tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
    all_data = []
    
    for ticker in tickers:
        df = scan_single_ticker(ticker, vol_oi, vol_avg, min_prem, otm_pct, iv_mult)
        if not df.empty:
            all_data.append(df)
            
    if not all_data: return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

# --- UI ---
st.title(" Unusual Options Activity Scanner")
st.markdown("**Detect unusual options flow with institutional-grade metrics**")

with st.sidebar:
    st.header("️ Configuration")
    tickers_input = st.text_input("Tickers (comma separated)", value="AAPL, NVDA, TSLA")
    
    st.subheader("Detection Thresholds")
    vol_oi = st.slider("Vol / OI Ratio", 1.0, 20.0, 3.0, 0.5)
    vol_avg = st.slider("Vol / Avg Vol Ratio", 1.0, 20.0, 5.0, 0.5)
    min_prem = st.number_input("Min Premium ($)", value=50000, step=10000)
    otm_pct = st.slider("OTM % Threshold (0 = all)", 0.0, 0.30, 0.0, 0.05)
    iv_mult = st.slider("IV Multiplier (1.0 = all)", 1.0, 5.0, 1.0, 0.1)
    
    scan_btn = st.button("🚀 Run Scan", type="primary", use_container_width=True)

# --- MAIN APP LOGIC ---
if scan_btn:
    with st.spinner("Fetching options data..."):
        df = scan_options_batch(tickers_input, vol_oi, vol_avg, min_prem, otm_pct, iv_mult)
    
    if df.empty:
        st.warning("No unusual activity detected. Try lowering Min Premium.")
    else:
        st.success(f"Identified {len(df)} unusual contracts!")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏆 Large Cap Tech Screener", "📈 Historical Trend", "📊 Executive Summary", 
            "💰 Premium Flow", "📋 Contract Details", " Block Trades"
        ])
        
        # TAB 1: SECTOR SCREENER
        with tab1:
            st.subheader("🏆 Large Cap Tech Unusual Activity Screener")
            st.markdown(f"Scans {len(LARGE_CAP_TECH)} major tech stocks. (Takes ~15 seconds)")
            
            if st.button("Scan Full Large Cap Tech Sector", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                sector_results = []
                
                for i, ticker in enumerate(LARGE_CAP_TECH):
                    status_text.text(f"Scanning {ticker} ({i+1}/{len(LARGE_CAP_TECH)})...")
                    
                    tkr_df = scan_single_ticker(ticker, vol_oi, vol_avg, min_prem, otm_pct, iv_mult)
                    
                    if not tkr_df.empty:
                        total_prem = float(tkr_df['premium'].sum())
                        total_vol = int(tkr_df['volume'].sum())
                        blocks = int(tkr_df['is_block_trade'].sum())
                        call_vol = int(tkr_df[tkr_df['type'] == 'Call']['volume'].sum())
                        put_vol = int(tkr_df[tkr_df['type'] == 'Put']['volume'].sum())
                        bias = "🟢 BULLISH" if call_vol > put_vol else "🔴 BEARISH"
                        
                        sector_results.append({
                            'Ticker': ticker,
                            'Unusual Premium': total_prem,
                            'Unusual Volume': total_vol,
                            'Block Trades': blocks,
                            'Bias': bias
                        })
                    
                    # 1 FULL SECOND SLEEP TO PREVENT YAHOO BLOCKING US
                    time.sleep(1) 
                    progress_bar.progress((i + 1) / len(LARGE_CAP_TECH))
                
                status_text.text("Scan Complete!")
                
                if sector_results:
                    sector_df = pd.DataFrame(sector_results).sort_values('Unusual Premium', ascending=False)
                    st.subheader("Top Companies by Unusual Premium")
                    st.dataframe(sector_df.style.format({'Unusual Premium': '${:,.0f}', 'Unusual Volume': '{:,}'}), use_container_width=True, hide_index=True)
                else:
                    st.warning("No unusual activity found. Try lowering Min Premium in the sidebar to $10,000.")

        # TAB 2: HISTORICAL TREND (FIXED WITH YF.TICKER)
        with tab2:
            st.subheader("📈 Historical Volume Trend")
            st.markdown("Identify the exact day or week where volume spiked massively.")
            
            hist_ticker = st.text_input("Enter Ticker for Historical View", value="AAPL").upper()
            time_range = st.radio("Select Time Range", ["Last 5 Days", "Last 30 Days", "Last 6 Months"], horizontal=True)
            
            if st.button("Generate Historical Graph", type="primary"):
                period_map = {"Last 5 Days": "5d", "Last 30 Days": "1mo", "Last 6 Months": "6mo"}
                period = period_map[time_range]
                
                with st.spinner(f"Fetching {time_range} data for {hist_ticker}..."):
                    try:
                        # USING YF.TICKER.HISTORY() TO AVOID MULTIINDEX BUGS
                        hist_data = yf.Ticker(hist_ticker).history(period=period)
                        
                        if not hist_data.empty and 'Volume' in hist_data.columns:
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=hist_data.index,
                                y=hist_data['Volume'],
                                name='Underlying Stock Volume (Proxy)',
                                marker_color='rgba(0, 255, 0, 0.6)'
                            ))
                            
                            fig.update_layout(
                                title=f"{hist_ticker} Historical Volume ({time_range})",
                                xaxis_title="Date",
                                yaxis_title="Volume",
                                template="plotly_dark",
                                hovermode="x unified"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            max_day = hist_data['Volume'].idxmax()
                            max_vol = hist_data.loc[max_day, 'Volume']
                            st.success(f"🔍 **Biggest Spike Detected:** {max_day.strftime('%Y-%m-%d')} with {int(max_vol):,} shares traded.")
                        else:
                            st.error("Could not fetch volume data. Check ticker symbol.")
                    except Exception as e:
                        st.error(f"Graph failed to load. Error: {str(e)}")

        # TAB 3: EXECUTIVE SUMMARY
        with tab3:
            st.subheader("Executive Summary - Key Findings")
            total_premium = float(df['premium'].sum())
            total_volume = int(df['volume'].sum())
            block_trades = df[df['is_block_trade']]
            bullish_volume = int(df[df['type'] == 'Call']['volume'].sum())
            bearish_volume = int(df[df['type'] == 'Put']['volume'].sum())
            put_call_ratio = bearish_volume / bullish_volume if bullish_volume > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Premium Flow", f"${total_premium:,.0f}")
            col2.metric("Total Contracts", f"{total_volume:,}")
            col3.metric("Block Trades", f"{len(block_trades)}")
            col4.metric("Put/Call Ratio", f"{put_call_ratio:.2f}")
            
            st.subheader("Call vs Put Volume")
            call_put_data = []
            for ticker in df['ticker'].unique():
                tkr_df = df[df['ticker'] == ticker]
                call_put_data.append({'Ticker': ticker, 'Type': 'Calls', 'Volume': int(tkr_df[tkr_df['type'] == 'Call']['volume'].sum())})
                call_put_data.append({'Ticker': ticker, 'Type': 'Puts', 'Volume': int(tkr_df[tkr_df['type'] == 'Put']['volume'].sum())})
            
            fig = px.bar(pd.DataFrame(call_put_data), x='Ticker', y='Volume', color='Type',
                         barmode='group', color_discrete_map={'Calls': '#00FF00', 'Puts': '#FF0000'})
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

        # TAB 4: PREMIUM FLOW
        with tab4:
            st.subheader("Premium Flow Analysis by Ticker")
            premium_data = []
            for ticker in df['ticker'].unique():
                tkr_df = df[df['ticker'] == ticker]
                calls_prem = float(tkr_df[tkr_df['type'] == 'Call']['premium'].sum())
                puts_prem = float(tkr_df[tkr_df['type'] == 'Put']['premium'].sum())
                total_prem = calls_prem + puts_prem
                _, baseline_prem = get_baseline_volume(ticker)
                prem_multiplier = min(total_prem / baseline_prem, 100.0) if baseline_prem > 0 else 0
                
                premium_data.append({
                    'Ticker': ticker, 'Call Premium': f"${calls_prem:,.0f}", 'Put Premium': f"${puts_prem:,.0f}",
                    'Total Premium': f"${total_prem:,.0f}", 'Baseline Premium': f"${baseline_prem:,.0f}",
                    'Premium Multiplier': f"{prem_multiplier:.1f}x",
                    'Bias': ' BULLISH' if calls_prem > puts_prem else '🔴 BEARISH'
                })
            st.dataframe(pd.DataFrame(premium_data), use_container_width=True, hide_index=True)

        # TAB 5: CONTRACT DETAILS
        with tab5:
            st.subheader("Complete Contract-Level Dataset")
            display_cols = ['ticker', 'type', 'strike', 'expiry', 'moneyness', 'volume', 'open_interest', 'iv', 'premium', 'vol_oi_ratio', 'unusualness_score', 'is_block_trade', 'is_sweep']
            display_df = df[display_cols].copy()
            display_df['moneyness'] = display_df['moneyness'].apply(lambda x: f"{x*100:+.1f}%")
            display_df['premium'] = display_df['premium'].round(2)
            display_df['unusualness_score'] = display_df['unusualness_score'].round(2)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Dataset", data=csv, file_name="unusual_options.csv", mime="text/csv")

        # TAB 6: BLOCK TRADES
        with tab6:
            st.subheader("🐋 Block Trades & Sweeps")
            block_trades_df = df[df['is_block_trade'] | df['is_sweep']].copy()
            if not block_trades_df.empty:
                block_summary = []
                for _, row in block_trades_df.iterrows():
                    size_cat = '🐋 MEGA BLOCK' if row['volume'] >= 5000 else ('Large Block' if row['is_block_trade'] else '🔄 Sweep')
                    block_summary.append({
                        'Ticker': row['ticker'], 'Type': row['type'], 'Strike': f"${row['strike']}",
                        'Expiry': row['expiry'], 'Moneyness': f"{row['moneyness']*100:+.1f}%",
                        'Volume': f"{int(row['volume']):,}", 'Premium': f"${float(row['premium']):,.0f}",
                        'Size Category': size_cat, 'Days to Expiry': int(row['days_to_expiry'])
                    })
                st.dataframe(pd.DataFrame(block_summary), use_container_width=True, hide_index=True)
            else:
                st.warning("No block trades detected.")