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
    menu_items={'Get help': None, 'Report a bug': None, 'About': None}
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# --- SECTOR WATCHLISTS ---
SECTOR_WATCHLISTS = {
    "Large Cap Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "ORCL", "CRM", "AMD", "INTC", "CSCO", "ADBE", "NFLX"],
    "Semiconductors": ["NVDA", "AMD", "INTC", "AVGO", "TSM", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL"],
    "Financials": ["JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "SCHW", "CB"],
    "Healthcare": ["UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN"]
}

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
            baseline_options_volume = avg_stock_volume * 0.02
            baseline_premium = baseline_options_volume * 2.0 * 100
            return baseline_options_volume, baseline_premium
        return 50000, 1000000
    except:
        return 50000, 1000000

def calculate_iv_rank(ticker, current_iv):
    try:
        if current_iv <= 0: return 50
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) > 30:
            hist_vol = hist['Close'].pct_change().dropna().std() * np.sqrt(252)
            if hist_vol <= 0: return 50
            iv_ratio = current_iv / hist_vol
            if iv_ratio >= 2.5: return 95
            elif iv_ratio >= 2.0: return 85
            elif iv_ratio >= 1.5: return 70
            elif iv_ratio >= 1.0: return 50
            elif iv_ratio >= 0.7: return 30
            else: return 15
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
                    for opt_type, opt_df in [('Call', chain.calls), ('Put', chain.puts)]:
                        if opt_df.empty: continue
                        opt_df = opt_df.copy()
                        opt_df['ticker'], opt_df['expiry'], opt_df['type'] = ticker, exp, opt_type
                        opt_df['underlying_price'] = current_price
                        opt_df = opt_df.rename(columns={'openInterest': 'open_interest', 'impliedVolatility': 'iv'})
                        all_data.append(opt_df)
                except Exception as e:
                    failed_tick.append(f"{ticker} {exp} ({str(e)[:30]})")
                    continue
        except Exception as e:
            failed_tick.append(f"{ticker} ({str(e)[:30]})")
            continue

    if not all_data:
        return pd.DataFrame(), failed_tick
    
    df = pd.concat(all_data, ignore_index=True)
    df['volume'] = df['volume'].fillna(0).astype(int)
    df['open_interest'] = df['open_interest'].fillna(0).astype(int)
    df['iv'] = df['iv'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    df['lastPrice'] = df['lastPrice'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    df['premium'] = df['volume'] * df['lastPrice'] * 100
    
    # Fixed: Realistic proxy for daily volume (5% of Open Interest)
    df['avg_volume_20d'] = df['open_interest'].apply(lambda x: max(x * 0.05, 10.0))
    
    df['avg_oi_for_ticker'] = df.groupby('ticker')['open_interest'].transform('mean')
    df['is_block_trade'] = df.apply(lambda row: row['volume'] >= max(1000, row['avg_oi_for_ticker'] * 0.1), axis=1)
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
    
    # Fixed: Actually using the sidebar parameters
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
    result = df[df['is_unusual']].sort_values('unusualness_score', ascending=False)
    return result, failed_tick

# --- UI ---
st.title("🔍 Unusual Options Activity Scanner")
st.markdown("**Detect unusual options flow with institutional-grade metrics**")
st.info("⚠️ Volume data is snapshot-based. Historical options data requires paid feeds; stock volume used as proxy.")

with st.sidebar:
    st.header("⚙️ Configuration")
    tickers_input = st.text_input("Tickers (comma separated)", value="AAPL, NVDA, TSLA")
    
    st.subheader("Detection Thresholds")
    vol_oi = st.slider("Vol / OI Ratio", 1.0, 20.0, 3.0, 0.5)
    vol_avg = st.slider("Vol / Avg Vol Ratio", 1.0, 20.0, 5.0, 0.5)
    min_prem = st.number_input("Min Premium ($)", value=100000, step=50000)
    otm_pct = st.slider("OTM % Threshold (0 = all)", 0.0, 0.30, 0.0, 0.05)
    iv_mult = st.slider("IV Multiplier (1.0 = all)", 1.0, 5.0, 1.0, 0.1)
    
    scan_btn = st.button("🚀 Run Scan", type="primary", use_container_width=True)

# --- MAIN APP LOGIC ---
if scan_btn:
    with st.spinner("Fetching options data..."):
        df, failed_tickers = scan_options(tickers_input, vol_oi, vol_avg, min_prem, otm_pct, iv_mult)
    
    if failed_tickers:
        st.warning(f"⚠️ Failed to fetch: {', '.join(failed_tickers[:3])}")
    
    if df.empty:
        st.warning("No unusual activity detected.")
    else:
        st.success(f"Identified {len(df)} unusual contracts!")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏆 Sector Screener", " Historical Trend", "📊 Executive Summary", 
            "💰 Premium Flow", "📋 Contract Details", "🐋 Block Trades"
        ])
        
        # TAB 1: SECTOR SCREENER
        with tab1:
            st.subheader("🏆 Sector Unusual Activity Screener")
            st.markdown("Scans top companies in a sector to find the highest unusual options flow.")
            
            selected_sector = st.selectbox("Select Sector", list(SECTOR_WATCHLISTS.keys()))
            sector_tickers = SECTOR_WATCHLISTS[selected_sector]
            
            if st.button(f"Scan {selected_sector} Sector", type="primary"):
                progress_bar = st.progress(0)
                sector_results = []
                
                for i, ticker in enumerate(sector_tickers):
                    try:
                        # Unpack correctly to avoid errors
                        tkr_df, _ = scan_options(ticker, vol_oi, vol_avg, min_prem, otm_pct, iv_mult)
                        if not tkr_df.empty:
                            total_prem = float(tkr_df['premium'].sum())
                            total_vol = int(tkr_df['volume'].sum())
                            blocks = int(tkr_df['is_block_trade'].sum())
                            
                            call_vol = int(tkr_df[tkr_df['type'] == 'Call']['volume'].sum())
                            put_vol = int(tkr_df[tkr_df['type'] == 'Put']['volume'].sum())
                            bias = "🟢 BULLISH" if call_vol > put_vol else " BEARISH"
                            
                            sector_results.append({
                                'Ticker': ticker,
                                'Unusual Premium': total_prem,
                                'Unusual Volume': total_vol,
                                'Block Trades': blocks,
                                'Bias': bias
                            })
                    except:
                        pass
                    progress_bar.progress((i + 1) / len(sector_tickers))
                
                if sector_results:
                    sector_df = pd.DataFrame(sector_results).sort_values('Unusual Premium', ascending=False)
                    st.subheader(f"Top {min(10, len(sector_df))} Companies in {selected_sector}")
                    st.dataframe(sector_df.head(10).style.format({'Unusual Premium': '${:,.0f}', 'Unusual Volume': '{:,}'}), use_container_width=True, hide_index=True)
                else:
                    st.warning("No unusual activity found in this sector at current thresholds. Try lowering Min Premium.")

        # TAB 2: HISTORICAL TREND
        with tab2:
            st.subheader("📈 Historical Volume Trend")
            st.markdown("Identify the exact day or week where volume spiked massively.")
            
            hist_ticker = st.text_input("Enter Ticker for Historical View", value="AAPL").upper()
            time_range = st.radio("Select Time Range", ["Last 5 Days", "Last 30 Days", "Last 6 Months"], horizontal=True)
            
            if st.button("Generate Historical Graph", type="primary"):
                period_map = {"Last 5 Days": "5d", "Last 30 Days": "1mo", "Last 6 Months": "6mo"}
                period = period_map[time_range]
                
                with st.spinner(f"Fetching {time_range} data for {hist_ticker}..."):
                    hist_data = yf.Ticker(hist_ticker).history(period=period)
                    
                    if not hist_data.empty:
                        # Using stock volume as a proxy for options activity
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
                        st.error("Could not fetch historical data.")

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
                    'Bias': '🟢 BULLISH' if calls_prem > puts_prem else '🔴 BEARISH'
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
            st.download_button(" Download Dataset", data=csv, file_name="unusual_options.csv", mime="text/csv")

        # TAB 6: BLOCK TRADES
        with tab6:
            st.subheader("🐋 Block Trades & Sweeps")
            block_trades_df = df[df['is_block_trade'] | df['is_sweep']].copy()
            if not block_trades_df.empty:
                block_summary = []
                for _, row in block_trades_df.iterrows():
                    size_cat = '🐋 MEGA BLOCK' if row['volume'] >= 5000 else ('Large Block' if row['is_block_trade'] else ' Sweep')
                    block_summary.append({
                        'Ticker': row['ticker'], 'Type': row['type'], 'Strike': f"${row['strike']}",
                        'Expiry': row['expiry'], 'Moneyness': f"{row['moneyness']*100:+.1f}%",
                        'Volume': f"{int(row['volume']):,}", 'Premium': f"${float(row['premium']):,.0f}",
                        'Size Category': size_cat, 'Days to Expiry': int(row['days_to_expiry'])
                    })
                st.dataframe(pd.DataFrame(block_summary), use_container_width=True, hide_index=True)
            else:
                st.warning("No block trades detected.")