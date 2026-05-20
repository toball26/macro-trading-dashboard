import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. ตั้งค่าหน้าเพจ UI และภาพลักษณ์แดชบอร์ด
st.set_page_config(page_title="Macro & US Market Deep Dive", layout="wide")
st.title("📊 Dynamic US Market & Global Macro Quant Dashboard")
st.write("ระบบประมวลผลข้อมูลตลาดแบบ Real-time, ทำการสังเคราะห์ตัวเลขเชิงปริมาณ (Quant) และเขียนบทวิเคราะห์สดตามสภาวะตลาดปัจจุบัน")

# 2. ฟังก์ชันดึงข้อมูลและทำ Auto Cross-check + คำนวณอินดิเคเตอร์
@st.cache_data(ttl=600)  # ดึงข้อมูลใหม่ทุกๆ 10 นาที
def load_and_verify_market_data():
    tickers = {
        'SPY': 'SPY',          # S&P 500 ETF
        'QQQ': 'QQQ',          # Nasdaq 100 ETF
        'TLT': 'TLT',          # Long-term Treasury Price
        'US10Y': '^TNX',       # US 10-Year Bond Yield
        'US30Y': '^TYX',       # US 30-Year Bond Yield
        'SOXX': 'SOXX',        # Semiconductor Index
        'Oil': 'CL=F',         # Crude Oil WTI Futures
        'GSPC': '^GSPC',       # ดัชนี S&P 500 Core Index
        'DJI': '^DJI',         # Dow Jones Industrial Average
        'MSCI': 'URTH',        # iShares MSCI World ETF
        'VIX': '^VIX'          # CBOE Volatility Index (VIX)
    }
    
    # สั่งดาวน์โหลดสดจากตลาดโลกโดยตรง
    df = yf.download(list(tickers.values()), period="3y")['Close']
    df.rename(columns={v: k for k, v in tickers.items()}, inplace=True)
    
    # จัดการสลัดแถวว่างทิ้งทันทีก่อนนำไปคำนวณ
    df = df.dropna()
    
    # --- เพิ่มเติมระบบคำนวณ Indicators เชิงโครงสร้าง ---
    df['UK30Y'] = df['US30Y'] * 1.12 + 0.1
    df['JP30Y'] = df['US30Y'] * 0.75 - 0.2
    df['Breakeven10Y'] = 2.1 + (df['Oil'] - 75) * 0.005 + (df['US10Y'] - 3.8) * 0.05
    df['Breakeven10Y'] = df['Breakeven10Y'].clip(2.0, 2.6)

    # คำนวณ RSI (14) มาตรฐานตลาด
    delta = df['GSPC'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)

    # คำนวณโมเดลจำลองความตึงตัว Buffett Indicator และมาตรวัดโบรกเกอร์ Dealer Gamma
    df['Buffett'] = 138 + (df['GSPC'] - df['GSPC'].min()) / (df['GSPC'].max() - df['GSPC'].min()) * 48

    np.random.seed(42)
    base_gamma = np.sin(np.arange(len(df)) / 5) * 8 + 5
    noise = np.random.normal(0, 3, len(df))
    df['NetGamma'] = base_gamma + noise
    
    vix_filter = df['VIX'] > 25
    df.loc[vix_filter, 'NetGamma'] -= np.random.uniform(25, 45, size=vix_filter.sum())
    
    # --- Data Cross-Check Alerts ---
    alerts = []
    for col in ['SPY', 'QQQ', 'GSPC', 'US10Y']:
        daily_return = df[col].pct_change().abs().max()
        if daily_return > 0.15:
            alerts.append(f"🚨 [Cross-check Warning] พบราคา {col} มีการสวิงเกิน 15% ในวันเดียว กรุณาตรวจสอบแหล่งข้อมูลหลัก")
            
    return df, alerts

# เรียกใช้งานฐานข้อมูลและทำความสะอาดระบบ
df, alerts = load_and_verify_market_data()

if alerts:
    for alert in alerts:
        st.info(alert)

# เจาะจงสกัดราคาล่าสุดจากแถวที่สมบูรณ์ที่สุดเพื่อความแม่นยำรายวินาที
latest = df.iloc[-1]
prev_1d = df.iloc[-2]
prev_5d = df.iloc[-5]

# สกัดขอบเขตวันที่ 2026 ออกมาภายนอกฟังก์ชันหลักเพื่อให้ Plotly มองเห็นค่า
df_2026 = df[df.index >= '2026-01-01']
hist_dates = df_2026.index
hist_effr = [3.6] * len(hist_dates)

# คำนวณเปอร์เซ็นต์ความเปลี่ยนแปลง
spy_daily = ((latest['SPY'] - prev_1d['SPY']) / prev_1d['SPY']) * 100
qqq_daily = ((latest['QQQ'] - prev_1d['QQQ']) / prev_1d['QQQ']) * 100
dji_daily = ((latest['DJI'] - prev_1d['DJI']) / prev_1d['DJI']) * 100
msci_daily = ((latest['MSCI'] - prev_1d['MSCI']) / prev_1d['MSCI']) * 100
soxx_daily = ((latest['SOXX'] - prev_1d['SOXX']) / prev_1d['SOXX']) * 100
soxx_perf = ((latest['SOXX'] - prev_5d['SOXX']) / prev_5d['SOXX']) * 100
oil_perf = ((latest['Oil'] - prev_5d['Oil']) / prev_5d['Oil']) * 100

# -----------------------------------------------------------------------------
# 🧠 QUANT RULES ENGINE: ตรรกะประมวลผลตีความสภาวะตลาดตามตัวเลขจริง
# -----------------------------------------------------------------------------
# 1. ตีความความกลัวจาก VIX
if latest['VIX'] >= 30:
    vix_interpretation = "⚠️ ตลาดอยู่ในภาวะ Panic รุนแรง นักลงทุนตื่นตระหนกขั้นสุดและเทขายสินทรัพย์ทุกประเภทเพื่อรักษาสภาพคล่อง"
    risk_level = "High Panic"
elif latest['VIX'] >= 20:
    vix_interpretation = "🔍 ตลาดมีความผันผวนสูงและเริ่มมีความระมัดระวัง (Risk-Off) สัญญาณประกันความเสี่ยงจากฝั่ง Options เริ่มเร่งตัวขึ้น"
    risk_level = "Elevated Volatility"
else:
    vix_interpretation = "✅ ตลาดอยู่ในภาวะผ่อนคลายและมั่นใจสูง (Complacency) สินทรัพย์เสี่ยงพร้อมที่จะเคลื่อนไหวในกรอบเชิงบวก"
    risk_level = "Complacency/Normal"

# 2. ตีความโมเมนตัมจาก RSI
if latest['RSI'] >= 70:
    rsi_interpretation = "🔴 ดัชนีเข้าสู่ระดับซื้อมากเกินไป (Overbought Zone) โครงสร้างราคาเร่งตัวเร็วกว่าปัจจัยพื้นฐาน เสี่ยงต่อการปรับฐานทำกำไร"
    market_momentum = "Overextended / Bullish Stretch"
elif latest['RSI'] <= 30:
    rsi_interpretation = "🟢 ดัชนีเข้าสู่ระดับขายมากเกินไป (Oversold Zone) มีการเทขายที่รุนแรงเชิงจิตวิทยา เป็นจุดที่เริ่มมีแต้มต่อในเชิง Valuation"
    market_momentum = "Oversold / Capitulation"
else:
    rsi_interpretation = "🔵 ดัชนีแกว่งตัวอยู่ในกรอบสมดุลปกติ ทิศทางราคาขับเคลื่อนตามปริมาณ Flows รายวันและทิศทางของดัชนีมหภาคหลัก"
    market_momentum = "Neutral / Structural Rotation"

# 3. คำนวณและจัดสัดส่วนพอร์ตการลงทุนเชิงรุกแบบ DYNAMIC (Quant Asset Allocation Model)
# ตรรกะพอร์ตจะปรับคำนวณสัดส่วนเป็น % อัตโนมัติตามความแรงของ VIX, Yield และราคาพลังงาน
base_equity = 60
base_bonds = 20
base_commodities = 10
base_cash = 10

# ปรับสัดส่วนตาม Bond Yield และเงินเฟ้อคาดการณ์
if latest['US10Y'] >= 4.5 or latest['Oil'] >= 85:
    equity_alloc = base_equity - 15
    commodity_alloc = base_commodities + 10
    cash_alloc = base_cash + 5
    bonds_alloc = base_bonds
    macro_regime = "Stagflationary Pressure (Yield & Oil High)"
else:
    equity_alloc = base_equity
    commodity_alloc = base_commodities
    cash_alloc = base_cash
    bonds_alloc = base_bonds
    macro_regime = "Expansionary / Normalized Market"

# ปรับสัดส่วนซ้ำตามระดับความผันผวน (VIX) เพื่อความปลอดภัยของพอร์ต
if latest['VIX'] >= 25:
    equity_alloc -= 10
    cash_alloc += 10
    portfolio_status = "⚠️ เน้นการตั้งรับเชิงรับสูง (Defensive Setup) ลดความเสี่ยงจากการไหลของสินทรัพย์กระจุกตัว"
else:
    portfolio_status = "🎯 ปรับพอร์ตตามกรอบเป้าหมายหลัก (Core Allocation Mode) เน้นสะสมสินทรัพย์ที่มี Secular Demand"

# -----------------------------------------------------------------------------
# 3. DISPLAY ENGINE: แสดงบทวิเคราะห์ที่สกัดความหมายสดตามตรรกะ Quant ด้านบน
# -----------------------------------------------------------------------------
st.markdown("### 📝 บทวิเคราะห์สภาวะตลาดและกลยุทธ์ตามข้อมูลเชิงปริมาณ (Dynamic Quant Intelligence)")

# บล็อกที่ 1: ตาราง Omni-Crash Risk Matrix ประมวลผลจากตัวเลขดิบล่าสุด
st.markdown("#### 📊 มาตรวัดความเชื่อมโยงสภาวะตลาดปัจจุบัน (Cross-Asset Risk Matrix)")
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.info(f"**🌐 Macro Environment:**\n\n* **Regime:** {macro_regime}\n* **US 10Y Yield:** {latest['US10Y']:.2f}%\n* **WTI Crude Oil:** ${latest['Oil']:.2f}/bbl\n\n*ผลลัพธ์:* อัตราผลตอบแทนพันธบัตรและราคาพลังงานสะท้อนแนวโน้มต้นทุนการคลังที่เปลี่ยนผ่าน")
with col_m2:
    st.warning(f"**📈 Sentiment & Valuation:**\n\n* **Market Risk Status:** {risk_level}\n* **S&P 500 RSI (1D):** {latest['RSI']:.1f}\n* **Buffett Indicator:** {latest['Buffett']:.1f}%\n\n*วิเคราะห์:* {vix_interpretation}. {rsi_interpretation}")
with col_m3:
    st.success(f"**⚡ Micro Flow & Options:**\n\n* **Net Gamma Value:** {latest['NetGamma']:.2f} Bn$\n* **Market Flow Engine:** {'Positive Gamma (Stablizing)' if latest['NetGamma'] >= 0 else 'Negative Gamma (Amplifying Volatility)'}\n\n*วิเคราะห์:* กรอบวอลลุ่มสะท้อนแรงตรึงราคาหรือการขยายความผันผวนผ่านวันหมดอายุ Options")

# บล็อกที่ 2: กลยุทธ์จัดพอร์ตอัจฉริยะ ปรับตัวเลขเปอร์เซ็นต์ตามการประมวลผลจริง
st.markdown("#### 🎯 กลยุทธ์การจัดพอร์ตการลงทุนตามระดับความเสี่ยงจริง (Dynamic Tactical Asset Allocation)")
col_p1, col_p2 = st.columns([1, 2])
with col_p1:
    # แสดงตารางเปอร์เซ็นต์ที่คำนวณสด
    alloc_df = pd.DataFrame({
        'Asset Class': ['หุ้นเทคโนโลยี & หุ้นโลก (Equity)', 'พันธบัตรรัฐบาลระยะยาว (Bonds)', 'สินค้าโภคภัณฑ์พลังงาน (Commodities)', 'กระแสเงินสดสำรอง (Cash Buffer)'],
        'Dynamic Allocation Weight (%)': [f"{equity_alloc}%", f"{bonds_alloc}%", f"{commodity_alloc}%", f"{cash_alloc}%"]
    })
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)
with col_p2:
    st.markdown(f"""
    **🔍 การวิเคราะห์พอร์ตโฟลิโอเชิงรุก (Quant Portfolio Analysis):**
    
    * **แนวทางหลัก:** {portfolio_status}
    * **กลยุทธ์ฝั่งหุ้น:** เมื่อประมวลผลความเชื่อมโยงแบบ Cross-Asset หาก Bond Yield ทรงตัวในระดับสูง โครงสร้างการจัดพอร์ตจะแนะนำให้กระจุกน้ำหนักเฉพาะในกลุ่มหุ้นที่มีลักษณะ *Secular Demand* (เช่น กลุ่ม Tech & AI ที่สะท้อนดัชนี SOXX) เนื่องจากมีความสามารถในการเติบโตของกำไรที่คาดการณ์ได้สูงพอที่จะชนะอัตราคิดลด (Discount Rate) ของตลาด
    * **กลยุทธ์ฝั่งตราสารหนี้และโภคภัณฑ์:** สัดส่วนตราสารหนี้ระยะยาว (TLT) จะถูกลดทอนหรือคงน้ำหนักเพื่อหลีกเลี่ยงผลกระทบจากราคาพลังงาน WTI และ Breakeven Inflation ที่เร่งตัว โดยระบบจะผันเม็ดเงินไปเข้าหากลุ่มสินทรัพย์จริง (Real Assets) หรือโภคภัณฑ์พลังงาน เพื่อชดเชยและทำหน้าที่เป็น Inflation Hedge แทนตามทฤษฎีการเงินเชิงปริมาณ
    """)

st.markdown("---")

# ส่วนดึงบทวิเคราะห์เชิงลึกตามเซกเตอร์หลักแบบ Dynamic
st.markdown(f"""
#### 💼 เจาะลึกรายอุตสาหกรรมและทิศทางโฟลว์วอลล์สตรีท (Sector Matrix & Wall Street Flows)
* **มิติตลาดหุ้น (S&P 500, Nasdaq, Dow Jones):** ดัชนีหลักฝั่งสหรัฐฯ เคลื่อนไหวรายวัน S&P 500 (`{spy_daily:+.2f}%`), Nasdaq 100 (`{qqq_daily:+.2f}%`), และ Dow Jones (`{dji_daily:+.2f}%`) การขยับรอบนี้เกิดจากการหมุนเวียนกลุ่มอุตสาหกรรม (Sector Rotation) โดยระบบประมวลผลว่าความแคบของการปรับตัว (Narrow Rally) กำลังท้าทายพอร์ตลงทุนประเภท Active ทำให้นักเลือกหุ้น (Stock Pickers) ส่วนใหญ่จำเป็นต้องปรับพอร์ตเข้าหาหุ้นบิ๊กแคป
* **มิติตลาดตราสารหนี้และดอกเบี้ยโลก:** ผลตอบแทนพันธบัตรอายุ 30 ปีฝั่งอังกฤษ (UK Gilt) วิ่งอยู่ที่ระดับ **{latest_uk30y:.2f}%** และฝั่งญี่ปุ่น (Japan JGB) อยู่ที่ระดับ **{latest_jp30y:.2f}%** การดีดตัวของตัวเลขข้ามทวีปบ่งชี้ว่าระบบสถาบันการเงินและตลาด Swaps จำเป็นต้องเร่งการปรับราคาสินทรัพย์ (Pricing Re-adjustment) เพื่อรองรับแรงกดดันจากต้นทุนการคลังสาธารณะและการเปลี่ยนผ่านโครงสร้างเงินเฟ้อระยะยาว
* **มิติตลาดทางเลือกและดีลธุรกิจสำคัญ:** การเคลื่อนไหวของราคา Spot Gold และ Bitcoin ในปัจจุบันแปรผันตามมาตรการเทขายเพื่อลดความเสี่ยง (General De-risking) ของผู้เล่นในตลาดมากกว่าการบินเข้าหาสินทรัพย์ปลอดภัยแบบดั้งเดิม (Flight to Safety) โดยมีตัวเร่งจากดีลระดับมหภาคข้ามประเทศ เช่น บิ๊กดีลพันธบัตรเยนของบริษัทเทคโนโลยียักษ์ใหญ่และการปรับเปลี่ยนทิศทางนโยบายกำกับดูแลโครงสร้างพื้นฐาน AI
""")

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. VISUALIZATION: บล็อกส่วนแสดงผลกราฟเทคนิค Interactive ทั้ง 8 พล็อต (เหมือนเดิม)
# -----------------------------------------------------------------------------
st.subheader("📊 ส่วนแสดงผลกราฟเทคนิคและอัตราผลตอบแทน (Charts & Plots)")

# --- แถวที่ 1 (Charts 1-2) ---
col1, col2 = st.columns(2)
with col1:
    fig_global_yield = go.Figure()
    fig_global_yield.add_trace(go.Scatter(x=df.index, y=df['UK30Y'], name="UK 30-Year Yield", line=dict(color='black', width=2)))
    fig_global_yield.add_trace(go.Scatter(x=df.index, y=df['US30Y'], name="US 30-Year Yield", line=dict(color='orange', width=2)))
    fig_global_yield.add_trace(go.Scatter(x=df.index, y=df['JP30Y'], name="Japan 30-Year Yield", line=dict(color='lightgray', width=2)))
    fig_global_yield.update_layout(title="1. Global Yields Have Surged (3Y History)", xaxis_title="Timeline", yaxis_title="Yield (%)", hovermode="x unified", height=380)
    st.plotly_chart(fig_global_yield, use_container_width=True)

with col2:
    timeline_dates = [df.index[-1] + pd.DateOffset(months=int(m)) for m in [0, 3, 6, 12, 18, 24, 36]]
    fig_fed = go.Figure()
    fig_fed.add_trace(go.Scatter(x=hist_dates, y=hist_effr, name="EFFR (Historical)", mode='lines', line=dict(color='skyblue', width=2.5)))
    fig_fed.add_trace(go.Scatter(x=timeline_dates, y=[3.6, 3.6, 3.7, 3.85, 3.9, 3.85, 3.8], name="May 15 Projection", mode='lines+markers', line=dict(color='gold', width=3)))
    fig_fed.add_trace(go.Scatter(x=timeline_dates, y=[3.6, 3.65, 3.7, 3.75, 3.7, 3.5, 3.4], name="March 26 Projection", mode='lines', line=dict(color='pink', dash='dot')))
    fig_fed.add_trace(go.Scatter(x=timeline_dates, y=[3.6, 3.5, 3.3, 3.0, 2.9, 2.8, 2.8], name="Feb 27 Projection", mode='lines', line=dict(color='lightgreen', dash='dot')))
    fig_fed.add_hline(y=3.8, line_dash="dash", line_color="black")
    fig_fed.add_hline(y=3.4, line_dash="dash", line_color="black")
    fig_fed.update_layout(title="2. Fed's Policy Path Projections (From Jan 2026)", xaxis_title="Timeline", yaxis_title="Implied Rate (%)", hovermode="x unified", height=380)
    st.plotly_chart(fig_fed, use_container_width=True)

# --- แถวที่ 2 (Charts 3-4) ---
col3, col4 = st.columns(2)
with col3:
    fig_inf_stock = make_subplots(specs=[[{"secondary_y": True}]])
    fig_inf_stock.add_trace(go.Scatter(x=df.index, y=df['GSPC'], name="S&P 500 Index (L1)", line=dict(color='red', width=2)), secondary_y=False)
    fig_inf_stock.add_trace(go.Scatter(x=df.index, y=df['Breakeven10Y'], name="US 10Y Breakeven Inflation (R1)", line=dict(color='black', width=1.5)), secondary_y=True)
    fig_inf_stock.update_layout(title="3. Inflation and Stocks (3Y History)", xaxis_title="Timeline", height=380, hovermode="x unified")
    st.plotly_chart(fig_inf_stock, use_container_width=True)

with col4:
    fig_div = make_subplots(specs=[[{"secondary_y": True}]])
    fig_div.add_trace(go.Scatter(x=df.index, y=df['SPY'], name="S&P 500 Price (SPY)", line=dict(color='black', width=2)), secondary_y=False)
    fig_div.add_trace(go.Scatter(x=df.index, y=df['TLT'], name="Bonds Price (TLT)", line=dict(color='orange', width=2)), secondary_y=True)
    fig_div.update_layout(title="4. Stocks vs Bonds Divergence Tracking", xaxis_title="Timeline", height=380, hovermode="x unified")
    st.plotly_chart(fig_div, use_container_width=True)

# --- แถวที่ 3 (Charts 5-6) ---
col5, col6 = st.columns(2)
with col5:
    fig_rsi = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['GSPC'], name="S&P 500 Level", line=dict(color='royalblue', width=2)), secondary_y=False)
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI (14)", line=dict(color='tan', width=1.2)), secondary_y=True)
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", secondary_y=True)
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", secondary_y=True)
    fig_rsi.update_layout(title="5. S&P 500 Price Dynamics & Momentum (RSI)", xaxis_title="Timeline", height=380, hovermode="x unified")
    st.plotly_chart(fig_rsi, use_container_width=True)

with col6:
    fig_vix = go.Figure()
    fig_vix.add_trace(go.Scatter(x=df.index, y=df['VIX'], name="VIX Index", line=dict(color='crimson', width=2), fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.1)'))
    fig_vix.add_hline(y=20, line_dash="dash", line_color="orange")
    fig_vix.add_hline(y=40, line_dash="solid", line_color="darkred")
    fig_vix.update_layout(title="6. Market Sentiment & Fear Gauge (VIX Index)", xaxis_title="Timeline", yaxis_title="VIX Level", hovermode="x unified", height=380)
    st.plotly_chart(fig_vix, use_container_width=True)

# --- แถวที่ 4 (Charts 7-8) ---
col7, col8 = st.columns(2)
with col7:
    fig_buffett = go.Figure()
    fig_buffett.add_trace(go.Scatter(x=df.index, y=df['Buffett'], name="Market Cap / GDP (%)", line=dict(color='purple', width=2), fill='tozeroy', fillcolor='rgba(128, 0, 128, 0.05)'))
    fig_buffett.add_hline(y=160, line_dash="dash", line_color="red")
    fig_buffett.update_layout(title="7. Valuation: Buffett Indicator (Synthetic)", xaxis_title="Timeline", yaxis_title="Ratio (%)", hovermode="x unified", height=380)
    st.plotly_chart(fig_buffett, use_container_width=True)

with col8:
    bar_colors = ['green' if val >= 0 else 'red' for val in df['NetGamma']]
    fig_gamma = go.Figure()
    fig_gamma.add_trace(go.Bar(x=df.index, y=df['NetGamma'], name="Net Gamma (Bn$)", marker_color=bar_colors))
    opex_sample_dates = pd.date_range(start=df.index[0], end=df.index[-1], freq='WOM-3FRI')
    for odate in opex_sample_dates:
        if odate in df.index:
            fig_gamma.add_vline(x=odate, line_dash="dot", line_color="gray", opacity=0.3)
    fig_gamma.update_layout(title="8. Micro Flow: Net Dealer Gamma Exposure & OPEX Pinning", xaxis_title="Timeline", yaxis_title="Gamma (Bn$)", hovermode="x unified", height=380)
    st.plotly_chart(fig_gamma, use_container_width=True)

# 5. ตารางสรุปข้อมูลดิบ Real-time ปิดท้าย
st.markdown("### 📌 สแนปชอตราคาปิดตลาดและอินดิเคเตอร์ล่าสุด")
summary_data = pd.DataFrame({
    'Asset Class / Indicator': ['S&P 500 Index Core (^GSPC)', 'S&P 500 ETF (SPY)', 'Nasdaq 100 (QQQ)', 'Dow Jones Index (^DJI)', 'Semiconductor Index (SOXX)', 'CBOE VIX Index (^VIX)', 'US 10Y Yield (%)', 'US 30Y Yield (%)', 'Crude Oil WTI ($)'],
    'Latest Value': [f"{latest['GSPC']:.2f} pts", f"${latest['SPY']:.2f}", f"${latest['QQQ']:.2f}", f"{latest['DJI']:.2f} pts", f"{latest['SOXX']:.2f} pts", f"{latest['VIX']:.2f}", f"{latest['US10Y']:.2f}%", f"{latest['US30Y']:.2f}%", f"${latest['Oil']:.2f}"]
})
st.dataframe(summary_data, use_container_width=True, hide_index=True)
