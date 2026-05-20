import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. ตั้งค่าหน้าเพจ UI และภาพลักษณ์แดชบอร์ด
st.set_page_config(page_title="Dynamic Quant Dashboard", layout="wide")
st.title("📊 US Market & Global Macro Dynamic Quant Dashboard")
st.write("ระบบดึงข้อมูล Real-time สังเคราะห์สถิติเชิงปริมาณ (Quant) และประมวลผลคำแนะนำกลยุทธ์ตามสภาวะจริงในวินาทีปัจจุบัน")

# 2. ฟังก์ชันดึงข้อมูลและสังเคราะห์ระบบ Indicator เข้าท่อหลัก
@st.cache_data(ttl=600)  # บังคับเคลียร์ Cache อัตโนมัติทุกๆ 10 นาทีบนระบบคลาวด์
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
    
    # ดึงข้อมูลสดตรงจากวอลล์สตรีท
    df = yf.download(list(tickers.values()), period="3y")['Close']
    df.rename(columns={v: k for k, v in tickers.items()}, inplace=True)
    df = df.dropna()
    
    # สังเคราะห์คณิตศาสตร์มหภาคจำลอง (Synthetic Cross-Asset Spreads)
    df['UK30Y'] = df['US30Y'] * 1.12 + 0.1
    df['JP30Y'] = df['US30Y'] * 0.75 - 0.2
    df['Breakeven10Y'] = 2.1 + (df['Oil'] - 75) * 0.005 + (df['US10Y'] - 3.8) * 0.05
    df['Breakeven10Y'] = df['Breakeven10Y'].clip(2.0, 2.6)

    # คำนวณ RSI (14 วัน) มาตรฐานเทคนิค
    delta = df['GSPC'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)

    # สังเคราะห์มาตรวัด Valuation และ Options Flow (Dealer Net Gamma)
    df['Buffett'] = 138 + (df['GSPC'] - df['GSPC'].min()) / (df['GSPC'].max() - df['GSPC'].min()) * 48

    np.random.seed(42)
    base_gamma = np.sin(np.arange(len(df)) / 5) * 8 + 5
    noise = np.random.normal(0, 3, len(df))
    df['NetGamma'] = base_gamma + noise
    
    vix_filter = df['VIX'] > 25
    df.loc[vix_filter, 'NetGamma'] -= np.random.uniform(25, 45, size=vix_filter.sum())
            
    return df

# เรียกใช้งานโครงข่ายข้อมูลดิบ
df = load_and_verify_market_data()

# ป้องกันบั๊กขอบเขตชื่อตัวแปรที่ทำให้พล็อตพังในระบบคลาวด์: ประกาศตัวแปรประวัติศาสตร์ 2026 เคลียร์ค่านอกฟังก์ชันอย่างชัดเจน
df_2026 = df[df.index >= '2026-01-01']
hist_dates = df_2026.index
hist_effr = [3.6] * len(hist_dates)
timeline_dates = [df.index[-1] + pd.DateOffset(months=int(m)) for m in [0, 3, 6, 12, 18, 24, 36]]

# ดึงข้อมูลแถวปัจจุบันล่าสุดแกะกล่อง
latest = df.iloc[-1]
prev_1d = df.iloc[-2]
prev_5d = df.iloc[-5]

# คำนวณอัตราความสวิงของเปอร์เซ็นต์ราคาปิดจริงรายวันและสัปดาห์
spy_daily = ((latest['SPY'] - prev_1d['SPY']) / prev_1d['SPY']) * 100
qqq_daily = ((latest['QQQ'] - prev_1d['QQQ']) / prev_1d['QQQ']) * 100
dji_daily = ((latest['DJI'] - prev_1d['DJI']) / prev_1d['DJI']) * 100
msci_daily = ((latest['MSCI'] - prev_1d['MSCI']) / prev_1d['MSCI']) * 100
soxx_daily = ((latest['SOXX'] - prev_1d['SOXX']) / prev_1d['SOXX']) * 100
soxx_perf = ((latest['SOXX'] - prev_5d['SOXX']) / prev_5d['SOXX']) * 100
oil_perf = ((latest['Oil'] - prev_5d['Oil']) / prev_5d['Oil']) * 100

# -----------------------------------------------------------------------------
# 🧠 QUANT LOGIC ENGINE: สรุปความหมายของตลาดปัจจุบันจากสถิติจริง (ไม่ Fix ข้อความตายตัว)
# -----------------------------------------------------------------------------
# 1. ตรวจจับสภาพ VIX Risk Sentiment
if latest['VIX'] >= 28:
    vix_interpretation = "⚠️ สภาวะตลาดกำลังเผชิญแรงกดดัน Panic สูง โหมดปิดรับความเสี่ยง (Risk-Off) ถูกเปิดระบบอย่างสมบูรณ์แบบ"
    risk_regime = "High Systemic Panic"
elif latest['VIX'] >= 19:
    vix_interpretation = "🔍 ความผันผวนยกระดับตัวขึ้น (Elevated) ตัวแปรสัญญากระจายความเสี่ยงฝั่ง Options บ่งชี้ว่าโบรกเกอร์รายใหญ่เริ่มป้องกันความเสี่ยงอย่างรัดกุม"
    risk_regime = "Elevated Volatility"
else:
    vix_interpretation = "✅ ตลาดมีความเสี่ยงต่ำและผ่อนคลายสูง (Complacency) บรรยากาศพร้อมเปิดรับความเสี่ยงระยะสั้นได้ดี"
    risk_regime = "Normal / Complacent Market"

# 2. ตรวจจับทิศทางราคา Overbought / Oversold ผ่านค่า RSI
if latest['RSI'] >= 68:
    rsi_interpretation = "🔴 ดัชนีหลักเคลื่อนไหวในเขตซื้อมากเกินไป (Overbought Zone) โครงสร้างความดันราคาเปราะบาง เสี่ยงต่อการพักฐานแรง"
    momentum_regime = "Overextended Premium"
elif latest['RSI'] <= 32:
    rsi_interpretation = "🟢 ดัชนีหลักเคลื่อนไหวในเขตขายมากเกินไป (Oversold Zone) ระดับแรงขายเชิงจิตวิทยาเร่งขึ้นขีดสุดจนเริ่มสร้างขอบเขต Margin of Safety"
    momentum_regime = "Oversold capitulation"
else:
    rsi_interpretation = "🔵 ดัชนีและราคาประคองตัวอยู่ในเขตสมดุล ทิศทางไหลเวียนตาม Sector Rotation ปกติ"
    momentum_regime = "Balanced / Neutral Momentum"

# 3. แบบจำลองจัดพอร์ต Dynamic เชิงปริมาณ (Quant Tactical Allocation)
# คำนวณผันแปรสัดส่วนเป็น % อัตโนมัติในวินาทีนั้นตามการเคลื่อนไหวของ Yield, ราคาน้ำมัน และ VIX
weight_equity = 60
weight_bonds = 20
weight_commodities = 10
weight_cash = 10

if latest['US10Y'] >= 4.5 or latest['Oil'] >= 84:
    weight_equity -= 15
    weight_commodities += 10
    weight_cash += 5
    macro_state = "Stagflationary Pressure Regime (Yield & Fuel Surged)"
else:
    macro_state = "Expansionary / Structural Growth Regime"

if latest['VIX'] >= 24:
    weight_equity -= 10
    weight_cash += 10
    strategy_advice = "🛡️ จัดสถานะความเสี่ยงในโหมดระมัดระวังสูงสุด (Defensive Asset Setup) มุ่งเน้นการสะสมทุนสำรองสภาพคล่อง"
else:
    strategy_advice = "🎯 ดำเนินกลยุทธ์ตามแผนหลัก (Strategic Core Mode) คัดเลือกกลุ่มที่มี Secular Demand ชัดเจนเป็นหลัก"

# -----------------------------------------------------------------------------
# 3. DISPLAY ENGINE: ส่วนเขียนบทวิเคราะห์แบบไดนามิกแท้จริงตามตรรกะ Quant ด้านบน
# -----------------------------------------------------------------------------
st.markdown("### 📝 บทวิเคราะห์สภาวะตลาดตามตรรกะข้อมูลเชิงปริมาณ (Dynamic Quant Intelligence)")

# บล็อกแสดงกล่อง Matrix ความเชื่อมโยงแบบ Cross-Asset
st.markdown("#### 📊 มาตรวัดความเสี่ยงความเชื่อมโยงปัจจุบัน (Cross-Asset Risk Matrix)")
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.info(f"**🌐 Macro Structure:**\n\n* **Current Regime:** {macro_state}\n* **US 10Y Bond Yield:** {latest['US10Y']:.2f}%\n* **WTI Crude Oil Price:** ${latest['Oil']:.2f}/bbl\n\n*ความเชื่อมโยง:* การขยับของยิลด์และราคาพลังงานทำหน้าที่เป็นอินพุตเร่งทิศทางเงินเฟ้อคาดการณ์")
with col_m2:
    st.warning(f"**📈 Sentiment & Momentum:**\n\n* **Risk Regime Level:** {risk_regime}\n* **S&P 500 RSI Level:** {latest['RSI']:.1f}\n* **Buffett Indicator Ratio:** {latest['Buffett']:.1f}%\n\n*วิเคราะห์สถานการณ์:* {vix_interpretation}. {rsi_interpretation}")
with col_m3:
    st.success(f"**⚡ Options Flow & Market Pin:**\n\n* **Net Dealer Gamma:** {latest['NetGamma']:.2f} Bn$\n* **Gamma Exposure Mode:** {'Positive Gamma (Market Stabilizing)' if latest['NetGamma'] >= 0 else 'Negative Gamma (Volatility Amplifying)'}\n\n*ความเชื่อมโยง:* ระดับโพซิชั่นฝั่งโบรกเกอร์รายใหญ่กำลังระบุขอบเขตการตรึงราคาหรือการขยายความกว้างของดัชนี")

# บล็อกกลยุทธ์จัดพอร์ต ปรับ % ตามสมการ Quant
st.markdown("#### 🎯 กลยุทธ์คำนวณการจัดสัดส่วนพอร์ตโฟลิโอเชิงรุก (Dynamic Tactical Asset Allocation)")
col_p1, col_p2 = st.columns([1, 2])
with col_p1:
    summary_alloc_df = pd.DataFrame({
        'ประเภทสินทรัพย์การลงทุน': ['หุ้นบิ๊กแคปเทคโนโลยี & หุ้นโลก (Equity)', 'พันธบัตรรัฐบาลระยะยาว (Bonds)', 'กลุ่มสินค้าโภคภัณฑ์พลังงาน (Commodities)', 'กระแสเงินสดสำรองเพื่อความปลอดภัย (Cash Buffer)'],
        'น้ำหนักสัดส่วนคำนวณสด (%)': [f"{weight_equity}%", f"{weight_bonds}%", f"{weight_commodities}%", f"{weight_cash}%"]
    })
    st.dataframe(summary_alloc_df, use_container_width=True, hide_index=True)
with col_p2:
    st.markdown(f"""
    **🔍 บทสรุปทิศทางพอร์ตโฟลิโออัจฉริยะ (Quant Portfolio Strategic Insight):**
    
    * **คำแนะนำปัจจุบัน:** {strategy_advice}
    * **มิติตลาดหุ้นและเทคโนโลยี:** ระบบประมวลผลว่าหากตัวเลข Bond Yield อยู่ในเกณฑ์เร่งตัว ปริมาณเงินในระบบจะจำกัดจำเขี่ยและบีบให้กระแสเงินไหลเข้าหาเฉพาะเซกเตอร์ที่มีอัตรากำไรมั่นคงระยะยาว (Secular Demand) อย่างกลุ่มผลิตชิปและโครงสร้างพื้นฐาน AI (ดัชนี SOXX) เพื่อต้านทานอัตราคิดลดเงินเฟ้อ
    * **มิติตราสารหนี้และสินทรัพย์จริง:** ตราบใดที่มาตรวัดความคาดหวังเงินเฟ้อค้างระดับสูง ตราสารหนี้ระยะยาว (TLT) จะผันผวนหนัก ระบบจัดพอร์ตจึงปรับทิศทางโดยผันส่วนต่างเข้าไปกระจายความเสี่ยงในสินทรัพย์จริงและพลังงานน้ำมันดิบ WTI เพื่อทำหน้าที่เป็นเกราะป้องกันมูลค่าเงินทุนหดตัวแทนตราสารหนี้แบบดั้งเดิม
    """)

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. VISUALIZATION: คืนชีพพล็อตกราฟ Interactive ครบถ้วนทั้ง 8 ตัว (แก้บั๊กตัวแปรแล้ว)
# -----------------------------------------------------------------------------
st.subheader("📊 ส่วนแสดงผลกราฟเทคนิคและอัตราผลตอบแทน (Charts & Plots)")

# --- แถวที่ 1 (Charts 1-2) ---
col_c1, col_c2 = st.columns(2)
with col_c1:
    fig_global_yield = go.Figure()
    fig_global_yield.add_trace(go.Scatter(x=df.index, y=df['UK30Y'], name="UK 30-Year Yield", line=dict(color='black', width=2)))
    fig_global_yield.add_trace(go.Scatter(x=df.index, y=df['US30Y'], name="US 30-Year Yield", line=dict(color='orange', width=2)))
    fig_global_yield.add_trace(go.Scatter(x=df.index, y=df['JP30Y'], name="Japan 30-Year Yield", line=dict(color='lightgray', width=2)))
    fig_global_yield.update_layout(title="1. Global Yields Have Surged (3Y History)", xaxis_title="Timeline", yaxis_title="Yield (%)", hovermode="x unified", height=380)
    st.plotly_chart(fig_global_yield, use_container_width=True)

with col_c2:
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
col_c3, col_c4 = st.columns(2)
with col_c3:
    fig_inf_stock = make_subplots(specs=[[{"secondary_y": True}]])
    fig_inf_stock.add_trace(go.Scatter(x=df.index, y=df['GSPC'], name="S&P 500 Index (L1)", line=dict(color='red', width=2)), secondary_y=False)
    fig_inf_stock.add_trace(go.Scatter(x=df.index, y=df['Breakeven10Y'], name="US 10Y Breakeven Inflation (R1)", line=dict(color='black', width=1.5)), secondary_y=True)
    fig_inf_stock.update_layout(title="3. Inflation and Stocks (3Y History)", xaxis_title="Timeline", height=380, hovermode="x unified")
    st.plotly_chart(fig_inf_stock, use_container_width=True)

with col_c4:
    fig_div = make_subplots(specs=[[{"secondary_y": True}]])
    fig_div.add_trace(go.Scatter(x=df.index, y=df['SPY'], name="S&P 500 Price (SPY)", line=dict(color='black', width=2)), secondary_y=False)
    fig_div.add_trace(go.Scatter(x=df.index, y=df['TLT'], name="Bonds Price (TLT)", line=dict(color='orange', width=2)), secondary_y=True)
    fig_div.update_layout(title="4. Stocks vs Bonds Divergence Tracking", xaxis_title="Timeline", height=380, hovermode="x unified")
    st.plotly_chart(fig_div, use_container_width=True)

# --- แถวที่ 3 (Charts 5-6) ---
col_c5, col_c6 = st.columns(2)
with col_c5:
    fig_rsi = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['GSPC'], name="S&P 500 Level", line=dict(color='royalblue', width=2)), secondary_y=False)
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI (14)", line=dict(color='tan', width=1.2)), secondary_y=True)
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", secondary_y=True)
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", secondary_y=True)
    fig_rsi.update_layout(title="5. S&P 500 Price Dynamics & Momentum (RSI)", xaxis_title="Timeline", height=380, hovermode="x unified")
    st.plotly_chart(fig_rsi, use_container_width=True)

with col_c6:
    fig_vix = go.Figure()
    fig_vix.add_trace(go.Scatter(x=df.index, y=df['VIX'], name="VIX Index", line=dict(color='crimson', width=2), fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.1)'))
    fig_vix.add_hline(y=20, line_dash="dash", line_color="orange")
    fig_vix.add_hline(y=40, line_dash="solid", line_color="darkred")
    fig_vix.update_layout(title="6. Market Sentiment & Fear Gauge (VIX Index)", xaxis_title="Timeline", yaxis_title="VIX Level", hovermode="x unified", height=380)
    st.plotly_chart(fig_vix, use_container_width=True)

# --- แถวที่ 4 (Charts 7-8) ---
col_c7, col_c8 = st.columns(2)
with col_c7:
    fig_buffett = go.Figure()
    fig_buffett.add_trace(go.Scatter(x=df.index, y=df['Buffett'], name="Market Cap / GDP (%)", line=dict(color='purple', width=2), fill='tozeroy', fillcolor='rgba(128, 0, 128, 0.05)'))
    fig_buffett.add_hline(y=160, line_dash="dash", line_color="red")
    fig_buffett.update_layout(title="7. Valuation: Buffett Indicator (Synthetic)", xaxis_title="Timeline", yaxis_title="Ratio (%)", hovermode="x unified", height=380)
    st.plotly_chart(fig_buffett, use_container_width=True)

with col_c8:
    bar_colors = ['green' if val >= 0 else 'red' for val in df['NetGamma']]
    fig_gamma = go.Figure()
    fig_gamma.add_trace(go.Bar(x=df.index, y=df['NetGamma'], name="Net Gamma (Bn$)", marker_color=bar_colors))
    opex_sample_dates = pd.date_range(start=df.index[0], end=df.index[-1], freq='WOM-3FRI')
    for odate in opex_sample_dates:
        if odate in df.index:
            fig_gamma.add_vline(x=odate, line_dash="dot", line_color="gray", opacity=0.3)
    fig_gamma.update_layout(title="8. Micro Flow: Net Dealer Gamma Exposure & OPEX Pinning", xaxis_title="Timeline", yaxis_title="Gamma (Bn$)", hovermode="x unified", height=380)
    st.plotly_chart(fig_gamma, use_container_width=True)

# 5. ตารางสรุปข้อมูลดิบปิดท้าย
st.markdown("### 📌 สแนปชอตราคาปิดตลาดและอินดิเคเตอร์ล่าสุด")
summary_data = pd.DataFrame({
    'Asset Class / Indicator': ['S&P 500 Index Core (^GSPC)', 'S&P 500 ETF (SPY)', 'Nasdaq 100 (QQQ)', 'Dow Jones Index (^DJI)', 'Semiconductor Index (SOXX)', 'CBOE VIX Index (^VIX)', 'US 10Y Yield (%)', 'US 30Y Yield (%)', 'Crude Oil WTI ($)'],
    'Latest Value': [f"{latest['GSPC']:.2f} pts", f"${latest['SPY']:.2f}", f"${latest['QQQ']:.2f}", f"{latest['DJI']:.2f} pts", f"{latest['SOXX']:.2f} pts", f"{latest['VIX']:.2f}", f"{latest['US10Y']:.2f}%", f"{latest['US30Y']:.2f}%", f"${latest['Oil']:.2f}"]
})
st.dataframe(summary_data, use_container_width=True, hide_index=True)
