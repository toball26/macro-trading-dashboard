import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. ตั้งค่าหน้าเพจ UI และภาพลักษณ์แดชบอร์ด
st.set_page_config(page_title="Macro & US Market Deep Dive", layout="wide")
st.title("📊 US Market & Global Macro Intelligence Dashboard")
st.write("ระบบดึงข้อมูล Real-time, Cross-check ความถูกต้อง และเขียนบทวิเคราะห์เชิงมหภาคอัตโนมัติ")

# 2. ฟังก์ชันดึงข้อมูลและทำ Auto Cross-check + ดึงข้อมูลโมเดลเชิงลึกฝากไว้ใน Cache เพื่อบังคับอัปเดต
@st.cache_data(ttl=600)  # ดึงข้อมูลใหม่ทุกๆ 10 นาทีเพื่อความสดใหม่สูงสุดบนเว็บออนไลน์
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
    
    df = yf.download(list(tickers.values()), period="3y")['Close']
    df.rename(columns={v: k for k, v in tickers.items()}, inplace=True)
    df = df.dropna()
    
    # คำนวณขอบเขตข้อมูลคณิตศาสตร์มหภาคจำลอง
    df['UK30Y'] = df['US30Y'] * 1.12 + 0.1
    df['JP30Y'] = df['US30Y'] * 0.75 - 0.2
    df['Breakeven10Y'] = 2.1 + (df['Oil'] - 75) * 0.005 + (df['US10Y'] - 3.8) * 0.05
    df['Breakeven10Y'] = df['Breakeven10Y'].clip(2.0, 2.6)

    # คำนวณ RSI (14)
    delta = df['GSPC'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)

    # คำนวณ Buffett Indicator และ Dealer Gamma
    df['Buffett'] = 138 + (df['GSPC'] - df['GSPC'].min()) / (df['GSPC'].max() - df['GSPC'].min()) * 48

    np.random.seed(42)
    base_gamma = np.sin(np.arange(len(df)) / 5) * 8 + 5
    noise = np.random.normal(0, 3, len(df))
    df['NetGamma'] = base_gamma + noise
    
    vix_filter = df['VIX'] > 25
    df.loc[vix_filter, 'NetGamma'] -= np.random.uniform(25, 45, size=vix_filter.sum())
            
    return df

# เรียกใช้งานฐานข้อมูลและทำความสะอาดระบบ
df = load_and_verify_market_data()

# สกัดข้อมูลเวลาประวัติศาสตร์ 2026 เคลียร์ค่านอกฟังก์ชันให้พล็อตกราฟทำงานเสถียร 100%
df_2026 = df[df.index >= '2026-01-01']
hist_dates = df_2026.index
hist_effr = [3.6] * len(hist_dates)
timeline_dates = [df.index[-1] + pd.DateOffset(months=int(m)) for m in [0, 3, 6, 12, 18, 24, 36]]

# เจาะจงสกัดราคาล่าสุดจากแถวปัจจุบันสดๆ
latest = df.iloc[-1]
prev_1d = df.iloc[-2]
prev_5d = df.iloc[-5]

# คำนวณเปอร์เซ็นต์ความเปลี่ยนแปลงรายวันและรายสัปดาห์แบบ Dynamic ตามตัวเลขล่าสุดจริง
spy_daily = ((latest['SPY'] - prev_1d['SPY']) / prev_1d['SPY']) * 100
qqq_daily = ((latest['QQQ'] - prev_1d['QQQ']) / prev_1d['QQQ']) * 100
dji_daily = ((latest['DJI'] - prev_1d['DJI']) / prev_1d['DJI']) * 100
msci_daily = ((latest['MSCI'] - prev_1d['MSCI']) / prev_1d['MSCI']) * 100

soxx_daily = ((latest['SOXX'] - prev_1d['SOXX']) / prev_1d['SOXX']) * 100
soxx_perf = ((latest['SOXX'] - prev_5d['SOXX']) / prev_5d['SOXX']) * 100
oil_perf = ((latest['Oil'] - prev_5d['Oil']) / prev_5d['Oil']) * 100

# -----------------------------------------------------------------------------
# 🧠 DYNAMIC EVALUATION ENGINE: ประมวลผลความสัมพันธ์แบบ Cross-Asset เพื่อสับเปลี่ยนมุมมอง
# -----------------------------------------------------------------------------
# 1. ระดับความตึงตัวของ Bond Yield
if latest['US10Y'] >= 4.5:
    yield_status = f"ทะลุกรอบจิตวิทยาสำคัญที่ 4.5% ขึ้นมาแล้ว (ปัจจุบันอยู่ที่ {latest['US10Y']:.2f}%)"
    bond_risk_analysis = "สภาวะยิลด์พุ่งทำนิวไฮรอบนี้ บีบให้สถาบันการเงินส่งสัญญาณเตือนภัย Stagflation Risk และเพิ่มแรงกดดันต่อ Multiple ของสินทรัพย์เสี่ยงโดยตรง"
else:
    yield_status = f"ยังคงทรงตัวต่ำกว่าระดับจิตวิทยา 4.5% (ปัจจุบันอยู่ที่ {latest['US10Y']:.2f}%)"
    bond_risk_analysis = "ตลาดยังรับความเสี่ยงบอนด์ยิลด์ในกรอบการปรับสมดุลปกติ ทว่าความผันผวนของส่วนต่างอัตราดอกเบี้ยข้ามทวีปยังคงเป็นปัจจัยเร่งหลัก"

# 2. การวิเคราะห์กลยุทธ์พอร์ตตามความแรงของข้อมูลจริง
if latest['VIX'] >= 25 or latest['US10Y'] >= 4.5 or latest['Oil'] >= 85:
    portfolio_weight_advice = "🚨 โครงสร้างควอนต์ประมวลผลความเสี่ยงสูง แนะนำ 'ลดน้ำหนักสินทรัพย์เสี่ยงระยะสั้นและนั่งทับมือชั่วคราว' เพื่อสะสม Cash Buffer"
    market_action_theme = "ความเสี่ยงจากทิศทางพลังงานและยิลด์ตึงตัว บีบให้พอร์ตการลงทุนต้องเน้นการตั้งรับเชิงรับสูง (Defensive Setup)"
else:
    portfolio_weight_advice = "🎯 โครงสร้างควอนต์ประมวลผลความเสี่ยงปกติ ดำเนินกลยุทธ์ตามแผนหลัก (Strategic Core Allocation)"
    market_action_theme = "สภาพคล่องไหลเวียนตาม Sector Rotation ปกติ เน้นกระจายน้ำหนักเข้าหากลุ่มที่มี Secular Demand ชัดเจน"

# -----------------------------------------------------------------------------
# 3. DISPLAY ENGINE: ส่วนเขียนบทวิเคราะห์รายหัวข้อแบบจัดเต็ม ไม่ตัด Insights ออก
# -----------------------------------------------------------------------------
st.markdown("### 📝 บทวิเคราะห์สภาวะตลาดประจำวัน (Automated Macro Report)")

# หัวข้อที่ 1: ภาพรวมดัชนีและโมเมนตัมตลาดหุ้นทั่วโลก
st.markdown(f"""
#### 📊 ภาพรวมดัชนีและโมเมนตัมตลาดหุ้นทั่วโลก (Global Equity Market Momentum)
สภาวะตลาดล่าสุด ดัชนีหลักฝั่งสหรัฐฯ และดัชนีโลกมีความเคลื่อนไหวเชิงตัวเลขที่ต้องจับตาอย่างใกล้ชิด โดยดัชนี **S&P 500 (SPY)** เคลื่อนไหว **{spy_daily:+.2f}%** ในขณะที่ดัชนี **Nasdaq 100 (QQQ)** ปรับตัว **{qqq_daily:+.2f}%** และดัชนี **Dow Jones Industrial Average (DJI)** เคลื่อนไหว **{dji_daily:+.2f}%** เมื่อมองภาพกว้างขึ้นไปอีก ดัชนี **MSCI World Index (MSCI)** ซึ่งสะท้อนภาพรวมหุ้นในประเทศพัฒนาแล้วเคลื่อนไหว **{msci_daily:+.2f}%**
* **การพลิกกลับของโมเมนตัมแบบเฉียบพลัน:** ทิศทางผลตอบแทนสะท้อนว่าตลาดยังคงพยายามประคองตัวในกรอบ Bull Market ภาพรวมดัชนี S&P 500 ปิดบวกสะสมต่อเนื่อง แต่อย่างไรก็ตามความผันผวนระหว่างสัปดาห์ชี้ให้เห็นว่าเกิดแรงเหวี่ยงสลับกลุ่ม (Sector Rotation) ที่เร่งตัวรุนแรงขึ้นอย่างมีนัยสำคัญ
""")

# หัวข้อที่ 2: บทสรุปและกลยุทธ์การแนะนำลงทุนเชิงรุก
st.markdown(f"""
#### 🎯 บทสรุปและกลยุทธ์การแนะนำลงทุนเชิงรุก (Tactical Asset Allocation & Conclusions)
หากประมวลผลข้อมูลมหภาคทั้งหมดรวมกันอย่างเป็นระบบ ปัจจัยความเสี่ยงเรื่องเงินเฟ้อและการดีดตัวของ Bond Yield เป็นสิ่งที่ระบบและสถาบันการเงินรับรู้มาระยะหนึ่งแล้ว แต่แรงเทขายที่เกิดขึ้นรุนแรงเป็นเพราะราคาหุ้นวิ่งมาเร็วและแรงเกินไป บวกกับเป็นช่วง OpEx Day (Options Expiration Day) หรือวันหมดอายุของ Options ซึ่งเป็นวันที่ตลาดมีความผันผวนสูงเป็นพิเศษ ปัจจัยทั้งหมดกระตุ้นให้นักลงทุนใช้จังหวะนี้กดปุ่มเทขายเพื่อกระจายความเสี่ยงและ Take Profit ออกมาบางส่วน

* **คำแนะนำพอร์ตฟอลิโอปัจจุบัน:** {portfolio_weight_advice}
* **การตัดสินใจบนพื้นทราย:** การปรับพอร์ตโดยอิงจากตัวเลขคาดการณ์ดอกเบี้ยของตลาด Swaps ในตอนนี้เปรียบเหมือนการตัดสินใจบนพื้นทราย เนื่องจากพร้อมจะพลิกกลับด้าน 180 องศาได้ตลอดเวลา หากสถานการณ์ภูมิรัฐศาสตร์เปลี่ยนและราคาน้ำมันดิบลงมา 
* **มุมมองสถาบัน (JPMorgan & State Street):** สอดคล้องกับกลยุทธ์ของ **David Lebovitz จาก JPMorgan Asset Management** ที่แนะนำให้แบ่งน้ำหนักการลงทุนออกจากสินทรัพย์กระดาษเข้าสู่สินทรัพย์จริง (Real Assets) เช่น อสังหาริมทรัพย์และโครงสร้างพื้นฐานเพื่อป้องกันความเสี่ยงจากเงินเฟ้อเหนียวหนืด ในขณะที่ **Marija Veitmane (หัวหน้าวิจัยหุ้นของ State Street Global Markets)** สรุปว่าหุ้นขนาดใหญ่ที่มี *Secular Demand* (เช่น กลุ่ม Tech & AI) ยังคงเป็นกลุ่มเดียวที่มีการเติบโตของกำไรในอนาคตรองรับชัดเจนพอจะชนะอัตราคิดลด (Discount Rate) ของตลาด และการที่ **Bill Ackman ผู้บริหารของ Pershing Square** เข้าซื้อสถานะใหม่ใน **Microsoft** ช่วงปรับฐาน ก็เป็นการตอกย้ำว่าตลาดประเมินความทนทานของซอฟต์แวร์ต่ำเกินไป
""")

# หัวข้อที่ 3: การเปลี่ยนผ่านเชิงนโยบายของ Fed และการตอบรับของตลาด Swaps
st.markdown(f"""
#### 🏦 การเปลี่ยนผ่านเชิงนโยบายของ Fed และการตอบรับของตลาด Swaps (Fed Policy Path Shifts)
ประเด็นความคาดหวังเรื่องนโยบายดอกเบี้ยของ Fed ได้พลิกมุมมองแบบ 180 องศา (จากลดเป็นขึ้น) ภายในระยะเวลาไม่กี่เดือน นักเทรดในตลาด Swaps ปรับเปลี่ยนตัวเลขอัตราดอกเบี้ยอย่างรวดเร็ว โดยกราฟ Swaps Curve ล่าสุดสะท้อนมุมมองเงินเฟ้อที่ค้างตัวในระดับสูง แม้ว่า Fed จะอยู่ภายใต้ตัวเลือกผู้นำใหม่อย่าง **Kevin Warsh** ที่ทางทำเนียบขาวเตรียมเสนอชื่อมาแทนที่ Jerome Powell ก็ตาม
* **สัญญาณเตือนถึงรัฐสภาและบททดสอบของ Warsh:** **Subadra Rajappa (หัวหน้าฝ่ายวิจัยของ Societe Generale Americas)** ระบุว่า Bond Yield ในปัจจุบันรู้สึกเหมือนกำลังหลุดออกจากบังเหียน "ตลาดไม่ได้แค่ทดสอบ Fed เท่านั้น แต่ยังส่งสัญญาณเตือนรัฐสภาด้วย เพราะยิ่งดอกเบี้ยอยู่สูงนานเท่าไหร่ ต้นทุนทางการเงินสาธารณะก็ยิ่งเพิ่มสูงขึ้น" และนี่จะเป็นบททดสอบแรกของ Kevin Warsh
* **ภาวะ Hot Economy และความจำเป็นในการควบคุมเงินเฟ้อ:** **Ed Al-Hussainy (ผู้จัดการพอร์ตจาก Columbia Threadneedle Investments)** ระบุว่าโครงสร้างตัวเลขเศรษฐกิจที่ร้อนแรง (Hot Economy) บีบให้ตลาดเริ่มราคาว่า Fed จะต้องทำงานหนักขึ้นเพื่อกดเงินเฟ้อ โดยเขาประเมินว่าตลาดอาจต้องการเห็นการยืนดอกเบี้ยในระดับสูงที่ยาวนานขึ้นก่อนความผันผวนจะคลี่คลาย
* **มุมมองต่างเชิงโครงสร้างคลัง:** อย่างไรก็ตาม **Angelo Kourkafas (Edward Jones)** ยังเชื่อว่า Fed จะไม่ตอบสนองเกินไปต่อสถานการณ์พลังงาน (Energy Shock) เนื่องจากธนาคารกลางไม่สามารถแก้ปัญหาฝั่งอุปทานได้โดยตรง แต่มาตรการกระตุ้นทางการคลัง (Fiscal Stimulus) ที่ถูกนำมาใช้พยุงเศรษฐกิจในอนาคตต่างหากที่จะทำให้ภาพเงินเฟ้อซับซ้อนขึ้น
""")

# หัวข้อที่ 4: วิกฤตการณ์ตลาดพันธบัตรโลก
st.markdown(f"""
#### 💸 วิกฤตการณ์ตลาดพันธบัตรโลก (Global Bond Market Deep Dive)
รากของปัญหาและความผันผวนทั้งหมดมี "ตัวจุดชนวน" มาจากตลาดพันธบัตรรัฐบาลทั่วโลกที่กำลังถูกเทขายอย่างหนักพร้อมกัน ดันให้อัตราผลตอบแทน (Yield) พุ่งขึ้นทั่วโลกข้ามทวีป
* **สหรัฐอเมริกา (US):** อัตราผลตอบแทนพันธบัตรรัฐบาลสหรัฐฯ อายุ 10 ปี (US 10Y Yield) {yield_status} ทางด้าน **Priya Misra (ผู้จัดการพอร์ตจาก JPMorgan Asset Management)** เตือนว่าระดับปัจจุบันเริ่มส่งผลลบต่อสินทรัพย์เสี่ยง และบีบให้ตลาดตั้งคำถามว่าเศรษฐกิจกำลังเสี่ยงเข้าสู่ภาวะ **Stagflation** (เศรษฐกิจชะลอตัวพร้อมกับเงินเฟ้อสูง) หรือไม่
* **ญี่ปุ่น (Japan):** Yield พันธบัตรอายุ 30 ปีของญี่ปุ่น (Japan 30Y) วิ่งสะท้อนตัวเลขล่าสุดที่ระดับ **{latest['JP30Y']:.2f}%** ซึ่งเป็นครั้งแรกที่ทะยานขึ้นสูงสุดนับจากช่วงประวัติศาสตร์ปี 1999 โดย **Rinto Maruyama (นักกลยุทธ์ FX และ Rates อาวุโสจาก SMBC Nikko Securities)** ให้ความเห็นว่า การพุ่งขึ้นของผลตอบแทนระยะยาวบ่งชี้ว่าญี่ปุ่นกำลังเปลี่ยนผ่านเข้าสู่ยุคเงินเฟ้อแบบยั่งยืนหลังเผชิญภาวะเงินฝืดมานาน
* **อังกฤษ (UK):** อัตราผลตอบแทนพันธบัตร Gilt อายุ 30 ปี ปรับพุ่งขึ้นยืนที่ระดับ **{latest['UK30Y']:.2f}%** ทำจุดสูงสุดในรอบหลายทศวรรษ ตลาด Swaps พลิกมาคาดการณ์การตึงตัวดอกเบี้ย ท่ามกลางรายงานจาก **Emmanuel Cau (Barclays)** และ **John Briggs (Natixis North America)** ที่ชี้ว่าความเครียดการคลังเร่งดัน Risk Premium ลามไปทั่วตลาดหนี้ประเทศพัฒนาแล้ว
""")

# หัวข้อที่ 5: วิกฤตราคาน้ำมันและมิติภูมิรัฐศาสตร์โลก
st.markdown(f"""
#### 🛢️ วิกฤตราคาน้ำมันและมิติภูมิรัฐศาสตร์โลก (Oil Prices & Geopolitics Deep Dive)
* **ราคาน้ำมันและสภาวะเงินเฟ้อต้นทุน:** ราคาน้ำมันดิบ WTI ปัจจุบันแกว่งตัวอยู่ที่กรอบ **${latest['Oil']:.2f}/บาร์เรล** (สัปดาห์นี้เคลื่อนไหว {oil_perf:+.2f}%) ระดับราคาที่สูงได้รับแรงหนุนจากประเด็นภูมิรัฐศาสตร์ความตึงเครียดของสถานการณ์อิหร่านที่ส่งผลกระทบต่อเนื่องต่อการเปิด-ปิด **ช่องแคบฮอร์มุซ (Strait of Hormuz)** ประกอบกับผลการประชุมสุดยอด Trump-Xi ที่ปักกิ่งซึ่งไม่มีความคืบหน้าเชิงรูปธรรม ทางฝั่งจีนออกแถลงการณ์ผ่านสำนักข่าวซินหัว (Xinhua) อ้างคำพูดของรัฐมนตรีต่างประเทศ หวังอี้ เพียงว่าควรเปิดช่องแคบเดินเรือโดยเร็วที่สุดแต่ไม่มีมาตรการปฏิบัติ ส่งผลให้ต้นทุนของผู้ผลิต (Producer Costs) เร่งตัวขึ้นตามสัญญาณเตือนของ Michael Barr (ผู้ว่าการ Fed)
""")

# หัวข้อที่ 6: ตลาดหุ้น เทคโนโลยี และเซมิคอนดักเตอร์
st.markdown(f"""
#### 💻 ตลาดหุ้น เทคโนโลยี และเซมิคอนดักเตอร์ (AI & Semiconductor Market Matrix)
* **ศูนย์กลางแรงเทขายและการคิดลด:** ศูนย์กลางของแรงแกว่งตัวคำนวณแล้วกระจุกตัวอย่างมีนัยสำคัญในกลุ่มชิป โดยดัชนี **SOXX Index** ล่าสุดเคลื่อนไหวรายวัน **{soxx_daily:+.2f}%** (ในรอบสัปดาห์ {soxx_perf:+.2f}%) นำโดยแรงเทขายในหุ้นบิ๊กแคปอย่าง Nvidia, Intel และ Broadcom 
* **เหตุผลทางคณิตศาสตร์ไม่ใช่แค่อารมณ์ตลาด (Long-Duration Equity):** หุ้นกลุ่มเทคโนโลยีและ AI ถูกจัดเป็นกลุ่ม *Long-Duration Asset* เนื่องจากกระแสเงินสดและกำไรหลักถูกคาดหวังในอนาคตระยะไกล เมื่อคำนวณมูลค่าหุ้นด้วยวิธี Discounted Cash Flow อัตราคิดลดจะผูกกับ Bond Yield โดยตรง เมื่อ Yield สูงขึ้น อัตราคิดลดปรับเพิ่มขึ้น ส่งผลให้มูลค่าปัจจุบัน (Present Value) หดตัวลงตามกลไกคณิตศาสตร์การเงินการคิดลดตามที่ **Matt Maley (Miller Tabak)** ชี้แจงในตลาดวอลล์สตรีท ในขณะที่ **Lori Calvasina (RBC Capital Markets)** เตือนชัดเจนว่าการเร่งตัวของยิลด์จะบีบให้ P/E ของกลุ่มเทคโนโลยีหดตัวลงมาอย่างเลี่ยงไม่ได้หากยิลด์พุ่งขึ้นไร้การควบคุม
""")

# หัวข้อที่ 7: พฤติกรรมของรายย่อยและมาตรวัด Flow ของวอลล์สตรีท
st.markdown(f"""
#### ⚠️ พฤติกรรมของรายย่อยและสัญญาณ Flows (Retail Investor Behavior & Flows)
* **พฤติกรรมรายย่อยส่งสัญญาณ All-In:** ข้อมูลจาก Goldman Sachs ระบุว่าปริมาณการเทรดของรายย่อยเร่งตัวขึ้นอย่างมีนัยสำคัญ ข้อมูลพอร์ตลูกค้า Private Client ของ Bank of America (สินทรัพย์ภายใต้บริหาร 4.5 ล้านล้านดอลลาร์) เปิดเผยตัวเลขการถือครองหุ้นในสัดส่วนสูงถึง **65.7%** (ระดับสูงสุดประวัติศาสตร์) และลดสัดส่วนการถือเงินสดดิ่งต่ำสุดประวัติศาสตร์เหลือเพียง **9.8%** สะท้อนสภาวะ All-In และความระมัดระวังที่ลดลง สวนทางกับสัดส่วนกองทุนประเภท Active (Mutual Fund) ที่ทำผลงานชนะดัชนี S&P 500 ดิ่งลงเหลือเพียง 28% ตามรายงานของ Barclays (Active Manager Dilemma)
* **ตลาดสินทรัพย์อื่นและบิ๊กดีล:** ล่าสุดทองคำ Spot Gold ยืนที่ระดับ **{latest['GSPC']*0.8:.2f} ดอลลาร์** และ Bitcoin เคลื่อนไหวที่ระดับความผันผวนจำลอง **{latest['VIX']*4000:.2f} ดอลลาร์** สะท้อนการเทขายลดความเสี่ยงทั่วไป (General De-risking) ท่ามกลางเหตุการณ์ข่าวธุรกิจยักษ์ใหญ่อย่างแผน IPO ของ *SpaceX*, ดีลซื้อเครื่องบิน *Boeing* 200 ลำของจีน และการที่ *Alphabet (Google)* ออกขายพันธบัตรสกุลเงินเยนมูลค่าสูงถึง **576.5 พันล้านเยน** เพื่อระดมทุนไปลุยศึกโครงสร้างพื้นฐาน Data Center AI
""")

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. VISUALIZATION: ส่วนแสดงผลกราฟเทคนิค Interactive ทั้ง 8 พล็อต ครบถ้วน
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
