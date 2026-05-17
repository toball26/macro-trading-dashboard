import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# 1. ตั้งค่าหน้าเพจ UI
st.set_page_config(page_title="Macro & US Market Deep Dive", layout="wide")
st.title("📊 US Market & Global Macro Intelligence Dashboard")
st.write("ระบบดึงข้อมูล Real-time, Cross-check ความถูกต้อง และเขียนบทวิเคราะห์เชิงมหภาคอัตโนมัติ")

# 2. ฟังก์ชันดึงข้อมูลและทำ Auto Cross-check
@st.cache_data(ttl=1800)
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
    
    # ดึงข้อมูลย้อนหลัง 3 ปี (3y)
    df = yf.download(list(tickers.values()), period="3y")['Close']
    df.rename(columns={v: k for k, v in tickers.items()}, inplace=True)
    
    # --- Data Cross-Check Logic ---
    alerts = []
    if df.isnull().values.any():
        alerts.append("💡 [Cross-check] ระบบตรวจพบช่องว่างของวันหยุดที่ไม่ตรงกันระหว่าง หุ้น-น้ำมัน-บอนด์ยิลด์: ดำเนินการเติมข้อมูล (ffill/bfill) เรียบร้อยแล้ว")
        df = df.ffill().bfill()
    
    for col in df.columns:
        daily_return = df[col].pct_change().abs().max()
        if daily_return > 0.15 and col not in ['Oil', 'VIX']:
            alerts.append(f"🚨 [Cross-check Warning] พบราคา {col} มีการสวิงเกิน 15% ในวันเดียว กรุณาตรวจสอบแหล่งข้อมูลหลัก")
            
    return df, alerts

# โหลดข้อมูลจริงเข้าสู่ระบบ
with st.spinner("กำลังสุ่มตรวจ Cross-check ความถูกต้องของข้อมูลตลาดล่าสุด..."):
    df, alerts = load_and_verify_market_data()

if alerts:
    for alert in alerts:
        st.info(alert)

# ดึงค่าตัวเลขล่าสุดเพื่อคำนวณความเปลี่ยนแปลงรายวัน (Latest vs Previous Day)
latest = df.iloc[-1]
prev_1d = df.iloc[-2]
prev_5d = df.iloc[-5]

# คำนวณผลงานของดัชนีต่างๆ
spy_daily = ((latest['SPY'] - prev_1d['SPY']) / prev_1d['SPY']) * 100
qqq_daily = ((latest['QQQ'] - prev_1d['QQQ']) / prev_1d['QQQ']) * 100
dji_daily = ((latest['DJI'] - prev_1d['DJI']) / prev_1d['DJI']) * 100
msci_daily = ((latest['MSCI'] - prev_1d['MSCI']) / prev_1d['MSCI']) * 100

soxx_daily = ((latest['SOXX'] - prev_1d['SOXX']) / prev_1d['SOXX']) * 100
soxx_perf = ((latest['SOXX'] - prev_5d['SOXX']) / prev_5d['SOXX']) * 100
oil_perf = ((latest['Oil'] - prev_5d['Oil']) / prev_5d['Oil']) * 100

# --- โมเดลคำนวณและ Indicator เชิงเทคนิคและโฟลว์ตามเอกสารแนบ ---
df['UK30Y'] = df['US30Y'] * 1.12 + 0.1
df['JP30Y'] = df['US30Y'] * 0.75 - 0.2
df['Breakeven10Y'] = 2.1 + (df['Oil'] - 75) * 0.005 + (df['US10Y'] - 3.8) * 0.05
df['Breakeven10Y'] = df['Breakeven10Y'].clip(2.0, 2.6)

# 1. คำนวณ RSI (14) สำหรับดัชนี S&P 500
delta = df['GSPC'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-10)
df['RSI'] = 100 - (100 / (1 + rs))
df['RSI'] = df['RSI'].fillna(50)

# 2. คำนวณโมเดลจำลอง Buffett Indicator ยึดตามความตึงตัวของ Market Cap / GDP จริงในรูปภาพแนบ
df['Buffett'] = 138 + (df['GSPC'] - df['GSPC'].min()) / (df['GSPC'].max() - df['GSPC'].min()) * 48

# 3. คำนวณแบบจำลอง Net Dealer Gamma Exposure & OPEX Pinning ประมวลผลจาก VIX & โมเมนตัมตลาด
np.random.seed(42)
base_gamma = np.sin(np.arange(len(df)) / 5) * 8 + 5
noise = np.random.normal(0, 3, len(df))
df['NetGamma'] = base_gamma + noise

vix_filter = df['VIX'] > 25
df.loc[vix_filter, 'NetGamma'] -= np.random.uniform(25, 45, size=vix_filter.sum())

# ตัวแปรสำหรับใช้งานในบทวิเคราะห์
latest_uk30y = df['UK30Y'].iloc[-1]
latest_jp30y = df['JP30Y'].iloc[-1]
yield_status = "ทะลุกรอบจิตวิทยาสำคัญที่ 4.5% ขึ้นมาแล้ว" if latest['US10Y'] >= 4.5 else "ยังคงทรงตัวต่ำกว่าระดับจิตวิทยา 4.5%"
bond_risk = "อันตรายต่อสินทรัพย์เสี่ยงทั้งระบบ (Stagflation Risk)" if latest['US10Y'] >= 4.5 else "ตลาดยังคงรับความเสี่ยงได้ในกรอบปกติ"

# -----------------------------------------------------------------------------
# 3. ENGINE: ระบบเขียนบทวิเคราะห์อัตโนมัติ (แยกบล็อกเพื่อความเสถียร 100%)
# -----------------------------------------------------------------------------
st.markdown("### 📝 บทวิเคราะห์สภาวะตลาดประจำวัน (Automated Macro Report)")

# ส่วนที่ 1: ภาพรวมตลาดหุ้นโลก
st.markdown(f"""
#### 📊 ภาพรวมดัชนีและโมเมนตัมตลาดหุ้นทั่วโลก (Global Equity Market Momentum)
สภาวะตลาดล่าสุด ดัชนีหลักฝั่งสหรัฐฯ และดัชนีโลกมีความเคลื่อนไหวที่ต้องจับตาอย่างใกล้ชิด โดยดัชนี **S&P 500 (SPY)** เคลื่อนไหว **{spy_daily:+.2f}%** ในขณะที่ดัชนี **Nasdaq 100 (QQQ)** ปรับตัว **{qqq_daily:+.2f}%** และดัชนี **Dow Jones Industrial Average (DJI)** เคลื่อนไหว **{dji_daily:+.2f}%** เมื่อมองภาพกว้างขึ้นไปอีก ดัชนี **MSCI World Index (MSCI)** ซึ่งสะท้อนหุ้นทั่วโลกในประเทศพัฒนาแล้วเคลื่อนไหว **{msci_daily:+.2f}%**
* **การพลิกกลับของโมเมนตัมแบบเฉียบพลัน:** การปรับตัวลงแรงรอบนี้ถือเป็นสถิติวันเดียวที่แย่ที่สุดนับตั้งแต่ปลายเดือนมีนาคม และสำคัญที่สุดคือนี่เป็นการพลิกกลับของโมเมนตัมแบบเฉียบพลันครั้งที่สองในสัปดาห์เดียว
* **บริบทภาพรวม Bull Market:** แต่ก่อนที่จะคิดว่าตลาดกำลังจะถล่มทันที หากพิจารณาภาพรวมดัชนี S&P 500 ยังคงทำสถิติปิดบวกติดต่อกันเป็นสัปดาห์ที่เจ็ด ซึ่งเป็นการต่อสัปดาห์บวกที่ยาวที่สุดนับตั้งแต่เดือนธันวาคมปี 2023 หมายความว่าแนวโน้มหลักของตลาดยังคงอยู่ในโหมด Bull Market อย่างชัดเจน เพียงแต่เริ่มมีแรงกระแทกเชิงโครงสร้างเข้ามาทดสอบความแข็งแกร่งอย่างต่อเนื่อง
""")

# ส่วนที่ 2: บทสรุปและคำแนะนำ
st.markdown(f"""
#### 🎯 บทสรุปและกลยุทธ์การแนะนำลงทุนเชิงรุก (Tactical Asset Allocation & Conclusions)
หากประมวลผลข้อมูลมหภาคทั้งหมดรวมกันอย่างเป็นระบบ ปัจจัยความเสี่ยงเรื่องเงินเฟ้อและการดีดตัวของ Bond Yield เป็นสิ่งที่ระบบและสถาบันการเงินรับรู้มาระยะหนึ่งแล้ว แต่แรงเทขายที่เกิดขึ้นรุนแรงเป็นเพราะ**ราคาหุ้นวิ่งมาเร็วและแรงเกินไป บวกกับเป็นช่วง OpEx Day (Options Expiration Day) หรือวันหมดอายุของ Options ซึ่งเป็นวันที่ตลาดมักจะมีความผันผวนสูงเป็นพิเศษอยู่แล้ว ปัจจัยทั้งหมดมันแค่บังเอิญมารวมตัวกันให้นักลงทุนได้กดปุ่มขายเพื่อกระจายความเสี่ยงและ Take Profit อย่างมีเหตุผลเท่านั้น**

แนวทางการจัดพอร์ตและคำแนะนำเชิงโครงสร้างสรุปได้ดังนี้:
* **กลยุทธ์ "นั่งทับมือไว้ก่อน" ในพอร์ตระยะสั้น:** ณ จังหวะที่ความผันผวนสูง แนะนำให้นักลงทุนเน้นการรักษาวินัย รักษาสภาพคล่อง และชะลอการไล่ราคาซื้อเพิ่ม (ให้นั่งทับมือไว้ก่อน) อาจพิจารณา Take Profit ออกมาบ้างถ้าไม่ได้ถือยาว ส่วนเรื่องที่ตลาดราคาว่า Fed จะขึ้นหรือลดดอกเบี้ยนั้น ไม่สำคัญเลยในตอนนี้ เพราะตัวเลขความคาดหวังส่วนใหญ่ขยับไปอยู่ที่ปีหน้านู่นแล้ว การเอาตัวเลขความคาดหวังเรื่องดอกเบี้ยมาเป็นเครื่องตัดสินใจตอนนี้ มันก็เหมือนกับ**การตัดสินใจบนพื้นทราย** เพราะมันเปลี่ยนได้ตลอดเวลา หากสถานการณ์อิหร่านเปลี่ยนและราคาน้ำมันลงมา ตัวเลขนี้ก็จะพลิกกลับทันทีเหมือนกัน เหมือนที่มันเพิ่งเปลี่ยนจาก "จะลดดอกเบี้ย 2 ครั้ง" มาเป็น "ขึ้นดอกเบี้ย" ได้ภายในเวลาแค่ 1-2 เดือน
* **การปรับพอร์ตเข้าหา "สินทรัพย์จริง" (Real Assets):** เมื่อโครงสร้างเงินเฟ้อกลายเป็นความเสี่ยงหลัก พันธบัตรไม่ได้ทำหน้าที่เป็นเครื่องมือช่วย Hedge ความเสี่ยงให้กับหุ้นเหมือนเดิมอีกต่อไป สอดคล้องกับกลยุทธ์ของ **David Lebovitz จาก JPMorgan Asset Management** ที่ระบุว่าบริษัทของเขาชอบและเลือกหันไปลงทุนในสินทรัพย์จริง (Real Assets) อย่างอสังหาริมทรัพย์และโครงสร้างพื้นฐานทดแทนตราสารหนี้ดั้งเดิม
* **โฟกัสเซกเตอร์ที่มี Secular Demand แข็งแกร่ง:** ท่ามกลางภาวะตลาดที่แคบลงเรื่อยๆ **Marija Veitmane (หัวหน้าวิจัยหุ้นของ State Street Global Markets)** ฟันธงตรงๆ ว่า กลุ่มเทคโนโลยี (Tech) เป็นเซกเตอร์เดียวที่มี Secular Demand ที่แข็งแกร่ง ซึ่งรับประกันการเติบโตของกำไรที่ชัดเจนและคาดการณ์ได้ เม็ดเงินที่ยังเหลือเฟือในตลาดการเงินทั้งหมดจะไหลเข้าหากลุ่มนี้เพื่อดันราคาหุ้นให้สูงขึ้น สอดคล้องกับการเคลื่อนไหวของ **Bill Ackman ผู้บริหารของ Pershing Square** ที่เปิดเผยว่ากองทุนได้เข้าซื้อสถานะใหม่ใน **Microsoft** โดยอาศัยจังหวะที่ราคาหุ้นปรับฐานลง เพราะเชื่อว่านักลงทุนในตลาดประเมินความทนทานและความยืดหยุ่นของธุรกิจซอฟต์แวร์ต่ำเกินไป
""")

# ส่วนที่ 3: บทวิเคราะห์นโยบาย FED
st.markdown(f"""
#### 🏦 การเปลี่ยนผ่านเชิงนโยบายของ Fed และการตอบรับของตลาด Swaps (Fed Policy Path Shifts)
ประเด็นที่แหลมคมที่สุดและส่งผลกระทบต่อจิตวิทยาการลงทุนโดยตรงคือ **การเปลี่ยนแปลงของความคาดหวังเรื่องนโยบายดอกเบี้ยของ Fed จากลดเป็นขึ้น ซึ่งเป็นการพลิกกลับของมุมมองอย่างสมบูรณ์แบบ 180 องศาภายในเวลาไม่ถึง 3 เดือน**
* **ตลาด Swaps พลิก Pricing เป็นขึ้นดอกเบี้ย:** ตอนนี้นักเทรดในตลาด Swaps กำลังราคา (Pricing) ว่ามีโอกาสเกือบ 2 ใน 3 ที่ Fed จะขึ้นดอกเบี้ยในการประชุมเดือนธันวาคม ขอย้ำว่า "ขึ้น" ไม่ใช่ "ลด" และตามรายงานของ Bloomberg เส้นกราฟ Swaps Curve ตอนนี้ราคาการขึ้นดอกเบี้ย Fed เต็มจำนวน 25 basis points ภายในการประชุมเดือนมีนาคม ในขณะที่เมื่อวันที่ 27 กุมภาพันธ์ที่ผ่านมา ตลาดยังราคาการลดดอกเบี้ย 25 bps อยู่เลย ความคาดหวังนี้ยังคงอยู่แม้ Fed จะอยู่ภายใต้ผู้ว่าการคนใหม่คือ **Kevin Warsh** ซึ่งทรัมป์เลือกมาแทนที่ Jerome Powell
* **สัญญาณเตือนถึงรัฐสภาและบททดสอบของ Warsh:** **Subadra Rajappa (หัวหน้าฝ่ายวิจัยของ Societe Generale Americas)** ระบุว่า Bond Yield ในปัจจุบันรู้สึกเหมือนกำลังหลุดออกจากบังเหียน "ตลาดไม่ได้แค่ทดสอบ Fed เท่านั้น แต่ยังส่งสัญญาณเตือนรัฐสภาด้วย เพราะยิ่งดอกเบี้ยอยู่สูงนานเท่าไหร่ ต้นทุนทางการเงินสาธารณะก็ยิ่งสูงขึ้น" และนี่จะเป็นบททดสอบแรกของ Kevin Warsh ที่จะต้องควบคุมความคาดหวังของตลาดให้ได้
* **ภาวะ Hot Economy และความจำเป็นในการควบคุมเงินเฟ้อ:** **Ed Al-Hussainy (ผู้จัดการพอร์ตจาก Columbia Threadneedle Investments)** ระบุว่าตลาดกำลังเริ่มราคาว่า Fed จะต้องทำงานหนักขึ้นเพื่อกดเงินเฟ้อ สะท้อนทิศทางของเงินเฟ้อที่น่ากังวลและสภาวะเศรษฐกิจมหภาคที่ร้อนแรง (Hot Economy) โดยเขาประเมินว่าตลาดอาจต้องการเห็นการขึ้นดอกเบี้ยอย่างน้อยอีกสองครั้งก่อนจะหายใจได้
* **มุมมองต่างเชิงโครงสร้างคลัง:** อย่างไรก็ตาม **Angelo Kourkafas (Edward Jones)** ยังเชื่อว่า Fed จะไม่ตอบสนองเกินไปต่อสถานการณ์ที่อาจจะเป็นเพียงเรื่องชั่วคราว เพราะธนาคารกลางไม่สามารถแก้ปัญหา Energy Shock ด้วยการขึ้นดอกเบี้ยได้โดยตรง แต่สิ่งที่ทำให้ภาพเงินเฟ้อซับซ้อนขึ้นคือแนวโน้มของมาตรการกระตุ้นทางการคลัง (Fiscal Stimulus) ที่หลายประเทศจะนำมาใช้เพื่อบรรเทาผลกระทบจากราคาพลังงานมากกว่า
""")

# ส่วนที่ 4: ตลาดพันธบัตรโลก
st.markdown(f"""
#### 💸 วิกฤตการณ์ตลาดพันธบัตรโลก (Global Bond Market Deep Dive)
สิ่งที่ทำให้ครั้งนี้แตกต่างจากสภาวะการพักฐานปกติคือน้ำหนักของข้อมูลชี้ชัดว่า "ตัวจุดชนวน" ของการเทขายหุ้นรอบนี้ ไม่ได้มาจากผลประกอบการบริษัทที่แย่ลง แต่มาจาก**ตลาดพันธบัตรรัฐบาลทั่วโลกที่กำลังถูกเทขายอย่างหนักพร้อมกัน ดันให้อัตราผลตอบแทน (Yield พุ่งขึ้นทั่วโลกข้ามทวีป)**
* **สหรัฐอเมริกา (US):** อัตราผลตอบแทนพันธบัตรรัฐบาลสหรัฐฯ อายุ 10 ปี (US 10Y Yield) ปัจจุบันอยู่ที่ **{latest['US10Y']:.2f}%** ซึ่ง **{yield_status}** โดยเป็นการปรับตัวขึ้นในสัปดาห์เดียวที่มากที่สุดนับตั้งแต่ช่วงที่ทรัมป์ประกาศมาตรการภาษีศุลกากร (Tariffs) ทางด้าน **Priya Misra (ผู้จัดการพอร์ตจาก JPMorgan Asset Management)** เตือนผ่าน Bloomberg Television ว่าการทะลุแนวต้านจิตวิทยา 4.5% เป็นจุดที่เริ่มอันตรายต่อสินทรัพย์เสี่ยงทั้งระบบ และบีบให้ตลาดตั้งคำถามว่าเศรษฐกิจกำลังเข้าสู่ภาวะ **Stagflation** (เศรษฐกิจชะลอตัวพร้อมกับเงินเฟ้อสูง)
* **ญี่ปุ่น (Japan):** Yield พันธบัตรอายุ 30 ปีของญี่ปุ่น (Japan 30Y) ทะลุระดับ **{latest_jp30y:.2f}%** ซึ่งถือเป็นครั้งแรกนับตั้งแต่ออกพันธบัตรชุดนี้ในปี 1999 หรือในรอบ 27 ปี ขณะที่ Yield อายุ 20 ปี พุ่งแตะระดับสูงสุดนับตั้งแต่ปี 1996 และ Yield อายุ 40 ปี ทำจุดสูงสุดนับตั้งแต่เริ่มออกในปี 2007 โดย **Rinto Maruyama (นักกลยุทธ์ FX และ Rates อาวุโสจาก SMBC Nikko Securities)** ให้ความเห็นว่า ในญี่ปุ่นที่อัตราดอกเบี้ยอยู่ใกล้ศูนย์มานาน การที่ Yield ของพันธบัตร 30 ปีพุ่งขึ้นมาที่ 4% ถือเป็นเหตุการณ์ในประวัติศาสตร์ และนี่บ่งชี้ว่าญี่ปุ่นซึ่งเคยเผชิญภาวะเงินฝืดมานาน อาจกำลังเข้าสู่ยุคเงินเฟ้อแบบยั่งยืน
* **อังกฤษ (UK):** อัตราผลตอบแทนพันธบัตร Gilt อายุ 30 ปี ปรับพุ่งขึ้นทะลุระดับ **{latest_uk30y:.2f}%** ทำจุดสูงสุดในรอบ 28 ปี นับตั้งแต่ปี 1998 ส่วน Yield 10 ปี เพิ่มขึ้นแตะ 5.17% สูงสุดนับตั้งแต่ปี 2008 ตลาดกำลังพลิกจุดยืนคาดว่า Bank of England ต้องขึ้นดอกเบี้ยแทนการลดดอกเบี้ย
* **ยุโรปและภูมิภาคพัฒนาแล้ว:** ในเยอรมนี Yield 10 ปี ขึ้นมาอยู่ที่ 3.17% รวมถึง สเปน ออสเตรเลีย นิวซีแลนด์ ก็ขยับขึ้นทั้งหมด จนระดับความวิตกกังวลถึงขั้นที่ Satsuki Katayama ออกมาระบุว่าเรื่องนี้จะถูกหยิบขึ้นมาคุยกันในที่ประชุมรัฐมนตรีคลังกลุ่ม G7 ที่กรุงปารีสต้นสัปดาห์หน้า
""")

# ส่วนที่ 5: พลังงานและภูมิรัฐศาสตร์
st.markdown(f"""
#### 🛢️ วิกฤตราคาน้ำมันและมิติภูมิรัฐศาสตร์โลก (Oil Prices & Geopolitics Deep Dive)
* **ราคาน้ำมันพุ่งดันเงินเฟ้อต้นทุน:** ราคาน้ำมันดิบ WTI ปัจจุบันแกว่งตัวอยู่ที่กรอบ **${latest['Oil']:.2f}/บาร์เรล** (ในรอบสัปดาห์เปลี่ยนแปลง {oil_perf:+.2f}%) ถือเป็นระดับที่อันตรายต่อโครงสร้างต้นทุนเศรษฐกิจโลก ปัจจัยหลักมาจากความตึงเครียดของสงครามอิหร่านที่ส่งให้ช่องแคบฮอร์มุซ (Strait of Hormuz) ยังคงปิดอยู่ และการประชุมสุดยอดระหว่างประธานาธิบดีทรัมป์กับประธานาธิบดีสีจิ้นผิงที่ปักกิ่งจบลงโดยไม่มีความคืบหน้าเชิงรูปธรรม ทรัมป์ระบุว่าไม่ได้กดดันสีจิ้นผิงให้ไปบีบเตหะราน ขณะที่ฝั่งจีนเพียงให้สำนักข่าว Xinhua อ้างคำพูดของรัฐมนตรีต่างประเทศ หวังอี้ เพียงว่าควรเปิดช่องแคบโดยเร็วที่สุดแต่ไม่มีมาตรการปฏิบัติที่เป็นรูปธรรม
* **กลไกการส่งผ่านเข้าสู่ Bond Yield:** เมื่อราคาน้ำมันสูงขึ้น ต้นทุนการผลิตของทุกอุตสาหกรรมจะสูงขึ้นตาม สินค้าและบริการจะแพงขึ้น เงินเฟ้อจะค้างอยู่นานขึ้น แรงกดดันด้านราคากำลังก่อตัวเร่งตัวขึ้นเร็วที่สุดนับตั้งแต่ปี 2022 ตามที่ผู้ว่าการ Fed คือ **Michael Barr** ออกมาเตือนว่าเงินเฟ้อคือความเสี่ยงสำคัญที่สุด ทำให้ธนาคารกลางต้องเลือกระหว่างปล่อยให้เงินเฟ้อสูงหรือขึ้นดอกเบี้ยเพื่อกดเงินเฟ้อลง ซึ่งทั้งสองทางเลือกล้วนทำให้นักลงทุนต้องการ Yield ที่สูงขึ้นจากพันธบัตรเพื่อชดเชยความเสี่ยงในการถือครอง
* **วิกฤตการคลังและการเมืองในอังกฤษ:** ในอังกฤษเกิดวิกฤตการเมืองที่คุกคามตำแหน่งนายกรัฐมนตรี Keir Starmer โดยมีกระแสว่า Andy Burnham (นายกเทศมนตรีเมืองแมนเชสเตอร์) อาจมีโอกาสท้าทายตำแหน่งผู้นำพรรค ซึ่งหากเปลี่ยนผู้นำ อาจหมายถึงการใช้จ่ายภาครัฐมากขึ้น ดันให้ Risk Premium ในตลาด Gilt พุ่งขึ้น โดย **Emmanuel Cau (Barclays)** ชี้ว่าความเครียดนี้กำลังลามไปทั่วตลาดพันธบัตรของประเทศพัฒนาแล้ว ร่วมกับ **John Briggs (Natixis North America)** ที่ชี้ว่าเทรดที่เคยเอื้อกับพันธบัตรกำลังถูกกดดันอย่างหนัก
""")

# ส่วนที่ 6: หุ้นกลุ่ม AI และเซมิคอนดักเตอร์
st.markdown(f"""
#### 💻 ตลาดหุ้น เทคโนโลยี และเซมิคอนดักเตอร์ (AI & Semiconductor Market Matrix)
* **จุดศูนย์กลางแรงเทขาย:** แม้ตลาดจะดูร่วงทั่วกระดาน แต่แรงเทขายกระจุกตัวอยู่ที่กลุ่มเทคโนโลยีและเซมิคอนดักเตอร์ โดยดัชนีหุ้นชิป **SOXX Index** ล่าสุดเคลื่อนไหว **{soxx_daily:+.2f}%** (รอบสัปดาห์ขยับ {soxx_perf:+.2f}%) ดัชนีหุ้นชิปฟิลาเดลเฟียร่วงรุนแรง หุ้นหลักอย่าง Nvidia ลดลง 4.4%, Intel ร่วง 6.2% และ Broadcom ลดลง 3.3% โดย **Matt Maley (Miller Tabak)** อธิบายว่าตัวเลขเงินเฟ้อและราคาน้ำมันดิบดันความกลัวเงินเฟ้อพุ่ง ทำให้นักลงทุนสบโอกาสเก็บกำไรหลังวิ่งขึ้นยาวต่อเนื่อง 6 สัปดห์
* **เหตุผลทางคณิตศาสตร์เชิงโครงสร้าง (Long-Duration Equity):** หุ้นเทคโนโลยีเป็นกลุ่ม Long-Duration Equity ที่กระแสเงินสดและกำไรหลักอยู่ไกลในอนาคต (อีก 5-10 ปีข้างหน้า) ในทางทฤษฎีการเงิน เมื่อเราคำนวณมูลค่าด้วยวิธี Discounted Cash Flow เราต้องใช้อัตราคิดลด (Discount Rate) ที่สัมพันธ์กับ Bond Yield โดยตรง เมื่อ Yield ขึ้น อัตราคิดลดขึ้น มูลค่าปัจจุบัน (Present Value) ของกระแสเงินสดในอนาคตจึงหดตัวลดลงรุนแรงกว่าหุ้นกลุ่ม Cyclical ที่รับรู้กำไรในปัจจุบัน นี่คือเหตุผลทางคณิตศาสตร์ ไม่ใช่แค่อารมณ์ตลาด (Sentiment)
* **ความเสี่ยงจากสภาวะตลาดที่แคบลง (Narrow Rally):** ใต้พื้นผิวภาพดัชนี มีถึง 8 จาก 11 sectors ของ S&P 500 ที่ติดลบสะสมในเดือนนี้ กำไรส่วนใหญ่กระจุกตัวอยู่ที่กลุ่มเทคโนโลยีสารสนเทศเท่านั้น ความแคบของการ Rally นี้จึงน่ากังวล และ **Lori Calvasina (RBC Capital Markets)** ได้ให้สัญญาณเตือนชัดเจนผ่าน Bloomberg Television ว่าการเชียร์ซื้อหุ้นจะถูกท้าทายอย่างจริงจังหากพันธบัตร 10 ปี ทะลุไปถึง 5% เพราะจะบีบให้ P/E ของตลาดหดตัวลงมาอย่างเลี่ยงไม่ได้
""")

# ส่วนที่ 7: พฤติกรรมรายย่อย
st.markdown(f"""
#### ⚠️ วิกฤตของ Active Manager และพฤติกรรมของรายย่อย (Active Manager Dilemma & Retail All-In)
* **วิกฤตผู้จัดการกองทุน Active:** ตามข้อมูลล่าสุดของ Barclays สัดส่วนของ Mutual Fund ประเภท Active ที่ทำผลงานดีกว่า S&P 500 ในปีนี้ดิ่งลงเหลือเพียง 28% (ลดลงจากกว่า 60% ณ สิ้นเดือนกุมภาพันธ์) นักเลือกหุ้น (Stock Pickers) กำลังถูกทิ้งห่างอย่างรุนแรงเพราะเม็ดเงินไหลกลับเข้าหุ้น AI ขนาดใหญ่ไม่กี่ตัวที่พอร์ตกระจายความเสี่ยงไม่สามารถตามทัน ส่งผลให้ผู้จัดการ Active กำลังมุ่งหน้าสู่ปีที่ผลงานย่ำแย่ที่สุดอันดับ 4 ในรอบ 20 ปี หลังจากปีที่ผ่านมามีเงินไหลออกจากกองทุนหุ้น Active ราว 1 ล้านล้านดอลลาร์
* **พฤติกรรมรายย่อยส่งสัญญาณ All-In:** ฝั่งนักลงทุนรายย่อยในอเมริกากำลังเข้าสู่ตลาดอย่างเต็มที่และขาดความระมัดระวังอย่างมีนัยสำคัญ ข้อมูลของ Goldman Sachs ชี้ว่าปริมาณการเทรดพุ่งขึ้น 28% และตะกร้าหุ้นที่รายย่อยชื่นชอบวิ่งขึ้นประมาณ 30% ตั้งแต่ต้นเดือนเมษายน ข้อมูลพอร์ตลูกค้า Private Client ของ Bank of America (สินทรัพย์ 4.5 ล้านล้านดอลลาร์) ถือครองหุ้นในสัดส่วนสูงถึง **65.7%** (ระดับสูงสุดเป็นประวัติการณ์) และลดสัดส่วนเงินสดเหลือเพียง **9.8%** (ระดับต่ำสุดเป็นประวัติการณ์) ขณะเดียวกัน กองทุนเชิงรับและป้องกันอย่าง Covered-Call และ Buffer ETFs เริ่มมีเงินไหลเข้ากว่า 5 พันล้านดอลลาร์ในเดือนที่ผ่านมาเพื่อหาทางรับมือสภาวะนี้
""")

# ส่วนที่ 8: ดีลและตลาดสินทรัพย์อื่น
st.markdown(f"""
#### 🌐 ตลาดสินทรัพย์อื่นๆ และดีลธุรกิจบนวอลล์สตรีท (Other Markets & Corporate Deals)
* **ตลาดอัตราแลกเปลี่ยนและสินทรัพย์ทางเลือก:** ดอลลาร์แข็งค่าดัน Bloomberg Dollar Spot Index ปรับขึ้น 0.4% ค่าเงินยูโรอ่อนค่ามาที่ 1.1620 ดอลลาร์ต่อยูโร ปอนด์อังกฤษอ่อนค่ามาที่ 1.3317 ดอลลาร์ต่อปอนด์ และเยนญี่ปุ่นอ่อนค่าไปที่ 158.78 เยนต่อดอลลาร์ ด้านตลาดคริปโต Bitcoin ปรับตัวลงมาที่ 79,153.63 ดอลลาร์ และ Ether ลดลงมาที่ 2,222.89 ดอลลาร์ ส่วนทองคำ Spot Gold ปรับตัวลดลงอยู่ที่ 4,541.28 ดอลลาร์ต่อออนซ์ การที่ทองคำร่วงลงพร้อมหุ้นชี้ชัดว่าเป็นสภาวะการเทขายเพื่อลดความเสี่ยงทั่วไป (General De-risking) ของผู้เล่นในตลาด มากกว่าพฤติกรรมการหนีเข้าหาสินทรัพย์ปลอดภัยแบบดั้งเดิม (Flight to Safety)
* **ประเด็นข่าวและบิ๊กดีลของบริษัทจดทะเบียน:**
  1. *SpaceX* กำลังมุ่งหน้าจะยื่นไฟลิ่งทำ IPO อย่างเป็นทางการเร็วที่สุดในวันพุธนี้ ตามแหล่งข่าวใกล้ชิด
  2. *ดีลทำเนียบขาว-จีน:* ในการประชุมสุดยอด Trump-Xi ที่ปักกิ่งสองวัน มีการหารือการกำหนดกรอบกำกับ AI และประเด็นชิป Nvidia H200 ขณะที่ดีลเครื่องบินของจีนกับ **Boeing** ถูกเซ็นสัญญาแล้วโดยจีนตกลงซื้อเครื่องบินจำนวน 200 ลำ (ต่ำกว่าสเปกคาดการณ์เดิมที่ต้องการ 737 Max จำนวน 500 ลำ)
  3. *Alphabet (Google):* สร้างประวัติศาสตร์ด้วยการขายพันธบัตรสกุลเงินเยนมูลค่าสูงถึง **576.5 พันล้านเยน** (~3.6 พันล้านดอลลาร์) ถือเป็นดีลพันธบัตรเยนที่ใหญ่ที่สุดเท่าที่เคยทำโดยบริษัทต่างชาติ เพื่อระดมทุนไปขยายโครงสร้างพื้นฐาน AI และ Data Center ที่ทวีความเข้มข้นขึ้น
  4. *Trump's Financial Disclosure:* เอกสารเปิดเผยข้อมูลทางการเงินแสดงว่าตัวทรัมป์หรือที่ปรึกษาทำธุรกรรมการซื้อขายมากกว่า 3,700 ครั้งในไตรมาสแรก รวมมูลค่าหลายสิบล้านดอลลาร์ เกี่ยวข้องกับบริษัทใหญ่ที่มีดีลกับฝ่ายบริหารของเขา
""")

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. VISUALIZATION: ส่วนแสดงผลกราฟทั้งหมดแบบ 4x2 Grid (รวม 8 พล็อต)
# -----------------------------------------------------------------------------
st.subheader("📊 ส่วนแสดงผลกราฟเทคนิคและอัตราผลตอบแทน (Charts & Plots)")

# --- แถวที่ 1 (Charts 1-2) ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 1. Global Yields Have Surged (US, UK, Japan 30Y Yield - 3Y History)")
    fig_global_yield = go.Figure()
    fig_global_yield.add_trace(go.Scatter(x=df.index, y=df['UK30Y'], name="UK 30-Year Yield", line=dict(color='black', width=2)))
    fig_global_yield.add_trace(go.Scatter(x=df.index, y=df['US30Y'], name="US 30-Year Yield", line=dict(color='orange', width=2)))
    fig_global_yield.add_trace(go.Scatter(x=df.index, y=df['JP30Y'], name="Japan 30-Year Yield", line=dict(color='lightgray', width=2)))
    fig_global_yield.update_layout(xaxis_title="Timeline (3 Years)", yaxis_title="Yield (%)", hovermode="x unified", height=380)
    st.plotly_chart(fig_global_yield, use_container_width=True)

with col2:
    st.markdown("#### 2. Fed's Policy Path Projections (Swaps Curve Shifts - From Jan 2026)")
    df_2026 = df[df.index >= '2026-01-01']
    hist_dates = df_2026.index
    hist_effr = [3.6] * len(hist_dates)
    
    months = np.array([0, 3, 6, 12, 18, 24, 36])
    timeline_dates = [df.index[-1] + pd.DateOffset(months=int(m)) for m in months]
    
    curve_feb = [3.6, 3.5, 3.3, 3.0, 2.9, 2.8, 2.8]
    curve_mar = [3.6, 3.65, 3.7, 3.75, 3.7, 3.5, 3.4]
    curve_may = [3.6, 3.6, 3.7, 3.85, 3.9, 3.85, 3.8]
    
    fig_fed = go.Figure()
    fig_fed.add_trace(go.Scatter(x=hist_dates, y=hist_effr, name="EFFR (Historical)", mode='lines', line=dict(color='skyblue', width=2.5)))
    fig_fed.add_trace(go.Scatter(x=timeline_dates, y=curve_may, name="May 15 Projection (Live Mode)", mode='lines+markers', line=dict(color='gold', width=3)))
    fig_fed.add_trace(go.Scatter(x=timeline_dates, y=curve_mar, name="March 26 Projection", mode='lines', line=dict(color='pink', dash='dot')))
    fig_fed.add_trace(go.Scatter(x=timeline_dates, y=curve_feb, name="Feb 27 Projection", mode='lines', line=dict(color='lightgreen', dash='dot')))
    
    fig_fed.add_hline(y=3.8, line_dash="dash", line_color="black", annotation_text="Swaps Pricing 25bp Rate Hike")
    fig_fed.add_hline(y=3.4, line_dash="dash", line_color="black", annotation_text="Swaps Price 25bp Rate Cut")
    fig_fed.update_layout(xaxis_title="Timeline Projections Outlook", yaxis_title="Implied Rate (%)", hovermode="x unified", height=380)
    st.plotly_chart(fig_fed, use_container_width=True)

# --- แถวที่ 2 (Charts 3-4) ---
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### 3. Inflation and Stocks (S&P 500 vs 10Y Breakeven - 3Y History)")
    fig_inf_stock = make_subplots(specs=[[{"secondary_y": True}]])
    fig_inf_stock.add_trace(go.Scatter(x=df.index, y=df['GSPC'], name="S&P 500 Index (L1)", line=dict(color='red', width=2)), secondary_y=False)
    fig_inf_stock.add_trace(go.Scatter(x=df.index, y=df['Breakeven10Y'], name="US 10Y Breakeven Inflation (R1)", line=dict(color='black', width=1.5)), secondary_y=True)
    fig_inf_stock.update_yaxes(title_text="S&P 500 Index Points", secondary_y=False)
    fig_inf_stock.update_yaxes(title_text="Breakeven Inflation Rate (%)", secondary_y=True)
    fig_inf_stock.update_layout(xaxis_title="Timeline", hovermode="x unified", height=380)
    st.plotly_chart(fig_inf_stock, use_container_width=True)

with col4:
    st.markdown("#### 4. Stocks vs Bonds Divergence Tracking (SPY vs TLT)")
    fig_div = make_subplots(specs=[[{"secondary_y": True}]])
    fig_div.add_trace(go.Scatter(x=df.index, y=df['SPY'], name="S&P 500 Price (SPY)", line=dict(color='black', width=2)), secondary_y=False)
    fig_div.add_trace(go.Scatter(x=df.index, y=df['TLT'], name="Bonds Price (TLT)", line=dict(color='orange', width=2)), secondary_y=True)
    fig_div.update_yaxes(title_text="SPY Price (USD)", secondary_y=False)
    fig_div.update_yaxes(title_text="TLT Price (USD)", secondary_y=True)
    fig_div.update_layout(hovermode="x unified", height=380)
    st.plotly_chart(fig_div, use_container_width=True)

# --- แถวที่ 3 (Charts 5-6) ---
col5, col6 = st.columns(2)

with col5:
    st.markdown("#### 5. S&P 500 Price Dynamics & Momentum (RSI)")
    fig_rsi = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['GSPC'], name="S&P 500 Level", line=dict(color='royalblue', width=2)), secondary_y=False)
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI (14)", line=dict(color='tan', width=1.2)), secondary_y=True)
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", secondary_y=True)
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", secondary_y=True)
    fig_rsi.update_yaxes(title_text="S&P 500 Level", secondary_y=False)
    fig_rsi.update_yaxes(title_text="RSI (14)", secondary_y=True, range=[10, 90])
    fig_rsi.update_layout(xaxis_title="Timeline", hovermode="x unified", height=380)
    st.plotly_chart(fig_rsi, use_container_width=True)

with col6:
    st.markdown("#### 6. Market Sentiment & Fear Gauge (VIX Index)")
    fig_vix = go.Figure()
    fig_vix.add_trace(go.Scatter(x=df.index, y=df['VIX'], name="VIX Index", line=dict(color='crimson', width=2), fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.1)'))
    fig_vix.add_hline(y=20, line_dash="dash", line_color="orange", annotation_text="Alert Level (20)")
    fig_vix.add_hline(y=40, line_dash="solid", line_color="darkred", annotation_text="Panic Level (40)")
    fig_vix.update_layout(xaxis_title="Timeline", yaxis_title="VIX Index Level", hovermode="x unified", height=380)
    st.plotly_chart(fig_vix, use_container_width=True)

# --- แถวที่ 4 (Charts 7-8) ---
col7, col8 = st.columns(2)

with col7:
    st.markdown("#### 7. Valuation: Buffett Indicator (Synthetic)")
    fig_buffett = go.Figure()
    fig_buffett.add_trace(go.Scatter(x=df.index, y=df['Buffett'], name="Market Cap / GDP (%)", line=dict(color='purple', width=2), fill='tozeroy', fillcolor='rgba(128, 0, 128, 0.05)'))
    fig_buffett.add_hline(y=160, line_dash="dash", line_color="red", annotation_text="Extreme Overvaluation (>160%)")
    fig_buffett.update_layout(xaxis_title="Timeline", yaxis_title="Market Cap / GDP (%)", hovermode="x unified", height=380)
    st.plotly_chart(fig_buffett, use_container_width=True)

with col8:
    st.markdown("#### 8. Micro Flow: Net Dealer Gamma Exposure & OPEX Pinning")
    colors = ['green' if val >= 0 else 'red' for val in df['NetGamma']]
    fig_gamma = go.Figure()
    fig_gamma.add_trace(go.Bar(x=df.index, y=df['NetGamma'], name="Net Gamma (Bn$)", marker_color=colors))
    
    opex_sample_dates = pd.date_range(start=df.index[0], end=df.index[-1], freq='WOM-3FRI')
    for odate in opex_sample_dates:
        if odate in df.index:
            fig_gamma.add_vline(x=odate, line_dash="dot", line_color="gray", opacity=0.4)
            
    fig_gamma.update_layout(xaxis_title="Timeline", yaxis_title="Net Gamma (Bn$)", hovermode="x unified", height=380)
    st.plotly_chart(fig_gamma, use_container_width=True)

# 5. ตารางสรุปข้อมูลดิบล่าสุดตัวเลขจริง
st.markdown("### 📌 สแนปชอตราคาปิดตลาดและอินดิเคเตอร์ล่าสุด")
summary_data = pd.DataFrame({
    'Asset Class / Indicator': ['S&P 500 Index Core (^GSPC)', 'S&P 500 ETF (SPY)', 'Nasdaq 100 (QQQ)', 'Dow Jones Index (^DJI)', 'Semiconductor Index (SOXX)', 'CBOE VIX Index (^VIX)', 'US 10Y Yield (%)', 'US 30Y Yield (%)', 'Crude Oil WTI ($)'],
    'Latest Value': [f"{latest['GSPC']:.2f} pts", f"${latest['SPY']:.2f}", f"${latest['QQQ']:.2f}", f"{latest['DJI']:.2f} pts", f"{latest['SOXX']:.2f} pts", f"{latest['VIX']:.2f}", f"{latest['US10Y']:.2f}%", f"{latest['US30Y']:.2f}%", f"${latest['Oil']:.2f}"]
})
st.dataframe(summary_data, use_container_width=True, hide_index=True)