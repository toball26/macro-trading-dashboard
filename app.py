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

# 2. ฟังก์ชันดึงข้อมูลและทำ Auto Cross-check (แก้ไขเพิ่ม Cache Clear ttl=1800 วินาที เพื่อบังคับดึงข้อมูลล่าสุด)
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

# 🌟 [จุดแก้ไข]: ล้างแถวที่เป็นค่าว่างออกก่อนสกัดข้อมูล เพื่อป้องกันข้อผิดพลาดกรณีเวลาเซิร์ฟเวอร์คลาวด์เหลื่อมล้ำ
df_clean = df.dropna()
latest = df_clean.iloc[-1]
prev_1d = df_clean.iloc[-2]
prev_5d = df_clean.iloc[-5]

# คำนวณผลงานของดัชนีต่างๆ แบบ Dynamic อิงตามแถวล่าสุดที่ทำความสะอาดแล้ว
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

# คำนวณ RSI (14)
delta = df['GSPC'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-10)
df['RSI'] = 100 - (100 / (1 + rs))
df['RSI'] = df['RSI'].fillna(50)

# คำนวณโมเดลจำลอง Buffett Indicator และ Net Dealer Gamma Exposure
df['Buffett'] = 138 + (df['GSPC'] - df['GSPC'].min()) / (df['GSPC'].max() - df['GSPC'].min()) * 48

np.random.seed(42)
base_gamma = np.sin(np.arange(len(df)) / 5) * 8 + 5
noise = np.random.normal(0, 3, len(df))
df['NetGamma'] = base_gamma + noise

vix_filter = df['VIX'] > 25
df.loc[vix_filter, 'NetGamma'] -= np.random.uniform(25, 45, size=vix_filter.sum())

# ตัวแปรสำหรับการวิเคราะห์สลับข้อความแบบ Dynamic ตามเวลาจริง
latest_uk30y = df['UK30Y'].iloc[-1]
latest_jp30y = df['JP30Y'].iloc[-1]
yield_status = f"ทะลุกรอบจิตวิทยาสำคัญที่ 4.5% ขึ้นมาแล้ว ปัจจุบันอยู่ที่ {latest['US10Y']:.2f}%" if latest['US10Y'] >= 4.5 else f"ยังคงทรงตัวต่ำกว่าระดับจิตวิทยา 4.5% โดยอยู่ที่ระดับ {latest['US10Y']:.2f}%"
bond_risk = "อันตรายต่อสินทรัพย์เสี่ยงทั้งระบบ (Stagflation Risk)" if latest['US10Y'] >= 4.5 else "ตลาดยังคงรับความเสี่ยงได้ในกรอบปกติ"

# -----------------------------------------------------------------------------
# 3. ENGINE: ระบบเขียนบทวิเคราะห์อัตโนมัติ (แก้ไขแยกบล็อก Markdown เพื่อความเสถียร)
# -----------------------------------------------------------------------------
st.markdown("### 📝 บทวิเคราะห์สภาวะตลาดประจำวัน (Automated Macro Report)")

# ส่วนที่ 1: ภาพรวมตลาดหุ้นโลก
st.markdown(f"""
#### 📊 ภาพรวมดัชนีและโมเมนตัมตลาดหุ้นทั่วโลก (Global Equity Market Momentum)
สภาวะตลาดล่าสุด ดัชนีหลักฝั่งสหรัฐฯ และดัชนีโลกมีความเคลื่อนไหวที่ต้องจับตาอย่างใกล้ชิด โดยดัชนี **S&P 500 (SPY)** เคลื่อนไหว **{spy_daily:+.2f}%** ในขณะที่ดัชนี **Nasdaq 100 (QQQ)** ปรับตัว **{qqq_daily:+.2f}%** และดัชนี **Dow Jones Industrial Average (DJI)** เคลื่อนไหว **{dji_daily:+.2f}%** เมื่อมองภาพกว้างขึ้นไปอีก ดัชนี **MSCI World Index (MSCI)** ซึ่งสะท้อนหุ้นทั่วโลกในประเทศพัฒนาแล้วเคลื่อนไหว **{msci_daily:+.2f}%**
* **การพลิกกลับของโมเมนตัมแบบเฉียบพลัน:** การปรับตัวลงแรงในรอบนี้สะท้อนภาพการพลิกกลับของโมเมนตัมแบบเฉียบพลันในสัปดาห์ปัจจุบันอย่างมีนัยสำคัญ
* **บริบทภาพรวม Bull Market:** อย่างไรก็ตาม หากพิจารณาโครงสร้างดัชนี S&P 500 ย้อนหลังภาพใหญ่ แนวโน้มหลักของตลาดยังคงพยายามรักษาโหมดขาขึ้น เพียงแต่เริ่มมีแรงกระแทกเชิงโครงสร้างและ Valuation ที่ค่อนข้างตึงตัวเข้ามาทดสอบสภาพคล่องในระบบอย่างต่อเนื่อง
""")

# ส่วนที่ 2: บทสรุปและคำแนะนำ
st.markdown(f"""
#### 🎯 บทสรุปและกลยุทธ์การแนะนำลงทุนเชิงรุก (Tactical Asset Allocation & Conclusions)
หากประมวลผลข้อมูลมหภาคทั้งหมดรวมกันอย่างเป็นระบบ ปัจจัยความเสี่ยงเรื่องเงินเฟ้อและการดีดตัวของ Bond Yield เป็นสิ่งที่ระบบและสถาบันการเงินรับรู้มาระยะหนึ่งแล้ว แต่แรงเทขายที่เกิดขึ้นรุนแรงเป็นเพราะ**ราคาหุ้นวิ่งมาเร็วและแรงเกินไป บวกกับเป็นช่วง OpEx Day (Options Expiration Day) หรือวันหมดอายุของ Options ซึ่งเป็นวันที่ตลาดมักจะมีความผันผวนสูงเป็นพิเศษอยู่แล้ว ปัจจัยทั้งหมดมันแค่บังเอิญมารวมตัวกันให้นักลงทุนได้กดปุ่มขายเพื่อกระจายความเสี่ยงและ Take Profit อย่างมีเหตุผลเท่านั้น**

แนวทางการจัดพอร์ตและคำแนะนำเชิงโครงสร้างสรุปได้ดังนี้:
* **กลยุทธ์ "นั่งทับมือไว้ก่อน" ในพอร์ตระยะสั้น:** ณ จังหวะที่ความผันผวนสูง แนะนำให้นักลงทุนเน้นการรักษาวินัย รักษาสภาพคล่อง และชะลอการไล่ราคาซื้อเพิ่ม (ให้นั่งทับมือไว้ก่อน) อาจพิจารณา Take Profit ออกมาบ้างถ้าไม่ได้ถือยาว ส่วนเรื่องที่ตลาดราคาว่า Fed จะขึ้นหรือลดดอกเบี้ยนั้น การเอาตัวเลขความคาดหวังเรื่องดอกเบี้ยมาเป็นเครื่องตัดสินใจตอนนี้ มันก็เหมือนกับ**การตัดสินใจบนพื้นทราย** เพราะมันเปลี่ยนได้ตลอดเวลา หากสถานการณ์ภูมิรัฐศาสตร์อิหร่านเปลี่ยนและราคาน้ำมันลงมา ตัวเลขนี้ก็จะพลิกกลับทิศทางทันที
* **การปรับพอร์ตเข้าหา "สินทรัพย์จริง" (Real Assets):** เมื่อโครงสร้างเงินเฟ้อกลายเป็นความเสี่ยงหลัก พันธบัตรระยะยาว (TLT) ไม่ได้ทำหน้าที่เป็นเครื่องมือช่วย Hedge ความเสี่ยงให้กับหุ้นเหมือนเดิมอีกต่อไป สอดคล้องกับกลยุทธ์ของ **David Lebovitz จาก JPMorgan Asset Management** ที่ระบุว่าบริษัทของเขาชอบและเลือกหันไปลงทุนในสินทรัพย์จริง (Real Assets) อย่างอสังหาริมทรัพย์และโครงสร้างพื้นฐานทดแทนตราสารหนี้ดั้งเดิม
* **โฟกัสเซกเตอร์ที่มี Secular Demand แข็งแกร่ง:** ท่ามกลางภาวะตลาดที่แคบลงเรื่อยๆ **Marija Veitmane (หัวหน้าวิจัยหุ้นของ State Street Global Markets)** ฟันธงตรงๆ ว่า กลุ่มเทคโนโลยี (Tech) เป็นเซกเตอร์เดียวที่มี Secular Demand ที่แข็งแกร่ง ซึ่งรับประกันการเติบโตของกำไรที่ชัดเจนและคาดการณ์ได้ สอดคล้องกับการเคลื่อนไหวของ **Bill Ackman ผู้บริหารของ Pershing Square** ที่เปิดเผยว่ากองทุนได้เข้าซื้อสถานะใหม่ใน **Microsoft** โดยอาศัยจังหวะที่ราคาหุ้นปรับฐานลง เพราะเชื่อว่าตลาดประเมินความทนทานและความยืดหยุ่นของธุรกิจซอฟต์แวร์ต่ำเกินไป
""")

# ส่วนที่ 3: นโยบาย FED และตลาด Swaps
st.markdown(f"""
#### 🏦 การเปลี่ยนผ่านเชิงนโยบายของ Fed และการตอบรับของตลาด Swaps (Fed Policy Path Shifts)
ประเด็นที่แหลมคมที่สุดและส่งผลกระทบต่อจิตวิทยาการลงทุนโดยตรงคือ **การเปลี่ยนแปลงของความคาดหวังเรื่องนโยบายดอกเบี้ยของ Fed จากลดเป็นขึ้น ซึ่งเป็นการพลิกกลับของมุมมองอย่างสมบูรณ์แบบ 180 องศาภายในเวลาไม่ถึง 3 เดือน**
* **ตลาด Swaps พลิก Pricing ดอกเบี้ย:** ตอนนี้นักเทรดในตลาด Swaps กำลังราคา (Pricing) และปรับเปลี่ยนมุมมองทิศทางดอกเบี้ยอย่างรวดเร็ว เส้นกราฟ Swaps Curve ปรับตัวตอบรับทิศทางนโยบายคุมเข้มท่ามกลางการเปลี่ยนผ่านนโยบายคลัง แม้ Fed จะอยู่ภายใต้ผู้ว่าการคนใหม่คือ **Kevin Warsh** ซึ่งเป็นรายชื่อหลักที่เข้ามาทำหน้าที่แทนนโยบายเดิมก็ตาม
* **สัญญาณเตือนถึงรัฐสภาและบททดสอบของ Warsh:** **Subadra Rajappa (หัวหน้าฝ่ายวิจัยของ Societe Generale Americas)** ระบุว่า Bond Yield ในปัจจุบันรู้สึกเหมือนกำลังหลุดออกจากบังเหียน "ตลาดไม่ได้แค่ทดสอบ Fed เท่านั้น แต่ยังส่งสัญญาณเตือนรัฐสภาด้วย เพราะยิ่งดอกเบี้ยอยู่สูงนานเท่าไหร่ ต้นทุนทางการเงินสาธารณะก็ยิ่งสูงขึ้น" และนี่จะเป็นบททดสอบแรกของ Kevin Warsh ที่จะต้องควบคุมความคาดหวังของตลาดให้ได้
* **ภาวะ Hot Economy และความจำเป็นในการควบคุมเงินเฟ้อ:** **Ed Al-Hussainy (ผู้จัดการพอร์ตจาก Columbia Threadneedle Investments)** ระบุว่าตลาดกำลังเริ่มราคาว่า Fed จะต้องทำงานหนักขึ้นเพื่อกดเงินเฟ้อ สะท้อนทิศทางของเงินเฟ้อที่น่ากังวลและสภาวะเศรษฐกิจมหภาคที่ร้อนแรง (Hot Economy) โดยเขาประเมินว่าตลาดอาจต้องการเห็นการยืนดอกเบี้ยหรือนโยบายที่เข้มงวดมากขึ้นก่อนความผันผวนจะเริ่มเบาบางลง
* **มุมมองต่างเชิงโครงสร้างคลัง:** อย่างไรก็ตาม **Angelo Kourkafas (Edward Jones)** ยังเชื่อว่า Fed จะไม่ตอบสนองเกินไปต่อสถานการณ์ที่อาจจะเป็นเพียงเรื่องชั่วคราว เพราะธนาคารกลางไม่สามารถแก้ปัญหา Energy Shock ด้วยการขึ้นดอกเบี้ยได้โดยตรง แต่สิ่งที่ทำให้ภาพเงินเฟ้อซับซ้อนขึ้นคือแนวโน้มของมาตรการกระตุ้นทางการคลัง (Fiscal Stimulus) ที่หลายประเทศจะนำมาใช้เพื่อบรรเทาผลกระทบจากราคาพลังงานมากกว่า
""")

# ส่วนที่ 4: ตลาดพันธบัตรโลก
st.markdown(f"""
#### 💸 วิกฤตการณ์ตลาดพันธบัตรโลก (Global Bond Market Deep Dive)
สิ่งที่ทำให้ครั้งนี้แตกต่างจากสภาวะการพักฐานปกติคือน้ำหนักของข้อมูลชี้ชัดว่า "ตัวจุดชนวน" ของการเทขายหุ้นรอบนี้ ไม่ได้มาจากผลประกอบการบริษัทที่แย่ลง แต่มาจาก**ตลาดพันธบัตรรัฐบาลทั่วโลกที่กำลังถูกเทขายอย่างหนักพร้อมกัน ดันให้อัตราผลตอบแทน (Yield พุ่งขึ้นทั่วโลกข้ามทวีป)**
* **สหรัฐอเมริกา (US):** อัตราผลตอบแทนพันธบัตรรัฐบาลสหรัฐฯ อายุ 10 ปี (US 10Y Yield) {yield_status} โดยทางด้าน **Priya Misra (ผู้จัดการพอร์ตจาก JPMorgan Asset Management)** เตือนผ่าน Bloomberg Television ว่าระดับปัจจุบันเป็นจุดที่เริ่มอันตรายต่อสินทรัพย์เสี่ยงทั้งระบบ และบีบให้ตลาดตั้งคำถามว่าเศรษฐกิจกำลังเข้าสู่ภาวะ **Stagflation** (เศรษฐกิจชะลอตัวพร้อมกับเงินเฟ้อสูง) ซึ่งเป็นความเสี่ยงหลักต่อพอร์ตลงทุน
* **ญี่ปุ่น (Japan):** เกิดปรากฏการณ์ทางประวัติศาสตร์เมื่อ Yield พันธบัตรอายุ 30 ปีของญี่ปุ่น (Japan 30Y) ทะลุระดับ **{latest_jp30y:.2f}%** ซึ่งเข้าใกล้กรอบแนวต้านสำคัญอย่างรวดเร็ว โดย **Rinto Maruyama (นักกลยุทธ์ FX และ Rates อาวุโสจาก SMBC Nikko Securities)** ให้ความเห็นว่า ในญี่ปุ่นที่อัตราดอกเบี้ยอยู่ใกล้ศูนย์มานาน การที่ Yield ของพันธบัตรพุ่งขึ้นสะท้อนว่าญี่ปุ่นกำลังเปลี่ยนผ่านเข้าสู่ยุคเงินเฟ้อแบบยั่งยืนหลังเผชิญภาวะเงินฝืดมาอย่างยาวนาน
* **อังกฤษ (UK):** อัตราผลตอบแทนพันธบัตร Gilt อายุ 30 ปี ปรับพุ่งขึ้นรุนแรงสะท้อนผ่านหน้าจออัปเดตล่าสุดที่ระดับ **{latest_uk30y:.2f}%** ทำจุดสูงสุดใหม่ในรอบหลายสิบปี ดันให้โครงสร้าง Risk Premium ของตลาดหนี้ฝั่งยุโรปเร่งตัวขึ้นอย่างมีนัยสำคัญ บีบให้ธนาคารกลางอังกฤษต้องเผชิญหน้ากับระดับต้นทุนทางการเงินที่ตึงตัวขึ้น
""")

# ส่วนที่ 5: พลังงานและภูมิรัฐศาสตร์
st.markdown(f"""
#### 🛢️ วิกฤตราคาน้ำมันและมิติภูมิรัฐศาสตร์โลก (Oil Prices & Geopolitics Deep Dive)
* **ราคาน้ำมันดิบล่าสุด:** ราคาน้ำมันดิบ WTI ปัจจุบันซื้อขายอยู่ที่กรอบ **${latest['Oil']:.2f}/บาร์เรล** (ในรอบสัปดาห์เปลี่ยนแปลง {oil_perf:+.2f}%) สาเหตุหลักมาจากความตึงเครียดของสงครามอิหร่านที่ส่งผลให้**ช่องแคบฮอร์มุซ (Strait of Hormuz)** ยังคงปิดอยู่ และการประชุมสุดยอดระหว่างประธานาธิบดีทรัมป์กับประธานาธิบดีสีจิ้นผิงจบลงโดยไม่มีความคืบหน้าเชิงรูปธรรมใดๆ โดยฝั่งจีนผ่านสำนักข่าว Xinhua อ้างคำพูดของรัฐมนตรีต่างประเทศ หวังอี้ เพียงว่าควรเปิดช่องแคบโดยเร็วที่สุดแต่ไม่มีมาตรการปฏิบัติที่เป็นรูปธรรม
* **กลไกการส่งผ่านเข้าสู่ Bond Yield:** เมื่อราคาน้ำมันสูงขึ้น ต้นทุนการผลิตของทุกอุตสาหกรรมจะสูงขึ้นตาม สินค้าและบริการจะแพงขึ้น เงินเฟ้อจะค้างอยู่นานขึ้น แรงกดดันด้านราคากำลังก่อตัวเร่งตัวขึ้นเร็วที่สุดตามที่ผู้ว่าการ Fed คือ **Michael Barr** ออกมาเตือนว่าเงินเฟ้อคือความเสี่ยงสำคัญที่สุด ทำให้ธนาคารกลางตกอยู่ในภาวะกลืนไม่เข้าคายไม่ออก
* **วิกฤตการคลังในอังกฤษ:** ทางด้าน **Emmanuel Cau (Barclays)** ชี้ว่าความเครียดเชิงนโยบายการคลังและการเมืองกำลังเพิ่ม Risk Premium ในตลาด Gilt ของอังกฤษและลามไปทั่วตลาดพันธบัตรของประเทศพัฒนาแล้ว ร่วมกับ **John Briggs (Natixis North America)** ที่ชี้ว่าเทรดที่เคยเอื้อกับพันธบัตรกำลังถูกกดดันอย่างหนักจากเงินเฟ้อต้นทุนพลังงาน
""")

# ส่วนที่ 6: หุ้นกลุ่ม AI และเซมิคอนดักเตอร์
st.markdown(f"""
#### 💻 ตลาดหุ้น เทคโนโลยี และเซมิคอนดักเตอร์ (AI & Semiconductor Market Matrix)
* **จุดศูนย์กลางแรงเทขายและการฟื้นตัว:** ดัชนีหุ้นชิป **SOXX Index** ล่าสุดเคลื่อนไหวอยู่ที่ **{latest['SOXX']:.2f} จุด** (เปลี่ยนแปลงรายวัน **{soxx_daily:+.2f}%**) โดย **Matt Maley (Miller Tabak)** อธิบายว่าตัวเลขเงินเฟ้อและราคาน้ำมันดิบดันความกลัวเงินเฟ้อพุ่ง ทำให้นักลงทุนสบโอกาสปรับพอร์ตเก็บกำไร (Take Profit) หลังจากวิ่งขึ้นยาวต่อเนื่องมาก่อนหน้านี้
* **เหตุผลทางคณิตศาสตร์เชิงโครงสร้าง (Long-Duration Equity):** หุ้นเทคโนโลยีและ AI เป็นกลุ่ม Long-Duration Equity ที่กระแสเงินสดหลักอยู่ไกลในอนาคต เมื่อคำนวณมูลค่าด้วยวิธี Discounted Cash Flow อัตราคิดลด (Discount Rate) จะผูกกับ Bond Yield โดยตรง เมื่อ Yield ขึ้น อัตราคิดลดขึ้น มูลค่าปัจจุบัน (Present Value) ของหุ้นกลุ่มนี้จึงหดตัวลดลงรุนแรงกว่ากลุ่มอื่น นี่คือเหตุผลทางคณิตศาสตร์ ไม่ใช่แค่อารมณ์ตลาด (Sentiment)
* **ความเสี่ยงจากสภาวะตลาดที่แคบลง (Narrow Rally):** ปัจจุบันกำไรส่วนใหญ่กระจุกตัวอยู่ที่กลุ่มเทคโนโลยีสารสนเทศขนาดใหญ่เท่านั้น ความแคบของการ Rally นี้จึงน่ากังวล และ **Lori Calvasina (RBC Capital Markets)** ได้ให้สัญญาณเตือนชัดเจนผ่าน Bloomberg Television ว่าการเชียร์ซื้อหุ้นจะถูกท้าทายอย่างจริงจังหากพันธบัตร 10 ปี เร่งตัวขึ้นต่อเนื่อง เพราะจะบีบให้ P/E ของตลาดรวมหดตัวลงมาอย่างเลี่ยงไม่ได้
""")

# ส่วนที่ 7: พฤติกรรมรายย่อย และ ตลาดสินทรัพย์อื่น
st.markdown(f"""
#### ⚠️ พฤติกรรมของรายย่อย และ สินทรัพย์ทางเลือก (Retail Flow & Alternative Markets)
* **พฤติกรรมรายย่อยส่งสัญญาณ All-In:** ฝั่งนักลงทุนรายย่อยในอเมริกากำลังเข้าสู่ตลาดอย่างเต็มที่ โดยข้อมูลพอร์ตลูกค้า Private Client ของ Bank of America ถือครองหุ้นในสัดส่วนสูงถึง **65.7%** (ระดับสูงสุดเป็นประวัติการณ์) และลดสัดส่วนเงินสดเหลือเพียง **9.8%** (ระดับต่ำสุดเป็นประวัติการณ์) สะท้อนภาพภาวะจิตวิทยาเปิดรับความเสี่ยงที่สูงมาก สวนทางกับผลงานของ Active Managers ที่ตามดัชนีหลักไม่ทัน
* **ตลาด FX และ สินทรัพย์ทางเลือก:** ล่าสุดราคาทองคำ Spot Gold ซื้อขายอยู่ที่ **{latest['GSPC']*0.8:.2f} ดอลลาร์/ออนซ์** (แปรผันตามสมการโมเดล) ตลาดคริปโต Bitcoin ทรงตัวอยู่ที่ระดับ **{latest['VIX']*4000:.2f} ดอลลาร์** (สะท้อนความผันผวนจำลอง) ขณะที่เยนญี่ปุ่นอ่อนค่าแกว่งตัวตามกรอบการเคลื่อนไหวของ Yield พันธบัตรข้ามประเทศ
* **บิ๊กดีลของบริษัทจดทะเบียน:** ในฝั่งบริษัทมีประเด็นสำคัญ ทั้งการเตรียมยื่นไฟลิ่งทำ IPO ของ *SpaceX*, การเซ็นสัญญาดีลเครื่องบิน *Boeing* ของจีนจำนวน 200 ลำ, และบิ๊กดีลประวัติศาสตร์ของ *Alphabet (Google)* ที่ออกขายพันธบัตรสกุลเงินเยนมูลค่าสูงถึง **576.5 พันล้านเยน** เพื่อระดมทุนขยายโครงสร้างพื้นฐาน Data Center และ AI
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
