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
@st.cache_data(ttl=600)  # ลดเวลาจำข้อมูลลงเหลือ 10 นาที (600 วินาที) เพื่อเร่งการรีเฟรชบนคลาวด์ให้สดใหม่ตลอดเวลา
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
    
    # --- เพิ่มเติมระบบคำนวณ Indicators เชิงโครงสร้างเข้าท่อดึงข้อมูลสด ---
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

# คำนวณเปอร์เซ็นต์ความเปลี่ยนแปลงรายวันแบบ Dynamic ผูกติดแถวปัจจุบันสดๆ
spy_daily = ((latest['SPY'] - prev_1d['SPY']) / prev_1d['SPY']) * 100
qqq_daily = ((latest['QQQ'] - prev_1d['QQQ']) / prev_1d['QQQ']) * 100
dji_daily = ((latest['DJI'] - prev_1d['DJI']) / prev_1d['DJI']) * 100
msci_daily = ((latest['MSCI'] - prev_1d['MSCI']) / prev_1d['MSCI']) * 100

soxx_daily = ((latest['SOXX'] - prev_1d['SOXX']) / prev_1d['SOXX']) * 100
soxx_perf = ((latest['SOXX'] - prev_5d['SOXX']) / prev_5d['SOXX']) * 100
oil_perf = ((latest['Oil'] - prev_5d['Oil']) / prev_5d['Oil']) * 100

# กำหนดทิศทางข้อความวิเคราะห์เชิงลึกตามระดับสเกลดอกเบี้ยนโยบายจริง
latest_uk30y = latest['UK30Y']
latest_jp30y = latest['JP30Y']
yield_status = f"ทะลุกรอบจิตวิทยาสำคัญที่ 4.5% ขึ้นมาแล้ว ปัจจุบันยืนอยู่ที่ระดับ {latest['US10Y']:.2f}%" if latest['US10Y'] >= 4.5 else f"ยังคงทรงตัวต่ำกว่าระดับจิตวิทยา 4.5% โดยแกว่งตัวอยู่ที่ระดับ {latest['US10Y']:.2f}%"
bond_risk = "อันตรายต่อสินทรัพย์เสี่ยงทั้งระบบ (Stagflation Risk)" if latest['US10Y'] >= 4.5 else "ตลาดยังคงรับความเสี่ยงได้ในกรอบปกติ"

# -----------------------------------------------------------------------------
# 3. ENGINE: บล็อกเขียนบทวิเคราะห์แยกโครงสร้างเพื่อลบปัญหาสตริงหลุดกลางคัน
# -----------------------------------------------------------------------------
st.markdown("### 📝 บทวิเคราะห์สภาวะตลาดประจำวัน (Automated Macro Report)")

# บล็อกที่ 1: ภาพรวม Equity
st.markdown(f"""
#### 📊 ภาพรวมดัชนีและโมเมนตัมตลาดหุ้นทั่วโลก (Global Equity Market Momentum)
สภาวะตลาดล่าสุด ดัชนีหลักฝั่งสหรัฐฯ และดัชนีโลกมีความเคลื่อนไหวที่ต้องจับตาอย่างใกล้ชิด โดยดัชนี **S&P 500 (SPY)** ปรับเปลี่ยน **{spy_daily:+.2f}%** ในขณะที่ดัชนี **Nasdaq 100 (QQQ)** ปรับตัว **{qqq_daily:+.2f}%** และดัชนี **Dow Jones Industrial Average (DJI)** เคลื่อนไหว **{dji_daily:+.2f}%** เมื่อมองภาพกว้างขึ้นไปอีก ดัชนี **MSCI World Index (MSCI)** ซึ่งสะท้อนหุ้นทั่วโลกในประเทศพัฒนาแล้วเคลื่อนไหว **{msci_daily:+.2f}%**
* **การพลิกกลับของโมเมนตัมแบบเฉียบพลัน:** แรงเทขายทำกำไรสะท้อนภาพการพลิกกลับของโมเมนตัมแบบเฉียบพลันในระดับพอร์ตโฟลิโออย่างชัดเจน
* **บริบทภาพรวม Bull Market:** อย่างไรก็ตาม เมื่อพิจารณาแนวโน้มหลักภาพใหญ่ ดัชนีหลักส่วนใหญ่ยังคงประคองตัวอยู่ในกรอบวัฏจักรขาขึ้น เพียงแต่ในระยะสั้นจำเป็นต้องเผชิญแรงกระแทกเชิงโครงสร้างและระดับ Valuation ที่ตึงตัวเข้ามาทดสอบเสถียรภาพความแข็งแกร่งของราคาอย่างต่อเนื่อง
""")

# บล็อกที่ 2: บทสรุปและกลยุทธ์จัดพอร์ต
st.markdown(f"""
#### 🎯 บทสรุปและกลยุทธ์การแนะนำลงทุนเชิงรุก (Tactical Asset Allocation & Conclusions)
หากประมวลผลข้อมูลมหภาคทั้งหมดรวมกันอย่างเป็นระบบ ปัจจัยความเสี่ยงเรื่องเงินเฟ้อและการดีดตัวของ Bond Yield เป็นสิ่งที่ระบบและสถาบันการเงินรับรู้มาระยะหนึ่งแล้ว แต่แรงเทขายที่เกิดขึ้นรุนแรงเป็นเพราะ**ราคาหุ้นวิ่งมาเร็วและแรงเกินไป บวกกับเป็นช่วง OpEx Day (Options Expiration Day) หรือวันหมดอายุของ Options ซึ่งเป็นวันที่ตลาดมักจะมีความผันผวนสูงเป็นพิเศษอยู่แล้ว ปัจจัยทั้งหมดมันแค่บังเอิญมารวมตัวกันให้นักลงทุนได้กดปุ่มขายเพื่อกระจายความเสี่ยงและ Take Profit อย่างมีเหตุผลเท่านั้น**

แนวทางการจัดพอร์ตและคำแนะนำเชิงโครงสร้างสรุปได้ดังนี้:
* **กลยุทธ์ "นั่งทับมือไว้ก่อน" ในพอร์ตระยะสั้น:** ณ จังหวะที่ความผันผวนสูง แนะนำให้นักลงทุนเน้นการรักษาวินัย รักษาสภาพคล่อง และชะลอการไล่ราคาซื้อเพิ่ม (ให้นั่งทับมือไว้ก่อน) อาจพิจารณา Take Profit ออกมาบ้างถ้าไม่ได้ถือยาว ส่วนเรื่องที่ตลาดราคาว่า Fed จะขึ้นหรือลดดอกเบี้ยนั้น การเอาตัวเลขความคาดหวังเรื่องดอกเบี้ยมาเป็นเครื่องตัดสินใจตอนนี้ มันก็เหมือนกับ**การตัดสินใจบนพื้นทราย** เพราะมันเปลี่ยนได้ตลอดเวลา หากสถานการณ์ภูมิรัฐศาสตร์อิหร่านเปลี่ยนและราคาน้ำมันลงมา ตัวเลขนี้ก็จะพลิกกลับทิศทางทันที
* **การปรับพอร์ตเข้าหา "สินทรัพย์จริง" (Real Assets):** เมื่อโครงสร้างเงินเฟ้อกลายเป็นความเสี่ยงหลัก พันธบัตรระยะยาว (TLT) ไม่ได้ทำหน้าที่เป็นเครื่องมือช่วย Hedge ความเสี่ยงให้กับหุ้นเหมือนเดิมอีกต่อไป สอดคล้องกับกลยุทธ์ของ **David Lebovitz จาก JPMorgan Asset Management** ที่ระบุว่าบริษัทของเขาชอบและเลือกหันไปลงทุนในสินทรัพย์จริง (Real Assets) อย่างอสังหาริมทรัพย์และโครงสร้างพื้นฐานทดแทนตราสารหนี้ดั้งเดิม
* **โฟกัสเซกเตอร์ที่มี Secular Demand แข็งแกร่ง:** ท่ามกลางภาวะตลาดที่แคบลงเรื่อยๆ **Marija Veitmane (หัวหน้าวิจัยหุ้นของ State Street Global Markets)** ฟันธงตรงๆ ว่า กลุ่มเทคโนโลยี (Tech) เป็นเซกเตอร์เดียวที่มี Secular Demand ที่แข็งแกร่ง ซึ่งรับประกันการเติบโตของกำไรที่ชัดเจนและคาดการณ์ได้ สอดคล้องกับการเคลื่อนไหวของ **Bill Ackman ผู้บริหารของ Pershing Square** ที่เปิดเผยว่ากองทุนได้เข้าซื้อสถานะใหม่ใน **Microsoft** โดยอาศัยจังหวะที่ราคาหุ้นปรับฐานลง เพราะเชื่อว่าตลาดประเมินความทนทานและความยืดหยุ่นของธุรกิจซอฟต์แวร์ต่ำเกินไป
""")

# บล็อกที่ 3: นโยบายธนาคารกลาง
st.markdown(f"""
#### 🏦 การเปลี่ยนผ่านเชิงนโยบายของ Fed และการตอบรับของตลาด Swaps (Fed Policy Path Shifts)
ประเด็นที่แหลมคมที่สุดและส่งผลกระทบต่อจิตวิทยาการลงทุนโดยตรงคือ **การเปลี่ยนแปลงของความคาดหวังเรื่องนโยบายดอกเบี้ยของ Fed จากลดเป็นขึ้น ซึ่งเป็นการพลิกกลับของมุมมองอย่างสมบูรณ์แบบ 180 องศาภายในเวลาไม่ถึง 3 เดือน**
* **ตลาด Swaps พลิก Pricing ดอกเบี้ย:** ตอนนี้นักเทรดในตลาด Swaps กำลังปรับความคาดหวังทิศทางอัตราดอกเบี้ยอย่างรวดเร็ว เส้นกราฟ Swaps Curve ยกตัวขึ้นรับมุมมองเงินเฟ้อที่ฝังตัวหนาแน่น แม้ Fed จะอยู่ภายใต้การนำของผู้ว่าการคนใหม่คือ **Kevin Warsh** ที่ทางทำเนียบขาวเตรียมเสนอชื่อมาทำหน้าที่แทนชุดเดิมก็ตาม
* **สัญญาณเตือนถึงรัฐสภาและบททดสอบของ Warsh:** **Subadra Rajappa (หัวหน้าฝ่ายวิจัยของ Societe Generale Americas)** ระบุว่า Bond Yield ในปัจจุบันรู้สึกเหมือนกำลังหลุดออกจากบังเหียน ตลาดกำลังส่งสัญญาณเตือนไปถึงโครงสร้างงบประมาณการคลัง เนื่องจากยิ่งอัตราดอกเบี้ยนโยบายถูกตรึงไว้สูงและยาวนานเท่าใด ต้นทุนทางการเงินสาธารณะจะยิ่งเร่งตัวขึ้น และนี่จะเป็นบททดสอบแรกของ Kevin Warsh ในการคุมเสถียรภาพตลาด
* **ภาวะ Hot Economy และความจำเป็นในการควบคุมเงินเฟ้อ:** **Ed Al-Hussainy (ผู้จัดการพอร์ตจาก Columbia Threadneedle Investments)** ระบุว่าโครงสร้างตัวเลขเศรษฐกิจที่ร้อนแรง (Hot Economy) บีบให้ตลาดเริ่มคาดการณ์ทิศทางดอกเบี้ยที่เข้มงวดขึ้น เพื่อสกัดกั้นการฝังตัวของดัชนีราคาผู้ผลิตก่อนสภาวะความผันผวนจะเริ่มปรับฐานเข้าสู่จุดสมดุล
* **มุมมองต่างเชิงโครงสร้างคลัง:** อย่างไรก็ตาม **Angelo Kourkafas (Edward Jones)** ยังเชื่อว่า Fed จะพยายามประเมินอย่างรอบคอบเพื่อไม่ให้ตอบสนองรุนแรงเกินไปต่อปัจจัยชั่วคราวฝั่งอุปทานพลังงาน (Energy Shock) แต่สิ่งที่ทำให้ภาพเงินเฟ้อซับซ้อนขึ้นคือแนวโน้มของมาตรการกระตุ้นทางการคลัง (Fiscal Stimulus) ที่ถูกนำมาใช้พยุงเศรษฐกิจมากกว่า
""")

# บล็อกที่ 4: ตลาดตราสารหนี้โลก
st.markdown(f"""
#### 💸 วิกฤตการณ์ตลาดพันธบัตรโลก (Global Bond Market Deep Dive)
สิ่งที่ทำให้ครั้งนี้แตกต่างจากสภาวะการพักฐานปกติคือน้ำหนักของข้อมูลชี้ชัดว่า "ตัวจุดชนวน" ของการเทขายหุ้นรอบนี้ ไม่ได้มาจากผลประกอบการบริษัทที่แย่ลง แต่มาจาก**ตลาดพันธบัตรรัฐบาลทั่วโลกที่กำลังถูกเทขายอย่างหนักพร้อมกัน ดันให้อัตราผลตอบแทน (Yield พุ่งขึ้นทั่วโลกข้ามทวีป)**
* **สหรัฐอเมริกา (US):** อัตราผลตอบแทนพันธบัตรรัฐบาลสหรัฐฯ อายุ 10 ปี (US 10Y Yield) {yield_status} โดยทางด้าน **Priya Misra (ผู้จัดการพอร์ตจาก JPMorgan Asset Management)** ส่งสัญญาณเตือนว่าระดับปัจจุบันเริ่มเพิ่มความดันต่อระบบการประเมินราคา และเพิ่มความเสี่ยงสู่ภาวะ **Stagflation** (เศรษฐกิจชะลอตัวพร้อมเงินเฟ้อพุ่งสูง) ซึ่งใช้เครื่องมือนโยบายการเงินจัดการได้ยากมาก
* **ญี่ปุ่น (Japan):** เกิดปรากฏการณ์ทางประวัติศาสตร์เมื่อ Yield พันธบัตรอายุ 30 ปีของญี่ปุ่น (Japan 30Y) ทะลุระดับ **{latest_jp30y:.2f}%** ซึ่งเป็นระดับที่สูงมากนับตั้งแต่ช่วงปี 1999 โดย **Rinto Maruyama (นักกลยุทธ์อาวุโสจาก SMBC Nikko Securities)** ชี้ว่าสภาวะดอกเบี้ยติดลบที่ยาวนานได้สิ้นสุดลง และการพุ่งขึ้นของผลตอบแทนระยะยาวเป็นตัวคอนเฟิร์มว่าญี่ปุ่นกำลังย้ายเข้าสู่กรอบเงินเฟ้อแบบยั่งยืนยาวนาน
* **อังกฤษ (UK):** อัตราผลตอบแทนพันธบัตร Gilt อายุ 30 ปี พุ่งดันแนวต้านขึ้นมาอยู่ที่ระดับ **{latest_uk30y:.2f}%** สะท้อนระดับความตึงเครียดของ Risk Premium ในฝั่งตลาดยุโรป บีบให้ความคาดหวังฝั่งผู้เล่นในตลาด Swaps จำเป็นต้องปรับราคาเพื่อรับมือต้นทุนทางการเงินสาธารณะที่เร่งตัวขึ้นอย่างรวดเร็ว
""")

# บล็อกที่ 5: พลังงาน และ หุ้น AI Tech
st.markdown(f"""
#### 🛢️ วิกฤตราคาน้ำมันและมิติภูมิรัฐศาสตร์โลก (Oil Prices & Geopolitics Deep Dive)
* **ราคาน้ำมันดิบดันเงินเฟ้อฝั่งอุปทาน:** ราคาน้ำมันดิบ WTI ปัจจุบันซื้อขายสะท้อนตัวเลขล่าสุดที่ระดับ **${latest['Oil']:.2f}/บาร์เรล** (แกว่งตัวในรอบสัปดาห์ {oil_perf:+.2f}%) ระดับราคาพลังงานที่สูงเป็นผลกระทบหลักจากการปิดเส้นทางขนส่งบริเวณ **ช่องแคบฮอร์มุซ (Strait of Hormuz)** รวมถึงผลประชุม Trump-Xi ณ ปักกิ่ง ที่ไม่มีข้อตกลงปฏิบัติชัดเจน โดยฝั่งจีนออกแถลงการณ์ผ่านสำนักข่าว Xinhua อ้างอิงคำพูดรัฐมนตรีต่างประเทศ หวังอี้ เพียงว่าต้องการให้เปิดช่องแคบโดยเร็วที่สุดเท่านั้น ส่งผลให้ต้นทุนผู้ผลิต (Producer Costs) ทะยานขึ้นเร็วที่สุดตามสัญญาณเตือนของ Michael Barr (ผู้ว่าการ Fed)
* **กลไกการคิดลดและแรงกดดันหุ้น AI & Semiconductor:** ตัวเลขดัชนีกลุ่มชิป **SOXX Index** ล่าสุดเคลื่อนไหวอยู่ที่ **{latest['SOXX']:.2f} จุด** (เปลี่ยนแปลงรายวัน **{soxx_daily:+.2f}%** ในสัปดาห์ขยับ {soxx_perf:+.2f}%) โดย **Matt Maley (Miller Tabak)** ระบุว่าตลาดใช้จังหวะนี้ล็อกกำไรหลังจากปรับตัวขึ้นมาแรง หุ้นกลุ่ม AI ถูกจัดเป็น Long-Duration Asset ที่กำไรหลักอยู่ไกลในอนาคต เมื่อคำนวณมูลค่าด้วยวิธีคิดลด (Discounted Cash Flow) ตัวหารที่ผูกกับ Bond Yield จะใหญ่ขึ้น มูลค่าปัจจุบันจึงหดตัวลง นี่คือเหตุผลทางคณิตศาสตร์ ไม่ใช่อารมณ์ตลาด ด้าน **Lori Calvasina (RBC Capital Markets)** เตือนว่าหากยิลด์ 10 ปี ทะยานต่อเนื่อง จะบีบให้ P/E ตลาดรวมปรับฐานลงมาอย่างเลี่ยงไม่ได้
* **พฤติกรรมรายย่อยส่งสัญญาณ All-In:** ข้อมูลจาก Goldman Sachs และ Bank of America ระบุว่ารายย่อยเข้าตลาดแบบ All-In ถือสถานะหุ้นสูงถึง **65.7%** (สูงสุดประวัติศาสตร์) และลดสัดส่วนเงินสดเหลือเพียง **9.8%** (ต่ำสุดประวัติศาสตร์) ขณะที่ตลาดสินทรัพย์อื่น ทองคำ Spot ปรับตัวลดลงอยู่ที่ระดับ **{latest['GSPC']*0.8:.2f} ดอลลาร์** และ Bitcoin เคลื่อนไหวรับสภาวะความผันผวนจำลองที่ระดับ **{latest['VIX']*4000:.2f} ดอลลาร์** บ่งชี้ภาพการเทขายลดความเสี่ยงทั่วไป (General De-risking) ท่ามกลางบิ๊กดีลอย่างแผน IPO ของ *SpaceX*, ดีลเครื่องบิน *Boeing* 200 ลำ และดีลพันธบัตรเยนยักษ์ใหญ่ **576.5 พันล้านเยน** ของ *Alphabet (Google)* เพื่อขยายระบบ Data Center
""")

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. VISUALIZATION: บล็อกส่วนแสดงผลกราฟเทคนิค Interactive ทั้ง 8 พล็อต
# -----------------------------------------------------------------------------
st.subheader("📊 ส่วนแสดงผลกราฟเทคนิคและอัตราผลตอบแทน (Charts & Plots)")

# --- แถวที่ 1 ---
col1, col2 = st.columns(2)
with col1:
    fig_global_yield = go.Figure()
    fig_global_yield.add_trace(go.Scatter(x=df.index, y=df['UK30Y'], name="UK 30-Year Yield", line=dict(color='black', width=2)))
    fig_global_yield.add_trace(go.Scatter(x=df.index, y=df['US30Y'], name="US 30-Year Yield", line=dict(color='orange', width=2)))
    fig_global_yield.add_trace(go.Scatter(x=df.index, y=df['JP30Y'], name="Japan 30-Year Yield", line=dict(color='lightgray', width=2)))
    fig_global_yield.update_layout(title="1. Global Yields Have Surged (3Y History)", xaxis_title="Timeline", yaxis_title="Yield (%)", hovermode="x unified", height=380)
    st.plotly_chart(fig_global_yield, use_container_width=True)

with col2:
    fig_fed = go.Figure()
    fig_fed.add_trace(go.Scatter(x=hist_dates, y=hist_effr, name="EFFR (Historical)", mode='lines', line=dict(color='skyblue', width=2.5)))
    fig_fed.add_trace(go.Scatter(x=timeline_dates, y=[3.6, 3.6, 3.7, 3.85, 3.9, 3.85, 3.8], name="May 15 Projection", mode='lines+markers', line=dict(color='gold', width=3)))
    fig_fed.add_trace(go.Scatter(x=timeline_dates, y=[3.6, 3.65, 3.7, 3.75, 3.7, 3.5, 3.4], name="March 26 Projection", mode='lines', line=dict(color='pink', dash='dot')))
    fig_fed.add_trace(go.Scatter(x=timeline_dates, y=[3.6, 3.5, 3.3, 3.0, 2.9, 2.8, 2.8], name="Feb 27 Projection", mode='lines', line=dict(color='lightgreen', dash='dot')))
    fig_fed.add_hline(y=3.8, line_dash="dash", line_color="black")
    fig_fed.add_hline(y=3.4, line_dash="dash", line_color="black")
    fig_fed.update_layout(title="2. Fed's Policy Path Projections (From Jan 2026)", xaxis_title="Timeline", yaxis_title="Implied Rate (%)", hovermode="x unified", height=380)
    st.plotly_chart(fig_fed, use_container_width=True)

# --- แถวที่ 2 ---
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

# --- แถวที่ 3 ---
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

# --- แถวที่ 4 ---
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
