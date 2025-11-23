import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
import re
from datetime import datetime
from scipy.stats import linregress

# ==========================================
# 這是確保程式碼可以順利執行，但不會被實際調用的導入，因為我們使用 requests 庫
# 這是一個空的佔位符，確保 Python 啟動時不報錯
# ==========================================
class MockGenai:
    def configure(self, api_key): pass
    def GenerativeModel(self, model):
        class MockModel:
            def generate_content(self, prompt):
                raise Exception("Gemini SDK 導入失敗，無法連接 AI 服務。")
        return MockModel()
genai = MockGenai()

# ==========================================
# 0. 頁面設定與初始化
# ==========================================
st.set_page_config(page_title="GALAXY | 區塊鏈羅盤分析 v3.2", layout="wide", page_icon="🧭")

# 初始化 Session State 來儲存連線狀態和 API Key 輸入
if 'gemini_connected' not in st.session_state:
    st.session_state.gemini_connected = False
if 'gemini_message' not in st.session_state:
    st.session_state.gemini_message = ""
if 'api_key_input' not in st.session_state:
    st.session_state.api_key_input = ""
if 'last_used_model' not in st.session_state:
    st.session_state.last_used_model = "N/A" # 儲存實際用於生成報告的模型

# --- 賽博龐克風格 CSS ---
st.markdown("""
<style>
    /* 1. 引入 Google Font (Fira Code - 更具科技感) */
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;500;700&display=swap');

    /* 基礎設置 */
    .stApp {
        background-color: #0d0d0d; 
        color: #00e5ff; 
        font-family: 'Fira Code', monospace; /* 🚀 全局字體替換 */
    }

    /* 2. 深度定制 Sidebar */
    .st-emotion-cache-1d391kg { /* Sidebar 容器 Class */
        background-color: #0d0d0d !important; 
        border-right: 2px solid #ff00ff; /* 霓虹粉色邊框 */
        box-shadow: 2px 0 10px #ff00ff55; /* 霓虹陰影 */
    }
    
    /* Sidebar 內部容器顏色修正 */
    .st-emotion-cache-r69vge { /* Sidebar 內部內容區塊 */
        background-color: #0d0d0d !important;
    }

    /* 全局文本顏色覆蓋 */
    h1, h2, h3, h4, h5, h6, label, .stMarkdown, .stButton>button {
        color: #00e5ff !important; 
    }
    
    /* Sidebar 標題 */
    .css-1d391kg h1 {
        color: #ff00ff !important; 
        text-shadow: 0 0 5px #ff00ff, 0 0 10px #ff00ff; 
    }

    /* Streamlit 原生輸入框 (Text Input, Selectbox) */
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        background-color: #1a1a1a; 
        color: #00e5ff; 
        border: 1px solid #00e5ff; 
        border-radius: 5px;
        box-shadow: 0 0 5px #00e5ff55; 
        font-family: 'Fira Code', monospace;
    }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus {
        border-color: #ff00ff; 
        box-shadow: 0 0 8px #ff00ff;
    }

    /* 3. 分隔線顏色優化 (更明顯的霓虹效果) */
    .st-emotion-cache-1px212h { /* st.divider 的 Class */
        border-top: 1px dashed #ffff00 !important;
    }

    /* 按鈕樣式 (通用) */
    .stButton>button {
        background-color: #1a1a1a;
        color: #00e5ff !important; 
        border: 1px solid #00e5ff; 
        border-radius: 8px;
        font-weight: 500; /* 中等粗細，更優雅 */
        box-shadow: 0 0 5px #00e5ff88;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #00e5ff; 
        color: #1a1a1a !important;
        border-color: #ff00ff;
        box-shadow: 0 0 10px #ff00ff;
    }

    /* 主要分析按鈕 */
    .stButton[data-testid*="stFormSubmitButton"]>button, .stButton>button[data-testid*="primary"] {
        background-color: #ff00ff; 
        color: #1a1a1a !important;
        border: 1px solid #ff00ff;
        box-shadow: 0 0 15px #ff00ff; /* 更強烈的陰影 */
        font-weight: 700; /* 加粗 */
    }
    .stButton[data-testid*="stFormSubmitButton"]>button:hover, .stButton>button[data-testid*="primary"]:hover {
        background-color: #00e5ff; 
        color: #1a1a1a !important;
        border-color: #00e5ff;
        box-shadow: 0 0 20px #00e5ff;
    }

    /* 輔助資訊 (st.caption) 優化 */
    .stText .stCaption {
        color: #ffff00 !important; /* 改為霓虹黃 */
        font-size: 0.75rem; /* 稍微縮小 */
    }

    /* 4. 評分卡 (Score Card) 邊框與陰影優化 */
    .score-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border-radius: 15px; padding: 20px; text-align: center;
        border: 3px solid; /* 邊框加粗 */
        box-shadow: 0 0 20px rgba(0,229,255,0.7); /* 陰影更強 */
    }
    .score-val { 
        font-size: 64px; /* 字體更大 */
        font-weight: 900; 
        margin: 0; line-height: 1;
    }
    
    /* 5. Metric 數據卡片優化 (體積感) */
    div[data-testid="stMetric"] {
        background-color: #1a1a1a;
        border: 1px solid #00e5ff33; 
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 10px;
        box-shadow: 0 0 5px #00e5ff55; 
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important; /* 數據值放大 */
        color: #ffff00 !important; /* 數據值顏色 */
        text-shadow: 0 0 6px #ffff00; /* 霓虹黃數據陰影 */
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 數學算法庫
# ==========================================
class Indicators:
    @staticmethod
    def calc_ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def calc_l3_banker(df):
        close, low, high, open_ = df['close'], df['low'], df['high'], df['open']
        tp = (2 * close + high + low + open_) / 5
        lowest_low = low.rolling(34).min()
        highest_high = high.rolling(34).max()
        denominator = highest_high - lowest_low
        bull_bear = Indicators.calc_ema(
            ((tp - lowest_low) / denominator).replace([np.inf, -np.inf], np.nan).fillna(0) * 100, 
            13
        )
        
        up_diff = close.diff().clip(lower=0)
        down_diff = close.diff().clip(upper=0).abs()
        avg_up = up_diff.rolling(14).mean()
        avg_down = down_diff.rolling(14).mean()
        rs = avg_up / avg_down.replace(0, 1e-9) 
        rsi = 100 - (100 / (1 + rs))
        
        fund_trend = Indicators.calc_ema(rsi, 5)
        
        if len(fund_trend) < 2:
            return {"trend": np.nan, "bull_bear": np.nan, "status": "數據不足", "entry": False}
        
        curr_trend, curr_bb = fund_trend.iloc[-1], bull_bear.iloc[-1]
        status = "莊家控盤 (多)" if curr_trend > curr_bb else "莊家撤退 (空)"
        
        if len(fund_trend) > 1:
            entry_signal = (fund_trend.iloc[-1] > bull_bear.iloc[-1]) and \
                           (fund_trend.iloc[-2] <= bull_bear.iloc[-2]) and \
                           (bull_bear.iloc[-1] < 30)
        else:
            entry_signal = False
            
        return {"trend": curr_trend, "bull_bear": curr_bb, "status": status, "entry": entry_signal}

    @staticmethod
    def calc_log_regression(df, length=100):
        subset = df.tail(length).copy()
        if len(subset) < length: return None
        y = np.log(subset['close'].values)
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        reg_line = np.exp(intercept + slope * x)
        current_reg = reg_line[-1]
        std_dev = subset['close'].std()
        upper = current_reg + (std_dev * 2)
        lower = current_reg - (std_dev * 2)
        price = subset['close'].iloc[-1]
        pos = "強勢區" if price > current_reg else "弱勢區"
        if price > upper: pos = "超買 (壓力)"
        elif price < lower: pos = "超賣 (支撐)"
        trend = "上升" if slope > 0 else "下降"
        return {"reg_price": current_reg, "trend": trend, "position": pos, "upper": upper, "lower": lower}

# ==========================================
# 2. 數據引擎
# ==========================================
class MarketEngine:
    def __init__(self): self.base = "https://fapi.binance.com"
    
    def get_klines(self, symbol, interval, limit=1000):
        try:
            url = f"{self.base}/fapi/v1/klines"
            res = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=5).json()
            df = pd.DataFrame(res).iloc[:, :8]
            df.columns = ["ts", "open", "high", "low", "close", "vol", "ct", "qvol"]
            df = df.astype(float)
            df['time'] = pd.to_datetime(df['ts'], unit='ms')
            return df
        except: 
            return pd.DataFrame()

    def get_metrics(self, symbol):
        try:
            fr_res = requests.get(f"{self.base}/fapi/v1/premiumIndex", params={"symbol": symbol}, timeout=3).json()
            fr = float(fr_res['lastFundingRate'])
            
            oi_res = requests.get(f"{self.base}/fapi/v1/openInterest", params={"symbol": symbol}, timeout=3).json()
            oi = float(oi_res['openInterest'])
            
            depth = requests.get(f"{self.base}/fapi/v1/depth", params={"symbol": symbol, "limit": 50}, timeout=3).json()
            bids = sum([float(x[1]) for x in depth['bids']])
            asks = sum([float(x[1]) for x in depth['asks']])
            ratio = bids / asks if asks > 0 else 1
            return {"fr": fr, "oi": oi, "depth": ratio}
        except: 
            return {"fr": 0, "oi": 0, "depth": 1}

    def get_fng(self):
        try: 
            res = requests.get("https://api.alternative.me/fng/", timeout=3).json()
            return res['data'][0]['value']
        except: 
            return "50"

    def analyze_structure(self, df, trend_bias):
        recent = df.tail(100)
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        diff = swing_high - swing_low
        
        if len(df) < 5: 
            sup_struct, res_struct = df['low'].iloc[-1], df['high'].iloc[-1]
        else:
            sup_struct = df['low'].rolling(5).min().iloc[-3:-1].max()  
            res_struct = df['high'].rolling(5).max().iloc[-3:-1].min()

        df['ema9'] = Indicators.calc_ema(df['close'], 9)
        df['ema13'] = Indicators.calc_ema(df['close'], 13)
        
        fib_levels = {}
        if diff > 0.0001:
            if trend_bias == "BULL":
                fib_levels = {
                    "0.618": swing_high - (diff * 0.618),
                    "0.500": swing_high - (diff * 0.500),
                    "0.382": swing_high - (diff * 0.382),
                    "type": "Support (回調接多)"
                }
            else:
                fib_levels = {
                    "0.618": swing_low + (diff * 0.618),
                    "0.500": swing_low + (diff * 0.500),
                    "0.382": swing_low + (diff * 0.382),
                    "type": "Resistance (反彈做空)"
                }
            
        return {
            "qvol": df['qvol'].iloc[-1],
            "ema9": df['ema9'].iloc[-1], "ema13": df['ema13'].iloc[-1],
            "res_struct": res_struct, "sup_struct": sup_struct,
            "fibs": fib_levels, "swing_high": swing_high, "swing_low": swing_low,
        }

# ==========================================
# 3. AI 分析師 (使用 Requests 庫直接調用 API)
# ==========================================
class AnalystAI:
    def __init__(self, key): 
        self.key = key
        # 降級順序：Pro -> Flash -> Flash 2.0 (將在 API 調用中使用)
        self.models = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash']
    
    def test_connection(self):
        if not self.key: 
            return False, "未輸入 Key", ""
        
        # 使用 requests 庫測試連線到 Gemini API
        test_model = 'gemini-2.5-flash' 
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{test_model}:generateContent?key={self.key}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": "Hello"}]}]}
        
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=8)
            
            if res.status_code == 200 and 'candidates' in res.json():
                return True, "連線成功", test_model
            else:
                error_msg = res.json().get('error', {}).get('message', 'API 連接/驗證錯誤。')
                return False, f"API 驗證失敗：{error_msg}", ''
        except Exception as e:
            return False, f"網絡連線錯誤: {str(e)}", ""

    def generate_report(self, symbol, interval, htf, tech_curr, tech_htf, market, fng, l3, log_reg, struct):
        if not self.key: 
            return {"error": "無 Key"}
        
        # --- 原始數據準備 ---
        qvol_str = f"{struct['qvol']/1000000:.2f}M" if struct['qvol'] > 1000000 else f"{struct['qvol']/1000:.2f}K"
        current_price = tech_curr['close']
        
        # 報告內文會使用 **粗體** 來強調關鍵數據
        prompt = f"""
        你是一位華爾街操盤手。請為 {symbol} ({interval}) 撰寫交易分析報告。
        
        【關鍵數據】
        - 現價: **{current_price:.4f}**
        - 宏觀趨勢 ({htf}): EMA100(**{tech_htf.get('ema100', current_price):.4f}**)
        - Fib黃金位: **{struct['fibs'].get('0.618', current_price):.4f}**
        - 資金量: **{qvol_str}** / 費率: **{market['fr']*100:.4f}%** / 恐慌: **{fng}**
        - SMC支撐/壓力: **{struct['sup_struct']:.4f}** / **{struct['res_struct']:.4f}**

        【任務：輸出單一文本報告】
        請直接輸出一個包含所有信息的純文本報告 (不要使用任何 Markdown 格式, 也不要輸出任何 JSON 結構，只輸出以下內容):
        
        SCORE: [0-100的數字]
        DIRECTION: [LONG/SHORT/WAIT]
        ENTRY: [掛單價格, 精確到4位]
        SL: [止損價格, 精確到4位]
        TP: [止盈價格, 精確到4位]
        
        ANALYSIS_START
        ## 📊 綜合評估
        請根據所有數據，判斷是否 LONG/SHORT/WAIT。如果建議進場，請確保 ENTRY 遵循回調原則 (LONG Entry < 現價; SHORT Entry > 現價)。
        
        ## 🌍 宏觀趨勢 ({htf}) - 長線趨勢分析。
        
        ## 🔬 微觀結構與短線趨勢 - SMC結構與Fib點位解讀。
        
        ## 💰 資金與籌碼 - 資金量、費率與L3資金流向的解讀。
        ANALYSIS_END
        """
        
        # 執行模型降級 (API 降級調用)
        for m in self.models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={self.key}"
            headers = {"Content-Type": "application/json"}
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=30)
                
                if res.status_code == 200:
                    json_data = res.json()
                    if 'candidates' in json_data and len(json_data['candidates']) > 0 and 'parts' in json_data['candidates'][0]['content']:
                        text = json_data['candidates'][0]['content']['parts'][0]['text']
                    else:
                        continue 
                    
                    # 執行結果解析 (保持不變)
                    score_match = re.search(r'SCORE:\s*([\d\.]+)', text, re.IGNORECASE)
                    dir_match = re.search(r'DIRECTION:\s*(LONG|SHORT|WAIT)', text, re.IGNORECASE)
                    entry_match = re.search(r'ENTRY:\s*([\d\.]+)', text, re.IGNORECASE)
                    sl_match = re.search(r'SL:\s*([\d\.]+)', text, re.IGNORECASE)
                    tp_match = re.search(r'TP:\s*([\d\.]+)', text, re.IGNORECASE)
                    
                    report_data = {
                        "score": int(float(score_match.group(1))) if score_match else 0,
                        "direction": dir_match.group(1).upper() if dir_match else "WAIT",
                        "summary_report": text, 
                        "setup": {
                            "entry": float(entry_match.group(1)) if entry_match else "N/A",
                            "sl": float(sl_match.group(1)) if sl_match else "N/A",
                            "tp": float(tp_match.group(1)) if tp_match else "N/A",
                        },
                        "used_model": m
                    }
                    return report_data
                else:
                    continue
            except Exception as e:
                continue
        return {"error": "AI分析失敗或無法解析關鍵數據"}

# ==========================================
# 4. UI 介面 (已優化結構)
# ==========================================
def run_connection_test(api_key):
    tester = AnalystAI(api_key)
    ok, msg, model_name = tester.test_connection()
    
    st.session_state.gemini_connected = ok
    if ok:
        st.session_state.gemini_message = f"✅ 連線成功！**{msg}**。測試模型: `{model_name}`"
    else:
        st.session_state.gemini_message = f"❌ 連線失敗！原因: {msg}"

with st.sidebar:
    st.title("GALAXY | 區塊鏈羅盤分析 v3.2")
    
    api_key = st.text_input("Gemini API Key", type="password", 
                            value=st.session_state.api_key_input, 
                            key="api_key_input_widget")
    
    st.session_state.api_key_input = api_key

    st.button("🔌 連線測試", on_click=run_connection_test, args=(api_key,), use_container_width=True)
    
    if st.session_state.gemini_message:
        if st.session_state.gemini_connected:
            st.success(st.session_state.gemini_message)
        else:
            st.error(st.session_state.gemini_message)
    
    st.divider()
    
    st.markdown("### 查詢幣種")
    symbol_in = st.text_input("輸入幣種代碼 (例如 BTC, XRP)", "XRP").upper()
    symbol = f"{symbol_in}USDT" if not symbol_in.endswith("USDT") else symbol_in
    
    st.markdown("### 交易週期")
    # 這裡只保留一行正確的字典定義
    tf_map = {"15m": "1h", "1h": "4h", "4h": "1d"} 
    interval = st.selectbox("選擇分析週期", list(tf_map.keys()), index=0)
    htf = tf_map[interval]
    
    st.caption(f"自動對應大局觀週期: {htf}")
    
    analyze_btn = st.button("🔍 進行分析", type="primary", use_container_width=True)

if analyze_btn and api_key:
    engine = MarketEngine()
    ai = AnalystAI(api_key)
    
    with st.spinner(f"正在將所有技術數據餵給 GALAXY AI 分析..."):
        df_curr = engine.get_klines(symbol, interval)
        df_htf = engine.get_klines(symbol, htf)
        
        if df_curr.empty or df_htf.empty: st.error(f"數據獲取失敗，請檢查幣種 {symbol} 或週期是否正確。"); st.stop()
            
        htf_close = df_htf['close']
        tech_htf = {
            "ema20": Indicators.calc_ema(htf_close, 20).iloc[-1] if len(htf_close) >= 20 else np.nan,
            "ema50": Indicators.calc_ema(htf_close, 50).iloc[-1] if len(htf_close) >= 50 else np.nan,
            "ema100": Indicators.calc_ema(htf_close, 100).iloc[-1] if len(htf_close) >= 100 else np.nan
        }
        
        curr_price = df_curr['close'].iloc[-1]
        ema100_htf = tech_htf.get('ema100', curr_price)
        trend_bias = "BULL" if curr_price > ema100_htf else "BEAR"

        struct_data = engine.analyze_structure(df_curr, trend_bias)
        tech_curr = {
            "close": curr_price, "qvol": struct_data['qvol'],
            "high": struct_data['swing_high'], "low": struct_data['swing_low'], 
            "fib": struct_data['fibs'], "ema9": struct_data['ema9'], "ema13": struct_data['ema13']
        }
        
        l3_res = Indicators.calc_l3_banker(df_curr)
        log_res = Indicators.calc_log_regression(df_curr)
        market = engine.get_metrics(symbol)
        fng = engine.get_fng()
        
        # 3. AI 分析
        report = ai.generate_report(symbol, interval, htf, tech_curr, tech_htf, market, fng, l3_res, log_res, struct_data)
        
        if "error" in report: st.error(f"AI 分析失敗: {report['error']}"); st.stop()
        else:
            used_model = report.get('used_model', 'N/A')
            st.session_state.last_used_model = used_model
            st.success(f"✅ AI 分析報告生成完成！模型: **{used_model}**")
            
            # --- 顯示報告 ---
            score = report.get('score', 0)
            s_color_hex = "#00e5ff" if score >= 75 else ("#ff00ff" if score <= 40 else "#ffff00")
            direction = report.get('direction', 'WAIT')
            
            c1, c2 = st.columns([1, 3])
            
            # 評分卡 (動態顏色與陰影)
            with c1: st.markdown(f"""
                <div class='score-card' style='border-color: {s_color_hex}; box-shadow: 0 0 15px {s_color_hex}aa;'>
                    <div class='score-val' style='color: {s_color_hex}; text-shadow: 0 0 8px {s_color_hex}, 0 0 15px {s_color_hex}aa;'>{score}</div>
                    <div class='score-label'>AI 信心評分</div>
                </div>
            """, unsafe_allow_html=True)
            
            with c2: st.markdown(f"## {symbol} 深度分析報告", unsafe_allow_html=True); st.subheader(f"週期: {interval} | 當前價格: {curr_price:.4f}")
            
            # 輔助數據總覽 (Metric 優化排版)
            st.markdown("---")
            col_data_top = st.columns(4)
            col_data_bottom = st.columns(5) # 雖然只用一個，但保留 col_data_bottom[0] 讓排版更靈活
            
            vol_str = f"{struct_data['qvol']/1000000:.2f}M" if struct_data['qvol'] > 1000000 else f"{struct_data['qvol']/1000:.2f}K"
            
            col_data_top[0].metric("成交額 (資金量)", f"${vol_str}", help="當前週期的 USDT 成交總額")
            col_data_top[1].metric("資金費率", f"{market.get('fr', 0)*100:.4f}%")
            col_data_top[2].metric("買賣比", f"{market.get('depth', 1):.2f}", help="深度圖 Bid/Ask 交易量比")
            col_data_top[3].metric("恐慌指數", f"{fng}")
            col_data_bottom[0].metric("L3資金流狀態", f"{l3_res['status']}")
            
            # 詳細報告區 
            c_l, c_r = st.columns([2, 1])
            with c_l:
                st.markdown("<div class='report-container'>", unsafe_allow_html=True)
                st.markdown(f"<div class='report-header'>🎯 總結分析 (AI評估方向: {direction})</div>", unsafe_allow_html=True)
                
                raw_text = report.get('summary_report', 'AI未提供完整分析報告。')
                content_match = re.search(r'ANALYSIS_START\s*(.*?)\s*ANALYSIS_END', raw_text, re.DOTALL | re.IGNORECASE)
                if content_match:
                    analysis_content = content_match.group(1).strip()
                    # 渲染 Markdown 標題，並保留您的 report-text div
                    st.markdown(f"<div class='report-text'>{analysis_content}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='report-text'>{raw_text}</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

            with c_r:
                st.markdown("<div class='report-header'>🎯 交易計畫 (SETUP)</div>", unsafe_allow_html=True)
                setup = report.get('setup', {})
                
                # 交易方向標籤
                dir_class = "dir-long" if direction == "LONG" else ("dir-short" if direction == "SHORT" else "dir-wait")
                st.markdown(f"<div class='direction-tag {dir_class}'>建議方向: {direction}</div>", unsafe_allow_html=True)

                # 交易計畫 Metric 橫向排版 (視覺升級)
                col_setup = st.columns(3)
                col_setup[0].metric("掛單 (Entry)", f"{setup.get('entry', 'N/A'):.4f}")
                col_setup[1].metric("止損 (SL)", f"{setup.get('sl', 'N/A'):.4f}")
                col_setup[2].metric("止盈 (TP)", f"{setup.get('tp', 'N/A'):.4f}")
                
                st.markdown("---")
                
                # 關鍵點位總覽 (使用 Expander 提高整潔度)
                with st.expander("🧮 關鍵點位總覽"):
                    fib_0618 = struct_data['fibs'].get('0.618', 'N/A')
                    st.metric("Fib 0.618", f"{fib_0618:.4f}" if isinstance(fib_0618, float) else "N/A")
                    st.metric("SMC 壓力位 (R)", f"{struct_data['res_struct']:.4f}")
                    st.metric("SMC 支撐位 (S)", f"{struct_data['sup_struct']:.4f}")
                
                # EMA 趨勢參考
                with st.expander("📊 EMA 趨勢參考"):
                    ema_text = (
                        f"{tech_htf['ema20']:.4f}" if not np.isnan(tech_htf['ema20']) else "N/A"
                    ) + " / " + (
                        f"{tech_htf['ema50']:.4f}" if not np.isnan(tech_htf['ema50']) else "N/A"
                    ) + " / " + (
                        f"{tech_htf['ema100']:.4f}" if not np.isnan(tech_htf['ema100']) else "N/A"
                    )
                    st.metric(f"宏觀 {htf} EMA20/50/100", ema_text)
                    st.metric("微觀 EMA9/13", f"{struct_data['ema9']:.4f} / {struct_data['ema13']:.4f}")

elif not api_key:
    if not st.session_state.gemini_connected:
        st.info("👈 請先輸入 Gemini API Key，然後點擊「連線測試」按鈕。")
elif analyze_btn and not api_key:
    st.error("請輸入 Gemini API Key 後再進行分析！")
