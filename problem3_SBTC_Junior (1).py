
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from FiinQuantX import FiinSession
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# ===================== CONFIG =====================
st.set_page_config(page_title="FiinQuant – Stock Signals Dashboard", layout="wide")
# =================== SBTC JUNIOR LOGO (CỐ ĐỊNH) ===================


def build_sbtc_logo_svg():
    return '''
<svg width="280" height="88" viewBox="0 0 560 176" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="560" height="176" rx="18" fill="black" fill-opacity="0.15"/>
  <!-- Khiên -->
  <g transform="translate(20,18)">
    <path d="M60 0 L120 22 V74 C120 106 94 132 60 142 C26 132 0 106 0 74 V22 Z"
          fill="none" stroke="#22D3EE" stroke-width="6"/>
    <!-- Sóng giá -->
    <polyline points="12,78 30,60 46,74 64,46 84,58 104,34"
              fill="none" stroke="#A78BFA" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="30" cy="60" r="5" fill="#A78BFA"/>
    <circle cx="64" cy="46" r="5" fill="#22D3EE"/>
    <circle cx="104" cy="34" r="5" fill="#A78BFA"/>
  </g>

  <!-- Chữ SBTC Junior chính -->
  <g font-family="Inter,Segoe UI,system-ui" font-weight="800" fill="#E5E7EB">
    <text x="170" y="74" font-size="60" letter-spacing="2">
      <tspan fill="#22D3EE">S</tspan><tspan>BTC</tspan>
      <tspan dx="12" dy="-10" font-size="26" fill="#A78BFA">Junior</tspan>
    </text>
  </g>
</svg>
'''.strip()

logo_svg = build_sbtc_logo_svg()
st.markdown(f"""
<style>
#sbtc-logo-wrap {{
  position: fixed;
  top: 70px;   /* 👈 chỉnh xuống dưới, mặc định 14px */
  left: 14px;
  z-index: 99999;
  border-radius: 20px;
  box-shadow: 0 8px 24px rgba(0,0,0,.35);
  background: rgba(0,0,0,0.0);
  pointer-events: none; /* không che nút bấm bên dưới */
}}
@media (max-width: 768px) {{
  #sbtc-logo-wrap {{ transform: scale(.8); transform-origin: top left; top: 60px; }}
}}
</style>
<div id="sbtc-logo-wrap">{logo_svg}</div>
""", unsafe_allow_html=True)
# ------------------ LOGIN ------------------
username = 'DSTC_12@fiinquant.vn'
password = 'Fiinquant0606'
client = FiinSession(username=username, password=password).login()

# ------------------ STYLE ------------------
st.markdown(
    """
    <style>
    .main {background: #0f172a;}
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {color: #e5e7eb !important;}
    .metric-box {background:#111827;border:1px solid #1f2937;border-radius:16px;padding:16px}
    .dataframe tbody tr:nth-child(even) {background-color: #0b1220}
    .dataframe {color:#e5e7eb}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ HEADER ------------------
st.title("FiinQuant – Dashboard Đánh Giá & Tín hiệu")
st.caption("Nguồn dữ liệu: FiinQuantX | Khung tín hiệu: 30 phút")

# ===================== SIDEBAR =====================
# ================== BỘ LỌC CHÍNH ==================
st.sidebar.header("⚙️ Cấu hình")

TOP_DEFAULT = ['VCB', 'ACB', 'VNR', 'BVH', 'VHM', 'BCM', 'AST', 'DGW', 'CMG', 'ONE']
all_tickers = TOP_DEFAULT

# Lựa chọn danh sách mã
selected_tickers = st.sidebar.multiselect(
    "Chọn mã theo dõi",
    options=all_tickers,
    default=all_tickers,
)

# Bộ lọc thời gian
start_from = st.sidebar.date_input(
    "Lọc từ ngày",
    value=pd.to_datetime("2025-03-01").date()
)

# Số lượng thanh 30 phút (period)
period = st.sidebar.slider(
    "Số thanh 30m (period)", 
    min_value=500, 
    max_value=3000, 
    value=2300, 
    step=100
)

# Tuỳ chọn hiển thị thêm
show_obv = st.sidebar.toggle("Hiển thị OBV panel phụ", value=False)
# ===================== FUNCTIONS =====================
@st.cache_data(show_spinner=True)
def fetch_30m(tickers: list, period: int) -> pd.DataFrame:
    data = client.Fetch_Trading_Data(
        realtime=False,
        tickers=tickers,
        fields=["open","high","low","close","volume","bu","sd","fb","fs","fn"],
        adjusted=True,
        by="30m",
        period=period,
        lasted=True,
    ).get_data()
    # đảm bảo kiểu thời gian
    data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.tz_localize(None)
    return data


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # SMA
    for w in [50, 100, 150, 200]:
        df[f"SMA{w}"] = df["close"].rolling(window=w).mean()
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    # RSI14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    RS = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + RS))
    # OBV
    df["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["OBV_EMA20"] = df["OBV"].ewm(span=20, adjust=False).mean()
    # Buy/Sell rules
    df["Buy_Signal"] = ((df["close"] > df["SMA150"]) &
                         (df["MACD"] > df["Signal"]) & (df["MACD"] > 0) &
                         (df["RSI14"].between(45, 65)) &
                         (df["OBV"] > df["OBV_EMA20"])).astype(int)
    df["Sell_Signal"] = ((df["close"] < df["SMA100"]) &
                          (df["MACD"] < df["Signal"]) & (df["MACD"] < 0)).astype(int)
    # Position
    pos = 0
    positions = []
    for _, row in df.iterrows():
        if row["Buy_Signal"] == 1 and pos == 0:
            pos = 1
        elif row["Sell_Signal"] == 1 and pos == 1:
            pos = 0
        positions.append(pos)
    df["Position"] = positions
    return df

# ===================== DATA =====================
if len(selected_tickers) == 0:
    st.info("Hãy chọn ít nhất 1 mã ở sidebar.")
    st.stop()

with st.spinner("Đang tải dữ liệu 30m từ FiinQuant..."):
    raw = fetch_30m(selected_tickers, period)

ticker_frames = {tk: add_features(df.reset_index(drop=True))
                 for tk, df in raw.groupby("ticker")}

# Lọc theo ngày bắt đầu
start_dt = pd.to_datetime(start_from)
for tk in list(ticker_frames.keys()):
    df_t = ticker_frames[tk]
    df_t = df_t[df_t["timestamp"] >= start_dt]
    ticker_frames[tk] = df_t.reset_index(drop=True)

# ===================== OVERVIEW METRICS =====================
st.subheader("📌 Tổng quan nhanh")
cols = st.columns(4)
first_tk = selected_tickers[0]
last_close = ticker_frames[first_tk]["close"].iloc[-1] if len(ticker_frames[first_tk]) else np.nan
prev_close = ticker_frames[first_tk]["close"].iloc[-2] if len(ticker_frames[first_tk]) > 1 else np.nan
chg = ((last_close/prev_close - 1)*100) if pd.notna(prev_close) else 0
cols[0].markdown(f"<div class='metric-box'><h4>{first_tk}</h4><h2>{last_close:,.0f}</h2><p>Close gần nhất</p></div>", unsafe_allow_html=True)
cols[1].markdown(f"<div class='metric-box'><h4>Thay đổi</h4><h2>{chg:,.2f}%</h2><p>So với thanh trước</p></div>", unsafe_allow_html=True)
cols[2].markdown(f"<div class='metric-box'><h4>Số mã đang nắm giữ</h4><h2>{sum((f["Position"].iloc[-1] if len(f) else 0) for f in ticker_frames.values())}</h2><p>Position==1</p></div>", unsafe_allow_html=True)
cols[3].markdown(f"<div class='metric-box'><h4>Mã được chọn</h4><h2>{len(selected_tickers)}</h2><p>theo dõi</p></div>", unsafe_allow_html=True)

# ===================== TABS =====================
tab1, tab2, tab3, tab_lstm, tab_thresholds,tab_similar = st.tabs(["📈 Biểu đồ & Tín hiệu", "📋 Tóm tắt tín hiệu", "🧾 Dữ liệu chi tiết","🤖 LSTM dự báo (Daily)","🎯 Ngưỡng giá (1D)","🧩 Similar Chart (1D)"]) 

# ---- TAB 1: CHARTS ----
with tab1:
    st.markdown("### Biểu đồ giá (30m) + Buy/Sell + SMA")
    n = len(selected_tickers)
    cols_per_row = 2
    rows = (n + cols_per_row - 1)//cols_per_row
    for r in range(rows):
        row_cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            idx = r*cols_per_row + c
            if idx >= n: break
            tk = selected_tickers[idx]
            df_t = ticker_frames[tk]
            if df_t.empty:
                row_cols[c].warning(f"Không có dữ liệu cho {tk}")
                continue
            # Plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_t["timestamp"], y=df_t["close"], name="Close", mode="lines", line=dict(color="#60a5fa")))
            for w, col in [(50, "#a78bfa"), (100, "#22d3ee"), (150, "#f472b6"), (200, "#34d399")]:
                if df_t[f"SMA{w}"].notna().any():
                    fig.add_trace(go.Scatter(x=df_t["timestamp"], y=df_t[f"SMA{w}"], name=f"SMA{w}", mode="lines", line=dict(width=1, color=col)))
            # Buy/Sell markers
            buys = df_t[df_t["Buy_Signal"]==1]
            sells= df_t[df_t["Sell_Signal"]==1]
            fig.add_trace(go.Scatter(x=buys["timestamp"], y=buys["close"], mode="markers", name="Buy", marker=dict(symbol="triangle-up", size=10, color="#10b981", line=dict(color="#064e3b", width=1))))
            fig.add_trace(go.Scatter(x=sells["timestamp"], y=sells["close"], mode="markers", name="Sell", marker=dict(symbol="triangle-down", size=10, color="#ef4444", line=dict(color="#7f1d1d", width=1))))
            fig.update_layout(height=400, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#e5e7eb"))
            row_cols[c].plotly_chart(fig, use_container_width=True)
            if show_obv:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df_t["timestamp"], y=df_t["OBV"], name="OBV", line=dict(color="#f59e0b")))
                fig2.add_trace(go.Scatter(x=df_t["timestamp"], y=df_t["OBV_EMA20"], name="OBV_EMA20", line=dict(color="#fde68a")))
                fig2.update_layout(height=160, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#e5e7eb"))
                row_cols[c].plotly_chart(fig2, use_container_width=True)

# ---- TAB 2: SUMMARY ----
with tab2:
    st.markdown("### Tóm tắt tín hiệu hiện tại")
    rows = []
    for tk, df_t in ticker_frames.items():
        if df_t.empty: continue
        rows.append({
            "ticker": tk,
            "Close": df_t["close"].iloc[-1],
            "Buy tín hiệu gần nhất": int(df_t["Buy_Signal"].iloc[-1]),
            "Sell tín hiệu gần nhất": int(df_t["Sell_Signal"].iloc[-1]),
            "Position": int(df_t["Position"].iloc[-1]),
            "RSI14": round(df_t["RSI14"].iloc[-1], 2) if pd.notna(df_t["RSI14"].iloc[-1]) else None,
        })
    if len(rows):
        df_sum = pd.DataFrame(rows).sort_values(["Position","Buy tín hiệu gần nhất"], ascending=False)
        st.dataframe(df_sum, use_container_width=True)
        st.download_button("📥 Tải CSV tóm tắt", df_sum.to_csv(index=False).encode("utf-8"), file_name="signal_summary.csv")
    else:
        st.info("Chưa có dữ liệu để tóm tắt.")

# ---- TAB 3: RAW ----
with tab3:
    st.markdown("### Dữ liệu chi tiết (per ticker)")
    pick = st.selectbox("Chọn mã để xem dữ liệu", options=selected_tickers)
    st.dataframe(ticker_frames[pick], use_container_width=True)
    st.download_button("📥 Tải CSV dữ liệu chi tiết", ticker_frames[pick].to_csv(index=False).encode("utf-8"), file_name=f"{pick}_30m.csv")
# ---- TAB 4: LSTM ----
with tab_lstm :
    st.markdown("### 🤖 LSTM – Giá thật vs. Dự đoán (1D)")

    # ----- MẶC ĐỊNH CỐ ĐỊNH (có thể sửa trực tiếp trong code) -----
    lstm_tickers = ['VCB','ACB','DGW']        # ➜ sửa list nếu muốn
    from_date   = "2023-01-01"                # ➜ sửa mốc bắt đầu
    lookback    = 30
    train_months= 12
    test_months = 3
    epochs      = 15
    batch_size  = 64

    @st.cache_data(show_spinner=True)
    def fetch_daily_fixed(tickers: list, from_date_str: str) -> pd.DataFrame:
        data = client.Fetch_Trading_Data(
            realtime=False,
            tickers=tickers,
            fields=['open','high','low','close','volume'],
            adjusted=True,
            by='1d',
            from_date=from_date_str
        ).get_data()
        data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.tz_localize(None)
        return data

    def prepare_features(data: pd.DataFrame):
        df = data.copy()
        df['return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['ret_ma5']  = df['return'].rolling(5).mean()
        df['ret_ma10'] = df['return'].rolling(10).mean()
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['vol_ratio']= df['volume'] / df['vol_ma20']
        df['volatility5']  = df['return'].rolling(5).std()
        df['volatility20'] = df['return'].rolling(20).std()

        # RSI14
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df['rsi14'] = 100 - (100 / (1 + rs))

        # MACD + Signal
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # ATR14
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr14'] = tr.rolling(14).mean()

        cols_to_scale = ['open','high','low','close','volume','return','log_return','ret_ma5','ret_ma10',
                         'vol_ma20','vol_ratio','volatility5','volatility20','rsi14','macd','macd_signal','atr14']

        scaler_all = RobustScaler()
        df_scaled = df.copy()
        df_scaled[cols_to_scale] = scaler_all.fit_transform(df[cols_to_scale])

        scaler_close = RobustScaler()
        df_scaled['close'] = scaler_close.fit_transform(df[['close']])

        return df, df_scaled, scaler_close

    def run_lstm_walkforward(df_goc, df_scaled,
                             features=['close','return','volume','rsi14','macd','macd_signal'],
                             target='close',
                             lookback=30, train_months=12, test_months=3,
                             days_in_month=21, epochs=15, batch_size=64):
        X_all = df_scaled[features].values
        y_all = df_scaled[target].values

        def seq(X, y, L):
            Xs, ys = [], []
            for i in range(len(X) - L):
                Xs.append(X[i:i+L]); ys.append(y[i+L])
            return np.array(Xs), np.array(ys)

        results = []
        start_idx = 0
        while start_idx + (train_months+test_months)*days_in_month < len(df_scaled):
            train_end = start_idx + train_months*days_in_month
            test_end  = train_end + test_months*days_in_month

            Xtr = X_all[start_idx:train_end]; ytr = y_all[start_idx:train_end]
            Xte = X_all[train_end:test_end];  yte = y_all[train_end:test_end]

            Xtr, ytr = seq(Xtr, ytr, lookback)
            Xte, yte = seq(Xte, yte, lookback)
            if len(Xtr)==0 or len(Xte)==0: break

            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(Xtr.shape[1], Xtr.shape[2])),
                Dropout(0.2),
                LSTM(32),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(Xtr, ytr, epochs=epochs, batch_size=batch_size, verbose=0)

            y_pred = model.predict(Xte, verbose=0)
            results.append(pd.DataFrame({
                "timestamp": df_goc['timestamp'][train_end+lookback:test_end].values,
                "y_true": yte,
                "y_pred": y_pred.flatten()
            }))
            start_idx += test_months*days_in_month

        return pd.concat(results).reset_index(drop=True) if results else pd.DataFrame()

    # ----- Tải dữ liệu, train & vẽ NGAY (không controls) -----
    with st.spinner("Đang xử lý LSTM gọn..."):
        raw = fetch_daily_fixed(lstm_tickers, from_date)
        dfs_by_ticker = {tk: df.reset_index(drop=True) for tk, df in raw.groupby("ticker")}

        cols_per_row = 2
        rows = (len(lstm_tickers) + cols_per_row - 1)//cols_per_row
        for r in range(rows):
            row_cols = st.columns(cols_per_row)
            for c in range(cols_per_row):
                idx = r*cols_per_row + c
                if idx >= len(lstm_tickers): break
                tk = lstm_tickers[idx]
                if tk not in dfs_by_ticker or dfs_by_ticker[tk].empty:
                    row_cols[c].warning(f"{tk}: không đủ dữ liệu.")
                    continue

                df_feat, df_scaled, scaler_close = prepare_features(dfs_by_ticker[tk])
                res = run_lstm_walkforward(df_feat, df_scaled,
                                           lookback=lookback, train_months=train_months,
                                           test_months=test_months, epochs=epochs, batch_size=batch_size)
                if res.empty:
                    row_cols[c].warning(f"{tk}: chưa đủ dữ liệu để tạo chuỗi.")
                    continue

                # inverse
                res['y_true_inv'] = scaler_close.inverse_transform(res[['y_true']])
                res['y_pred_inv'] = scaler_close.inverse_transform(res[['y_pred']])
                res['timestamp'] = pd.to_datetime(res['timestamp'])

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res['timestamp'], y=res['y_true_inv'].ravel(),
                                         name="Giá thật (close)", mode="lines",
                                         line=dict(color="#60a5fa")))
                fig.add_trace(go.Scatter(x=res['timestamp'], y=res['y_pred_inv'].ravel(),
                                         name="Giá dự đoán", mode="lines",
                                         line=dict(color="#f43f5e", dash="dash")))
                fig.update_layout(
                    title=f"{tk} — Giá thật vs. Dự đoán (LSTM, 1D)",
                    height=420, margin=dict(l=10,r=10,t=40,b=10),
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                    font=dict(color="#e5e7eb")
                )
                row_cols[c].plotly_chart(fig, use_container_width=True)
# ---- TAB 5: Ngưỡng Giá ----
with tab_thresholds:
    st.markdown("### 🎯 Biểu đồ giá & ngưỡng (base/TP/SL/đỉnh) — Plotly")

    if len(selected_tickers) == 0:
        st.info("Chưa chọn mã trong bộ lọc chính.")
        st.stop()

    # --- Controls gọn ở đầu tab ---
    c1, c2, c3, c4, c5 = st.columns([1.2,1.2,1.2,1.2,2.2])
    show_base = c1.toggle("Base", True)
    show_tp   = c2.toggle("TP", True)
    show_sl   = c3.toggle("SL", True)
    show_peak = c4.toggle("Đỉnh", True)
    tp_pct    = c5.slider("TP %", 5, 50, 30, step=5, help="Mức chốt lời (+%)")

    c6, c7, c8 = st.columns([1.5, 1.5, 5])
    sl1_pct = c6.slider("SL1 %", 5, 30, 10, step=1, help="Cắt lỗ 1 (-%)")
    sl2_pct = c7.slider("SL2 %", 10, 50, 20, step=1, help="Cắt lỗ 2 (-%)")
    show_vline = c8.toggle("Vạch thời điểm bắt đầu", True)

    from_date_str = pd.to_datetime(start_from).strftime("%Y-%m-%d")

    # Lấy dữ liệu daily close
    thresholds_data = client.Fetch_Trading_Data(
        realtime=False,
        tickers=selected_tickers,
        fields=['close'],
        adjusted=True,
        by='1d',
        from_date=from_date_str,
        lasted=True
    ).get_data()
    thresholds_data["timestamp"] = pd.to_datetime(thresholds_data["timestamp"]).dt.tz_localize(None)

    # helper: giá vốn = close tại mốc start_from
    def get_base_price(df: pd.DataFrame, ticker: str, start_dt: pd.Timestamp):
        dft = df[(df["ticker"] == ticker) & (df["timestamp"] >= start_dt)].sort_values("timestamp")
        return None if dft.empty else float(dft.iloc[0]["close"])

    # Màu
    COLOR_PRICE = "#60a5fa"
    COLOR_BASE  = "#e5e7eb"
    COLOR_TP    = "#f59e0b"
    COLOR_SL1   = "#ef4444"
    COLOR_SL2   = "#7f1d1d"
    COLOR_PEAK  = "#a78bfa"

    # Lưới 2 cột
    tickers = list(thresholds_data["ticker"].unique())
    cols_per_row = 2
    rows = (len(tickers) + cols_per_row - 1)//cols_per_row
    start_dt = pd.to_datetime(start_from)

    for r in range(rows):
        col_row = st.columns(cols_per_row)
        for c in range(cols_per_row):
            idx = r*cols_per_row + c
            if idx >= len(tickers): break
            tk = tickers[idx]
            dft = thresholds_data[thresholds_data["ticker"] == tk].copy()

            base_price = get_base_price(thresholds_data, tk, start_dt)
            peak_price = float(dft["close"].max()) if not dft.empty else None

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dft["timestamp"], y=dft["close"], name="Close",
                mode="lines", line=dict(color=COLOR_PRICE, width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:,.0f}<extra></extra>"
            ))

            shapes = []; annotations = []
            x0 = dft["timestamp"].min() if not dft.empty else start_dt
            x1 = dft["timestamp"].max() if not dft.empty else start_dt

            def add_hline(y, color, label):
                if y is None: return
                shapes.append(dict(type="line", xref="x", yref="y",
                                   x0=x0, x1=x1, y0=y, y1=y,
                                   line=dict(color=color, width=1.5, dash="dot")))
                annotations.append(dict(x=x0, y=y, xanchor="left",
                                        text=label, showarrow=False,
                                        font=dict(color=color, size=11)))

            # Vẽ theo toggle
            if show_base and base_price is not None:
                add_hline(base_price, COLOR_BASE, "Giá vốn")
            if show_tp and base_price is not None:
                add_hline(base_price*(1+tp_pct/100), COLOR_TP, f"TP +{tp_pct}%")
            if show_sl and base_price is not None:
                add_hline(base_price*(1-sl1_pct/100), COLOR_SL1, f"SL -{sl1_pct}%")
                add_hline(base_price*(1-sl2_pct/100), COLOR_SL2, f"SL -{sl2_pct}%")
            if show_peak and peak_price is not None:
                add_hline(peak_price, COLOR_PEAK, "Đỉnh gần nhất")

            if show_vline:
                shapes.append(dict(type="line", xref="x", yref="paper",
                                   x0=start_dt, x1=start_dt, y0=0, y1=1,
                                   line=dict(color="#ef4444", width=2)))
                annotations.append(dict(x=start_dt, y=1, xref="x", yref="paper",
                                        text=f"Start {start_dt.date()}",
                                        showarrow=False, xanchor="left", yanchor="top",
                                        font=dict(color="#ef4444", size=11)))

            fig.update_layout(
                title=f"{tk} — Giá & Ngưỡng",
                height=420,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font=dict(color="#e5e7eb"),
                xaxis=dict(showgrid=True, gridcolor="#1f2937"),
                yaxis=dict(showgrid=True, gridcolor="#1f2937", tickformat=","),
                legend=dict(orientation="h", y=-0.2),
                shapes=shapes,
                annotations=annotations,
            )
            col_row[c].plotly_chart(fig, use_container_width=True)

    st.download_button(
        "📥 Tải CSV dữ liệu ngưỡng",
        thresholds_data.sort_values(["ticker","timestamp"]).to_csv(index=False).encode("utf-8"),
        file_name="thresholds_data.csv"
    )
# ---- TAB 6: Similar Chart----
with tab_similar:
    st.markdown("### 🧩 So sánh mẫu hình giá – tìm chuỗi **tương đồng** (1D)")

    if len(selected_tickers) == 0:
        st.info("Chưa chọn mã trong bộ lọc chính.")
        st.stop()

    # Controls nhỏ gọn
    colA, colB, colC, colD = st.columns([1.8, 1.2, 1.2, 1.2])
    base_ticker = colA.selectbox("Chọn mã gốc (so sánh)", options=selected_tickers, index=0)
    lookback_days = colB.slider("Lookback (ngày)", 60, 360, 180, step=10)
    top_k = colC.slider("Top similar", 1, min(5, len(selected_tickers)-1), min(3, len(selected_tickers)-1))
    method = colD.selectbox("Chuẩn hoá", ["Rebase=100", "Z-score"], index=0)

    from_date_str = pd.to_datetime(start_from).strftime("%Y-%m-%d")

    @st.cache_data(show_spinner=True)
    def fetch_close_daily(tickers: list, from_date_str: str) -> pd.DataFrame:
        # Lấy dữ liệu 1D từ FiinQuant (đúng cú pháp)
        df = client.Fetch_Trading_Data(
            realtime=False,
            tickers=tickers,
            fields=['close'],
            adjusted=True,
            by='1d',
            from_date=from_date_str,
            lasted=True
        ).get_data()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        return df

    # Lấy dữ liệu
    raw = fetch_close_daily(selected_tickers, from_date_str)
    if raw.empty:
        st.warning("Không có dữ liệu close để so sánh.")
        st.stop()

    # Pivot sang wide: index=timestamp, columns=ticker, values=close
    wide = raw.pivot_table(index="timestamp", columns="ticker", values="close").sort_index()
    # Cắt lookback cuối
    if len(wide) > lookback_days:
        wide_lb = wide.iloc[-lookback_days:].copy()
    else:
        wide_lb = wide.copy()

    # Drop cột toàn NaN
    wide_lb = wide_lb.dropna(axis=1, how="all")
    usable_tickers = [t for t in selected_tickers if t in wide_lb.columns and t != base_ticker]
    if base_ticker not in wide_lb.columns:
        st.warning(f"Không đủ dữ liệu cho {base_ticker} trong lookback.")
        st.stop()

    # Chuẩn hoá chuỗi theo phương pháp chọn
    def normalize_rebase(df: pd.DataFrame) -> pd.DataFrame:
        # Rebase 100 từ điểm đầu: 100 * (P/P0)
        first = df.iloc[0]
        return 100 * (df / first)

    def normalize_zscore(df: pd.DataFrame) -> pd.DataFrame:
        mu = df.mean()
        sigma = df.std(ddof=0).replace(0, np.nan)
        return (df - mu) / sigma

    if method == "Rebase=100":
        norm = normalize_rebase(wide_lb.dropna(how="all"))
    else:
        norm = normalize_zscore(wide_lb.dropna(how="all"))

    # Tính tương đồng: dùng hệ số tương quan Pearson trên khoảng lookback
    base_series = norm[base_ticker].dropna()
    sims = []
    for tk in usable_tickers:
        s = norm[tk].dropna()
        aligned = pd.concat([base_series, s], axis=1, join="inner").dropna()
        if len(aligned) < max(30, int(0.3*lookback_days)):  # yêu cầu tối thiểu điểm dữ liệu
            continue
        corr = aligned.iloc[:,0].corr(aligned.iloc[:,1])
        sims.append((tk, corr, len(aligned)))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]

    # Bảng kết quả tương đồng
    st.markdown("#### 🔗 Top tương đồng")
    if len(sims) == 0:
        st.info("Chưa tìm được mã tương đồng (thiếu dữ liệu hoặc lookback quá ngắn).")
        st.stop()
    sim_df = pd.DataFrame(sims, columns=["ticker","corr","overlap_pts"])
    st.dataframe(sim_df.style.format({"corr":"{:.3f}"}), use_container_width=True)

    # Vẽ overlay Plotly: base + top_k similar
    fig = go.Figure()
    # Base
    fig.add_trace(go.Scatter(
        x=norm.index, y=norm[base_ticker], name=f"{base_ticker} (base)",
        mode="lines", line=dict(color="#60a5fa", width=2.5)
    ))
    # Similar series
    palette = ["#22d3ee","#34d399","#f59e0b","#a78bfa","#f43f5e","#eab308"]
    for i, (tk, corr, _) in enumerate(sims):
        if tk not in norm.columns: continue
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[tk], name=f"{tk} (corr={corr:.2f})",
            mode="lines", line=dict(color=palette[i % len(palette)], width=1.6, dash="solid")
        ))

    # Tô bóng vùng lookback (toàn bộ vì đã cắt lookback)
    fig.update_layout(
        title=f"Similar Chart — {base_ticker} vs. nhóm tương đồng ({method}, lookback={len(norm):,} ngày)",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#e5e7eb"),
        xaxis=dict(showgrid=True, gridcolor="#1f2937"),
        yaxis=dict(showgrid=True, gridcolor="#1f2937"),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("© SBTC Junior – FiinQuantX Dashboard")
