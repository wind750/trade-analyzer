import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io

# --- 網頁基本設定 ---
st.set_page_config(
    page_title="交易損益分析工具 v8.0 (MC風格版)",
    page_icon="📊",
    layout="wide"
)

# --- 圖表中文設定 ---
font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
CHINESE_FONT = 'Microsoft JhengHei'
for font_path in font_paths:
    if 'msjh.ttc' in font_path or 'msjh.ttf' in font_path:
        CHINESE_FONT = 'Microsoft JhengHei'
        break
    elif 'Heiti' in font_path or 'SimHei' in font_path:
        CHINESE_FONT = 'SimHei'

plt.rcParams['font.sans-serif'] = [CHINESE_FONT, 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 輔助函式：計算最大回檔 ---
def calculate_drawdown_info(equity_curve_series):
    peak = equity_curve_series.expanding(min_periods=1).max()
    drawdown = peak - equity_curve_series
    # 分母為 peak，避免除以 0
    drawdown_percent = (drawdown / peak).fillna(0)
    max_drawdown_value = drawdown.max()
    max_drawdown_percent = drawdown_percent.max()
    return max_drawdown_value, max_drawdown_percent, drawdown

# --- 輔助函式：計算連勝與連敗 ---
def calculate_consecutive(pnl_series):
    if pnl_series.empty:
        return 0, 0
    
    # 建立一個布林序列：賺錢為 True, 賠錢為 False
    is_win = pnl_series > 0
    
    # 計算連續次數
    # 邏輯：比較當前與前一個是否不同，不同時產生新的群組編號，再計算每個群組的長度
    groups = is_win.ne(is_win.shift()).cumsum()
    streaks = groups.map(groups.value_counts())
    
    # 分別找出勝和敗的最大連續次數
    max_consecutive_wins = streaks[is_win].max() if not streaks[is_win].empty else 0
    max_consecutive_losses = streaks[~is_win].max() if not streaks[~is_win].empty else 0
    
    return int(max_consecutive_wins), int(max_consecutive_losses)

# --- 蒙地卡羅模擬函式 ---
@st.cache_data
def run_monte_carlo_simulation(pnl_series, n_simulations=1000, n_trades=None):
    if n_trades is None:
        n_trades = len(pnl_series)
    pnl_array = pnl_series.to_numpy()
    sim_results_matrix = np.zeros((n_trades, n_simulations))
    for i in range(n_simulations):
        random_trades = np.random.choice(pnl_array, size=n_trades, replace=True)
        sim_results_matrix[:, i] = np.cumsum(random_trades)
    sim_df = pd.DataFrame(sim_results_matrix)
    final_equities = sim_df.iloc[-1, :]
    return sim_df, final_equities

# --- 夏普與風報比計算函式 ---
def calculate_risk_metrics(df, date_col, pnl_col, initial_capital):
    df = df.sort_values(by=date_col)
    daily_pnl = df.groupby(date_col)[pnl_col].sum()
    if daily_pnl.empty:
        return 0.0, 0.0, None, 0.0

    idx = pd.date_range(start=daily_pnl.index.min(), end=daily_pnl.index.max())
    daily_pnl = daily_pnl.reindex(idx, fill_value=0)
    
    equity_curve = initial_capital + daily_pnl.cumsum()
    daily_returns = equity_curve.pct_change().fillna(0)
    
    std_dev = daily_returns.std()
    annualized_volatility = std_dev * np.sqrt(252)
    
    if std_dev == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (daily_returns.mean() / std_dev) * np.sqrt(252)
        
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    
    if downside_std == 0 or pd.isna(downside_std):
        sortino_ratio = float('inf') if daily_returns.mean() > 0 else 0.0
    else:
        sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(252)
        
    return sharpe_ratio, sortino_ratio, equity_curve, annualized_volatility


# --- 通用分析邏輯 (整合個股與期貨) ---
def perform_mc_style_analysis(df_cleaned, pnl_col, date_col, trade_count_col, initial_capital, report_name):
    
    # 1. 基礎數據準備
    pnl_events_df = df_cleaned[df_cleaned[pnl_col] != 0]
    
    # 計算交易次數 (期貨用筆數去重，個股用序號去重或直接數列數)
    if trade_count_col:
        total_trades = df_cleaned[trade_count_col].nunique()
    else:
        # 如果是個股且沒有序號欄位，可能需要另一種算法，但在這裡假設個股分析函式傳入時已處理好
        # 為了兼容 v7.6 的邏輯：
        total_trades = len(pnl_events_df) # 預設回退方案

    # 分離獲利與虧損交易
    profitable_trades = pnl_events_df[pnl_events_df[pnl_col] > 0]
    losing_trades = pnl_events_df[pnl_events_df[pnl_col] < 0]
    
    num_winning_trades = len(profitable_trades)
    num_losing_trades = len(losing_trades)
    
    # --- 2. MC 關鍵指標計算 ---
    
    # 全期損益分析 (Performance Summary)
    total_net_profit = df_cleaned[pnl_col].sum()                   # 總淨利
    gross_profit = profitable_trades[pnl_col].sum()                # 毛利
    gross_loss = abs(losing_trades[pnl_col].sum())                 # 毛損 (取絕對值)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') # 獲利因子
    return_on_initial_capital = (total_net_profit / initial_capital) * 100 # 報酬率
    
    # 交易分析 (Trade Analysis)
    realized_trades_count = num_winning_trades + num_losing_trades
    percent_profitable = (num_winning_trades / realized_trades_count) * 100 if realized_trades_count > 0 else 0 # 勝率
    
    avg_trade_net_profit = total_net_profit / total_trades if total_trades > 0 else 0 # 平均單筆損益
    avg_winning_trade = gross_profit / num_winning_trades if num_winning_trades > 0 else 0 # 平均獲利交易
    avg_losing_trade = gross_loss / num_losing_trades if num_losing_trades > 0 else 0 # 平均虧損交易
    ratio_avg_win_avg_loss = avg_winning_trade / avg_losing_trade if avg_losing_trade > 0 else float('inf') # 平均賺賠比
    
    max_consecutive_wins, max_consecutive_losses = calculate_consecutive(pnl_events_df[pnl_col]) # 最大連勝/連敗
    
    # 風險分析 (Risk Analysis)
    sharpe, sortino, equity_curve, volatility = calculate_risk_metrics(df_cleaned, date_col, pnl_col, initial_capital)
    
    if equity_curve is None:
        st.error("無有效數據可計算風險指標。")
        return

    mdd_val, mdd_pct, underwater_series = calculate_drawdown_info(equity_curve)

    # --- 3. 介面呈現 (MC 風格) ---
    
    st.header(f"策略績效報告 ({report_name})")
    
    # 第一區：全期損益分析
    st.subheader("1. 全期損益分析 (Performance Summary)")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("總淨利 (Total Net Profit)", f"${total_net_profit:,.0f}")
    col2.metric("毛利 (Gross Profit)", f"${gross_profit:,.0f}")
    col3.metric("毛損 (Gross Loss)", f"${gross_loss:,.0f}")
    col4.metric("獲利因子 (Profit Factor)", f"{profit_factor:.2f}")
    col5.metric("報酬率 (Return on Capital)", f"{return_on_initial_capital:.2f}%")
    
    st.markdown("---")
    
    # 第二區：交易分析
    st.subheader("2. 交易分析 (Trade Analysis)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("總交易次數", f"{total_trades} 筆")
    col2.metric("勝率 (Percent Profitable)", f"{percent_profitable:.2f}%")
    col3.metric("平均單筆損益", f"${avg_trade_net_profit:,.0f}")
    col4.metric("平均賺賠比 (Avg Win/Loss)", f"{ratio_avg_win_avg_loss:.2f}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("獲利交易次數", f"{num_winning_trades}")
    col2.metric("虧損交易次數", f"{num_losing_trades}")
    col3.metric("最大連勝 (Max Consec. Wins)", f"{max_consecutive_wins} 次")
    col4.metric("最大連敗 (Max Consec. Losses)", f"{max_consecutive_losses} 次")
    
    st.markdown("---")
    
    # 第三區：風險分析
    st.subheader("3. 風險分析 (Risk Analysis)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最大策略回檔 ($)", f"${mdd_val:,.0f}")
    col2.metric("最大策略回檔 (%)", f"{mdd_pct * 100:.2f}%")
    col3.metric("夏普比率 (Sharpe Ratio)", f"{sharpe:.2f}")
    col4.metric("年化波動率", f"{volatility * 100:.2f}%")
    
    st.markdown("---")

    # --- 4. 圖表區 ---
    st.header("4. 權益曲線與回檔 (Equity Curve & Drawdown)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("權益曲線 (Equity Curve)")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        # 繪製包含初始資金的權益曲線
        equity_df = equity_curve.reset_index()
        equity_df.columns = ['日期', '資產淨值']
        ax1.plot(equity_df['日期'], equity_df['資產淨值'], marker='', linestyle='-', color='#1f77b4', linewidth=1.5)
        ax1.fill_between(equity_df['日期'], equity_df['資產淨值'], initial_capital, where=(equity_df['資產淨值'] >= initial_capital), facecolor='green', alpha=0.1)
        ax1.fill_between(equity_df['日期'], equity_df['資產淨值'], initial_capital, where=(equity_df['資產淨值'] < initial_capital), facecolor='red', alpha=0.1)
        ax1.set_title(f'策略權益曲線 (初始資金: ${initial_capital:,.0f})')
        ax1.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        
    with col2:
        st.subheader("水下圖 (Underwater Plot)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.fill_between(underwater_series.index, -underwater_series, 0, facecolor='red', alpha=0.7)
        ax2.set_title("策略回檔 (Drawdown)")
        ax2.set_ylabel("回檔金額 ($)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    st.markdown("---")
    
    # --- 5. 蒙地卡羅模擬 ---
    st.header("5. 蒙地卡羅分析 (Monte Carlo Analysis)")
    st.write("透過隨機重組交易順序，評估策略在不同運氣下的表現。")
    
    mc_pnl_source = pnl_events_df[pnl_col]
    # 期貨用 pnl_events_df 的長度作為交易次數較為準確(不含空交易日)
    mc_trade_count = len(pnl_events_df) 
    real_curve = pnl_events_df[pnl_col].cumsum().reset_index(drop=True)

    if mc_pnl_source.empty:
        st.warning("沒有足夠的損益數據來執行蒙地卡羅模擬。")
    else:
        n_sims = st.number_input("請選擇模擬次數：", min_value=100, max_value=5000, value=1000, step=100)
        
        if st.button(f"開始執行 {n_sims} 次模擬"):
            with st.spinner(f"正在執行 {n_sims} 次模擬，請稍候..."):
                sim_df, final_equities = run_monte_carlo_simulation(mc_pnl_source, n_sims, mc_trade_count)
                
                st.subheader(f"{n_sims} 次模擬 - 權益曲線堆疊圖")
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                # 畫模擬線 (淡藍色)
                ax3.plot(sim_df, color='lightblue', alpha=0.05)
                # 畫真實線 (紅色)
                ax3.plot(real_curve, color='red', linewidth=2, label=f"原始策略 (結存: ${total_net_profit:,.0f})")
                ax3.set_title("蒙地卡羅模擬 vs 原始策略")
                ax3.set_xlabel("交易次數")
                ax3.set_ylabel("累積損益 ($)")
                ax3.legend()
                ax3.grid(True, linestyle='--')
                st.pyplot(fig3)
                
                st.subheader("模擬統計摘要")
                median_final = final_equities.median()
                pct_5 = final_equities.quantile(0.05)
                pct_95 = final_equities.quantile(0.95)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("原始策略結存", f"${total_net_profit:,.0f}")
                col2.metric("模擬中位數結存", f"${median_final:,.0f}")
                col3.metric("5% 最差情境 (95%信心)", f"${pct_5:,.0f}")
                
                if total_net_profit > pct_5:
                    st.success("您的原始績效位於模擬結果的 95% 信心區間之上，顯示策略具有顯著優勢。")
                else:
                    st.warning("您的原始績效接近最差的 5% 模擬結果，請注意策略可能存在過度擬合或運氣成分。")


# --- 數據讀取與清理函式 ---

def analyze_stock_data(df, initial_capital):
    df_cleaned = df.copy()
    df_cleaned.columns = df_cleaned.columns.str.strip().str.replace('"', '').str.strip()
    
    # 必要的欄位檢查
    required_cols = ['交易日期', '股票名稱', '損益金額', '序號', '報酬率']
    missing_cols = [col for col in required_cols if col not in df_cleaned.columns]
    if missing_cols:
        st.error(f"上傳的個股報表缺少必要欄位：`{', '.join(missing_cols)}`")
        return

    # 數據轉換
    df_cleaned['交易日期'] = pd.to_datetime(df_cleaned['交易日期'], errors='coerce')
    df_cleaned['損益金額'] = pd.to_numeric(df_cleaned['損益金額'].astype(str).str.strip(), errors='coerce').fillna(0)
    df_cleaned['序號'] = pd.to_numeric(df_cleaned['序號'].astype(str).str.strip(), errors='coerce')
    df_cleaned.dropna(subset=['交易日期'], inplace=True)
    df_cleaned = df_cleaned.sort_values(by='交易日期').reset_index(drop=True)
    
    # 呼叫通用 MC 分析引擎
    # 個股報表通常 '序號' 就是交易次數計數器
    perform_mc_style_analysis(df_cleaned, '損益金額', '交易日期', '序號', initial_capital, "個股")


def analyze_futures_data(df, initial_capital):
    df_cleaned = df.copy()
    df_cleaned.columns = df_cleaned.columns.str.strip().str.replace('"', '').str.strip()
    
    required_cols = ['交易日期', '商品名稱', '筆數', '淨損益']
    missing_cols = [col for col in required_cols if col not in df_cleaned.columns]
    if missing_cols:
        st.error(f"上傳的期貨報表缺少必要欄位：`{', '.join(missing_cols)}`")
        return

    df_cleaned['交易日期'] = pd.to_datetime(df_cleaned['交易日期'], errors='coerce')
    # 期貨報表可能有空白列，筆數和淨損益需轉數值
    for col in ['筆數', '淨損益']:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col].astype(str).str.strip(), errors='coerce')
    
    df_cleaned.dropna(subset=['交易日期'], inplace=True)
    df_cleaned['淨損益'] = df_cleaned['淨損益'].fillna(0)
    
    # 呼叫通用 MC 分析引擎
    # 期貨報表 '筆數' 是交易ID
    perform_mc_style_analysis(df_cleaned, '淨損益', '交易日期', '筆數', initial_capital, "期貨")


# --- 網頁主體 v8.0 (MC 風格版) ---
st.title("📊 交易損益分析工具 v8.0 (MC風格版)")

st.subheader("1. 設定與報表類型：")

col1, col2 = st.columns([1, 2])
with col1:
    initial_capital = st.number_input("請輸入初始資金 (元)", min_value=10000, value=3000000, step=10000)
with col2:
    report_type = st.radio(
        "選擇報表類型",
        ["個股交易報表 (已總結)", "期貨交易報表 (逐筆)"],
        horizontal=True
    )

st.markdown("---")

st.subheader("2. 請上傳您的 Excel 或 CSV 報表：")
uploaded_file = st.file_uploader(
    "選擇一個 Excel 或 CSV 檔案",
    type=["xlsx", "xls", "csv"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        dataframe = None 
        if uploaded_file.name.endswith('.csv'):
            uploaded_file.seek(0)
            # 嘗試多種編碼讀取 CSV
            encodings = ['utf-8', 'utf-8-sig', 'cp950', 'big5']
            for enc in encodings:
                try:
                    dataframe = pd.read_csv(uploaded_file, encoding=enc)
                    uploaded_file.seek(0)
                    break
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    continue
        else:
            dataframe = pd.read_excel(uploaded_file)
        
        st.markdown("---")
        
        if dataframe is None:
            st.error("讀取檔案失敗。請確認 CSV 編碼格式。")
        else:
            if report_type == "個股交易報表 (已總結)":
                analyze_stock_data(dataframe, initial_capital)
            else:
                analyze_futures_data(dataframe, initial_capital)
            
    except Exception as e:
        st.error(f"讀取或分析檔案時發生錯誤：{e}")
