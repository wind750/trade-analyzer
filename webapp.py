import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
import re
import os

# ==========================================
# 0. 網頁與環境設定
# ==========================================
st.set_page_config(
    page_title="交易損益分析工具 v9.5 (欄位適配版)",
    page_icon="📊",
    layout="wide"
)

# 設定中文字體
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

# ==========================================
# 1. 核心計算函式庫 (保持不變)
# ==========================================

def calculate_drawdown_info(equity_curve_series):
    peak = equity_curve_series.expanding(min_periods=1).max()
    drawdown = peak - equity_curve_series
    drawdown_percent = (drawdown / peak).fillna(0)
    max_drawdown_value = drawdown.max()
    max_drawdown_percent = drawdown_percent.max()
    return max_drawdown_value, max_drawdown_percent, drawdown

def calculate_consecutive(pnl_series):
    if pnl_series.empty: return 0, 0
    is_win = pnl_series > 0
    groups = is_win.ne(is_win.shift()).cumsum()
    streaks = groups.map(groups.value_counts())
    max_wins = streaks[is_win].max() if not streaks[is_win].empty else 0
    max_losses = streaks[~is_win].max() if not streaks[~is_win].empty else 0
    return int(max_wins), int(max_losses)

@st.cache_data
def run_monte_carlo_simulation(pnl_series, n_simulations=1000, n_trades=None):
    if n_trades is None: n_trades = len(pnl_series)
    pnl_array = pnl_series.to_numpy()
    sim_results_matrix = np.zeros((n_trades, n_simulations))
    for i in range(n_simulations):
        random_trades = np.random.choice(pnl_array, size=n_trades, replace=True)
        sim_results_matrix[:, i] = np.cumsum(random_trades)
    sim_df = pd.DataFrame(sim_results_matrix)
    final_equities = sim_df.iloc[-1, :]
    return sim_df, final_equities

def calculate_risk_metrics(df, date_col, pnl_col, initial_capital):
    df = df.sort_values(by=date_col)
    daily_pnl = df.groupby(date_col)[pnl_col].sum()
    if daily_pnl.empty: return 0.0, 0.0, None, 0.0

    idx = pd.date_range(start=daily_pnl.index.min(), end=daily_pnl.index.max())
    daily_pnl = daily_pnl.reindex(idx, fill_value=0)
    
    equity_curve = initial_capital + daily_pnl.cumsum()
    daily_returns = equity_curve.pct_change().fillna(0)
    
    std_dev = daily_returns.std()
    annualized_volatility = std_dev * np.sqrt(252)
    
    if std_dev == 0: sharpe_ratio = 0.0
    else: sharpe_ratio = (daily_returns.mean() / std_dev) * np.sqrt(252)
        
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    
    if downside_std == 0 or pd.isna(downside_std):
        sortino_ratio = float('inf') if daily_returns.mean() > 0 else 0.0
    else:
        sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(252)
        
    return sharpe_ratio, sortino_ratio, equity_curve, annualized_volatility

# ==========================================
# 2. 資料讀取與前處理 (v9.5 支援「獲利金額」欄位)
# ==========================================

def find_header_and_clean(df):
    """
    智慧搜尋標題列
    """
    header_idx = -1
    # ★ v9.5 新增支援 '序號'
    target_cols = ['交易日期', '日期', 'Date', '進場時間', '出場時間', '序號']
    
    # 1. 檢查當前 column
    current_cols = df.columns.astype(str).str.strip().str.replace('"', '')
    if any(col in current_cols for col in target_cols):
        df.columns = current_cols
        return df

    # 2. 往下掃描
    for i in range(min(30, len(df))):
        row_values = df.iloc[i].astype(str).str.strip().str.replace('"', '').tolist()
        if any(col in row_values for col in target_cols):
            header_idx = i
            break
            
    if header_idx != -1:
        new_header = df.iloc[header_idx].astype(str).str.strip().str.replace('"', '')
        df = df[header_idx + 1:].copy()
        df.columns = new_header
        return df.reset_index(drop=True)
    
    return None

def check_if_trade_data(df):
    """
    檢查是否為交易明細表
    """
    if df is None: return False
    cols = df.columns
    # ★ v9.5：支援 '獲利金額'
    has_date = ('交易日期' in cols) or ('進場時間' in cols) or ('出場時間' in cols)
    has_pnl = ('淨損益' in cols) or ('損益金額' in cols) or ('獲利金額' in cols)
    return has_date and has_pnl

def load_data_smart(uploaded_file):
    """讀取檔案 (CSV/Excel + 智慧標題搜尋)"""
    try:
        # CSV
        if uploaded_file.name.lower().endswith('.csv'):
            uploaded_file.seek(0)
            encodings = ['utf-8', 'utf-8-sig', 'cp950', 'big5']
            for enc in encodings:
                try:
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    uploaded_file.seek(0)
                    df_clean = find_header_and_clean(df)
                    if check_if_trade_data(df_clean):
                        return df_clean
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    continue
            return None

        # Excel
        else:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            for sheet in sheet_names:
                try:
                    df = pd.read_excel(uploaded_file, sheet_name=sheet)
                    df_clean = find_header_and_clean(df)
                    if check_if_trade_data(df_clean):
                        return df_clean
                except Exception:
                    continue
            return None

    except Exception:
        return None

def preprocess_xq_data(df):
    """
    ★ v9.5 核心：支援 '獲利金額', '進場時間', '序號'
    """
    if df is None: return None, None, None, None

    df.columns = df.columns.str.strip()
    
    # 1. 處理日期欄位 (優先使用進場時間)
    if '進場時間' in df.columns:
        df['交易日期'] = df['進場時間']
    elif '出場時間' in df.columns:
        df['交易日期'] = df['出場時間']
    
    # 2. 判斷格式與欄位對應
    if '淨損益' in df.columns:
        date_col = '交易日期'
        pnl_col = '淨損益'
        if '筆數' in df.columns: trade_id_col = '筆數'
        elif '編號' in df.columns: trade_id_col = '編號'
        else: trade_id_col = None
            
    elif '損益金額' in df.columns:
        date_col, pnl_col = '交易日期', '損益金額'
        trade_id_col = '序號' if '序號' in df.columns else None
        
    # ★ v9.5 新增：支援您的回測報告格式 ('獲利金額', '序號')
    elif '獲利金額' in df.columns:
        date_col = '交易日期'
        pnl_col = '獲利金額'
        trade_id_col = '序號' if '序號' in df.columns else None
        
    else:
        return None, None, None, None

    # 格式轉換
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[pnl_col] = pd.to_numeric(df[pnl_col].astype(str).str.strip(), errors='coerce').fillna(0)
    
    if trade_id_col and trade_id_col in df.columns:
        df[trade_id_col] = pd.to_numeric(df[trade_id_col].astype(str).str.strip(), errors='coerce')
    
    df.dropna(subset=[date_col], inplace=True)
    df = df.sort_values(by=date_col).reset_index(drop=True)
    return df, date_col, pnl_col, trade_id_col

# ==========================================
# 3. 模式一：單一報表分析 (MC 風格)
# ==========================================

def perform_single_report_analysis(df_cleaned, pnl_col, date_col, trade_id_col, initial_capital, report_title):
    
    pnl_events = df_cleaned[df_cleaned[pnl_col] != 0]
    total_trades = df_cleaned[trade_id_col].nunique() if trade_id_col else len(pnl_events)
    
    profitable = pnl_events[pnl_events[pnl_col] > 0]
    losing = pnl_events[pnl_events[pnl_col] < 0]
    
    num_wins = len(profitable)
    num_losses = len(losing)
    realized_trades = num_wins + num_losses
    win_rate = (num_wins / realized_trades * 100) if realized_trades > 0 else 0
    
    net_profit = df_cleaned[pnl_col].sum()
    gross_profit = profitable[pnl_col].sum()
    gross_loss = abs(losing[pnl_col].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_win = gross_profit / num_wins if num_wins > 0 else 0
    avg_loss = gross_loss / num_losses if num_losses > 0 else 0
    avg_trade = net_profit / total_trades if total_trades > 0 else 0
    ratio_wl = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    max_con_w, max_con_l = calculate_consecutive(pnl_events[pnl_col])
    
    sharpe, sortino, equity_curve, volatility = calculate_risk_metrics(df_cleaned, date_col, pnl_col, initial_capital)
    
    if equity_curve is None:
        st.error("無法計算風險指標。")
        return
        
    mdd_val, mdd_pct, underwater = calculate_drawdown_info(equity_curve)
    total_return = (net_profit / initial_capital) * 100

    # --- 顯示報告 ---
    st.header(f"策略績效報告 ({report_title})")
    
    st.subheader("1. 全期損益分析")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("總淨利", f"${net_profit:,.0f}")
    c2.metric("毛利", f"${gross_profit:,.0f}")
    c3.metric("毛損", f"${gross_loss:,.0f}")
    c4.metric("獲利因子", f"{pf:.2f}")
    c5.metric("總報酬率", f"{total_return:.2f}%")
    
    st.markdown("---")
    st.subheader("2. 交易分析")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("總交易次數", f"{total_trades}")
    c2.metric("勝率", f"{win_rate:.2f}%")
    c3.metric("平均單筆損益", f"${avg_trade:,.0f}")
    c4.metric("平均賺賠比", f"{ratio_wl:.2f}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("獲利次數", f"{num_wins}")
    c2.metric("虧損次數", f"{num_losses}")
    c3.metric("最大連勝", f"{max_con_w} 次")
    c4.metric("最大連敗", f"{max_con_l} 次")
    
    st.markdown("---")
    st.subheader("3. 風險分析")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("夏普比率", f"{sharpe:.2f}")
    c2.metric("風報比", f"{sortino:.2f}")
    c3.metric("年化波動率", f"{volatility*100:.2f}%")
    c4.metric("最大策略回檔 ($)", f"${mdd_val:,.0f}")
    c1, c2 = st.columns(2)
    c1.metric("最大策略回檔 (%)", f"{mdd_pct*100:.2f}%")
    
    st.markdown("---")
    st.subheader("4. 權益曲線與回檔")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(10, 5))
        eq_df = equity_curve.reset_index()
        eq_df.columns = ['Date', 'Equity']
        ax.plot(eq_df['Date'], eq_df['Equity'], color='#1f77b4', linewidth=1.5)
        ax.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(eq_df['Date'], eq_df['Equity'], initial_capital, where=(eq_df['Equity'] >= initial_capital), facecolor='green', alpha=0.1)
        ax.fill_between(eq_df['Date'], eq_df['Equity'], initial_capital, where=(eq_df['Equity'] < initial_capital), facecolor='red', alpha=0.1)
        ax.set_title(f"權益曲線 (初始: ${initial_capital:,.0f})")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.fill_between(underwater.index, -underwater, 0, facecolor='red', alpha=0.7)
        ax2.set_title("水下圖 (Drawdown)")
        ax2.set_ylabel("回檔金額 ($)")
        ax2.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig2)

    # ==========================================
    # 在 4. 權益曲線與回檔 的區塊之後，加入以下程式碼
    # ==========================================
    
    st.markdown("---")
    st.subheader("5. 每月損益分佈")
    
    # 1. 資料前處理：按月分群加總
    # 確保資料中包含有效的日期欄位，並將其設為 Index 以利重新取樣
    if not df_cleaned.empty and date_col in df_cleaned.columns:
        df_monthly_calc = df_cleaned.copy()
        df_monthly_calc.set_index(date_col, inplace=True)
        
        # 使用 'ME' (Month End) 進行重新取樣，加總每月的損益金額
        try:
            monthly_pnl = df_monthly_calc[pnl_col].resample('ME').sum()
        except ValueError:
            # 兼容舊版 Pandas
            monthly_pnl = df_monthly_calc[pnl_col].resample('M').sum()
            
        # 過濾掉沒有交易的空白月份（可選，若希望保留時間連續性可將此行註解）
        monthly_pnl = monthly_pnl[monthly_pnl != 0]
        
        # 將 Index 轉為 'YYYY-MM' 的字串格式，讓 X 軸顯示更美觀
        monthly_pnl.index = monthly_pnl.index.strftime('%Y-%m')
        
        # 2. 繪製長條圖
        if not monthly_pnl.empty:
            fig_monthly, ax_monthly = plt.subplots(figsize=(12, 5))
            
            # 設定顏色條件：大於等於 0 為綠色，小於 0 為紅色
            colors = ['#2ca02c' if val >= 0 else '#d62728' for val in monthly_pnl]
            
            # 繪製 Bar Chart
            bars = ax_monthly.bar(monthly_pnl.index, monthly_pnl, color=colors, alpha=0.8)
            
            # 畫一條 y=0 的黑色基準線
            ax_monthly.axhline(0, color='black', linewidth=1.2)
            
            # 圖表美化設定
            ax_monthly.set_title("每月淨損益長條圖", fontsize=14, pad=15)
            ax_monthly.set_ylabel("淨損益金額 ($)", fontsize=12)
            ax_monthly.tick_params(axis='x', rotation=45) # X 軸標籤旋轉 45 度避免重疊
            ax_monthly.grid(axis='y', linestyle='--', alpha=0.4)
            
            # 在長條圖上方/下方標註具體數字
            for bar in bars:
                height = bar.get_height()
                # 判斷文字位置，正值在柱子上方，負值在柱子下方
                va_align = 'bottom' if height >= 0 else 'top'
                y_offset = height + (height * 0.02) if height >= 0 else height - (abs(height) * 0.02)
                
                ax_monthly.text(bar.get_x() + bar.get_width()/2., y_offset,
                                f'{height:,.0f}',
                                ha='center', va=va_align, fontsize=9, color='#333333')
            
            # 調整佈局並顯示在 Streamlit 上
            plt.tight_layout()
            st.pyplot(fig_monthly)
        else:
            st.info("該期間內無足夠的逐月交易資料可供繪製。")
    else:
        st.warning("無法解析日期欄位，無法繪製每月損益圖。")

    st.markdown("---")
    st.subheader("5. 蒙地卡羅模擬")
    n_sims = st.number_input("模擬次數", 100, 5000, 1000, 100)
    if st.button(f"執行 {n_sims} 次模擬"):
        with st.spinner("模擬運算中..."):
            sim_df, final_eq = run_monte_carlo_simulation(pnl_events[pnl_col], n_sims, len(pnl_events))
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(sim_df, color='lightblue', alpha=0.05)
            original_curve = pnl_events[pnl_col].cumsum().reset_index(drop=True)
            ax3.plot(original_curve, color='red', linewidth=2, label="原始策略")
            ax3.set_title("蒙地卡羅路徑")
            ax3.legend()
            st.pyplot(fig3)
            c1, c2, c3 = st.columns(3)
            c1.metric("原始結存", f"${net_profit:,.0f}")
            c2.metric("模擬中位數", f"${final_eq.median():,.0f}")
            c3.metric("5% 最差結存", f"${final_eq.quantile(0.05):,.0f}")

# ==========================================
# 4. 模式二：最佳化分析 (Batch Optimization)
# ==========================================

def parse_filename_params(filename):
    name_no_ext = os.path.splitext(filename)[0]
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", name_no_ext)
    params = {}
    if numbers:
        params['Param1'] = float(numbers[0])
        if len(numbers) > 1:
            params['Param2'] = float(numbers[1])
    else:
        params['Param1'] = name_no_ext
    return params

def analyze_optimization_batch(uploaded_files, initial_capital):
    
    results = []
    progress_bar = st.progress(0)
    st.info(f"正在分析 {len(uploaded_files)} 個檔案...")
    
    for i, file in enumerate(uploaded_files):
        df_clean = load_data_smart(file)
        if df_clean is None: continue
        
        df_ready, date_col, pnl_col, trade_id_col = preprocess_xq_data(df_clean)
        if df_ready is None: continue
        
        net_profit = df_ready[pnl_col].sum()
        total_trades = df_ready[trade_id_col].nunique() if trade_id_col else len(df_ready[df_ready[pnl_col]!=0])
        
        sharpe, sortino, equity, vol = calculate_risk_metrics(df_ready, date_col, pnl_col, initial_capital)
        mdd_val, mdd_pct, _ = calculate_drawdown_info(equity) if equity is not None else (0, 0, None)
        
        params = parse_filename_params(file.name)
        
        record = {
            'Filename': file.name,
            'Net Profit': net_profit,
            'Sharpe': sharpe,
            'MDD %': mdd_pct * 100,
            'Trades': total_trades,
        }
        record.update(params)
        results.append(record)
        progress_bar.progress((i + 1) / len(uploaded_files))
        
    if not results:
        st.error("無法讀取有效數據。請確認檔案內容包含『進場時間』與『獲利金額』。")
        return

    res_df = pd.DataFrame(results)
    
    st.header("🔬 策略最佳化分析 (Optimization Report)")
    
    st.subheader("1. 參數高原分析 (Parameter Plateau)")
    c1, c2 = st.columns(2)
    x_axis = c1.selectbox("選擇 X 軸 (參數)", ['Param1', 'Param2'] if 'Param2' in res_df.columns else ['Param1'])
    y_axis = c2.selectbox("選擇 Y 軸 (績效)", ['Net Profit', 'Sharpe', 'MDD %'])
    
    chart_data = res_df.sort_values(by=x_axis)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'Param2' in res_df.columns and x_axis == 'Param1':
        sc = ax.scatter(chart_data[x_axis], chart_data['Param2'], c=chart_data[y_axis], cmap='viridis', s=150, edgecolors='black')
        plt.colorbar(sc, label=y_axis)
        ax.set_ylabel("Param2")
        ax.set_title(f"{y_axis} Heatmap (Param1 vs Param2)")
    else:
        ax.plot(chart_data[x_axis], chart_data[y_axis], marker='o', linestyle='-', color='#1f77b4', linewidth=2)
        max_idx = chart_data[y_axis].idxmax()
        ax.annotate(f'Max: {chart_data.loc[max_idx, y_axis]:.2f}', 
                    xy=(chart_data.loc[max_idx, x_axis], chart_data.loc[max_idx, y_axis]),
                    xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='red'))
        ax.set_ylabel(y_axis)
        ax.set_title(f"{y_axis} vs {x_axis}")

    ax.set_xlabel(x_axis)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)
    
    st.subheader("2. 詳細數據列表")
    st.dataframe(res_df.style.background_gradient(subset=['Net Profit', 'Sharpe'], cmap='Greens'))
    
    best = res_df.loc[res_df['Net Profit'].idxmax()]
    st.success(f"🏆 最佳淨利參數: {best[x_axis]} (淨利: ${best['Net Profit']:,.0f}, 夏普: {best['Sharpe']:.2f})")


# ==========================================
# 5. 主程式入口
# ==========================================

st.title("📊 交易損益分析工具 v9.5 (欄位適配版)")

st.subheader("1. 設定與模式")
col1, col2 = st.columns([1, 2])
with col1:
    initial_capital = st.number_input("初始資金 (Initial Capital)", value=3000000, step=10000)
with col2:
    mode = st.radio("選擇功能模式", ["單一報表分析 (MC風格)", "XQ 策略最佳化 (批次上傳)"], horizontal=True)

st.markdown("---")

if mode == "單一報表分析 (MC風格)":
    st.subheader("2. 上傳單一報表 (Excel/CSV)")
    file = st.file_uploader("選擇檔案", type=["xlsx", "xls", "csv"])
    if file:
        df = load_data_smart(file)
        if df is not None:
            df_clean, date_col, pnl_col, trade_id_col = preprocess_xq_data(df)
            if df_clean is not None:
                # 這裡統稱為 "回測報告" 即可
                perform_single_report_analysis(df_clean, pnl_col, date_col, trade_id_col, initial_capital, "券商")
            else:
                st.error("欄位無法識別，請確認檔案包含「獲利金額」與「進場時間/交易日期」。")
        else:
            st.error("讀取失敗：無法在檔案中找到交易明細表。")

else: # 最佳化模式
    st.subheader("2. 批次上傳多個回測報表 (Excel/CSV)")
    st.info("💡 提示：請將檔名命名為 `參數1.xlsx` (例如 `60.xlsx`)，程式會自動抓取數字作為參數。")
    files = st.file_uploader("選擇多個檔案 (可拖曳)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    
    if files:
        analyze_optimization_batch(files, initial_capital)

