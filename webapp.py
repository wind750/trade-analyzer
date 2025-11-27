import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io

# --- 網頁基本設定 ---
st.set_page_config(
    page_title="交易損益分析工具 v7.4",
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

# --- v7.1 MDD 輔助函式 ---
def calculate_drawdown_info(equity_curve_series):
    peak = equity_curve_series.expanding(min_periods=1).max()
    drawdown = peak - equity_curve_series
    drawdown_percent = (drawdown / peak).fillna(0)
    max_drawdown_value = drawdown.max()
    max_drawdown_percent = drawdown_percent.max()
    return max_drawdown_value, max_drawdown_percent, drawdown

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

# --- v7.2 夏普與風報比計算函式 ---
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


# --- 個股報表分析函式 (v7.4) ---
def analyze_stock_data(df, initial_capital):
    
    st.header("1. 資料清理與預覽 (個股報表)")
    df_cleaned = df.copy()
    df_cleaned.columns = df_cleaned.columns.str.strip().str.replace('"', '').str.strip()
    required_cols = ['交易日期', '股票名稱', '損益金額', '序號', '報酬率']
    missing_cols = [col for col in required_cols if col not in df_cleaned.columns]
    if missing_cols:
        st.error(f"上傳的個股報表缺少必要欄位：`{', '.join(missing_cols)}`")
        return

    df_cleaned['交易日期'] = pd.to_datetime(df_cleaned['交易日期'], errors='coerce')
    df_cleaned['損益金額'] = pd.to_numeric(df_cleaned['損益金額'].astype(str).str.strip(), errors='coerce').fillna(0)
    df_cleaned['序號'] = pd.to_numeric(df_cleaned['序號'].astype(str).str.strip(), errors='coerce')
    df_cleaned['報酬率'] = pd.to_numeric(df_cleaned['報酬率'].astype(str).str.strip().str.replace('%', ''), errors='coerce').fillna(0) / 100.0
    df_cleaned.dropna(subset=['交易日期'], inplace=True)
    df_cleaned = df_cleaned.sort_values(by='交易日期').reset_index(drop=True)

    st.write("以下是系統清理並用於分析的資料預覽：")
    st.dataframe(df_cleaned.head(10))

    if df_cleaned.empty:
        st.warning("清理後沒有有效的交易數據可供分析。")
        return

    # --- 2. 總體統計報告 ---
    st.header("2. 總體統計報告 (個股)")
    pnl_events_df = df_cleaned[df_cleaned['損益金額'] != 0]
    total_trades = int(df_cleaned['序號'].max()) if not df_cleaned['序號'].dropna().empty else len(pnl_events_df)
    profitable_trades = pnl_events_df[pnl_events_df['損益金額'] > 0]
    losing_trades = pnl_events_df[pnl_events_df['損益金額'] < 0]
    
    num_winning_trades = len(profitable_trades)
    num_losing_trades = len(losing_trades)
    win_rate = (num_winning_trades / len(pnl_events_df)) * 100 if not pnl_events_df.empty else 0
    total_net_pnl = df_cleaned['損益金額'].sum()
    total_profit_from_wins = profitable_trades['損益金額'].sum()
    total_loss_from_losses = abs(losing_trades['損益金額'].sum())
    avg_win = total_profit_from_wins / num_winning_trades if num_winning_trades > 0 else 0
    avg_loss = total_loss_from_losses / num_losing_trades if num_losing_trades > 0 else 0
    profit_factor = total_profit_from_wins / total_loss_from_losses if total_loss_from_losses > 0 else float('inf')
    avg_return_rate = df_cleaned['報酬率'].mean() * 100
    
    sharpe, sortino, equity_curve, volatility = calculate_risk_metrics(df_cleaned, '交易日期', '損益金額', initial_capital)
    
    if equity_curve is None:
        st.error("無有效數據可計算風險指標。")
        return
        
    mdd_val, mdd_pct, underwater_series = calculate_drawdown_info(equity_curve)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("總淨損益", f"${total_net_pnl:,.0f}")
    col2.metric("總交易筆數", f"{total_trades} 筆")
    col3.metric("勝率", f"{win_rate:.2f}%")
    col4.metric("獲利因子", f"{profit_factor:.2f}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("獲利交易次數", f"{num_winning_trades} 次")
    col2.metric("虧損交易次數", f"{num_losing_trades} 次")
    col3.metric("平均獲利", f"${avg_win:,.0f}")
    col4.metric("平均虧損", f"${avg_loss:,.0f}")
    
    st.markdown("---")
    
    st.subheader("風險與報酬分析")
    col1, col2, col3 = st.columns(3)
    col1.metric("夏普比率 (Sharpe)", f"{sharpe:.2f}")
    col2.metric("風報比 (Sortino)", f"{sortino:.2f}")
    col3.metric("年化波動率", f"{volatility * 100:.2f}%")

    col1, col2, col3 = st.columns(3)
    col1.metric("最大回檔 (金額)", f"${mdd_val:,.0f}")
    col2.metric("最大回檔 (%)", f"{mdd_pct * 100:.2f}%")
    col3.metric("平均報酬率", f"{avg_return_rate:.2f}%")
    
    st.markdown("---")

    # --- 3. 視覺化圖表 ---
    st.header("3. 視覺化圖表分析")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("每日淨損益")
        daily_pnl = df_cleaned.groupby(df_cleaned['交易日期'].dt.date)['損益金額'].sum()
        daily_pnl = daily_pnl[daily_pnl != 0]
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        daily_pnl.plot(kind='bar', ax=ax1, color=['g' if x > 0 else 'r' for x in daily_pnl.values])
        ax1.set_title('每日淨損益分佈')
        ax1.grid(axis='y', linestyle='--')
        plt.xticks(rotation=45)
        st.pyplot(fig1)
    with col2:
        st.subheader("資產權益曲線 (Equity Curve)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        equity_df = equity_curve.reset_index()
        equity_df.columns = ['日期', '資產淨值']
        ax2.plot(equity_df['日期'], equity_df['資產淨值'], marker='', linestyle='-', color='orange', linewidth=2)
        ax2.set_title(f'帳戶淨值成長 (初始資金: ${initial_capital:,.0f})')
        ax2.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    # --- 4. 深度圖表 ---
    st.markdown("---")
    st.header("4. 深度圖表分析")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("水下圖 (資產回檔)")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.fill_between(underwater_series.index, -underwater_series, 0, facecolor='red', alpha=0.7)
        ax3.set_title("水下圖 (Drawdown)")
        ax3.set_ylabel("回檔金額 ($)")
        ax3.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig3)
    with col2:
        st.subheader("報酬分佈直方圖")
        pnl_data = pnl_events_df['損益金額']
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.hist(pnl_data, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax4.set_title("損益分佈")
        ax4.set_xlabel("損益金額 ($)")
        ax4.set_ylabel("次數")
        ax4.grid(axis='y', linestyle='--')
        st.pyplot(fig4)

    # --- 5. 詳細數據 ---
    st.markdown("---")
    st.header("5. 詳細數據分析")
    pnl_by_product = df_cleaned.groupby('股票名稱')['損益金額'].sum().sort_values(ascending=False).reset_index()
    st.subheader("各股票損益排名")
    st.dataframe(pnl_by_product[pnl_by_product['損益金額'] != 0])
    
    # --- 6. 蒙地卡羅 ---
    st.markdown("---")
    st.header("6. 蒙地卡羅模擬 (策略穩健性分析)")
    mc_pnl_source = pnl_events_df['損益金額']
    mc_trade_count = len(pnl_events_df) 
    real_curve = pnl_events_df['損益金額'].cumsum().reset_index(drop=True)

    if mc_pnl_source.empty:
        st.warning("沒有足夠的損益數據來執行蒙地卡羅模擬。")
    else:
        n_sims = st.number_input("請選擇模擬次數：", min_value=100, max_value=5000, value=1000, step=100)
        if st.button(f"開始執行 {n_sims} 次模擬"):
            with st.spinner(f"正在執行 {n_sims} 次模擬，請稍候..."):
                sim_df, final_equities = run_monte_carlo_simulation(mc_pnl_source, n_sims, mc_trade_count)
                st.subheader(f"{n_sims} 次模擬 - 權益曲線")
                fig5, ax5 = plt.subplots(figsize=(12, 7))
                ax5.plot(sim_df, color='lightblue', alpha=0.1)
                ax5.plot(real_curve, color='red', linewidth=2, label=f"原始績效 (結存: ${total_net_pnl:,.0f})")
                ax5.set_title("蒙地卡羅模擬 vs 原始績效")
                ax5.set_xlabel("交易次數")
                ax5.set_ylabel("累積損益 ($)")
                ax5.legend()
                ax5.grid(True, linestyle='--')
                st.pyplot(fig5)
                
                st.subheader("模擬統計")
                median_final = final_equities.median()
                pct_5 = final_equities.quantile(0.05)
                col1, col2, col3 = st.columns(3)
                col1.metric("原始結存", f"${total_net_pnl:,.0f}")
                col2.metric("模擬中位數", f"${median_final:,.0f}")
                col3.metric("5% 最差結存", f"${pct_5:,.0f}")
                if total_net_pnl > pct_5:
                    st.success("您的原始績效優於 95% 的隨機模擬結果，策略可能具有優勢！")
                else:
                    st.warning("您的原始績效落入 5% 的最差結果中，策略可能存在風險或運氣不佳。")

# --- 期貨報表分析函式 (v7.4 修正) ---
def analyze_futures_data(df, initial_capital):
    
    st.header("1. 資料清理與預覽 (期貨報表)")
    df_cleaned = df.copy()
    df_cleaned.columns = df_cleaned.columns.str.strip().str.replace('"', '').str.strip()
    required_cols = ['交易日期', '商品名稱', '筆數', '淨損益']
    missing_cols = [col for col in required_cols if col not in df_cleaned.columns]
    if missing_cols:
        st.error(f"上傳的期貨報表缺少必要欄位：`{', '.join(missing_cols)}`")
        return

    numeric_cols = ['筆數', '淨損益']
    for col in numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col].astype(str).str.strip(), errors='coerce')

    df_cleaned['交易日期'] = pd.to_datetime(df_cleaned['交易日期'], errors='coerce')
    df_cleaned.dropna(subset=['交易日期'], inplace=True)
    df_cleaned['淨損益'] = df_cleaned['淨損益'].fillna(0)
    
    st.write("以下是系統清理並用於分析的資料預覽：")
    st.dataframe(df_cleaned.head(10))

    if df_cleaned.empty:
        st.warning("清理後沒有有效的交易數據可供分析。")
        return

    # --- 2. 總體統計報告 ---
    st.header("2. 總體統計報告 (期貨)")
    pnl_events_df = df_cleaned[df_cleaned['淨損益'] != 0]
    total_trades = int(df_cleaned['筆數'].max()) if not df_cleaned['筆數'].dropna().empty else 0
    profitable_trades = pnl_events_df[pnl_events_df['淨損益'] > 0]
    losing_trades = pnl_events_df[pnl_events_df['淨損益'] < 0]
    num_winning_trades = len(profitable_trades)
    num_losing_trades = len(losing_trades)
    win_rate = (num_winning_trades / len(pnl_events_df)) * 100 if not pnl_events_df.empty else 0
    total_net_pnl = df_cleaned['淨損益'].sum()
    total_profit_from_wins = profitable_trades['淨損益'].sum()
    total_loss_from_losses = abs(losing_trades['淨損益'].sum())
    avg_win = total_profit_from_wins / num_winning_trades if num_winning_trades > 0 else 0
    avg_loss = total_loss_from_losses / num_losing_trades if num_losing_trades > 0 else 0
    profit_factor = total_profit_from_wins / total_loss_from_losses if total_loss_from_losses > 0 else float('inf')
    
    sharpe, sortino, equity_curve, volatility = calculate_risk_metrics(df_cleaned, '交易日期', '淨損益', initial_capital)

    if equity_curve is None:
        st.error("無有效數據可計算風險指標。")
        return
        
    mdd_val, mdd_pct, underwater_series = calculate_drawdown_info(equity_curve)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("總淨損益", f"${total_net_pnl:,.0f}")
    col2.metric("總交易筆數", f"{total_trades} 筆")
    col3.metric("勝率", f"{win_rate:.2f}%")
    col4.metric("獲利因子", f"{profit_factor:.2f}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("獲利交易次數", f"{num_winning_trades} 次")
    col2.metric("虧損交易次數", f"{num_losing_trades} 次")
    col3.metric("平均獲利", f"${avg_win:,.0f}")
    col4.metric("平均虧損", f"${avg_loss:,.0f}")
    
    st.markdown("---")
    
    st.subheader("風險與報酬分析")
    col1, col2, col3 = st.columns(3)
    col1.metric("夏普比率 (Sharpe)", f"{sharpe:.2f}")
    col2.metric("風報比 (Sortino)", f"{sortino:.2f}")
    col3.metric("年化波動率", f"{volatility * 100:.2f}%")

    # ★★★ v7.4 修正點：從 st.columns(3) 改為 st.columns(2) ★★★
    col1, col2 = st.columns(2) 
    col1.metric("最大回檔 (金額)", f"${mdd_val:,.0f}")
    col2.metric("最大回檔 (%)", f"{mdd_pct * 100:.2f}%")
    
    st.markdown("---")

    # --- 3. 視覺化圖表 ---
    st.header("3. 視覺化圖表分析")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("每日淨損益")
        daily_pnl = df_cleaned.groupby(df_cleaned['交易日期'].dt.date)['淨損益'].sum()
        daily_pnl = daily_pnl[daily_pnl != 0]
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        daily_pnl.plot(kind='bar', ax=ax1, color=['g' if x > 0 else 'r' for x in daily_pnl.values])
        ax1.set_title('每日淨損益分佈')
        ax1.grid(axis='y', linestyle='--')
        plt.xticks(rotation=45)
        st.pyplot(fig1)
    with col2:
        st.subheader("資產權益曲線 (Equity Curve)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        equity_df = equity_curve.reset_index()
        equity_df.columns = ['日期', '資產淨值']
        ax2.plot(equity_df['日期'], equity_df['資產淨值'], marker='', linestyle='-', color='orange', linewidth=2)
        ax2.set_title(f'帳戶淨值成長 (初始資金: ${initial_capital:,.0f})')
        ax2.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    # --- 4. 深度圖表 ---
    st.markdown("---")
    st.header("4. 深度圖表分析")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("水下圖 (資產回檔)")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.fill_between(underwater_series.index, -underwater_series, 0, facecolor='red', alpha=0.7)
        ax3.set_title("水下圖 (Drawdown)")
        ax3.set_ylabel("回檔金額 ($)")
        ax3.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig3)
    with col2:
        st.subheader("報酬分佈直方圖")
        pnl_data = pnl_events_df['淨損益']
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.hist(pnl_data, bins
