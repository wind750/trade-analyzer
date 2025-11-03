import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io

# --- 網頁基本設定 ---
st.set_page_config(
    page_title="交易損益分析工具 v4.0",
    page_icon="📊",
    layout="wide"
)

# --- 圖表中文設定 (邏輯不變) ---
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

# --- ★★★ 新增：個股報表分析函式 v4.0 ★★★ ---
def analyze_stock_data(df):
    
    st.header("1. 資料清理與預覽 (個股報表)")
    
    df_cleaned = df.copy()
    
    # 清理欄位名稱
    df_cleaned.columns = df_cleaned.columns.str.strip().str.replace('"', '').str.strip()
    
    required_cols = ['交易日期', '股票名稱', '損益金額', '序號', '報酬率']
    missing_cols = [col for col in required_cols if col not in df_cleaned.columns]
    
    if missing_cols:
        st.error(f"上傳的個股報表缺少必要欄位：`{', '.join(missing_cols)}`")
        return

    # --- 格式轉換 ---
    df_cleaned['交易日期'] = pd.to_datetime(df_cleaned['交易日期'], errors='coerce')
    
    # 清理並轉換 '損益金額'
    df_cleaned['損益金額'] = pd.to_numeric(
        df_cleaned['損益金額'].astype(str).str.strip(), 
        errors='coerce'
    ).fillna(0)
    
    # 清理並轉換 '序號'
    df_cleaned['序號'] = pd.to_numeric(
        df_cleaned['序號'].astype(str).str.strip(), 
        errors='coerce'
    )
    
    # 清理並轉換 '報酬率' (例如 "-5.63%" -> -0.0563)
    df_cleaned['報酬率'] = pd.to_numeric(
        df_cleaned['報酬率'].astype(str).str.strip().str.replace('%', ''), 
        errors='coerce'
    ).fillna(0) / 100.0

    df_cleaned.dropna(subset=['交易日期'], inplace=True)
    df_cleaned = df_cleaned.sort_values(by='交易日期').reset_index(drop=True)

    st.write("以下是系統清理並用於分析的資料預覽。請檢查「損益金額」是否已正確讀取：")
    st.dataframe(df_cleaned.head(10))

    if df_cleaned.empty:
        st.warning("清理後沒有有效的交易數據可供分析。")
        return

    # --- 計算統計數據 ---
    st.header("2. 總體統計報告 (個股)")
    
    # 找出所有實際產生損益的交易紀錄
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
    
    # 新增指標：平均報酬率
    avg_return_rate = df_cleaned['報酬率'].mean() * 100

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
    
    st.metric("平均報酬率", f"{avg_return_rate:.2f}%") # 顯示平均報酬率
    
    st.markdown("---")
    
    # --- 數據分析 & 圖表 ---
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
        st.subheader("累積淨損益曲線")
        df_cleaned['累積淨損益'] = df_cleaned['損益金額'].cumsum()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(df_cleaned['交易日期'], df_cleaned['累積淨損益'], marker='.', linestyle='-')
        ax2.set_title('資產曲線')
        ax2.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    st.markdown("---")
    
    st.header("4. 詳細數據分析")
    pnl_by_product = df_cleaned.groupby('股票名稱')['損益金額'].sum().sort_values(ascending=False).reset_index()
    st.subheader("各股票損益排名")
    st.dataframe(pnl_by_product[pnl_by_product['損益金額'] != 0])
    
    # ... (下載按鈕邏輯相同) ...


# --- ★★★ 舊的期貨報表分析函式 v3.0 ★★★ ---
# (我們把 v3.0 的程式碼完整搬移到這裡，並改名)
def analyze_futures_data(df):
    
    st.header("1. 資料清理與預覽 (期貨報表)")
    
    df_cleaned = df.copy()
    original_columns = df_cleaned.columns.tolist()
    df_cleaned.columns = df_cleaned.columns.str.strip().str.replace('"', '').str.strip()
    cleaned_columns = df_cleaned.columns.tolist()

    required_cols = ['交易日期', '商品名稱', '筆數', '淨損益']
    missing_cols = [col for col in required_cols if col not in cleaned_columns]
    
    if missing_cols:
        st.error(f"上傳的期貨報表缺少必要欄位：`{', '.join(missing_cols)}`")
        return

    # --- 格式轉換 ---
    numeric_cols = ['筆數', '淨損益']
    for col in numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col].astype(str).str.strip(), errors='coerce')

    df_cleaned['交易日期'] = pd.to_datetime(df_cleaned['交易日期'], errors='coerce')
    df_cleaned.dropna(subset=['交易日期'], inplace=True)
    df_cleaned['淨損益'] = df_cleaned['淨損益'].fillna(0)
    
    df_for_charts = df_cleaned.sort_values(by='交易日期').reset_index(drop=True)

    st.write("以下是系統清理並用於分析的資料預覽。請檢查「淨損益」是否已正確讀取：")
    st.dataframe(df_cleaned.head(10))

    if df_cleaned.empty:
        st.warning("清理後沒有有效的交易數據可供分析。")
        return

    # --- 計算統計數據 ---
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
    
    # --- 數據分析 & 圖表 ---
    st.header("3. 視覺化圖表分析")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("每日淨損益")
        daily_pnl = df_for_charts.groupby(df_for_charts['交易日期'].dt.date)['淨損益'].sum()
        daily_pnl = daily_pnl[daily_pnl != 0]
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        daily_pnl.plot(kind='bar', ax=ax1, color=['g' if x > 0 else 'r' for x in daily_pnl.values])
        ax1.set_title('每日淨損益分佈')
        ax1.grid(axis='y', linestyle='--')
        plt.xticks(rotation=45)
        st.pyplot(fig1)

    with col2:
        st.subheader("累積淨損益曲線")
        df_for_charts['累積淨損益'] = df_for_charts['淨損益'].cumsum()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(df_for_charts['交易日期'], df_for_charts['累積淨損益'], marker='.', linestyle='-')
        ax2.set_title('資產曲線')
        ax2.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    st.markdown("---")
    
    st.header("4. 詳細數據分析")
    pnl_by_product = df_cleaned.groupby('商品名稱')['淨損益'].sum().sort_values(ascending=False).reset_index()
    st.subheader("各商品損益排名")
    st.dataframe(pnl_by_product[pnl_by_product['淨損益'] != 0])
    
    # ... (下載按鈕邏輯相同) ...


# --- ★★★ 網頁主體 v4.0 ★★★ ---
st.title("📊 交易損益分析工具 v4.0")

# 讓使用者選擇報表類型
st.subheader("1. 請選擇您的報表類型：")
report_type = st.radio(
    "選擇報表",
    ["個股交易報表 (已總結)", "期貨交易報表 (逐筆)"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("---")

# 顯示檔案上傳器
st.subheader("2. 請上傳您的 Excel 報表：")
uploaded_file = st.file_uploader("選擇一個 Excel 檔案", type=["xlsx", "xls"], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        dataframe = pd.read_excel(uploaded_file)
        st.markdown("---")
        
        # ★★★ 核心改動：根據選擇，呼叫不同的分析函式 ★★★
        if report_type == "個股交易報表 (已總結)":
            analyze_stock_data(dataframe)
        else:
            analyze_futures_data(dataframe)
            
    except Exception as e:
        st.error(f"讀取或分析檔案時發生錯誤：{e}")
        st.error("請確認您的檔案為標準的 Excel 格式，且選擇了正確的報表類型。")