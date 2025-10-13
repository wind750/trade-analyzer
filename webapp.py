import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io

# --- 網頁基本設定 ---
st.set_page_config(
    page_title="交易損益分析工具 v3.0",
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

# --- 主程式函式 ---
def analyze_data(df):
    
    st.header("1. 資料清理與預覽")
    
    df_cleaned = df.copy()
    original_columns = df_cleaned.columns.tolist()
    df_cleaned.columns = df_cleaned.columns.str.strip().str.replace('"', '').str.strip()
    cleaned_columns = df_cleaned.columns.tolist()

    # ★★★ v3.0 核心改動：現在 '淨損益' 也是必要讀取欄位 ★★★
    required_cols = ['交易日期', '商品名稱', '筆數', '淨損益']
    missing_cols = [col for col in required_cols if col not in cleaned_columns]
    
    if missing_cols:
        st.error(f"上傳的 Excel 檔案中缺少必要的欄位：`{', '.join(missing_cols)}`")
        st.info(f"偵測到的欄位為：`{', '.join(original_columns)}`")
        return

    # --- 格式轉換 ---
    # ★★★ v3.0 核心改動：直接清理 '淨損益' 欄位，不再需要 '平倉損益' 等欄位 ★★★
    numeric_cols = ['筆數', '淨損益']
    for col in numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col].astype(str).str.strip(), errors='coerce')

    df_cleaned['交易日期'] = pd.to_datetime(df_cleaned['交易日期'], errors='coerce')
    df_cleaned.dropna(subset=['交易日期'], inplace=True)

    # ★★★ v3.0 核心改動：刪除了錯誤的重新計算公式 ★★★
    # (舊的錯誤程式碼已被刪除) df_cleaned['淨損益'] = df_cleaned['平倉損益']...
    
    # 將沒有淨損益紀錄的列(NaN)填補為 0
    df_cleaned['淨損益'] = df_cleaned['淨損益'].fillna(0)
    
    # 為了圖表繪製，我們只保留有交易活動的列
    df_for_charts = df_cleaned.sort_values(by='交易日期').reset_index(drop=True)

    st.write("以下是系統清理並用於分析的資料預覽。請檢查 K 欄「淨損益」是否已正確讀取：")
    st.dataframe(df_cleaned.head(10))

    if df_cleaned.empty:
        st.warning("清理後沒有有效的交易數據可供分析。")
        return

    # --- 計算統計數據 ---
    st.header("2. 總體統計報告")
    
    # 找出所有實際產生損益的交易紀錄 (淨損益不為0的)
    pnl_events_df = df_cleaned[df_cleaned['淨損益'] != 0]
    
    total_trades = int(df_cleaned['筆數'].max()) if not df_cleaned['筆數'].dropna().empty else 0
    profitable_trades = pnl_events_df[pnl_events_df['淨損益'] > 0]
    losing_trades = pnl_events_df[pnl_events_df['淨損益'] < 0]
    
    num_winning_trades = len(profitable_trades)
    num_losing_trades = len(losing_trades)
    
    win_rate = (num_winning_trades / len(pnl_events_df)) * 100 if not pnl_events_df.empty else 0
    
    total_net_pnl = df_cleaned['淨損益'].sum() # 總損益仍然是整個欄位的總和
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
        # 移除沒有損益的日期，讓圖表更乾淨
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
    
    @st.cache_data
    def convert_df_to_excel(df_to_convert):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_to_convert.to_excel(writer, index=False, sheet_name='分析數據')
        processed_data = output.getvalue()
        return processed_data

    excel_data = convert_df_to_excel(df_cleaned)
    st.download_button(
        label="📥 下載完整分析數據 (Excel)",
        data=excel_data,
        file_name='trade_analysis_processed.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheet-sheet'
    )

# --- 網頁主體 ---
st.title("📊 交易損益分析工具 v3.0")
st.write("請上傳您的期貨或證券帳戶 Excel 報表，系統將自動為您分析。")

uploaded_file = st.file_uploader("選擇一個 Excel 檔案", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        dataframe = pd.read_excel(uploaded_file)
        analyze_data(dataframe)
    except Exception as e:
        st.error(f"讀取或分析檔案時發生錯誤：{e}")
        st.error("請確認您的檔案為標準的 Excel 格式。")