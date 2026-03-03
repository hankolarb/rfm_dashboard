import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

# matplotlib設定
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# ページ設定
st.set_page_config(page_title="RFM分析ダッシュボード", layout="wide")

# データ読み込み
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "raw_rfm_sales_transactions_30000.csv")
    return pd.read_csv(file_path)

def process_data(raw_df):
    # 顧客情報抽出
    customer_rows = raw_df[raw_df['Transaction ID'].astype(str).str.contains('Customer-')]
    
    # 都市マッピング作成
    city_map = {}
    for idx, row in customer_rows.iterrows():
        city_map[str(row['Transaction ID'])] = str(row['Date'])
    
    # 取引データ抽出
    transactions = raw_df[~raw_df['Transaction ID'].astype(str).str.contains('Customer-')].copy()
    
    # 顧客ID割り当て
    current_cust = None
    cust_list = []
    for idx, row in raw_df.iterrows():
        tid = str(row['Transaction ID'])
        if 'Customer-' in tid:
            current_cust = tid
        else:
            cust_list.append(current_cust)
    
    transactions['CustomerID'] = cust_list[:len(transactions)]
    transactions['City'] = transactions['CustomerID'].map(city_map)
    
    return transactions

def convert_types(df):
    # 日付変換
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    
    # 数値変換
    def to_float(val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, str):
            return float(val.replace(',', '').replace('"', ''))
        return float(val)
    
    df['PPU'] = df['PPU'].apply(to_float)
    df['Amount'] = df['Amount'].apply(to_float)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
    df = df.dropna(subset=['Date'])
    
    return df

def calc_rfm(df):
    ref_date = df['Date'].max() + datetime.timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'Date': lambda x: (ref_date - x.max()).days,
        'Transaction ID': 'count',
        'Amount': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # スコア計算
    rfm['R_Score'] = pd.cut(rfm['Recency'], bins=5, labels=[5,4,3,2,1]).astype(int)
    rfm['F_Score'] = pd.cut(rfm['Frequency'].rank(pct=True), 
                            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                            labels=[1,2,3,4,5], include_lowest=True).astype(int)
    rfm['M_Score'] = pd.cut(rfm['Monetary'].rank(pct=True), 
                            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                            labels=[1,2,3,4,5], include_lowest=True).astype(int)
    
    rfm['RFM_Total'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    
    return rfm

def get_segment(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal'
    elif r >= 4 and f >= 2 and m >= 2:
        return 'Potential'
    elif r >= 4:
        return 'New'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    elif r <= 2 and f <= 2:
        return 'Lost'
    else:
        return 'Need Attention'

# メイン処理
def main():
    st.title("RFM顧客分析ダッシュボード")
    
    st.markdown("---")
    
    # データ読み込み
    raw_data = load_data()
    df = process_data(raw_data)
    df = convert_types(df)
    
    # RFM計算
    rfm = calc_rfm(df)
    rfm['Segment'] = rfm.apply(get_segment, axis=1)
    
    # 追加列
    df['Month'] = df['Date'].dt.strftime('%Y-%m')
    df['Weekday'] = df['Date'].dt.day_name()
    
    # フィルター
    st.sidebar.header("フィルター")
    
    selected_city = st.sidebar.selectbox(
        "都市を選択", 
        ['全て'] + sorted(df['City'].unique().tolist())
    )
    
    selected_segments = st.sidebar.multiselect(
        "セグメントを選択",
        rfm['Segment'].unique().tolist(),
        default=rfm['Segment'].unique().tolist()
    )
    
    # フィルタリング
    if selected_city != '全て':
        df = df[df['City'] == selected_city]
    
    filtered_cust = rfm[rfm['Segment'].isin(selected_segments)]['CustomerID']
    df = df[df['CustomerID'].isin(filtered_cust)]
    
    if len(df) == 0:
        st.warning("選択した条件に該当するデータがありません")
        return
    
    # RFM再計算
    rfm = calc_rfm(df)
    rfm['Segment'] = rfm.apply(get_segment, axis=1)
    
    # KPI
    st.header("基本指標")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("総取引数", f"{len(df):,}")
    col2.metric("顧客数", f"{df['CustomerID'].nunique()}")
    col3.metric("総売上", f"{df['Amount'].sum()/1e6:.1f}M")
    col4.metric("平均取引額", f"{df['Amount'].mean():,.0f}")
    
    st.markdown("---")
    
    # セクション1: 売上トレンド
    st.header("売上トレンド分析")
    
    # 日別売上
    daily = df.groupby('Date')['Amount'].sum().reset_index()
    daily['MA7'] = daily['Amount'].rolling(7).mean()
    
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(daily['Date'], daily['Amount']/1e6, 'b-', linewidth=0.5, alpha=0.5, label='日別売上')
    ax1.plot(daily['Date'], daily['MA7']/1e6, 'r-', linewidth=2, label='7日移動平均')
    ax1.fill_between(daily['Date'], daily['Amount']/1e6, alpha=0.2)
    ax1.set_xlabel('日付')
    ax1.set_ylabel('売上 (Million)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()
    
    # 月別売上
    monthly = df.groupby('Month')['Amount'].sum().reset_index()
    monthly['Growth'] = monthly['Amount'].pct_change() * 100
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    bar_colors = []
    for i in range(len(monthly)):
        if i == 0:
            bar_colors.append('steelblue')
        elif monthly['Growth'].iloc[i] > 0:
            bar_colors.append('green')
        else:
            bar_colors.append('red')
    ax2.bar(monthly['Month'], monthly['Amount']/1e6, color=bar_colors)
    ax2.set_xlabel('月')
    ax2.set_ylabel('売上 (Million)')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    
    # 分析コメント
    max_month = monthly.loc[monthly['Amount'].idxmax(), 'Month']
    st.write(f"最高売上月: {max_month} (緑: 前月比プラス、赤: 前月比マイナス)")
    
    st.markdown("---")
    
    # セクション2: RFM分析
    st.header("RFM分析")
    
    # ヒストグラム
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.hist(rfm['Recency'], bins=20, color='purple', edgecolor='white')
        ax3.axvline(rfm['Recency'].mean(), color='red', linestyle='--', 
                    label=f'平均: {rfm["Recency"].mean():.0f}日')
        ax3.set_xlabel('Recency (日)')
        ax3.set_ylabel('顧客数')
        ax3.set_title('Recency分布')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
    
    with col_b:
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        ax4.hist(rfm['Frequency'], bins=20, color='green', edgecolor='white')
        ax4.axvline(rfm['Frequency'].mean(), color='red', linestyle='--',
                    label=f'平均: {rfm["Frequency"].mean():.1f}回')
        ax4.set_xlabel('Frequency (回)')
        ax4.set_ylabel('顧客数')
        ax4.set_title('Frequency分布')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
    
    with col_c:
        fig5, ax5 = plt.subplots(figsize=(5, 3.5))
        ax5.hist(rfm['Monetary']/1e6, bins=20, color='blue', edgecolor='white')
        ax5.axvline(rfm['Monetary'].mean()/1e6, color='red', linestyle='--',
                    label=f'平均: {rfm["Monetary"].mean()/1e6:.1f}M')
        ax5.set_xlabel('Monetary (Million)')
        ax5.set_ylabel('顧客数')
        ax5.set_title('Monetary分布')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()
    
    # RFM解釈
    st.markdown("""
    **解釈:**
    - Recency: 値が小さいほど最近購入している（良好）
    - Frequency: 値が大きいほど頻繁に購入している（良好）
    - Monetary: 値が大きいほど高額購入者（良好）
    """)
    
    # ヒートマップ
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    hm_data = rfm.groupby(['R_Score', 'F_Score']).size().unstack(fill_value=0)
    sns.heatmap(hm_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax6,
                cbar_kws={'label': '顧客数'})
    ax6.set_xlabel('Frequency スコア')
    ax6.set_ylabel('Recency スコア')
    ax6.set_title('R-F ヒートマップ')
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()
    
    st.markdown("---")
    
    # セクション3: セグメント分析
    st.header("顧客セグメンテーション")
    
    seg_counts = rfm['Segment'].value_counts()
    
    col_d, col_e = st.columns(2)
    
    with col_d:
        fig7, ax7 = plt.subplots(figsize=(6, 5))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        ax7.pie(seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%',
                colors=colors[:len(seg_counts)])
        ax7.set_title('セグメント構成比')
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close()
    
    with col_e:
        fig8, ax8 = plt.subplots(figsize=(7, 5))
        ax8.barh(seg_counts.index, seg_counts.values, color=colors[:len(seg_counts)])
        ax8.set_xlabel('顧客数')
        ax8.set_title('セグメント別顧客数')
        for i, v in enumerate(seg_counts.values):
            ax8.text(v + 0.5, i, str(v), va='center')
        ax8.grid(True, alpha=0.3, axis='x', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig8)
        plt.close()
    
    # セグメント別売上
    df_seg = df.merge(rfm[['CustomerID', 'Segment']], on='CustomerID')
    seg_sales = df_seg.groupby('Segment')['Amount'].sum().sort_values()
    
    fig9, ax9 = plt.subplots(figsize=(10, 5))
    ax9.barh(seg_sales.index, seg_sales.values/1e6, color='coral')
    ax9.set_xlabel('売上 (Million)')
    ax9.set_title('セグメント別売上')
    for i, v in enumerate(seg_sales.values/1e6):
        ax9.text(v + 0.3, i, f'{v:.1f}M', va='center')
    ax9.grid(True, alpha=0.3, axis='x', linestyle='--')
    plt.tight_layout()
    st.pyplot(fig9)
    plt.close()
    
    # セグメント統計
    st.subheader("セグメント統計")
    seg_table = rfm.groupby('Segment').agg({
        'CustomerID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['sum', 'mean']
    }).round(1)
    seg_table.columns = ['顧客数', '平均Recency', '平均Frequency', '総売上', '平均売上']
    st.dataframe(seg_table)
    
    st.markdown("---")
    
    # セクション4: 地域分析
    st.header("地域分析")
    
    city_stats = df.groupby('City').agg({
        'Amount': 'sum',
        'CustomerID': 'nunique'
    }).reset_index()
    city_stats.columns = ['City', 'Revenue', 'Customers']
    city_stats = city_stats.sort_values('Revenue', ascending=False)
    
    col_f, col_g = st.columns(2)
    
    with col_f:
        fig10, ax10 = plt.subplots(figsize=(8, 5))
        top10 = city_stats.head(10)
        ax10.barh(top10['City'], top10['Revenue']/1e6, color='navy')
        ax10.invert_yaxis()
        ax10.set_xlabel('売上 (Million)')
        ax10.set_title('売上 Top 10 都市')
        ax10.grid(True, alpha=0.3, axis='x', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig10)
        plt.close()
    
    with col_g:
        fig11, ax11 = plt.subplots(figsize=(8, 5))
        ax11.barh(top10['City'], top10['Customers'], color='darkgreen')
        ax11.invert_yaxis()
        ax11.set_xlabel('顧客数')
        ax11.set_title('顧客数 Top 10 都市')
        ax11.grid(True, alpha=0.3, axis='x', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig11)
        plt.close()
    
    st.markdown("---")
    
    # セクション5: 商品分析
    st.header("商品分析")
    
    # カテゴリ別
    cat_sales = df.groupby('Product Category')['Amount'].sum()
    
    fig12, ax12 = plt.subplots(figsize=(6, 5))
    ax12.pie(cat_sales.values, labels=cat_sales.index, autopct='%1.1f%%',
             colors=['#ff9999', '#66b3ff', '#99ff99'])
    ax12.set_title('カテゴリ別売上構成')
    plt.tight_layout()
    st.pyplot(fig12)
    plt.close()
    
    # Top商品
    prod_sales = df.groupby('Product Name')['Amount'].sum().sort_values(ascending=False)
    
    fig13, ax13 = plt.subplots(figsize=(10, 5))
    ax13.barh(prod_sales.head(10).index, prod_sales.head(10).values/1e6, color='purple')
    ax13.invert_yaxis()
    ax13.set_xlabel('売上 (Million)')
    ax13.set_title('売上 Top 10 商品')
    ax13.grid(True, alpha=0.3, axis='x', linestyle='--')
    plt.tight_layout()
    st.pyplot(fig13)
    plt.close()
    
    st.markdown("---")
    
    # セクション6: パレート分析
    st.header("顧客価値の集中度")
    
    cust_val = rfm.sort_values('Monetary', ascending=False).copy()
    cust_val['CumPct'] = cust_val['Monetary'].cumsum() / cust_val['Monetary'].sum() * 100
    cust_val['CustPct'] = np.arange(1, len(cust_val)+1) / len(cust_val) * 100
    
    fig14, ax14 = plt.subplots(figsize=(10, 5))
    ax14.plot(cust_val['CustPct'], cust_val['CumPct'], 'b-', linewidth=2)
    ax14.axhline(80, color='red', linestyle='--', label='80%ライン')
    ax14.axvline(20, color='green', linestyle='--', label='20%ライン')
    ax14.fill_between(cust_val['CustPct'], cust_val['CumPct'], alpha=0.2)
    ax14.set_xlabel('顧客の累積割合 (%)')
    ax14.set_ylabel('売上の累積割合 (%)')
    ax14.legend(loc='lower right')
    ax14.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig14)
    plt.close()
    
    # パレート結果
    top20 = int(len(cust_val) * 0.2)
    top20_rev = cust_val.head(top20)['Monetary'].sum() / cust_val['Monetary'].sum() * 100
    
    st.write(f"上位20%の顧客が総売上の {top20_rev:.1f}% を占めています")
    
    st.markdown("---")
    
    # 主要発見
    st.header("分析から分かること")
    
    max_wd = df.groupby('Weekday')['Amount'].sum().idxmax()
    at_risk = seg_counts.get('At Risk', 0)
    
    st.markdown(f"""
    1. **売上トレンド:** 最も売上が高い曜日は {max_wd} です
    
    2. **顧客集中度:** 上位20%の顧客が {top20_rev:.1f}% の売上を生み出しています
    
    3. **離脱リスク:** {at_risk}人の顧客 ({at_risk/len(rfm)*100:.1f}%) が離脱リスクがあります
    """)
    
    st.markdown("---")
    
    # 推奨事項
    st.header("ビジネス推奨事項")
    
    st.markdown("""
    1. **リテンション施策:** At Riskセグメント向けの特別オファーを実施
    
    2. **ロイヤリティプログラム:** Championsセグメント向けのVIP特典を検討
    
    3. **成長戦略:** Potentialセグメントへのアップセル施策
    
    """)
    
    # フッター
    st.markdown("---")

if __name__ == "__main__":
    main()
