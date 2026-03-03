import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import datetime

# matplotlib設定
plt.rcParams['font.family'] = 'Meiryo'

# ページ設定
st.set_page_config(page_title="RFM分析", layout="wide")

# データ読み込み
@st.cache_data
def load_data():
    return pd.read_csv("raw_rfm_sales_transactions_30000.csv")

def process_data(raw_df):
    customer_rows = raw_df[raw_df['Transaction ID'].astype(str).str.contains('Customer-')]
    
    city_map = {}
    for idx, row in customer_rows.iterrows():
        city_map[str(row['Transaction ID'])] = str(row['Date'])
    
    transactions = raw_df[~raw_df['Transaction ID'].astype(str).str.contains('Customer-')].copy()
    
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
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    
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

# メイン
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
    
    df['Month'] = df['Date'].dt.strftime('%Y-%m')
    
    # KPI
    st.header("基本指標")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("総取引数", f"{len(df):,}")
    col2.metric("顧客数", f"{df['CustomerID'].nunique()}")
    col3.metric("総売上", f"{df['Amount'].sum()/1e6:.1f}M")
    col4.metric("平均取引額", f"{df['Amount'].mean():,.0f}")
    
    st.markdown("---")
    
    # 売上トレンド
    st.header("売上トレンド")
    
    daily = df.groupby('Date')['Amount'].sum().reset_index()
    daily['MA7'] = daily['Amount'].rolling(7).mean()
    
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(daily['Date'], daily['Amount']/1e6, 'b-', linewidth=0.5, alpha=0.5, label='Daily Sales')
    ax1.plot(daily['Date'], daily['MA7']/1e6, 'r-', linewidth=2, label='7-Day MA')
    ax1.fill_between(daily['Date'], daily['Amount']/1e6, alpha=0.2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Revenue (Million)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()
    
    # 月別売上
    st.subheader("月別売上")
    monthly = df.groupby('Month')['Amount'].sum().reset_index()
    monthly['Growth'] = monthly['Amount'].pct_change() * 100
    
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    colors = ['#2ecc71' if i > 0 and monthly['Growth'].iloc[i] > 0 else '#3498db' 
              for i in range(len(monthly))]
    ax2.bar(monthly['Month'], monthly['Amount']/1e6, color=colors, edgecolor='white')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Revenue (Million)')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    
    max_month = monthly.loc[monthly['Amount'].idxmax(), 'Month']
    st.write(f"最高売上月: {max_month} (緑: 前月比プラス)")
    
    st.markdown("---")
    
    # RFM分析
    st.header("RFM分析")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.hist(rfm['Recency'], bins=15, color='#9b59b6', edgecolor='white')
        ax3.axvline(rfm['Recency'].mean(), color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Recency (Days)')
        ax3.set_ylabel('Customer Count')
        ax3.set_title('Recency Distribution')
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
    
    with col_b:
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        ax4.hist(rfm['Frequency'], bins=15, color='#27ae60', edgecolor='white')
        ax4.axvline(rfm['Frequency'].mean(), color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Frequency (Count)')
        ax4.set_ylabel('Customer Count')
        ax4.set_title('Frequency Distribution')
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
    
    with col_c:
        fig5, ax5 = plt.subplots(figsize=(4, 3))
        ax5.hist(rfm['Monetary']/1e6, bins=15, color='#3498db', edgecolor='white')
        ax5.axvline(rfm['Monetary'].mean()/1e6, color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('Monetary (Million)')
        ax5.set_ylabel('Customer Count')
        ax5.set_title('Monetary Distribution')
        ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()
    
    st.markdown("**解釈:** Recencyは小さいほど良好、Frequency/Monetaryは大きいほど良好")
    
    st.markdown("---")
    
    # セグメント
    st.header("顧客セグメンテーション")
    
    seg_counts = rfm['Segment'].value_counts()
    
    col_d, col_e = st.columns(2)
    
    with col_d:
        fig6, ax6 = plt.subplots(figsize=(5, 4))
        colors_seg = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22']
        wedges, texts, autotexts = ax6.pie(seg_counts.values, labels=seg_counts.index, 
                                            autopct='%1.1f%%', colors=colors_seg[:len(seg_counts)],
                                            startangle=90)
        ax6.set_title('Segment Distribution')
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()
    
    with col_e:
        fig7, ax7 = plt.subplots(figsize=(6, 4))
        ax7.barh(seg_counts.index, seg_counts.values, color=colors_seg[:len(seg_counts)])
        ax7.set_xlabel('Customer Count')
        ax7.set_title('Customers by Segment')
        for i, v in enumerate(seg_counts.values):
            ax7.text(v + 0.5, i, str(v), va='center', fontsize=9)
        ax7.grid(True, alpha=0.3, axis='x', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close()
    
    # セグメント別売上
    df_seg = df.merge(rfm[['CustomerID', 'Segment']], on='CustomerID')
    seg_sales = df_seg.groupby('Segment')['Amount'].sum().sort_values()
    
    fig8, ax8 = plt.subplots(figsize=(8, 4))
    ax8.barh(seg_sales.index, seg_sales.values/1e6, color='#e67e22')
    ax8.set_xlabel('Revenue (Million)')
    ax8.set_title('Revenue by Segment')
    for i, v in enumerate(seg_sales.values/1e6):
        ax8.text(v + 0.3, i, f'{v:.1f}M', va='center', fontsize=9)
    ax8.grid(True, alpha=0.3, axis='x', linestyle='--')
    plt.tight_layout()
    st.pyplot(fig8)
    plt.close()
    
    # セグメント統計
    st.subheader("セグメント統計")
    seg_table = rfm.groupby('Segment').agg({
        'CustomerID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).round(1)
    seg_table.columns = ['顧客数', '平均R', '平均F', '平均M']
    st.dataframe(seg_table)
    
    st.markdown("---")
    
    # 地域
    st.header("地域分析")
    
    city_sales = df.groupby('City')['Amount'].sum().sort_values(ascending=False).head(10)
    
    fig9, ax9 = plt.subplots(figsize=(8, 4))
    ax9.barh(city_sales.index[::-1], city_sales.values[::-1]/1e6, color='#2c3e50')
    ax9.set_xlabel('Revenue (Million)')
    ax9.set_title('Top 10 Cities by Revenue')
    ax9.grid(True, alpha=0.3, axis='x', linestyle='--')
    plt.tight_layout()
    st.pyplot(fig9)
    plt.close()
    
    st.markdown("---")
    
    # 商品
    st.header("商品分析")
    
    col_f, col_g = st.columns(2)
    
    with col_f:
        cat_sales = df.groupby('Product Category')['Amount'].sum()
        fig10, ax10 = plt.subplots(figsize=(4, 4))
        ax10.pie(cat_sales.values, labels=cat_sales.index, autopct='%1.1f%%',
                 colors=['#e74c3c', '#3498db', '#2ecc71'])
        ax10.set_title('Revenue by Category')
        plt.tight_layout()
        st.pyplot(fig10)
        plt.close()
    
    with col_g:
        prod_sales = df.groupby('Product Name')['Amount'].sum().sort_values(ascending=False).head(10)
        fig11, ax11 = plt.subplots(figsize=(5, 4))
        ax11.barh(prod_sales.index[::-1], prod_sales.values[::-1]/1e6, color='#8e44ad')
        ax11.set_xlabel('Revenue (Million)')
        ax11.set_title('Top 10 Products')
        ax11.grid(True, alpha=0.3, axis='x', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig11)
        plt.close()
    
    st.markdown("---")
    
    # パレート
    st.header("顧客価値集中度")
    
    cust_val = rfm.sort_values('Monetary', ascending=False)
    total = cust_val['Monetary'].sum()
    cust_val['CumPct'] = cust_val['Monetary'].cumsum() / total * 100
    cust_val['CustPct'] = range(1, len(cust_val) + 1)
    cust_val['CustPct'] = cust_val['CustPct'] / len(cust_val) * 100
    
    fig12, ax12 = plt.subplots(figsize=(8, 4))
    ax12.plot(cust_val['CustPct'], cust_val['CumPct'], 'b-', linewidth=2)
    ax12.axhline(80, color='red', linestyle='--', label='80% Line')
    ax12.axvline(20, color='green', linestyle='--', label='20% Line')
    ax12.fill_between(cust_val['CustPct'], cust_val['CumPct'], alpha=0.2)
    ax12.set_xlabel('Customer Percentage (%)')
    ax12.set_ylabel('Revenue Percentage (%)')
    ax12.legend(loc='lower right')
    ax12.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig12)
    plt.close()
    
    top20_pct = int(len(cust_val) * 0.2)
    top20_share = cust_val.head(top20_pct)['Monetary'].sum() / total * 100
    st.write(f"上位20%顧客が売上の **{top20_share:.1f}%** を占める")
    
    st.markdown("---")
    
    # 発見
    st.header("主要発見")
    
    weekday_map = {'Monday': '月曜', 'Tuesday': '火曜', 'Wednesday': '水曜',
                   'Thursday': '木曜', 'Friday': '金曜', 'Saturday': '土曜', 'Sunday': '日曜'}
    max_wd_en = df.groupby(df['Date'].dt.day_name())['Amount'].sum().idxmax()
    max_wd = weekday_map.get(max_wd_en, max_wd_en)
    
    st.markdown(f"""
    1. **売上トレンド:** 最も売上が高い曜日は {max_wd} です
    2. **顧客集中度:** 上位20%の顧客が {top20_share:.1f}% の売上を生み出しています
    3. **離脱リスク:** {seg_counts.get('At Risk', 0)}人の顧客 ({seg_counts.get('At Risk', 0)/len(rfm)*100:.1f}%) が離脱リスクがあります
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
