import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 解决图表中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 

# 网页的基础设置，适配 iPad 宽屏
st.set_page_config(page_title="残差互信息网络", layout="wide", initial_sidebar_state="expanded")

def calc_mutual_information(x, y, bins):
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = c_xy / float(np.sum(c_xy))
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    p_x_p_y = p_x[:, None] * p_y[None, :]
    nzs = p_xy > 0
    return np.sum(p_xy[nzs] * np.log(p_xy[nzs] / p_x_p_y[nzs]))

@st.cache_data # 使用缓存加速重复计算
def analyze_network(data):
    returns_df = np.log(data / data.shift(1)).dropna()
    tickers = returns_df.columns.tolist()
    N_samples = len(returns_df)
    N_stocks = len(tickers)
    
    bins = max(3, int(np.floor(N_samples ** (1/3))))
    rho_matrix = returns_df.corr(method='pearson').values
    delta_I_matrix = np.zeros((N_stocks, N_stocks))
    
    progress_bar = st.progress(0)
    total_pairs = (N_stocks * (N_stocks - 1)) / 2
    current_pair = 0
    
    for i in range(N_stocks):
        for j in range(i + 1, N_stocks):
            x = returns_df.iloc[:, i].values
            y = returns_df.iloc[:, j].values
            
            I_emp = calc_mutual_information(x, y, bins)
            rho = np.clip(rho_matrix[i, j], -0.999, 0.999) 
            I_g = -0.5 * np.log(1 - rho**2)
            
            delta_I = max(0, I_emp - I_g)
            delta_I_matrix[i, j] = delta_I
            delta_I_matrix[j, i] = delta_I
            
            current_pair += 1
            progress_bar.progress(int(current_pair / total_pairs * 100))
            
    progress_bar.empty() # 计算完成后清空进度条
    delta_I_df = pd.DataFrame(delta_I_matrix, index=tickers, columns=tickers)
    mean_delta_I = delta_I_df.sum(axis=1) / (N_stocks - 1)
    
    return delta_I_df, mean_delta_I

def plot_mst(delta_I_df):
    epsilon = 1e-5
    dist_matrix = 1 / (delta_I_df + epsilon)
    tickers = delta_I_df.columns
    
    G = nx.Graph()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            stock_a = tickers[i]
            stock_b = tickers[j]
            weight = dist_matrix.loc[stock_a, stock_b]
            delta_i_val = delta_I_df.loc[stock_a, stock_b]
            if delta_i_val > 0:
                G.add_edge(stock_a, stock_b, weight=weight)
                
    MST = nx.minimum_spanning_tree(G)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(MST, k=0.5, seed=42)
    nx.draw_networkx_nodes(MST, pos, ax=ax, node_size=800, node_color='#1f77b4', alpha=0.9, edgecolors='white')
    nx.draw_networkx_labels(MST, pos, ax=ax, font_size=10, font_weight='bold', font_color='white')
    nx.draw_networkx_edges(MST, pos, ax=ax, width=2, alpha=0.6, edge_color='#ff7f0e')
    
    ax.axis('off')
    return fig

# ================= 界面设计 =================
st.title("📊 隐藏共振网络分析系统")
st.markdown("上传您的 CSV 数据，一键计算剥离大盘贝塔后的核心共振标的。")

st.sidebar.header("📁 数据输入区")
uploaded_file = st.sidebar.file_uploader("点击或拖拽上传 CSV", type=['csv'])
st.sidebar.markdown("""
**格式提示：**
- 第1列：日期
- 其余列：股票代码
- 单元格：每日收盘价
""")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    data = data.dropna(axis=1, how='all').dropna(axis=0, how='all')
    st.sidebar.success(f"成功导入 {data.shape[1]} 只标的的数据！")
    
    if st.button("🚀 开始计算非线性共动网络", type="primary", use_container_width=True):
        with st.spinner('正在执行矩阵张量运算，请稍候...'):
            delta_i_matrix, factor_scores = analyze_network(data)
            
            col1, col2 = st.columns([1, 2.5])
            
            with col1:
                st.subheader("🏆 核心共动得分排行")
                factor_df = factor_scores.sort_values(ascending=False).reset_index()
                factor_df.columns = ['标的代码', '隐藏共动得分']
                st.dataframe(factor_df, use_container_width=True, height=600)
                
            with col2:
                st.subheader("🕸️ 最小生成树 (MST) 拓扑结构")
                fig = plot_mst(delta_i_matrix)
                st.pyplot(fig)
else:
    st.info("👈 请在左侧上传您的历史行情 CSV 文件以启动分析。")
