"""
Stock Analysis System - Main Application Streamlit Interface
股票分析与策略对比系统 - 主应用
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import sys

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import StockDataFetcher
from strategies import TrendFollowingStrategy, MeanReversionStrategy, MomentumStrategy, MartingaleStrategy, GridTradingStrategy, StrategyAnalyzer
from strategy_analyzer import EnhancedStrategyAnalyzer
from visualizer import StockVisualizer


def init_session_state():
    """Initialize session state variables"""
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = StockDataFetcher()
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedStrategyAnalyzer(st.session_state.data_fetcher)
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None


def main():
    """Main application function"""
    # Streamlit page configuration
    st.set_page_config(
        page_title="A股股票分析与策略对比系统",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    init_session_state()

    # Main title
    st.title("📈 A股股票分析与策略对比系统")
    st.markdown("---")

    # Sidebar setup
    st.sidebar.title("🔧 系统配置")

    # API Token input
    api_token = st.sidebar.text_input(
        "Tushare API Token",
        value="",
        help="请输入Tushare API Token（可选，用于获取更全面的数据）"
    )
    if api_token:
        st.session_state.data_fetcher = StockDataFetcher(api_token)
        st.session_state.analyzer = EnhancedStrategyAnalyzer(st.session_state.data_fetcher)
        st.sidebar.success("API Token已更新")

    # Stock input section
    st.sidebar.subheader("🏢 股票选择")
    stock_input = st.sidebar.text_input(
        "股票代码或名称",
        placeholder="例如: 600519 或 茅台",
        value="600519"
    )

    # Date range selection
    st.sidebar.subheader("📅 时间范围")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_input = st.date_input(
            "开始日期",
            start_date,
            max_value=end_date
        )
    with col2:
        end_input = st.date_input(
            "结束日期",
            end_date,
            max_value=end_date
        )

    # Convert dates to string format
    start_date_str = start_input.strftime('%Y%m%d')
    end_date_str = end_input.strftime('%Y%m%d')

    # Initial capital
    initial_capital = st.sidebar.number_input(
        "初始资金 (元)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000
    )

    # Analysis button
    analyze_button = st.sidebar.button(
        "🚀 开始分析",
        type="primary",
        use_container_width=True
    )

    # Main content area
    if analyze_button and stock_input:
        with st.spinner("正在分析股票数据，请稍候..."):
            # Search and validate stock
            try:
                search_results = st.session_state.data_fetcher.search_stock(stock_input)

                if search_results.empty:
                    st.error(f"未找到股票: {stock_input}")
                    return

                # Get the first search result
                stock_code = search_results.iloc[0]['ts_code']
                stock_name = search_results.iloc[0]['name']

                # Show demo mode indicator
                if st.session_state.data_fetcher.demo_mode:
                    st.warning(f"演示模式：使用模拟数据进行分析。获取完整数据请配置Tushare API Token。")

                # Perform analysis
                st.session_state.analysis_result = st.session_state.analyzer.analyze_stock_strategies(
                    stock_code, start_date_str, end_date_str, initial_capital
                )

                st.success(f"分析完成! 股票: {stock_name} ({stock_code})")

            except Exception as e:
                st.error(f"分析过程中出现错误: {str(e)}")
                return

    # Display results
    if st.session_state.analysis_result and 'error' not in st.session_state.analysis_result:
        result = st.session_state.analysis_result

        # Stock information
        stock_info = result['stock_info']
        st.header(f"📊 {stock_info['name']} ({stock_info['code']})")

        # Time period
        st.write(f"**分析期间**: {stock_info['start_date']} 至 {stock_info['end_date']}")
        st.write(f"**初始资金**: ¥{initial_capital:,.0f}")
        st.markdown("---")

        # Investment recommendation
        recommendation = result['recommendation']['overall']
        st.subheader("💡 投资建议")

        # Recommendation score
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("综合评分", f"{recommendation['score']:.1f}/100")
        with col2:
            st.metric("投资建议", recommendation['recommendation'])
        with col3:
            st.metric("置信度", recommendation['confidence_level'])

        st.markdown("---")

        # Key insights
        st.subheader("🔍 关键洞察")
        for insight in result['recommendation'].get('key_insights', []):
            st.markdown(f"• {insight}")

        st.markdown("---")

        # Strategy comparison table
        st.subheader("📈 策略性能对比")
        comparison_df = result['comparison']

        # Color coding for performance
        def color_performance(val):
            if isinstance(val, (int, float)):
                if val > 10:
                    return 'background-color: #d4edda'
                elif val > 0:
                    return 'background-color: #fff3cd'
                else:
                    return 'background-color: #f8d7da'
            return ''

        styled_comparison = comparison_df.style.applymap(color_performance, subset=['total_return', 'annualized_return'])
        st.dataframe(styled_comparison, use_container_width=True)

        st.markdown("---")

        # Charts section
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 K线图")
            try:
                kline_chart = st.session_state.visualizer.generate_html_template(
                    result['visualizations']['kline'],
                    "kline_chart",
                    "100%",
                    "500px"
                )
                st.components.v1.html(kline_chart, height=500)
            except Exception as e:
                st.error(f"K线图生成失败: {str(e)}")

        with col2:
            st.subheader("📊 策略收益对比")
            try:
                strategy_chart = st.session_state.visualizer.generate_html_template(
                    result['visualizations']['strategy_comparison'],
                    "strategy_chart",
                    "100%",
                    "500px"
                )
                st.components.v1.html(strategy_chart, height=500)
            except Exception as e:
                st.error(f"策略对比图生成失败: {str(e)}")

        st.markdown("---")

        # Risk factors
        st.subheader("⚠️ 风险因素")
        risk_factors = result['recommendation'].get('risk_factors', [])
        if risk_factors:
            for risk in risk_factors:
                severity_color = {
                    'critical': '#dc3545',
                    'high': '#fd7e14',
                    'medium': '#ffc107'
                }.get(risk['severity'], '#6c757d')

                st.markdown(f"""
                <div style="background-color: #f8f9fa; border-left: 4px solid {severity_color}; padding: 15px; margin: 10px 0; border-radius: 4px;">
                    <strong style="color: {severity_color};">{risk['severity'].upper()}</strong>: {risk['description']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("未发现显著风险因素")

        st.markdown("---")

        # Technical indicators
        st.subheader("📋 技术指标")
        if result.get('technical_indicators'):
            latest_indicators = result['technical_indicators'][-1] if result['technical_indicators'] else {}
            if latest_indicators:
                indicator_cols = st.columns(3)
                for i, (indicator, value) in enumerate(latest_indicators.items()):
                    if i < 3:
                        with indicator_cols[i]:
                            st.metric(indicator, f"{value:.2f}" if isinstance(value, (int, float)) else str(value))

        # Export options
        st.sidebar.subheader("💾 导出结果")

        if st.sidebar.button("生成HTML报告"):
            try:
                report_file = st.session_state.analyzer.generate_html_report(
                    st.session_state.analysis_result,
                    f"analysis_report_{stock_info['code']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                )
                st.sidebar.success(f"报告已保存: {report_file}")

                # Provide download link
                with open(report_file, 'rb') as f:
                    st.sidebar.download_button(
                        label="下载报告",
                        data=f.read(),
                        file_name=f"analysis_report_{stock_info['code']}.html",
                        mime="text/html"
                    )
            except Exception as e:
                st.sidebar.error(f"报告生成失败: {str(e)}")

        # Individual strategy details
        with st.expander("📊 详细策略分析", expanded=False):
            st.write("各策略的详细回测结果：")

            for strategy_name, results in result['strategy_results'].items():
                if not results.empty:
                    st.subheader(f"{strategy_name}")

                    # Show recent trades
                    trades = st.session_state.analyzer.strategy_analyzer.extract_trades(results)
                    if trades:
                        trade_df = pd.DataFrame(trades)
                        st.dataframe(trade_df.tail(10), use_container_width=True)

                    # Show performance metrics
                    metrics = result['performances'][strategy_name]
                    metric_cols = st.columns(4)

                    col_idx = 0
                    for metric, value in metrics.items():
                        if col_idx < 4:
                            with metric_cols[col_idx]:
                                if isinstance(value, (int, float)):
                                    st.metric(metric, f"{value:.2f}%")
                                else:
                                    st.metric(metric, str(value))
                            col_idx += 1

    elif st.session_state.analysis_result and 'error' in st.session_state.analysis_result:
        st.error(f"分析错误: {st.session_state.analysis_result['error']}")

    # Help section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 使用说明")

    help_text = """
    **功能说明:**
    1. 输入股票代码或名称进行搜索
    2. 选择分析时间范围
    3. 设置初始投资资金
    4. 点击"开始分析"查看结果

    **5种策略:**
    - 🔄 趋势跟踪策略 (双均线交叉)
    - 📊 均值回归策略 (价格偏离均线)
    - ⚡ 动量策略 (价格变化速率)
    - 🎯 亏损就翻倍策略 (马丁格尔策略)
    - 📐 网格交易策略 (价格区间网格)

    **推荐设置:**
    - 新手: 建议使用1-3万初始资金
    - 有经验: 可使用5-10万初始资金
    - 长期投资: 建议选择1年以上时间范围
    """

    st.sidebar.text_area("使用帮助", help_text, height=300, disabled=True)




if __name__ == "__main__":
    # Run the application
    main()