"""
Strategy Comparison and Performance Evaluation Module
Enhanced strategy analysis and investment recommendation system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from data_fetcher import StockDataFetcher
from strategies import TrendFollowingStrategy, MeanReversionStrategy, MomentumStrategy, MartingaleStrategy, GridTradingStrategy, StrategyAnalyzer
from visualizer import StockVisualizer


class EnhancedStrategyAnalyzer:
    """Enhanced strategy analyzer with investment recommendations"""

    def __init__(self, data_fetcher: StockDataFetcher):
        self.data_fetcher = data_fetcher
        self.strategy_analyzer = StrategyAnalyzer()
        self.visualizer = StockVisualizer()
        self.strategies = {
            '趋势跟踪策略': TrendFollowingStrategy(),
            '均值回归策略': MeanReversionStrategy(),
            '动量策略': MomentumStrategy(),
            '亏损就翻倍策略': MartingaleStrategy(),
            '网格交易策略': GridTradingStrategy()
        }

    def analyze_stock_strategies(self, stock_code: str, start_date: str = None,
                               end_date: str = None, initial_capital: float = 100000) -> Dict:
        """
        Analyze all strategies for a specific stock

        Args:
            stock_code: Stock code
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            initial_capital: Initial investment capital

        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing stock {stock_code}...")

        # Fetch stock data
        stock_data = self.data_fetcher.get_kline_data(stock_code, 'D', start_date, end_date)

        if stock_data.empty:
            return {"error": f"No data available for stock {stock_code}"}

        # Calculate additional technical indicators
        stock_data = self._calculate_technical_indicators(stock_data)

        # Backtest all strategies
        strategy_results = {}
        strategy_performances = {}

        for strategy_name, strategy in self.strategies.items():
            print(f"Testing {strategy_name}...")
            try:
                results = strategy.backtest(stock_data, initial_capital)
                strategy_results[strategy_name] = results
                metrics = self.strategy_analyzer.calculate_metrics(results)
                strategy_performances[strategy_name] = metrics
            except Exception as e:
                print(f"Error in {strategy_name}: {e}")
                strategy_results[strategy_name] = pd.DataFrame()
                strategy_performances[strategy_name] = {}

        # Generate comparison
        comparison_df = self.strategy_analyzer.compare_strategies(strategy_results)

        # Generate investment recommendation
        recommendation = self._generate_investment_recommendation(
            strategy_performances, stock_data, start_date
        )

        # Create visualizations
        kline_options = self.visualizer.prepare_kline_data(stock_data)
        strategy_chart_options = self.visualizer.create_strategy_comparison_chart(strategy_results)

        return {
            'stock_info': {
                'code': stock_code,
                'name': self.data_fetcher.get_stock_name(stock_code),
                'start_date': start_date or stock_data['trade_date'].min().strftime('%Y%m%d'),
                'end_date': end_date or stock_data['trade_date'].max().strftime('%Y%m%d')
            },
            'strategy_results': strategy_results,
            'performances': strategy_performances,
            'comparison': comparison_df,
            'recommendation': recommendation,
            'visualizations': {
                'kline': kline_options,
                'strategy_comparison': strategy_chart_options
            },
            'technical_indicators': stock_data.tail(10)[['close', 'ma5', 'ma10', 'ma20', 'rsi', 'macd']].to_dict('records')
        }

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for better strategy analysis
        """
        df = data.copy()

        # Moving averages
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['ma60'] = df['close'].rolling(window=60, min_periods=1).mean()

        # RSI (Relative Strength Index)
        df['rsi'] = self._calculate_rsi(df['close'], period=14)

        # MACD (Moving Average Convergence Divergence)
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['close'].rolling(window=20, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Volume indicators
        df['volume_ma'] = df['vol'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['vol'] / df['volume_ma']

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _generate_investment_recommendation(self, performances: Dict, data: pd.DataFrame, start_date: str) -> Dict:
        """
        Generate comprehensive investment recommendation based on strategy analysis
        """
        recommendations = []

        # Best performing strategy
        if performances:
            best_strategy = max(performances.keys(),
                             key=lambda x: performances.get(x, {}).get('total_return', -float('inf')))
            best_performance = performances[best_strategy]

            if best_performance.get('total_return', -float('inf')) > 0:
                recommendations.append({
                    'type': 'best_strategy',
                    'strategy': best_strategy,
                    'reasoning': f"{best_strategy}表现最佳，总收益为{best_performance['total_return']:.2f}%",
                    'confidence': 'high' if best_performance['total_return'] > 10 else 'medium'
                })

        # Risk assessment
        if performances:
            max_dd = min([perf.get('max_drawdown', 0) for perf in performances.values()])
            if max_dd < -15:
                recommendations.append({
                    'type': 'risk_warning',
                    'message': "注意：部分策略最大回撤超过15%，投资风险较高",
                    'severity': 'high' if max_dd < -20 else 'medium'
                })

        # Market condition assessment
        current_price = data['close'].iloc[-1]
        ma20 = data['ma20'].iloc[-1]
        ma60 = data['ma60'].iloc[-1]

        if current_price > ma20 > ma60:
            market_condition = "上升趋势"
            recommendation_level = "积极"
        elif current_price < ma20 < ma60:
            market_condition = "下降趋势"
            recommendation_level = "谨慎"
        else:
            market_condition = "震荡市"
            recommendation_level = "中性"

        recommendations.append({
            'type': 'market_assessment',
            'condition': market_condition,
            'recommendation_level': recommendation_level,
            'price_levels': {
                'current': current_price,
                'ma20': ma20,
                'ma60': ma60
            }
        })

        # Overall recommendation
        overall_score = self._calculate_overall_score(performances)
        overall_recommendation = {
            'score': overall_score,
            'recommendation': self._get_recommendation_text(overall_score),
            'confidence_level': self._get_confidence_level(performances)
        }

        return {
            'recommendations': recommendations,
            'overall': overall_recommendation,
            'key_insights': self._generate_key_insights(performances, data),
            'risk_factors': self._identify_risk_factors(performances)
        }

    def _calculate_overall_score(self, performances: Dict) -> float:
        """Calculate overall investment score (0-100)"""
        if not performances:
            return 0

        scores = []
        for strategy, perf in performances.items():
            total_return = perf.get('total_return', 0)
            sharpe_ratio = perf.get('sharpe_ratio', 0)
            max_drawdown = perf.get('max_drawdown', 0)

            # Score calculation (weighted)
            return_score = max(0, min(50, total_return / 2))  # Max 50 points for return
            risk_score = max(0, min(30, -max_drawdown))  # Max 30 points for risk management
            efficiency_score = max(0, min(20, sharpe_ratio * 5))  # Max 20 points for efficiency

            total_score = return_score + risk_score + efficiency_score
            scores.append(total_score)

        return np.mean(scores) if scores else 0

    def _get_recommendation_text(self, score: float) -> str:
        """Get recommendation text based on score"""
        if score >= 70:
            return "强烈推荐"
        elif score >= 50:
            return "推荐"
        elif score >= 30:
            return "中性"
        elif score >= 15:
            return "谨慎"
        else:
            return "不推荐"

    def _get_confidence_level(self, performances: Dict) -> str:
        """Get confidence level based on strategy consensus"""
        if not performances:
            return "低"

        # Count strategies with positive returns
        positive_strategies = sum(1 for perf in performances.values()
                               if perf.get('total_return', 0) > 0)
        total_strategies = len(performances)

        if positive_strategies / total_strategies >= 0.7:
            return "高"
        elif positive_strategies / total_strategies >= 0.5:
            return "中"
        else:
            return "低"

    def _generate_key_insights(self, performances: Dict, data: pd.DataFrame) -> List[str]:
        """Generate key insights from strategy analysis"""
        insights = []

        if performances:
            # Best and worst performers
            best = max(performances.items(), key=lambda x: x[1].get('total_return', -float('inf')))
            worst = min(performances.items(), key=lambda x: x[1].get('total_return', float('inf')))

            insights.append(f"最佳策略: {best[0]} ({best[1].get('total_return', 0):.2f}% 收益)")
            insights.append(f"最差策略: {worst[0]} ({worst[1].get('total_return', 0):.2f}% 收益)")

            # Risk assessment
            max_drawdown = max([perf.get('max_drawdown', 0) for perf in performances.values()])
            if max_drawdown < -20:
                insights.append("高风险警告: 最大回撤超过20%")
            elif max_drawdown < -10:
                insights.append("中等风险: 最大回撤在10-20%之间")
            else:
                insights.append("风险相对可控: 最大回撤低于10%")

        # Market trend analysis
        current_price = data['close'].iloc[-1]
        ma20 = data['ma20'].iloc[-1]
        ma60 = data['ma60'].iloc[-1]

        if current_price > ma20 > ma60:
            insights.append("技术指标显示上升趋势")
        elif current_price < ma20 < ma60:
            insights.append("技术指标显示下降趋势")
        else:
            insights.append("技术指标显示震荡行情")

        return insights

    def _identify_risk_factors(self, performances: Dict) -> List[Dict]:
        """Identify potential risk factors"""
        risks = []

        if not performances:
            return risks

        # High volatility risk
        avg_volatility = np.mean([perf.get('volatility', 0) for perf in performances.values()])
        if avg_volatility > 30:
            risks.append({
                'type': 'high_volatility',
                'severity': 'high',
                'description': '平均波动率超过30%，投资风险较高'
            })

        # Maximum drawdown risk
        max_drawdown = max([perf.get('max_drawdown', 0) for perf in performances.values()])
        if max_drawdown < -20:
            risks.append({
                'type': 'large_drawdown',
                'severity': 'critical',
                'description': f'最大回撤达到{max_drawdown:.1f}%，存在巨大亏损风险'
            })

        # Low return risk
        avg_return = np.mean([perf.get('total_return', 0) for perf in performances.values()])
        if avg_return < 0:
            risks.append({
                'type': 'negative_return',
                'severity': 'medium',
                'description': f'平均收益为{avg_return:.2f}%，投资效果不佳'
            })

        return risks

    def generate_html_report(self, analysis_result: Dict, filename: str = "strategy_analysis_report.html"):
        """Generate comprehensive HTML report"""
        if 'error' in analysis_result:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>股票分析报告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                    .error {{ color: #d32f2f; padding: 20px; border: 1px solid #ffcdd2; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>股票分析报告</h1>
                <div class="error">
                    <h2>错误</h2>
                    <p>{analysis_result['error']}</p>
                </div>
            </body>
            </html>
            """
        else:
            stock_info = analysis_result['stock_info']
            recommendation = analysis_result['recommendation']
            comparison = analysis_result['comparison']
            visualizations = analysis_result['visualizations']

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{stock_info['name']} ({stock_info['code']}) - 策略分析报告</title>
                <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .chart-container {{ width: 100%; height: 500px; margin: 20px 0; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 20px; }}
                    .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                    h1 {{ color: #1976d2; }}
                    h2 {{ color: #424242; border-bottom: 2px solid #1976d2; padding-bottom: 10px; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #f5f5f5; font-weight: bold; }}
                    .positive {{ color: #4caf50; font-weight: bold; }}
                    .negative {{ color: #f44336; font-weight: bold; }}
                    .neutral {{ color: #ff9800; font-weight: bold; }}
                    .recommendation-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                    .insight {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 4px solid #1976d2; }}
                    .risk {{ margin: 10px 0; padding: 10px; background: #ffebee; border-left: 4px solid #f44336; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{stock_info['name']} ({stock_info['code']})</h1>
                    <p>分析期间: {stock_info['start_date']} 至 {stock_info['end_date']}</p>
                </div>

                <!-- Investment Recommendation -->
                <div class="recommendation-card">
                    <h2>投资建议</h2>
                    <p><strong>综合评分:</strong> {recommendation['overall']['score']:.1f}/100</p>
                    <p><strong>建议:</strong> {recommendation['overall']['recommendation']}</p>
                    <p><strong>置信度:</strong> {recommendation['overall']['confidence_level']}</p>
                </div>

                <!-- Strategy Comparison -->
                <div class="section">
                    <h2>策略对比</h2>
                    {comparison.to_html(classes='comparison-table', float_format="%.2f")}
                </div>

                <!-- K-line Chart -->
                <div class="chart-container">
                    <div id="kline_chart" style="width: 100%; height: 100%;"></div>
                </div>

                <!-- Strategy Comparison Chart -->
                <div class="chart-container">
                    <div id="strategy_chart" style="width: 100%; height: 100%;"></div>
                </div>

                <!-- Key Insights -->
                <div class="section">
                    <h2>关键洞察</h2>
                    {"".join([f'<div class="insight">{insight}</div>' for insight in analysis_result.get('recommendation', {}).get('key_insights', [])])}
                </div>

                <!-- Risk Factors -->
                <div class="section">
                    <h2>风险因素</h2>
                    {"".join([f'<div class="risk">{risk["description"]}</div>' for risk in analysis_result.get('recommendation', {}).get('risk_factors', [])])}
                </div>

                <!-- Technical Indicators -->
                <div class="section">
                    <h2>技术指标</h2>
                    <p>最新技术指标数据：</p>
                    <table>
                        <tr><th>指标</th><th>数值</th></tr>
                        {"".join([f'<tr><td>{ind}</td><td>{value}</td></tr>' for ind, value in analysis_result.get('technical_indicators', [{}])[0].items()])}
                    </table>
                </div>

                <script>
                    // Initialize K-line Chart
                    var klineChart = echarts.init(document.getElementById('kline_chart'));
                    klineChart.setOption({json.dumps(visualizations['kline'], ensure_ascii=False)});

                    // Initialize Strategy Comparison Chart
                    var strategyChart = echarts.init(document.getElementById('strategy_chart'));
                    strategyChart.setOption({json.dumps(visualizations['strategy_comparison'], ensure_ascii=False)});

                    // Responsive
                    window.addEventListener('resize', function() {{
                        klineChart.resize();
                        strategyChart.resize();
                    }});
                </script>
            </body>
            </html>
            """

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return filename


if __name__ == "__main__":
    # Test enhanced strategy analyzer
    fetcher = StockDataFetcher()
    analyzer = EnhancedStrategyAnalyzer(fetcher)

    # Test with sample stock (贵州茅台)
    result = analyzer.analyze_stock_strategies('600519.SH', '20230101', '20241231')

    if 'error' not in result:
        print("Analysis completed successfully!")
        print(f"Best strategy: {result['recommendation']['overall']['recommendation']}")

        # Generate HTML report
        report_file = analyzer.generate_html_report(result)
        print(f"Report saved to: {report_file}")
    else:
        print(f"Error: {result['error']}")