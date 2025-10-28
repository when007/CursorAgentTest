"""
Stock Data Visualizer
Handles K-line chart creation and visualization using ECharts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta


class StockVisualizer:
    def __init__(self):
        self.colors = {
            'up': '#ef5350',      # Red for up
            'down': '#26a69a',    # Green for down
            'volume': '#757575',  # Gray for volume
            'ma5': '#ff9800',      # Orange for 5-day MA
            'ma10': '#2196f3',    # Blue for 10-day MA
            'ma20': '#9c27b0',    # Purple for 20-day MA
            'ma60': '#795548',    # Brown for 60-day MA
            'grid': '#e0e0e0',     # Light gray for grid
            'text': '#424242'     # Dark gray for text
        }

    def prepare_kline_data(self, df: pd.DataFrame) -> Dict:
        """
        Prepare K-line data for ECharts

        Args:
            df: DataFrame with K-line data

        Returns:
            Dictionary ready for ECharts
        """
        if df.empty:
            return {}

        # Ensure datetime format
        df = df.copy()
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['date_str'] = df['trade_date'].dt.strftime('%Y-%m-%d')
        else:
            df['date_str'] = df.index.astype(str)

        # Prepare candlestick data
        candlestick_data = []
        for idx, row in df.iterrows():
            candlestick = [
                row['open'],
                row['close'],
                row['low'],
                row['high']
            ]
            candlestick_data.append([row['date_str']] + candlestick)

        # Prepare volume data
        volume_data = []
        for idx, row in df.iterrows():
            volume_data.append([row['date_str'], row['vol']])

        # Calculate moving averages
        for ma_period in [5, 10, 20, 60]:
            ma_col = f'ma{ma_period}'
            df[ma_col] = df['close'].rolling(window=ma_period, min_periods=1).mean()
            ma_data = []
            for idx, row in df.iterrows():
                ma_data.append([row['date_str'], row[ma_col]])

        # Prepare data series
        series = []

        # Add candlestick
        series.append({
            'name': 'K线',
            'type': 'candlestick',
            'data': candlestick_data,
            'itemStyle': {
                'color': self.colors['up'],
                'color0': self.colors['down'],
                'borderColor': self.colors['up'],
                'borderColor0': self.colors['down']
            }
        })

        # Add volume
        if 'vol' in df.columns:
            series.append({
                'name': '成交量',
                'type': 'bar',
                'yAxisIndex': 1,
                'data': volume_data,
                'itemStyle': {
                    'color': lambda params: self.colors['up'] if params.value[1] >= params.value[0] else self.colors['down']
                }
            })

        # Add moving averages
        ma_periods = [5, 10, 20, 60]
        ma_period_names = ['5日均线', '10日均线', '20日均线', '60日均线']
        ma_period_colors = ['ma5', 'ma10', 'ma20', 'ma60']

        for i, (period, name, color_key) in enumerate(zip(ma_periods, ma_period_names, ma_period_colors)):
            ma_col = f'ma{period}'
            if ma_col in df.columns and not df[ma_col].isna().all():
                ma_data = []
                for idx, row in df.iterrows():
                    ma_data.append([row['date_str'], row[ma_col]])

                series.append({
                    'name': name,
                    'type': 'line',
                    'data': ma_data,
                    'lineStyle': {
                        'color': self.colors[color_key],
                        'width': 1
                    },
                    'smooth': True
                })

        # Prepare options for ECharts
        options = {
            'title': {
                'text': '股票K线图',
                'left': 'center',
                'textStyle': {
                    'color': self.colors['text'],
                    'fontSize': 16
                }
            },
            'tooltip': {
                'trigger': 'axis',
                'axisPointer': {
                    'type': 'cross'
                },
                'formatter': self._kline_tooltip_formatter()
            },
            'legend': {
                'data': ['K线'] + ma_period_names + (['成交量'] if 'vol' in df.columns else []),
                'top': 30,
                'textStyle': {
                    'color': self.colors['text']
                }
            },
            'grid': [
                {
                    'left': '10%',
                    'right': '10%',
                    'bottom': '15%',
                    'height': '60%',
                    'containLabel': True
                },
                {
                    'left': '10%',
                    'right': '10%',
                    'bottom': '5%',
                    'height': '10%',
                    'containLabel': True
                }
            ],
            'xAxis': [
                {
                    'type': 'category',
                    'data': df['date_str'].tolist(),
                    'gridIndex': 0,
                    'axisLabel': {
                        'color': self.colors['text'],
                        'rotate': 45
                    },
                    'axisLine': {
                        'lineStyle': {
                            'color': self.colors['grid']
                        }
                    }
                },
                {
                    'type': 'category',
                    'data': df['date_str'].tolist(),
                    'gridIndex': 1,
                    'axisLabel': {
                        'show': False
                    }
                }
            ],
            'yAxis': [
                {
                    'scale': True,
                    'gridIndex': 0,
                    'axisLabel': {
                        'color': self.colors['text']
                    },
                    'axisLine': {
                        'lineStyle': {
                            'color': self.colors['grid']
                        }
                    },
                    'splitLine': {
                        'lineStyle': {
                            'color': self.colors['grid'],
                            'type': 'dashed'
                        }
                    }
                },
                {
                    'scale': True,
                    'gridIndex': 1,
                    'position': 'right',
                    'axisLabel': {
                        'color': self.colors['text']
                    },
                    'axisLine': {
                        'lineStyle': {
                            'color': self.colors['grid']
                        }
                    }
                }
            ],
            'dataZoom': [
                {
                    'type': 'inside',
                    'xAxisIndex': [0, 1],
                    'start': 50,
                    'end': 100
                },
                {
                    'show': True,
                    'xAxisIndex': [0, 1],
                    'type': 'slider',
                    'bottom': 0,
                    'start': 50,
                    'end': 100,
                    'handleIcon': 'M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
                    'handleSize': '80%',
                    'textStyle': {
                        'color': self.colors['text']
                    }
                }
            ],
            'series': series
        }

        return options

    def _kline_tooltip_formatter(self):
        """Custom tooltip formatter for K-line chart"""
        return """
        function (params) {
            var result = params[0];
            if (result.componentType === 'candlestick') {
                var data = result.data;
                var date = data[0];
                var open = data[1];
                var close = data[2];
                var lowest = data[3];
                var highest = data[4];

                var change = ((close - open) / open * 100).toFixed(2);
                var changeColor = change >= 0 ? '#ef5350' : '#26a69a';
                var changeStr = (change >= 0 ? '+' : '') + change + '%';

                return [
                    '<div style="font-weight:bold;margin-bottom:8px;">' + date + '</div>',
                    '<div>开盘: <span style="color:#333">' + open.toFixed(2) + '</span></div>',
                    '<div>收盘: <span style="color:' + changeColor + '">' + close.toFixed(2) + ' (' + changeStr + ')</span></div>',
                    '<div>最高: <span style="color:#333">' + highest.toFixed(2) + '</span></div>',
                    '<div>最低: <span style="color:#333">' + lowest.toFixed(2) + '</span></div>',
                    '<div>成交量: <span style="color:#333">' + (data[5] / 10000).toFixed(2) + '万手</span></div>'
                ].join('');
            }

            var resultStr = '<div style="font-weight:bold;margin-bottom:8px;">' + params[0].name + '</div>';
            params.forEach(function(item) {
                resultStr += '<div>' + item.seriesName + ': <span style="color:#333">' + item.value[1].toFixed(2) + '</span></div>';
            });
            return resultStr;
        }
        """

    def create_strategy_comparison_chart(self, strategy_results: Dict[str, pd.DataFrame]) -> Dict:
        """
        Create strategy comparison chart

        Args:
            strategy_results: Dictionary with strategy names as keys and result DataFrames as values

        Returns:
            ECharts options for strategy comparison
        """
        if not strategy_results:
            return {}

        # Prepare data
        dates = None
        datasets = []

        for strategy_name, df in strategy_results.items():
            if df.empty:
                continue

            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df['date_str'] = df['trade_date'].dt.strftime('%Y-%m-%d')
            else:
                df['date_str'] = df.index.astype(str)

            # Get portfolio value (assuming 'portfolio_value' column exists)
            if 'portfolio_value' in df.columns:
                portfolio_data = []
                for idx, row in df.iterrows():
                    portfolio_data.append([row['date_str'], row['portfolio_value']])
                datasets.append({
                    'name': strategy_name,
                    'type': 'line',
                    'data': portfolio_data,
                    'smooth': True,
                    'lineStyle': {
                        'width': 2
                    }
            })

            if dates is None:
                dates = df['date_str'].tolist()

        # Generate distinct colors for each strategy
        strategy_colors = [
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57',
            '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43'
        ]

        # Prepare options
        options = {
            'title': {
                'text': '策略收益对比',
                'left': 'center',
                'textStyle': {
                    'color': self.colors['text'],
                    'fontSize': 16
                }
            },
            'tooltip': {
                'trigger': 'axis',
                'formatter': self._strategy_tooltip_formatter()
            },
            'legend': {
                'data': list(strategy_results.keys()),
                'top': 30,
                'textStyle': {
                    'color': self.colors['text']
                }
            },
            'grid': {
                'left': '10%',
                'right': '10%',
                'bottom': '15%',
                'top': '15%',
                'containLabel': True
            },
            'xAxis': {
                'type': 'category',
                'data': dates,
                'axisLabel': {
                    'color': self.colors['text'],
                    'rotate': 45
                },
                'axisLine': {
                    'lineStyle': {
                        'color': self.colors['grid']
                    }
                }
            },
            'yAxis': {
                'type': 'value',
                'axisLabel': {
                    'color': self.colors['text'],
                    'formatter': '{value}'
                },
                'axisLine': {
                    'lineStyle': {
                        'color': self.colors['grid']
                    }
                },
                'splitLine': {
                    'lineStyle': {
                        'color': self.colors['grid'],
                        'type': 'dashed'
                    }
                }
            },
            'dataZoom': [
                {
                    'type': 'inside',
                    'start': 50,
                    'end': 100
                },
                {
                    'show': True,
                    'type': 'slider',
                    'bottom': 0,
                    'start': 50,
                    'end': 100,
                    'textStyle': {
                        'color': self.colors['text']
                    }
                }
            ],
            'series': datasets
        }

        return options

    def _strategy_tooltip_formatter(self):
        """Custom tooltip formatter for strategy comparison"""
        return """
        function (params) {
            var result = '<div style="font-weight:bold;margin-bottom:8px;">' + params[0].name + '</div>';
            params.forEach(function(item) {
                if (item.value[1] !== undefined) {
                    var changePercent = ((item.value[1] - 100000) / 100000 * 100).toFixed(2);
                    var changeColor = changePercent >= 0 ? '#ef5350' : '#26a69a';
                    var changeStr = (changePercent >= 0 ? '+' : '') + changePercent + '%';

                    result += '<div>' + item.seriesName + ': <span style="color:' + changeColor + '">' +
                              item.value[1].toFixed(2) + ' (' + changeStr + ')</span></div>';
                }
            });
            return result;
        }
        """

    def generate_html_template(self, chart_options: Dict, chart_id: str = "main_chart",
                              width: str = "1200px", height: str = "600px") -> str:
        """
        Generate HTML template with ECharts

        Args:
            chart_options: ECharts options
            chart_id: Chart ID
            width: Chart width
            height: Chart height

        Returns:
            HTML string
        """
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>股票分析系统</title>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
            <style>
                body {{
                    margin: 0;
                    padding: 20px;
                    font-family: Arial, sans-serif;
                    background-color: #f5f5f5;
                }}
                .chart-container {{
                    width: {width};
                    height: {height};
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    padding: 20px;
                }}
                .loading {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: {height};
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="chart-container">
                <div id="{chart_id}" style="width: 100%; height: 100%;"></div>
            </div>

            <script>
                // Initialize ECharts
                var chart = echarts.init(document.getElementById('{chart_id}'));

                // Chart options
                var option = {json.dumps(chart_options, ensure_ascii=False)};

                // Set options
                chart.setOption(option);

                // Responsive
                window.addEventListener('resize', function() {{
                    chart.resize();
                }});
            </script>
        </body>
        </html>
        """
        return html_template

    def save_chart_html(self, chart_options: Dict, filename: str = "chart.html", **kwargs):
        """
        Save chart as HTML file

        Args:
            chart_options: ECharts options
            filename: Output filename
            kwargs: Additional parameters for generate_html_template
        """
        html_content = self.generate_html_template(chart_options, **kwargs)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Chart saved to {filename}")


if __name__ == "__main__":
    # Test visualizer
    visualizer = StockVisualizer()

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100)
    np.random.seed(42)

    sample_data = pd.DataFrame({
        'trade_date': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(100, 200, 100),
        'low': np.random.uniform(100, 200, 100),
        'close': np.random.uniform(100, 200, 100),
        'vol': np.random.uniform(1000000, 5000000, 100)
    })

    # Generate chart
    options = visualizer.prepare_kline_data(sample_data)

    # Save as HTML
    visualizer.save_chart_html(options, "sample_kline.html")
    print("Sample K-line chart generated successfully!")