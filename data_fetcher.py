"""
Stock Data Fetcher
Tushare API wrapper for A-share data fetching
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from typing import Optional, Dict, List, Union


class StockDataFetcher:
    def __init__(self, api_token: str = None):
        """
        Initialize data fetcher with Tushare API token

        Args:
            api_token: Tushare API token. If None, will try to use environment variable TUSHARE_TOKEN
        """
        self.api_token = api_token
        self.pro = None
        self.demo_mode = True

        try:
            if api_token:
                ts.set_token(api_token)
                self.pro = ts.pro_api()
                self.demo_mode = False
                print("Using Tushare Pro API with provided token")
            else:
                token = os.getenv('TUSHARE_TOKEN')
                if token:
                    ts.set_token(token)
                    self.pro = ts.pro_api()
                    self.demo_mode = False
                    print("Using Tushare Pro API with environment variable token")
                else:
                    print("Warning: No Tushare API token provided. Using demo mode with limited functionality.")
        except Exception as e:
            print(f"Error initializing Tushare API: {e}")
            print("Running in demo mode with sample data only.")

    def get_stock_basic_info(self, stock_code: str = None) -> pd.DataFrame:
        """
        Get basic information for stocks

        Args:
            stock_code: Stock code (e.g., '000001' or '600519'). If None, returns all stocks

        Returns:
            DataFrame with stock basic information
        """
        if self.demo_mode:
            return self._get_demo_basic_info(stock_code)

        try:
            df = self.pro.stock_basic(exchange='', list_status='L')
            if stock_code:
                df = df[df.ts_code.str.startswith(stock_code)]
            return df
        except Exception as e:
            print(f"Error fetching stock basic info: {e}")
            return self._get_demo_basic_info(stock_code)

    def search_stock(self, keyword: str) -> pd.DataFrame:
        """
        Search stocks by name or code

        Args:
            keyword: Search keyword

        Returns:
            DataFrame with search results
        """
        try:
            stocks = self.get_stock_basic_info()
            # Search by code or name
            results = stocks[stocks.ts_code.str.contains(keyword, case=False) |
                           stocks.name.str.contains(keyword, case=False)]
            return results.head(20)  # Return top 20 results
        except Exception as e:
            print(f"Error searching stock: {e}")
            return pd.DataFrame()

    def get_kline_data(self, stock_code: str, period: str = 'D',
                       start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get K-line data for a specific stock

        Args:
            stock_code: Stock code (e.g., '000001.SZ')
            period: Data period ('1m' - 1 minute, '5m' - 5 minute, '15m' - 15 minute,
                   '30m' - 30 minute, '60m' - 1 hour, 'D' - daily, 'W' - weekly,
                   'M' - monthly, 'Y' - yearly)
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)

        Returns:
            DataFrame with K-line data
        """
        if self.demo_mode:
            return self._generate_demo_kline_data(stock_code, period, start_date, end_date)

        try:
            # Convert period codes
            period_map = {
                '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
                '60m': '60min', 'D': 'daily', 'W': 'weekly', 'M': 'monthly', 'Y': 'yearly'
            }

            if period not in period_map:
                print(f"Invalid period {period}, using daily")
                period = 'D'

            actual_period = period_map[period]

            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y%m%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

            # Fetch data
            df = self.pro.query_kline_data(
                ts_code=stock_code,
                fields='ts_code,trade_date,open,high,low,close,vol,amount',
                freq=actual_period,
                start_date=start_date,
                end_date=end_date
            )

            if df.empty:
                print(f"No data found for {stock_code}")
                return self._generate_demo_kline_data(stock_code, period, start_date, end_date)

            # Convert trade_date to datetime and sort
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.sort_values('trade_date')

            return df

        except Exception as e:
            print(f"Error fetching K-line data for {stock_code}: {e}")
            return self._generate_demo_kline_data(stock_code, period, start_date, end_date)

    def get_multiple_periods(self, stock_code: str, periods: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get K-line data for multiple time periods

        Args:
            stock_code: Stock code
            periods: List of periods to fetch

        Returns:
            Dictionary mapping period names to DataFrames
        """
        if periods is None:
            periods = ['D', 'W', 'M', 'Y']  # Default to daily, weekly, monthly, yearly

        period_names = {
            '1m': '1分钟', '5m': '5分钟', '15m': '15分钟', '30m': '30分钟',
            '60m': '1小时', 'D': '日线', 'W': '周线', 'M': '月线', 'Y': '年线'
        }

        results = {}
        for period in periods:
            period_name = period_names.get(period, period)
            print(f"Fetching {period_name} data...")
            df = self.get_kline_data(stock_code, period)
            if not df.empty:
                results[period_name] = df
            time.sleep(0.1)  # Avoid API rate limits

        return results

    def get_index_data(self, index_code: str = '000300.SH',
                      period: str = 'D', start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get index data

        Args:
            index_code: Index code
            period: Data period
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with index data
        """
        try:
            df = self.pro.index_weight(
                ts_code=index_code,
                start_date=start_date or (datetime.now() - timedelta(days=365)).strftime('%Y%m%d'),
                end_date=end_date or datetime.now().strftime('%Y%m%d')
            )
            return df
        except Exception as e:
            print(f"Error fetching index data: {e}")
            return pd.DataFrame()

    def validate_stock_code(self, code: str) -> str:
        """
        Validate and format stock code

        Args:
            code: Raw stock code

        Returns:
            Formatted stock code
        """
        # Remove spaces and common prefixes
        code = code.strip().upper()

        # Check if already in correct format (e.g., '000001.SZ')
        if '.' in code:
            return code

        # Convert 6-digit code to full format
        if len(code) == 6:
            # Check if it's a Shanghai stock (600xxx, 601xxx) or Shenzhen stock (000xxx, 001xxx)
            if code.startswith(('60', '68', '688')):  # Shanghai
                return f"{code}.SH"
            else:  # Shenzhen
                return f"{code}.SZ"

        return code

    def get_stock_name(self, stock_code: str) -> str:
        """
        Get stock name from stock code

        Args:
            stock_code: Stock code

        Returns:
            Stock name
        """
        try:
            df = self.get_stock_basic_info(stock_code)
            if not df.empty:
                return df.iloc[0]['name']
            else:
                return "Unknown"
        except Exception as e:
            print(f"Error getting stock name: {e}")
            return "Unknown"

    def _get_demo_basic_info(self, stock_code: str = None) -> pd.DataFrame:
        """Generate demo stock basic information"""
        demo_stocks = [
            {'ts_code': '600519.SH', 'name': '贵州茅台', 'area': '贵州', 'industry': '白酒', 'market': '主板', 'list_date': '20011120'},
            {'ts_code': '000001.SZ', 'name': '平安银行', 'area': '广东', 'industry': '银行', 'market': '主板', 'list_date': '19910403'},
            {'ts_code': '000002.SZ', 'name': '万科A', 'area': '广东', 'industry': '房地产', 'market': '主板', 'list_date': '19910129'},
            {'ts_code': '300750.SZ', 'name': '宁德时代', 'area': '福建', 'industry': '电池', 'market': '创业板', 'list_date': '20180611'},
            {'ts_code': '600036.SH', 'name': '招商银行', 'area': '广东', 'industry': '银行', 'market': '主板', 'list_date': '20020409'},
            {'ts_code': '000858.SZ', 'name': '五粮液', 'area': '四川', 'industry': '白酒', 'market': '主板', 'list_date': '19940327'},
            {'ts_code': '600887.SH', 'name': '伊利股份', 'area': '内蒙古', 'industry': '食品饮料', 'market': '主板', 'list_date': '19930628'},
            {'ts_code': '002415.SZ', 'name': '海康威视', 'area': '浙江', 'industry': '电子', 'market': '中小板', 'list_date': '2010072'}
        ]

        df = pd.DataFrame(demo_stocks)
        if stock_code:
            df = df[df.ts_code.str.contains(stock_code, case=False) |
                   df.name.str.contains(stock_code, case=False)]
        return df

    def _generate_demo_kline_data(self, stock_code: str, period: str = 'D',
                                  start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Generate realistic K-line data for demo mode"""
        # Generate sample data based on stock characteristics
        np.random.seed(hash(stock_code) % 10000)  # Different seed for each stock

        # Set default dates
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=365)

        # Calculate number of data points
        if period == 'D':
            n_points = (end_date - start_date).days
            n_points = max(n_points, 100)  # Minimum 100 points
        elif period == 'W':
            n_points = max((end_date - start_date).days // 7, 20)
        elif period == 'M':
            n_points = max((end_date - start_date).days // 30, 12)
        else:
            n_points = 252  # Default to daily

        # Generate realistic price movement
        base_price = self._get_stock_base_price(stock_code)
        dates = pd.date_range(start=start_date, periods=n_points, freq='D')

        # Simulate price with trend and volatility
        prices = [base_price]
        trend = np.random.uniform(-0.0005, 0.0005, n_points)  # Small trend
        volatility = np.random.uniform(0.015, 0.025, n_points)  # Realistic volatility

        for i in range(1, n_points):
            change = np.random.normal(trend[i], volatility[i])
            price = prices[-1] * (1 + change)
            prices.append(max(price, base_price * 0.5))  # Price won't go below 50% of base

        # Generate OHLC data
        ohlc_data = []
        for i, date in enumerate(dates):
            base = prices[i]
            ohlc_data.append({
                'ts_code': stock_code,
                'trade_date': date.strftime('%Y%m%d'),
                'open': base * (1 + np.random.uniform(-0.005, 0.005)),
                'high': base * (1 + abs(np.random.uniform(0, 0.015))),
                'low': base * (1 - abs(np.random.uniform(0, 0.015))),
                'close': base,
                'vol': np.random.uniform(1000000, 10000000),
                'amount': base * np.random.uniform(1000000, 10000000)
            })

        return pd.DataFrame(ohlc_data)

    def _get_stock_base_price(self, stock_code: str) -> float:
        """Get base price for demo data based on stock"""
        price_map = {
            '600519.SH': 1800,  # 茅台
            '000001.SZ': 12,    # 平安银行
            '000002.SZ': 20,    # 万科A
            '300750.SZ': 250,   # 宁德时代
            '600036.SH': 35,    # 招商银行
            '000858.SZ': 100,   # 五粮液
            '600887.SH': 30,    # 伊利股份
            '002415.SZ': 30     # 海康威视
        }
        return price_map.get(stock_code, np.random.uniform(20, 100))


if __name__ == "__main__":
    # Test data fetcher
    fetcher = StockDataFetcher()

    # Test search
    print("Searching for '茅台'...")
    results = fetcher.search_stock('茅台')
    print(results.head())

    # Test basic info
    print("\nGetting basic info...")
    basic_info = fetcher.get_stock_basic_info('600519')
    print(basic_info.head())