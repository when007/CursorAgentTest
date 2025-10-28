"""
Utility functions for the stock analysis system
股票分析系统工具函数
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import hashlib


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('stock_analysis.log'),
            logging.StreamHandler()
        ]
    )
    logging.info("Stock analysis system logging initialized")


def validate_stock_code(stock_code: str) -> str:
    """
    Validate and format stock code

    Args:
        stock_code: Raw stock code

    Returns:
        Formatted stock code
    """
    # Remove spaces and common prefixes
    code = stock_code.strip().upper()

    # Check if already in correct format
    if '.' in code:
        return code

    # Convert 6-digit code to full format
    if len(code) == 6:
        if code.startswith(('60', '68', '688')):  # Shanghai stocks
            return f"{code}.SH"
        else:  # Shenzhen stocks
            return f"{code}.SZ"

    return code


def format_currency(amount: Union[int, float]) -> str:
    """Format currency with Chinese Yuan symbol"""
    try:
        return f"¥{amount:,.2f}"
    except:
        return f"¥{amount}"


def format_percentage(value: Union[int, float]) -> str:
    """Format percentage with + sign for positive values"""
    try:
        if value >= 0:
            return f"+{value:.2f}%"
        else:
            return f"{value:.2f}%"
    except:
        return f"{value}%"


def calculate_performance_metrics(returns: pd.Series) -> Dict:
    """
    Calculate comprehensive performance metrics

    Args:
        returns: Series of returns

    Returns:
        Dictionary of performance metrics
    """
    if len(returns) == 0:
        return {}

    # Basic metrics
    total_return = (1 + returns).prod() - 1

    # Annualized return
    days = len(returns)
    annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # Volatility
    volatility = returns.std() * np.sqrt(252)

    # Sharpe ratio (assuming risk-free rate of 2%)
    sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0

    # Sortino ratio
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
    sortino_ratio = (annualized_return - 0.02) / downside_deviation if downside_deviation > 0 else 0

    # Win rate
    win_rate = (returns > 0).mean() * 100

    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    metrics = {
        'total_return': total_return * 100,
        'annualized_return': annualized_return * 100,
        'max_drawdown': max_drawdown * 100,
        'volatility': volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'win_rate': win_rate,
        'calmar_ratio': calmar_ratio,
        'profit_factor': profit_factor,
        'trade_count': len(returns)
    }

    return metrics


def generate_color_scheme(style: str = "default") -> Dict:
    """
    Generate color scheme for charts

    Args:
        style: Color scheme style ('default', 'dark', 'green')

    Returns:
        Dictionary of color values
    """
    schemes = {
        'default': {
            'up': '#ef5350',      # Red for up
            'down': '#26a69a',    # Green for down
            'background': '#ffffff',
            'grid': '#e0e0e0',
            'text': '#424242',
            'positive': '#4caf50',
            'negative': '#f44336'
        },
        'dark': {
            'up': '#ff5252',
            'down': '#69f0ae',
            'background': '#1a1a1a',
            'grid': '#404040',
            'text': '#ffffff',
            'positive': '#69f0ae',
            'negative': '#ff5252'
        },
        'green': {
            'up': '#4caf50',
            'down': '#ff9800',
            'background': '#f0f8f0',
            'grid': '#c8e6c9',
            'text': '#2e7d32',
            'positive': '#4caf50',
            'negative': '#ff9800'
        }
    }

    return schemes.get(style, schemes['default'])


def create_cache_key(stock_code: str, start_date: str, end_date: str) -> str:
    """Create cache key for stock data"""
    key_str = f"{stock_code}_{start_date}_{end_date}"
    return hashlib.md5(key_str.encode()).hexdigest()


def is_trading_day(date: datetime) -> bool:
    """Check if a given date is a trading day"""
    # Simple implementation - exclude weekends
    return date.weekday() < 5


def get_previous_trading_day(date: datetime, n: int = 1) -> datetime:
    """Get previous trading day(s)"""
    current_date = date
    for _ in range(n * 7):  # Check up to 7 weeks back
        current_date = current_date - timedelta(days=1)
        if is_trading_day(current_date):
            return current_date
    return current_date


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default


def truncate_string(text: str, max_length: int = 50) -> str:
    """Truncate string to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def create_backup_file(file_path: str, max_backups: int = 5) -> None:
    """Create backup of existing file"""
    if not os.path.exists(file_path):
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_{timestamp}"

    try:
        os.rename(file_path, backup_path)

        # Remove old backups
        backup_files = [f for f in os.listdir(os.path.dirname(file_path))
                       if f.startswith(os.path.basename(file_path) + ".backup_")]

        if len(backup_files) > max_backups:
            backup_files.sort()
            for old_backup in backup_files[:-max_backups]:
                os.remove(os.path.join(os.path.dirname(file_path), old_backup))

    except Exception as e:
        logging.error(f"Failed to create backup: {e}")


def validate_date_range(start_date: str, end_date: str) -> bool:
    """Validate date range format"""
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        return start <= end
    except:
        return False


def get_business_days(start_date: datetime, end_date: datetime) -> int:
    """Get number of business days between two dates"""
    days = 0
    current_date = start_date

    while current_date <= end_date:
        if is_trading_day(current_date):
            days += 1
        current_date += timedelta(days=1)

    return days


def calculate_rolling_metrics(data: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
    """Calculate rolling metrics"""
    rolling_metrics = {
        'mean': data.rolling(window=window).mean(),
        'std': data.rolling(window=window).std(),
        'min': data.rolling(window=window).min(),
        'max': data.rolling(window=window).max(),
        'median': data.rolling(window=window).median()
    }

    return rolling_metrics


def optimize_parameters(data: pd.DataFrame, strategy_func, param_grid: Dict) -> Dict:
    """
    Simple parameter optimization using grid search

    Args:
        data: Historical price data
        strategy_func: Strategy function to optimize
        param_grid: Dictionary of parameter ranges to test

    Returns:
        Best parameters and corresponding performance
    """
    best_score = -float('inf')
    best_params = None
    best_performance = None

    # Simple grid search (not optimized for large parameter spaces)
    from itertools import product

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for params in product(*param_values):
        param_dict = dict(zip(param_names, params))

        try:
            # Run strategy with current parameters
            strategy = strategy_func(**param_dict)
            results = strategy.backtest(data)

            if not results.empty:
                performance = calculate_performance_metrics(results['portfolio_value'].pct_change().dropna())
                score = performance.get('sharpe_ratio', 0)

                if score > best_score:
                    best_score = score
                    best_params = param_dict
                    best_performance = performance

        except Exception as e:
            logging.warning(f"Parameter combination {param_dict} failed: {e}")
            continue

    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_performance': best_performance
    }


if __name__ == "__main__":
    # Test utility functions
    setup_logging()

    # Test date validation
    print(f"Date validation: {validate_date_range('20230101', '20231231')}")

    # Test currency formatting
    print(f"Currency formatting: {format_currency(123456.78)}")

    # Test percentage formatting
    print(f"Percentage formatting: {format_percentage(15.5)}")
    print(f"Negative percentage: {format_percentage(-8.2)}")

    # Test performance metrics calculation
    sample_returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.025])
    metrics = calculate_performance_metrics(sample_returns)
    print(f"Performance metrics: {metrics}")

    print("Utility functions test completed successfully!")