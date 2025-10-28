"""
Stock Investment Strategies Implementation
5 different trading strategies for A-share analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class Position(Enum):
    """Position status"""
    LONG = 1
    SHORT = -1
    FLAT = 0


class TradingStrategy:
    """Base class for trading strategies"""

    def __init__(self, name: str):
        self.name = name
        self.positions = []
        self.returns = []
        self.portfolio_values = []

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals
        Returns: Series of signals (1 for buy, -1 for sell, 0 for hold)
        """
        raise NotImplementedError("Subclasses must implement generate_signals")

    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> pd.DataFrame:
        """
        Backtest the strategy
        """
        signals = self.generate_signals(data)
        capital = initial_capital
        position = Position.FLAT
        shares = 0
        results = []

        for i, (date, signal) in enumerate(zip(data.index, signals)):
            current_price = data['close'].iloc[i]
            trade_value = capital * 0.1  # 10% of capital per trade

            if signal == 1 and position == Position.FLAT:  # Buy signal
                shares = int(trade_value / current_price)
                capital -= shares * current_price
                position = Position.LONG
                self.positions.append(('BUY', date, current_price, shares))

            elif signal == -1 and position == Position.LONG:  # Sell signal
                capital += shares * current_price
                self.positions.append(('SELL', date, current_price, shares))
                shares = 0
                position = Position.FLAT

            # Calculate portfolio value
            portfolio_value = capital + shares * current_price
            results.append({
                'date': date,
                'price': current_price,
                'signal': signal,
                'position': position.value,
                'shares': shares,
                'capital': capital,
                'portfolio_value': portfolio_value,
                'return': (portfolio_value / initial_capital - 1) * 100
            })

        return pd.DataFrame(results)


class TrendFollowingStrategy(TradingStrategy):
    """趋势跟踪策略 - 双均线交叉"""

    def __init__(self, short_window: int = 20, long_window: int = 60):
        super().__init__("趋势跟踪策略")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if len(data) < self.long_window:
            return pd.Series([0] * len(data), index=data.index)

        # Calculate moving averages
        short_ma = data['close'].rolling(window=self.short_window, min_periods=1).mean()
        long_ma = data['close'].rolling(window=self.long_window, min_periods=1).mean()

        # Generate signals
        signals = pd.Series(0, index=data.index)

        # Golden cross: short MA crosses above long MA
        golden_cross = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        signals[golden_cross] = 1

        # Death cross: short MA crosses below long MA
        death_cross = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        signals[death_cross] = -1

        return signals


class MeanReversionStrategy(TradingStrategy):
    """均值回归策略 - 价格偏离均线"""

    def __init__(self, window: int = 20, threshold: float = 2.0):
        super().__init__("均值回归策略")
        self.window = window
        self.threshold = threshold  # Number of standard deviations

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if len(data) < self.window:
            return pd.Series([0] * len(data), index=data.index)

        # Calculate mean and standard deviation
        rolling_mean = data['close'].rolling(window=self.window, min_periods=1).mean()
        rolling_std = data['close'].rolling(window=self.window, min_periods=1).std()

        # Calculate z-score
        z_scores = (data['close'] - rolling_mean) / rolling_std

        # Generate signals
        signals = pd.Series(0, index=data.index)

        # Buy when price is undervalued (z-score < -threshold)
        buy_signals = z_scores < -self.threshold
        signals[buy_signals] = 1

        # Sell when price is overvalued (z-score > threshold)
        sell_signals = z_scores > self.threshold
        signals[sell_signals] = -1

        return signals


class MomentumStrategy(TradingStrategy):
    """动量策略 - 价格变化速率"""

    def __init__(self, lookback_period: int = 10, momentum_threshold: float = 0.02):
        super().__init__("动量策略")
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if len(data) < self.lookback_period:
            return pd.Series([0] * len(data), index=data.index)

        # Calculate momentum (rate of change)
        momentum = (data['close'] - data['close'].shift(self.lookback_period)) / data['close'].shift(self.lookback_period)

        # Generate signals
        signals = pd.Series(0, index=data.index)

        # Buy when positive momentum exceeds threshold
        buy_signals = momentum > self.momentum_threshold
        signals[buy_signals] = 1

        # Sell when negative momentum exceeds threshold
        sell_signals = momentum < -self.momentum_threshold
        signals[sell_signals] = -1

        return signals


class MartingaleStrategy(TradingStrategy):
    """亏损就翻倍策略 - 马丁格尔策略"""

    def __init__(self, base_position: float = 1000, multiplier: float = 2.0):
        super().__init__("亏损就翻倍策略")
        self.base_position = base_position
        self.multiplier = multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if len(data) < 2:
            return pd.Series([0] * len(data), index=data.index)

        signals = pd.Series(0, index=data.index)
        consecutive_losses = 0
        in_position = False
        current_position = self.base_position

        for i in range(1, len(data)):
            if not in_position:
                # Enter position
                signals.iloc[i] = 1
                in_position = True
                consecutive_losses = 0
                current_position = self.base_position
            else:
                # Check if current position is profitable
                entry_price = data['close'].iloc[i-1]
                current_price = data['close'].iloc[i]
                profit_loss = (current_price - entry_price) / entry_price

                if profit_loss > 0:
                    # Take profit
                    signals.iloc[i] = -1
                    in_position = False
                    consecutive_losses = 0
                    current_position = self.base_position
                else:
                    # Loss detected, increase position
                    consecutive_losses += 1
                    current_position = self.base_position * (self.multiplier ** consecutive_losses)
                    signals.iloc[i] = 1  # Double down

        return signals

    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> pd.DataFrame:
        """
        Modified backtest for martingale strategy with risk management
        """
        signals = self.generate_signals(data)
        capital = initial_capital
        position = Position.FLAT
        shares = 0
        results = []
        consecutive_losses = 0
        current_position_size = self.base_position

        for i, (date, signal) in enumerate(zip(data.index, signals)):
            current_price = data['close'].iloc[i]

            if signal == 1 and position == Position.FLAT:  # Buy signal
                # Risk management: don't risk more than 10% of capital
                max_investment = capital * 0.1
                shares = int(max_investment / current_price)
                shares = min(shares, int(current_position_size / current_price))

                cost = shares * current_price
                if cost > capital * 0.05:  # Minimum 5% capital per trade
                    capital -= cost
                    position = Position.LONG
                    self.positions.append(('BUY', date, current_price, shares))

            elif signal == -1 and position == Position.LONG:  # Sell signal
                capital += shares * current_price
                profit_loss = (current_price - self.positions[-1][3]) / self.positions[-1][3]

                if profit_loss > 0:
                    consecutive_losses = 0
                    current_position_size = self.base_position
                else:
                    consecutive_losses += 1
                    current_position_size = self.base_position * (self.multiplier ** consecutive_losses)
                    current_position_size = min(current_position_size, capital * 0.1)  # Cap at 10% of capital

                self.positions.append(('SELL', date, current_price, shares))
                shares = 0
                position = Position.FLAT

            # Calculate portfolio value
            portfolio_value = capital + shares * current_price
            results.append({
                'date': date,
                'price': current_price,
                'signal': signal,
                'position': position.value,
                'shares': shares,
                'capital': capital,
                'portfolio_value': portfolio_value,
                'return': (portfolio_value / initial_capital - 1) * 100
            })

        return pd.DataFrame(results)


class GridTradingStrategy(TradingStrategy):
    """网格交易策略 - 价格区间内的网格交易"""

    def __init__(self, grid_size: float = 0.02, upper_bound: float = None, lower_bound: float = None):
        super().__init__("网格交易策略")
        self.grid_size = grid_size  # Grid spacing (2%)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if len(data) < 10:
            return pd.Series([0] * len(data), index=data.index)

        signals = pd.Series(0, index=data.index)

        # Set bounds if not provided
        if self.upper_bound is None:
            self.upper_bound = data['close'].iloc[:10].max() * 1.2
        if self.lower_bound is None:
            self.lower_bound = data['close'].iloc[:10].min() * 0.8

        current_price = data['close'].iloc[0]
        grid_levels = np.arange(self.lower_bound, self.upper_bound, self.grid_size)
        position_levels = {level: 0 for level in grid_levels}
        current_level = None

        for i, price in enumerate(data['close']):
            # Find the nearest grid level
            nearest_levels = []
            for level in grid_levels:
                if abs(price - level) < self.grid_size / 2:
                    nearest_levels.append(level)

            if nearest_levels:
                signal_level = nearest_levels[0]

                if signal_level not in position_levels:
                    continue

                # Buy when price is below grid level
                if price < signal_level and current_level is None:
                    signals.iloc[i] = 1
                    current_level = signal_level
                    position_levels[signal_level] = 1

                # Sell when price is above grid level
                elif price > signal_level and current_level is None:
                    signals.iloc[i] = -1
                    current_level = signal_level
                    position_levels[signal_level] = -1

                # Close position when returning to grid
                elif current_level is not None and abs(price - current_level) < self.grid_size / 4:
                    signals.iloc[i] = -1 if position_levels[current_level] == 1 else 1
                    position_levels[current_level] = 0
                    current_level = None

        return signals

    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> pd.DataFrame:
        """
        Modified backtest for grid trading strategy
        """
        signals = self.generate_signals(data)
        capital = initial_capital
        position = Position.FLAT
        shares = 0
        results = []
        current_grid_level = None
        grid_levels = np.arange(self.lower_bound, self.upper_bound, self.grid_size)
        position_status = {level: 0 for level in grid_levels}

        for i, (date, signal) in enumerate(zip(data.index, signals)):
            current_price = data['close'].iloc[i]

            if signal == 1 and position == Position.FLAT:
                # Buy at grid level
                shares = int(capital * 0.05 / current_price)  # 5% of capital per trade
                cost = shares * current_price
                capital -= cost
                position = Position.LONG
                self.positions.append(('BUY', date, current_price, shares))
                current_grid_level = current_price

            elif signal == -1 and position == Position.LONG:
                # Sell at grid level
                capital += shares * current_price
                self.positions.append(('SELL', date, current_price, shares))
                shares = 0
                position = Position.FLAT

            # Calculate portfolio value
            portfolio_value = capital + shares * current_price
            results.append({
                'date': date,
                'price': current_price,
                'signal': signal,
                'position': position.value,
                'shares': shares,
                'capital': capital,
                'portfolio_value': portfolio_value,
                'return': (portfolio_value / initial_capital - 1) * 100
            })

        return pd.DataFrame(results)


class StrategyAnalyzer:
    """Strategy performance analyzer"""

    def __init__(self):
        pass

    def calculate_metrics(self, backtest_results: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics
        """
        portfolio_values = backtest_results['portfolio_value']
        returns = portfolio_values.pct_change().dropna()

        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100

        # Annualized return
        days_held = (backtest_results['date'].iloc[-1] - backtest_results['date'].iloc[0]).days
        annualized_return = (1 + total_return / 100) ** (365 / days_held) - 1 if days_held > 0 else 0

        # Maximum drawdown
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        # Volatility
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility

        # Sharpe ratio (assuming risk-free rate of 2%)
        sharpe_ratio = (annualized_return - 2) / volatility if volatility > 0 else 0

        # Win rate
        profitable_trades = backtest_results[backtest_results['return'] > 0]
        win_rate = len(profitable_trades) / len(backtest_results) * 100 if len(backtest_results) > 0 else 0

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'final_value': portfolio_values.iloc[-1],
            'trade_count': len(self.extract_trades(backtest_results))
        }

        return metrics

    def extract_trades(self, backtest_results: pd.DataFrame) -> List[Dict]:
        """
        Extract individual trades from backtest results
        """
        trades = []
        position = 0

        for i, row in backtest_results.iterrows():
            if row['position'] > position:  # Entered position
                trade_entry = {
                    'entry_date': row['date'],
                    'entry_price': row['price'],
                    'entry_type': 'LONG'
                }
            elif row['position'] < position:  # Exited position
                if 'trade_entry' in locals():
                    trade_entry['exit_date'] = row['date']
                    trade_entry['exit_price'] = row['price']
                    trade_entry['return'] = (row['price'] - trade_entry['entry_price']) / trade_entry['entry_price'] * 100
                    trades.append(trade_entry)
                    del trade_entry

            position = row['position']

        return trades

    def compare_strategies(self, strategy_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compare multiple strategies
        """
        comparison_data = []

        for strategy_name, results in strategy_results.items():
            if not results.empty:
                metrics = self.calculate_metrics(results)
                metrics['strategy'] = strategy_name
                comparison_data.append(metrics)

        comparison_df = pd.DataFrame(comparison_data)

        # Reorder columns
        columns_order = ['strategy', 'total_return', 'annualized_return', 'max_drawdown',
                        'volatility', 'sharpe_ratio', 'win_rate', 'final_value', 'trade_count']

        for col in columns_order:
            if col in comparison_df.columns:
                comparison_df = comparison_df[[col] + [c for c in comparison_df.columns if c != col]]

        return comparison_df


if __name__ == "__main__":
    # Test strategies
    analyzer = StrategyAnalyzer()

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252)
    np.random.seed(42)

    # Simulate stock price with trend and volatility
    prices = [100]
    for i in range(1, 252):
        change = np.random.normal(0.001, 0.02)  # 0.1% mean return, 2% std
        prices.append(prices[-1] * (1 + change))

    sample_data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'vol': np.random.uniform(1000000, 5000000, 252)
    })
    sample_data.set_index('date', inplace=True)

    # Test strategies
    strategies = [
        TrendFollowingStrategy(),
        MeanReversionStrategy(),
        MomentumStrategy(),
        MartingaleStrategy(),
        GridTradingStrategy()
    ]

    strategy_results = {}
    for strategy in strategies:
        print(f"Testing {strategy.name}...")
        results = strategy.backtest(sample_data)
        strategy_results[strategy.name] = results
        metrics = analyzer.calculate_metrics(results)
        print(f"{strategy.name}: Total Return = {metrics['total_return']:.2f}%")

    # Compare strategies
    comparison = analyzer.compare_strategies(strategy_results)
    print("\nStrategy Comparison:")
    print(comparison.to_string(float_format="%.2f"))

    print("Strategy testing completed successfully!")