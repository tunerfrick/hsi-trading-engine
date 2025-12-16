import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class EnhancedTradingEngine:
    def __init__(self, initial_cash: float = 100000, short_window: int = 10, long_window: int = 30):
        self.initial_cash = initial_cash
        self.short_window = short_window
        self.long_window = long_window
        self.data: Dict[str, pd.DataFrame] = {}
        self.portfolio_value = []
        self.symbols: List[str] = []
        self.is_live = False
    
    def fetch_hsi_constituents(self) -> List[str]:
        """Scrape full HSI constituents from Yahoo Finance"""
        try:
            url = "https://finance.yahoo.com/quote/%5EHSI/components/"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            symbols = []
            for row in soup.find_all('tr', limit=85):  # Top ~82 HSI stocks
                symbol_cell = row.find('td', {'data-symbol': True})
                if symbol_cell:
                    symbol = symbol_cell['data-symbol'] + '.HK'
                    if symbol not in ['N/A.HK']:  # Filter invalid
                        symbols.append(symbol)
            self.symbols = symbols[:20]  # Use top 20 for speed, adjust as needed
            print(f"Fetched {len(self.symbols)} HSI constituents: {self.symbols[:5]}...")
            return self.symbols
        except Exception as e:
            print(f"Scrape failed, using fallback: {e}")
            self.symbols = ['0005.HK', '0700.HK', '9988.HK', '0939.HK', '1299.HK']
            return self.symbols
    
    def fetch_data(self, period: str = '1y', live: bool = False):
        """Fetch data - historical or live"""
        self.is_live = live
        self.symbols = self.fetch_hsi_constituents() if not self.symbols else self.symbols
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                if live:
                    data = ticker.history(period='1d', interval='1m')  # Live intraday
                else:
                    data = ticker.history(period=period)
                self.data[symbol] = data['Close'].dropna()
                print(f"Fetched {symbol} ({len(self.data[symbol])} points)")
            except Exception as e:
                print(f"Error {symbol}: {e}")
    
    def markowitz_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """Covariance-based optimal weights using linear algebra"""
        n_assets = returns.shape[1]
        def portfolio_vol(weights, cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        def neg_sharpe(weights, returns, cov_matrix):
            port_return = np.sum(returns.mean() * weights) * 252
            port_vol = portfolio_vol(weights, cov_matrix) * np.sqrt(252)
            return -port_return / port_vol
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_guess = n_assets * [1. / n_assets]
        
        cov_matrix = returns.cov()
        opt_results = minimize(neg_sharpe, init_guess, args=(returns, cov_matrix),
                              method='SLSQP', bounds=bounds, constraints=constraints)
        return opt_results.x
    
    def generate_signals(self):
        """SMA crossover signals per stock"""
        for symbol, prices in self.data.items():
            if len(prices) < self.long_window:
                continue
            short_sma = prices.rolling(window=self.short_window).mean()
            long_sma = prices.rolling(window=self.long_window).mean()
            
            signal = 0
            signals = []
            for i in range(len(prices)):
                if i >= self.long_window and short_sma.iloc[i] > long_sma.iloc[i]:
                    signal = 1
                elif i >= self.long_window and short_sma.iloc[i] < long_sma.iloc[i]:
                    signal = -1
                signals.append(signal)
            
            self.data[symbol]['Signal'] = pd.Series(signals, index=prices.index)
            self.data[symbol]['Returns'] = prices.pct_change()
    
    def calculate_metrics(self, portfolio_df: pd.DataFrame) -> Dict:
        """Enhanced risk metrics"""
        returns = portfolio_df['Returns'].dropna()
        total_return = (portfolio_df['Portfolio'].iloc[-1] / self.initial_cash - 1) * 100
        
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        downside_returns = returns[returns < 0]
        sortino = (returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        rolling_max = portfolio_df['Portfolio'].expanding().max()
        drawdown = (portfolio_df['Portfolio'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        return {
            'Total Return (%)': total_return,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown (%)': max_drawdown
        }
    
    def backtest(self, use_markowitz: bool = True):
        """Enhanced backtest with covariance allocation"""
        cash = self.initial_cash
        holdings = {symbol: 0 for symbol in self.symbols}
        
        # Get aligned returns for Markowitz
        aligned_returns = pd.DataFrame({s: self.data[s]['Returns'].dropna() 
                                       for s in self.symbols if len(self.data[s]) > 50})
        
        weights = self.markowitz_weights(aligned_returns) if use_markowitz else None
        
        dates = sorted(set.intersection(*(set(self.data[s].index) for s in self.symbols if len(self.data[s]) > 0)))
        
        for date in dates:
            daily_value = cash
            for i, symbol in enumerate(self.symbols):
                if symbol not in self.data or date not in self.data[symbol].index:
                    continue
                
                signal = self.data[symbol].loc[date, 'Signal']
                price = self.data[symbol].loc[date]
                target_weight = weights[i] if weights is not None and i < len(weights) else 1/len(self.symbols)
                
                if signal == 1 and cash > 0:
                    alloc_cash = cash * target_weight * 0.95  # Commission
                    shares = alloc_cash / price
                    holdings[symbol] += shares
                    cash -= alloc_cash
                elif signal == -1 and holdings[symbol] > 0:
                    cash += holdings[symbol] * price * 0.95
                    holdings[symbol] = 0
                
                daily_value += holdings[symbol] * price
            
            self.portfolio_value.append(daily_value)
        
        portfolio_df = pd.DataFrame({
            'Portfolio': self.portfolio_value
        }, index=[dates[i] for i in range(min(len(dates), len(self.portfolio_value)))])
        portfolio_df['Returns'] = portfolio_df['Portfolio'].pct_change()
        
        metrics = self.calculate_metrics(portfolio_df)
        for k, v in metrics.items():
            print(f"{k}: {v:.2f}")
        
        return portfolio_df, metrics
    
    def plot(self, portfolio_df: pd.DataFrame, metrics: Dict):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        ax1.plot(portfolio_df['Portfolio'], linewidth=2)
        ax1.set_title('Enhanced HSI Trading Engine - SMA Crossover + Markowitz')
        ax1.set_ylabel('Portfolio Value (HKD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        rolling_max = portfolio_df['Portfolio'].expanding().max()
        drawdown = (portfolio_df['Portfolio'] - rolling_max) / rolling_max * 100
        ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def monitor_live(self):
        """Paper trading monitor"""
        if not self.is_live:
            print("Run fetch_data(live=True) first")
            return
        
        latest_prices = {s: self.data[s].iloc[-1] for s in self.symbols}
        latest_signals = {s: self.data[s]['Signal'].iloc[-1] for s in self.symbols}
        
        print("\n=== LIVE HSI MONITOR ===")
        for symbol in self.symbols:
            signal_text = "BUY" if latest_signals[symbol] == 1 else "SELL" if latest_signals[symbol] == -1 else "HOLD"
            print(f"{signal_text} {symbol}: ${latest_prices[symbol]:.2f}")

# Usage Examples
if __name__ == "__main__":
    # Backtest
    engine = EnhancedTradingEngine(initial_cash=100000, short_window=10, long_window=30)
    engine.fetch_data(period='1y')
    engine.generate_signals()
    portfolio, metrics = engine.backtest(use_markowitz=True)
    engine.plot(portfolio, metrics)
    
    # Live monitoring (uncomment to use)
    # engine_live = EnhancedTradingEngine(short_window=5, long_window=20)
    # engine_live.fetch_data(period='1y', live=True)
    # engine_live.generate_signals()
    # engine_live.monitor_live()  
