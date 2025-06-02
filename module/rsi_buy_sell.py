import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

class RSIBuySellIndicator:
    """
    RSI Buy/Sell Signal Indicator
    Identifies overbought and oversold conditions for trading signals
    
    Original Pine Script by: Duy Thanh Nguyen (Vietnam)
    Converted to Python
    """
    
    def __init__(self, rsi_length=14, rsi_upper=70, rsi_lower=30):
        """
        Initialize the RSI Buy/Sell indicator
        
        Parameters:
        rsi_length (int): Period for RSI calculation (default: 14)
        rsi_upper (int): RSI upper threshold for sell signals (default: 70)
        rsi_lower (int): RSI lower threshold for buy signals (default: 30)
        """
        self.rsi_length = rsi_length
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower
    
    def rma(self, series, length):
        """
        Calculate Running Moving Average (RMA) - equivalent to Pine Script's rma()
        This is an exponential moving average with alpha = 1/length
        """
        alpha = 1.0 / length
        return series.ewm(alpha=alpha, adjust=False).mean()
    
    def calculate_rsi(self, prices):
        """
        Calculate RSI using the same method as Pine Script
        
        Parameters:
        prices (pd.Series): Price series (typically close prices)
        
        Returns:
        pd.Series: RSI values
        """
        # Calculate price changes
        changes = prices.diff()
        
        # Separate gains and losses
        gains = np.maximum(changes, 0)
        losses = -np.minimum(changes, 0)
        
        # Calculate RMA of gains and losses
        avg_gains = self.rma(gains, self.rsi_length)
        avg_losses = self.rma(losses, self.rsi_length)
        
        # Calculate RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi = rsi.fillna(50)  # Fill NaN with neutral RSI
        
        return rsi
    
    def generate_signals(self, prices):
        """
        Generate buy/sell signals based on RSI crossovers
        
        Parameters:
        prices (pd.Series): Price series
        
        Returns:
        tuple: (rsi, buy_signals, sell_signals)
        """
        # Calculate RSI
        rsi = self.calculate_rsi(prices)
        
        # Generate signals based on Pine Script logic
        # Buy signal: RSI was below lower threshold and now crosses above it
        buy_signals = (rsi.shift(1) < self.rsi_lower) & (rsi >= self.rsi_lower)
        
        # Sell signal: RSI was above upper threshold and now crosses below it
        sell_signals = (rsi.shift(1) > self.rsi_upper) & (rsi <= self.rsi_upper)
        
        return rsi, buy_signals, sell_signals
    
    def analyze_data(self, data):
        """
        Analyze price data and return complete results
        
        Parameters:
        data (pd.DataFrame): DataFrame with 'Close' column
        
        Returns:
        pd.DataFrame: DataFrame with RSI and signals
        """
        # Ensure we have the right column name
        if 'Close' not in data.columns and 'close' in data.columns:
            data = data.rename(columns={'close': 'Close'})
        
        # Calculate RSI and signals
        rsi, buy_signals, sell_signals = self.generate_signals(data['Close'])
        
        # Create results DataFrame
        results = data.copy()
        results['RSI'] = rsi
        results['RSI_Buy_Signal'] = buy_signals
        results['RSI_Sell_Signal'] = sell_signals
        
        return results
    
    def plot_analysis(self, results, title="RSI Buy/Sell Analysis"):
        """
        Plot price chart with RSI and buy/sell signals
        
        Parameters:
        results (pd.DataFrame): Results from analyze_data()
        title (str): Chart title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot price and signals
        ax1.plot(results.index, results['Close'], label='Close Price', linewidth=1)
        
        # Plot buy signals
        buy_points = results[results['RSI_Buy_Signal']]
        if not buy_points.empty:
            ax1.scatter(buy_points.index, buy_points['Close'], 
                       color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        
        # Plot sell signals
        sell_points = results[results['RSI_Sell_Signal']]
        if not sell_points.empty:
            ax1.scatter(sell_points.index, sell_points['Close'], 
                       color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title(f'{title} - Price Chart')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot RSI
        ax2.plot(results.index, results['RSI'], label='RSI', color='purple')
        ax2.axhline(y=self.rsi_upper, color='r', linestyle='--', alpha=0.7, label=f'Overbought ({self.rsi_upper})')
        ax2.axhline(y=self.rsi_lower, color='g', linestyle='--', alpha=0.7, label=f'Oversold ({self.rsi_lower})')
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5, label='Neutral (50)')
        
        # Highlight RSI signal areas
        ax2.fill_between(results.index, self.rsi_upper, 100, alpha=0.2, color='red', label='Overbought Zone')
        ax2.fill_between(results.index, 0, self.rsi_lower, alpha=0.2, color='green', label='Oversold Zone')
        
        ax2.set_title('RSI Indicator')
        ax2.set_ylabel('RSI')
        ax2.set_xlabel('Date')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_signal_summary(self, results):
        """
        Get summary of buy/sell signals
        
        Parameters:
        results (pd.DataFrame): Results from analyze_data()
        
        Returns:
        dict: Summary statistics
        """
        buy_count = results['RSI_Buy_Signal'].sum()
        sell_count = results['RSI_Sell_Signal'].sum()
        
        buy_dates = results[results['RSI_Buy_Signal']].index.tolist()
        sell_dates = results[results['RSI_Sell_Signal']].index.tolist()
        
        current_rsi = results['RSI'].iloc[-1]
        
        summary = {
            'total_buy_signals': buy_count,
            'total_sell_signals': sell_count,
            'current_rsi': round(current_rsi, 2),
            'latest_buy_dates': buy_dates[-5:] if buy_dates else [],
            'latest_sell_dates': sell_dates[-5:] if sell_dates else [],
            'rsi_parameters': {
                'length': self.rsi_length,
                'upper_threshold': self.rsi_upper,
                'lower_threshold': self.rsi_lower
            }
        }
        
        return summary


# Example Usage
def example_usage():
    """
    Example of how to use the RSI Buy/Sell Indicator
    """
    print("RSI Buy/Sell Signal Indicator - Example Usage")
    print("=" * 50)
    
    # Initialize the indicator
    rsi_indicator = RSIBuySellIndicator(rsi_length=14, rsi_upper=70, rsi_lower=30)
    
    # Download sample data (Apple stock for last 6 months)
    print("Downloading sample data (AAPL)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    try:
        # Download data using yfinance
        data = yf.download('AAPL', start=start_date, end=end_date)
        print(f"Downloaded {len(data)} days of data")
        
        # Analyze the data
        print("Analyzing data...")
        results = rsi_indicator.analyze_data(data)
        
        # Get signal summary
        summary = rsi_indicator.get_signal_summary(results)
        
        # Print summary
        print("\nSignal Summary:")
        print(f"Total Buy Signals: {summary['total_buy_signals']}")
        print(f"Total Sell Signals: {summary['total_sell_signals']}")
        print(f"Current RSI: {summary['current_rsi']}")
        print(f"RSI Length: {summary['rsi_parameters']['length']}")
        print(f"Upper Threshold: {summary['rsi_parameters']['upper_threshold']}")
        print(f"Lower Threshold: {summary['rsi_parameters']['lower_threshold']}")
        
        if summary['latest_buy_dates']:
            print(f"\nLatest Buy Signals:")
            for date in summary['latest_buy_dates']:
                print(f"  - {date.strftime('%Y-%m-%d')}")
        
        if summary['latest_sell_dates']:
            print(f"\nLatest Sell Signals:")
            for date in summary['latest_sell_dates']:
                print(f"  - {date.strftime('%Y-%m-%d')}")
        
        # Plot the analysis
        print("\nGenerating chart...")
        rsi_indicator.plot_analysis(results, "AAPL - RSI Buy/Sell Analysis")
        
        # Show recent signals with prices
        print("\nRecent Signals with Prices:")
        recent_signals = results[results['RSI_Buy_Signal'] | results['RSI_Sell_Signal']].tail(10)
        for idx, row in recent_signals.iterrows():
            signal_type = "BUY" if row['RSI_Buy_Signal'] else "SELL"
            print(f"{idx.strftime('%Y-%m-%d')}: {signal_type} at ${row['Close']:.2f} (RSI: {row['RSI']:.2f})")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please install yfinance: pip install yfinance")
        
        # Create sample data for demonstration
        print("\nCreating sample data for demonstration...")
        dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        sample_data = pd.DataFrame({'Close': prices}, index=dates)
        
        results = rsi_indicator.analyze_data(sample_data)
        summary = rsi_indicator.get_signal_summary(results)
        
        print(f"Sample Analysis - Buy Signals: {summary['total_buy_signals']}, Sell Signals: {summary['total_sell_signals']}")
        rsi_indicator.plot_analysis(results, "Sample Data - RSI Analysis")


if __name__ == "__main__":
    example_usage()