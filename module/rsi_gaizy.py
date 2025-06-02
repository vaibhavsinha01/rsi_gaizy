import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class RSIGainzy:
    """
    RSI Gainzy Strategy - Python Implementation
    Converted from Pine Script RSI Integration Tool
    
    This indicator uses RSI with pivot point trend lines to generate trading signals.
    
    Signal Colors/Trends:
    - STRONG_BULL (Bright Pink): trend = -3
    - STRONG_BEAR (Bright Green): trend = 3  
    - WEAK_BULL (Green): trend = 1
    - WEAK_BEAR (Red): trend = -1
    - NEUTRAL (Blue): trend = 2
    - DEFAULT (Black): trend = 0
    """
    
    def __init__(self):
        self.results = {}
        # Color definitions matching Pine Script logic
        self.colors = {
            'STRONG_BULL': '#F85BE3',   # Bright Pink (trend = -3)
            'STRONG_BEAR': '#3EFF45',   # Bright Green (trend = 3)
            'WEAK_BULL': '#00C853',     # Green (trend = 1)
            'WEAK_BEAR': '#FF3D00',     # Red (trend = -1)
            'NEUTRAL': '#2196F3',       # Blue (trend = 2)
            'DEFAULT': '#000000'        # Black (trend = 0)
        }
    
    def calculate_rsi(self, close_prices: np.array, period: int = 14) -> np.array:
        """Calculate RSI (Relative Strength Index)"""
        deltas = np.diff(close_prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Initialize RSI array
        rsi = np.full_like(close_prices, np.nan, dtype=float)
        
        # Calculate RSI values
        for i in range(period, len(close_prices)):
            if i == period:
                current_avg_gain = avg_gain
                current_avg_loss = avg_loss
            else:
                current_avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
                current_avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
                avg_gain = current_avg_gain
                avg_loss = current_avg_loss
            
            if current_avg_loss == 0:
                rsi[i] = 100
            else:
                rs = current_avg_gain / current_avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def find_pivot_highs(self, values: np.array, lookback: int, lookforward: int) -> np.array:
        """Find pivot highs in the array"""
        pivots = np.full_like(values, np.nan)
        
        for i in range(lookback, len(values) - lookforward):
            if np.isnan(values[i]):
                continue
            
            is_pivot = True
            # Check lookback period
            for j in range(i - lookback, i):
                if not np.isnan(values[j]) and values[j] >= values[i]:
                    is_pivot = False
                    break
            
            # Check lookforward period
            if is_pivot:
                for j in range(i + 1, min(i + lookforward + 1, len(values))):
                    if not np.isnan(values[j]) and values[j] >= values[i]:
                        is_pivot = False
                        break
            
            if is_pivot:
                pivots[i] = values[i]
        
        return pivots
    
    def find_pivot_lows(self, values: np.array, lookback: int, lookforward: int) -> np.array:
        """Find pivot lows in the array"""
        pivots = np.full_like(values, np.nan)
        
        for i in range(lookback, len(values) - lookforward):
            if np.isnan(values[i]):
                continue
            
            is_pivot = True
            # Check lookback period
            for j in range(i - lookback, i):
                if not np.isnan(values[j]) and values[j] <= values[i]:
                    is_pivot = False
                    break
            
            # Check lookforward period
            if is_pivot:
                for j in range(i + 1, min(i + lookforward + 1, len(values))):
                    if not np.isnan(values[j]) and values[j] <= values[i]:
                        is_pivot = False
                        break
            
            if is_pivot:
                pivots[i] = values[i]
        
        return pivots
    
    def calculate_trend_line_value(self, x1: int, y1: float, x2: int, y2: float, x: int) -> float:
        """Calculate trend line value at position x"""
        if x2 == x1:
            return y1
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    def calculate_signals(self,
                         df: pd.DataFrame,
                         rsi_length: int = 14,
                         pivot_length: int = 10,
                         overbought_level: int = 70,
                         oversold_level: int = 30) -> pd.DataFrame:
        """
        Calculate RSI Gainzy signals based on Pine Script logic
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data (must contain 'Close' column)
        rsi_length : int
            RSI calculation period
        pivot_length : int
            Pivot lookback/forward period
        overbought_level : int
            Overbought threshold
        oversold_level : int
            Oversold threshold
            
        Returns:
        --------
        pd.DataFrame
            Original DataFrame with added RSI Gainzy columns
        """
        
        # Validate input DataFrame
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
        
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Extract close prices
        close_prices = df['Close'].values
        
        # Calculate RSI
        rsi = self.calculate_rsi(close_prices, rsi_length)
        
        # Find pivot points
        pivot_highs = self.find_pivot_highs(rsi, pivot_length, pivot_length)
        pivot_lows = self.find_pivot_lows(rsi, pivot_length, pivot_length)
        
        # Initialize arrays for pivot tracking
        n = len(rsi)
        prev_high_val = np.full(n, np.nan)
        last_high_val = np.full(n, np.nan)
        prev_high_bar = np.full(n, np.nan)
        last_high_bar = np.full(n, np.nan)
        
        prev_low_val = np.full(n, np.nan)
        last_low_val = np.full(n, np.nan)
        prev_low_bar = np.full(n, np.nan)
        last_low_bar = np.full(n, np.nan)
        
        # Track pivot points
        for i in range(n):
            if i > 0:
                prev_high_val[i] = prev_high_val[i-1]
                last_high_val[i] = last_high_val[i-1]
                prev_high_bar[i] = prev_high_bar[i-1]
                last_high_bar[i] = last_high_bar[i-1]
                
                prev_low_val[i] = prev_low_val[i-1]
                last_low_val[i] = last_low_val[i-1]
                prev_low_bar[i] = prev_low_bar[i-1]
                last_low_bar[i] = last_low_bar[i-1]
            
            # Update pivot highs
            if not np.isnan(pivot_highs[i]):
                prev_high_val[i] = last_high_val[i-1] if i > 0 else np.nan
                prev_high_bar[i] = last_high_bar[i-1] if i > 0 else np.nan
                last_high_val[i] = pivot_highs[i]
                last_high_bar[i] = i
            
            # Update pivot lows
            if not np.isnan(pivot_lows[i]):
                prev_low_val[i] = last_low_val[i-1] if i > 0 else np.nan
                prev_low_bar[i] = last_low_bar[i-1] if i > 0 else np.nan
                last_low_val[i] = pivot_lows[i]
                last_low_bar[i] = i
        
        # Calculate current trend line points
        current_high_line_point = np.full(n, np.nan)
        current_low_line_point = np.full(n, np.nan)
        
        for i in range(n):
            if not np.isnan(prev_high_val[i]) and not np.isnan(last_high_val[i]):
                current_high_line_point[i] = self.calculate_trend_line_value(
                    prev_high_bar[i], prev_high_val[i], 
                    last_high_bar[i], last_high_val[i], i
                )
            
            if not np.isnan(prev_low_val[i]) and not np.isnan(last_low_val[i]):
                current_low_line_point[i] = self.calculate_trend_line_value(
                    prev_low_bar[i], prev_low_val[i], 
                    last_low_bar[i], last_low_val[i], i
                )
        
        # Calculate trend based on Pine Script logic
        trend = np.zeros(n, dtype=int)
        
        for i in range(1, n):
            trend[i] = trend[i-1]  # Carry forward previous trend
            
            # Get current values
            curr_rsi = rsi[i]
            curr_high_line = current_high_line_point[i]
            curr_low_line = current_low_line_point[i]
            prev_high_line = current_high_line_point[i-1] if i > 0 else np.nan
            prev_low_line = current_low_line_point[i-1] if i > 0 else np.nan
            
            # Skip if values are NaN
            if np.isnan(curr_rsi) or np.isnan(curr_high_line) or np.isnan(curr_low_line):
                continue
            
            # Bullish conditions
            if (curr_rsi > curr_high_line and curr_high_line > curr_low_line and 
                np.isnan(pivot_highs[i]) and curr_high_line > prev_high_line and trend[i] <= 0):
                trend[i] = 3
            elif (curr_rsi > curr_low_line and curr_high_line < curr_low_line and 
                  np.isnan(pivot_highs[i]) and curr_low_line > prev_low_line and trend[i] <= 0):
                trend[i] = 3
            elif (curr_rsi > curr_high_line and curr_high_line > curr_low_line and 
                  np.isnan(pivot_highs[i]) and trend[i] < 3):
                trend[i] = 1
            elif (curr_rsi > curr_low_line and curr_high_line < curr_low_line and 
                  np.isnan(pivot_highs[i]) and trend[i] < 3):
                trend[i] = 1
            elif (curr_rsi > curr_high_line and curr_rsi < curr_low_line and 
                  curr_high_line < curr_low_line):
                trend[i] = 2
            
            # Reset bullish trend
            if curr_rsi < curr_high_line and trend[i] > 0:
                trend[i] = 0
            
            # Bearish conditions
            if (curr_rsi < curr_low_line and curr_high_line > curr_low_line and 
                np.isnan(pivot_lows[i]) and curr_low_line < prev_low_line and 
                trend[i] >= 0 and trend[i] != 2):
                trend[i] = -3
            elif (curr_rsi < curr_high_line and curr_high_line < curr_low_line and 
                  np.isnan(pivot_lows[i]) and curr_low_line < prev_low_line and 
                  trend[i] >= 0 and trend[i] != 2):
                trend[i] = -3
            elif (curr_rsi < curr_low_line and curr_high_line > curr_low_line and 
                  np.isnan(pivot_lows[i]) and trend[i] > -3):
                trend[i] = -1
            elif (curr_rsi < curr_high_line and curr_high_line < curr_low_line and 
                  np.isnan(pivot_lows[i]) and trend[i] > -3):
                trend[i] = -1
            elif (curr_rsi < curr_low_line and curr_rsi > curr_high_line and 
                  curr_high_line < curr_low_line):
                trend[i] = 2
            
            # Reset bearish trend
            if curr_rsi > curr_low_line and trend[i] < 0:
                trend[i] = 0
        
        # Map trend values to signal colors
        signal_colors = np.full(n, 'DEFAULT', dtype=object)
        for i in range(n):
            if trend[i] == -3:
                signal_colors[i] = 'pink' # 'STRONG_BULL'
            elif trend[i] == 3:
                signal_colors[i] = 'bright_green' # 'STRONG_BEAR'
            elif trend[i] == 1:
                signal_colors[i] = 'dark_green' # 'WEAK_BULL'
            elif trend[i] == -1:
                signal_colors[i] = 'red' # 'WEAK_BEAR'
            elif trend[i] == 2:
                signal_colors[i] = 'blue' # 'NEUTRAL'
            else:
                signal_colors[i] = 'black' # 'DEFAULT'
        
        # Generate trading signals
        long_signals = (trend == -3) | (trend == 1)  # Strong bull or weak bull
        short_signals = (trend == 3) | (trend == -1)  # Strong bear or weak bear
        
        # Add results to DataFrame
        result_df['RSI'] = rsi
        result_df['RSI_Smoothed'] = rsi  # Keep same for compatibility
        result_df['RSI_Momentum'] = trend  # Use trend instead of momentum
        result_df['Signal_Color'] = signal_colors
        result_df['Long_Signal'] = long_signals.astype(int)
        result_df['Short_Signal'] = short_signals.astype(int)
        
        # Additional columns for analysis
        result_df['Pivot_Highs'] = pivot_highs
        result_df['Pivot_Lows'] = pivot_lows
        result_df['High_Line'] = current_high_line_point
        result_df['Low_Line'] = current_low_line_point
        result_df['Trend'] = trend
        
        # Store results for plotting
        self.results = {
            'df': result_df,
            'rsi': rsi,
            'smoothed_rsi': rsi,
            'momentum': trend,
            'signal_colors': signal_colors,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'overbought_level': overbought_level,
            'oversold_level': oversold_level,
            'pivot_highs': pivot_highs,
            'pivot_lows': pivot_lows,
            'high_line': current_high_line_point,
            'low_line': current_low_line_point,
            'trend': trend
        }
        
        # return result_df
        return result_df['Signal_Color']
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 12)):
        """Plot the RSI Gainzy results"""
        if not self.results:
            raise ValueError("No results to plot. Run calculate_signals() first.")
        
        df = self.results['df']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
        
        # Main price chart with colored background
        ax1.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1, alpha=0.7)
        
        # Color the background based on signal colors
        for i in range(len(df)-1):
            color = self.colors[self.results['signal_colors'][i]]
            ax1.axvspan(df.index[i], df.index[i+1], alpha=0.1, color=color)
        
        # Plot signals
        long_mask = self.results['long_signals']
        short_mask = self.results['short_signals']
        
        if np.any(long_mask):
            ax1.scatter(df.index[long_mask], df['Close'][long_mask], 
                       color='green', marker='^', s=100, 
                       label='Long Signal', zorder=5, edgecolors='white', linewidth=1)
        
        if np.any(short_mask):
            ax1.scatter(df.index[short_mask], df['Close'][short_mask], 
                       color='red', marker='v', s=100, 
                       label='Short Signal', zorder=5, edgecolors='white', linewidth=1)
        
        ax1.set_title('RSI Gainzy Strategy - Price Chart')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # RSI subplot with trend lines
        # Plot RSI with color coding based on trend
        for i in range(len(df)-1):
            color = self.colors[self.results['signal_colors'][i]]
            if not np.isnan(self.results['rsi'][i]) and not np.isnan(self.results['rsi'][i+1]):
                ax2.plot([df.index[i], df.index[i+1]], 
                        [self.results['rsi'][i], self.results['rsi'][i+1]], 
                        color=color, linewidth=3, alpha=0.8)
        
        # Plot trend lines
        high_line = self.results['high_line']
        low_line = self.results['low_line']
        
        valid_high = ~np.isnan(high_line)
        valid_low = ~np.isnan(low_line)
        
        if np.any(valid_high):
            ax2.plot(df.index[valid_high], high_line[valid_high], 
                    color='red', linestyle='--', alpha=0.7, label='High Trend Line')
        
        if np.any(valid_low):
            ax2.plot(df.index[valid_low], low_line[valid_low], 
                    color='green', linestyle='--', alpha=0.7, label='Low Trend Line')
        
        # Plot pivot points
        pivot_high_mask = ~np.isnan(self.results['pivot_highs'])
        pivot_low_mask = ~np.isnan(self.results['pivot_lows'])
        
        if np.any(pivot_high_mask):
            ax2.scatter(df.index[pivot_high_mask], self.results['pivot_highs'][pivot_high_mask], 
                       color='red', marker='v', s=50, alpha=0.8, label='Pivot Highs')
        
        if np.any(pivot_low_mask):
            ax2.scatter(df.index[pivot_low_mask], self.results['pivot_lows'][pivot_low_mask], 
                       color='green', marker='^', s=50, alpha=0.8, label='Pivot Lows')
        
        # Add overbought/oversold levels
        ax2.axhline(y=self.results['overbought_level'], color='red', linestyle=':', alpha=0.5)
        ax2.axhline(y=self.results['oversold_level'], color='green', linestyle=':', alpha=0.5)
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        
        ax2.set_title('RSI with Pivot Trend Lines')
        ax2.set_ylabel('RSI Value')
        ax2.set_xlabel('Time')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.show()
    
    def get_signal_summary(self) -> Dict:
        """Get a summary of the signals generated"""
        if not self.results:
            raise ValueError("No results available. Run calculate_signals() first.")
        
        df = self.results['df']
        
        # Count signals
        total_long_signals = df['Long_Signal'].sum()
        total_short_signals = df['Short_Signal'].sum()
        
        # Count color distribution
        color_counts = {}
        for color in self.colors.keys():
            color_counts[color] = np.sum(self.results['signal_colors'] == color)
        
        # Calculate percentage of each color
        total_bars = len(df)
        color_percentages = {color: (count/total_bars)*100 for color, count in color_counts.items()}
        
        # Count trend distribution
        trend_counts = {}
        for trend_val in [-3, -1, 0, 1, 2, 3]:
            trend_counts[f'Trend_{trend_val}'] = np.sum(self.results['trend'] == trend_val)
        
        summary = {
            'total_bars': total_bars,
            'long_signals': total_long_signals,
            'short_signals': total_short_signals,
            'total_signals': total_long_signals + total_short_signals,
            'color_counts': color_counts,
            'color_percentages': color_percentages,
            'trend_counts': trend_counts,
            'signal_frequency': ((total_long_signals + total_short_signals) / total_bars) * 100
        }
        
        return summary


# Example usage function
def example_usage():
    """Example of how to use the updated RSI Gainzy indicator"""
    # Create sample data
    np.random.seed(42)
    n_periods = 500
    
    # Generate synthetic price data with trend
    price = 100
    data = []
    trend = 0
    
    for i in range(n_periods):
        # Add some trending behavior
        if i % 50 == 0:
            trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        
        change = np.random.normal(trend * 0.1, 1)
        price += change
        
        # Create OHLC from the price
        high = price + abs(np.random.normal(0, 0.5))
        low = price - abs(np.random.normal(0, 0.5))
        open_price = price + np.random.normal(0, 0.2)
        close = price
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close
        })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=n_periods, freq='D')
    
    # Initialize RSI Gainzy
    rsi_gainzy = RSIGainzy()
    
    # Calculate signals with custom parameters
    result = rsi_gainzy.calculate_signals(
        df,
        rsi_length=14,
        pivot_length=10,
        overbought_level=70,
        oversold_level=30
    )
    
    # Display results
    print("RSI Gainzy Results:")
    print(result[['Close', 'RSI', 'Trend', 'Signal_Color', 
                  'Long_Signal', 'Short_Signal']].tail(10))
    
    # Get signal summary
    summary = rsi_gainzy.get_signal_summary()
    print(f"\n=== SIGNAL SUMMARY ===")
    print(f"Total Bars: {summary['total_bars']}")
    print(f"Long Signals: {summary['long_signals']}")
    print(f"Short Signals: {summary['short_signals']}")
    print(f"Signal Frequency: {summary['signal_frequency']:.2f}%")
    print(f"\nColor Distribution:")
    for color, percentage in summary['color_percentages'].items():
        print(f"  {color}: {percentage:.1f}%")
    print(f"\nTrend Distribution:")
    for trend, count in summary['trend_counts'].items():
        print(f"  {trend}: {count}")
    
    # Plot results
    rsi_gainzy.plot_results(figsize=(15, 12))
    
    return result

if __name__ == "__main__":
    example_usage()