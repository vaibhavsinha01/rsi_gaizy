

import numpy as np
import pandas as pd
from typing import Tuple

class RangeFilter:
    """
    Range Filter Indicator - Python Implementation (Corrected)
    
    This is a technical analysis indicator that filters out minor price action 
    for a clearer view of trends. Originally created by DonovanWall for TradingView.
    
    The filter applies a volatility-based process directly to price movements,
    calculating target ranges that trigger filter movement.
    """
    
    def __init__(self):
        self.results = {}
        
    def conditional_ema(self, values: np.array, condition: np.array, period: int) -> np.array:
        """
        Conditional Sampling EMA - only calculates EMA when condition is True
        Fixed to maintain single EMA value across all bars like Pine Script
        """
        ema_val = np.nan  # Single EMA value that persists
        result = np.full_like(values, np.nan, dtype=float)
        
        for i in range(len(values)):
            if condition[i] and not np.isnan(values[i]):
                if np.isnan(ema_val):
                    ema_val = values[i]  # Initialize with first valid value
                else:
                    # Pine Script EMA formula: (x - ema) * (2/(n+1)) + ema
                    alpha = 2.0 / (period + 1)
                    ema_val = (values[i] - ema_val) * alpha + ema_val
            
            result[i] = ema_val
                
        return result
    
    def conditional_sma(self, values: np.array, condition: np.array, period: int) -> np.array:
        """
        Conditional Sampling SMA - only includes values when condition is True
        Fixed to maintain running array like Pine Script
        """
        result = np.full_like(values, np.nan, dtype=float)
        vals_array = []  # Maintains the rolling window
        
        for i in range(len(values)):
            if condition[i] and not np.isnan(values[i]):
                vals_array.append(values[i])
                if len(vals_array) > period:
                    vals_array.pop(0)  # Remove oldest value
            
            # Calculate average of current array
            if len(vals_array) > 0:
                result[i] = np.mean(vals_array)
                
        return result
    
    def standard_deviation(self, values: np.array, period: int) -> np.array:
        """
        Calculate standard deviation using conditional sampling
        Formula: sqrt(SMA(x^2, n) - SMA(x, n)^2)
        """
        condition = np.ones_like(values, dtype=bool)
        mean_sq = self.conditional_sma(values**2, condition, period)
        mean_val = self.conditional_sma(values, condition, period)
        
        # Calculate variance and handle potential negative values
        variance = mean_sq - mean_val**2
        std_dev = np.sqrt(np.maximum(variance, 0))
        return np.where(np.isnan(std_dev), 0, std_dev)
    
    def true_range(self, high: np.array, low: np.array, close: np.array) -> np.array:
        """
        Calculate True Range - fixed to match Pine Script tr(true) function
        """
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # Handle first bar
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    def calculate_range_size(self, 
                           values: np.array, 
                           high: np.array, 
                           low: np.array, 
                           close: np.array,
                           scale: str, 
                           quantity: float, 
                           period: int,
                           point_value: float = 1.0,
                           tick_size: float = 0.01) -> np.array:
        """
        Calculate range size based on different scaling methods
        Fixed to match Pine Script logic exactly
        """
        condition = np.ones_like(values, dtype=bool)
        
        if scale == "ATR":
            tr = self.true_range(high, low, close)
            atr = self.conditional_ema(tr, condition, period)
            return quantity * atr
            
        elif scale == "Average Change":
            # Calculate absolute change from previous bar
            changes = np.abs(values - np.roll(values, 1))
            changes[0] = 0  # First bar has no previous value
            ac = self.conditional_ema(changes, condition, period)
            return quantity * ac
            
        elif scale == "Standard Deviation":
            sd = self.standard_deviation(values, period)
            return quantity * sd
            
        elif scale == "% of Price":
            return close * quantity / 100
            
        elif scale == "Points":
            return np.full_like(values, quantity * point_value)
            
        elif scale == "Pips":
            return np.full_like(values, quantity * 0.0001)
            
        elif scale == "Ticks":
            return np.full_like(values, quantity * tick_size)
            
        else:  # Absolute
            return np.full_like(values, quantity)
    
    def range_filter(self,
                    high: np.array,
                    low: np.array,
                    range_size: np.array,
                    range_period: int,
                    filter_type: str = "Type 1",
                    smooth_range: bool = True,
                    smooth_period: int = 27,
                    average_filter: bool = True,
                    average_samples: int = 2) -> Tuple[np.array, np.array, np.array]:
        """
        Calculate the Range Filter with bands - Fixed to match Pine Script exactly
        """
        # Smooth range if requested
        if smooth_range:
            condition = np.ones_like(range_size, dtype=bool)
            r = self.conditional_ema(range_size, condition, smooth_period)
        else:
            r = range_size.copy()
        
        # Initialize filter array (rfilt) - matches Pine Script var array
        rfilt_current = (high[0] + low[0]) / 2  # Single persistent value
        rfilt_previous = rfilt_current
        
        rfilt1 = np.full_like(high, np.nan, dtype=float)
        rfilt1[0] = rfilt_current
        
        # Calculate filter based on type
        for i in range(1, len(high)):
            rfilt_previous = rfilt_current
            
            if filter_type == "Type 1":
                # Type 1 logic from Pine Script
                if high[i] - r[i] > rfilt_previous:
                    rfilt_current = high[i] - r[i]
                elif low[i] + r[i] < rfilt_previous:
                    rfilt_current = low[i] + r[i]
                # else rfilt_current remains unchanged
                    
            elif filter_type == "Type 2":
                # Type 2 logic from Pine Script
                if high[i] >= rfilt_previous + r[i]:
                    steps = np.floor(abs(high[i] - rfilt_previous) / r[i])
                    rfilt_current = rfilt_previous + steps * r[i]
                elif low[i] <= rfilt_previous - r[i]:
                    steps = np.floor(abs(low[i] - rfilt_previous) / r[i])
                    rfilt_current = rfilt_previous - steps * r[i]
                # else rfilt_current remains unchanged
            
            rfilt1[i] = rfilt_current
        
        # Calculate initial bands
        hi_band1 = rfilt1 + r
        lo_band1 = rfilt1 - r
        
        # Apply averaging if requested (matches Pine Script av_rf logic)
        if average_filter:
            # Find filter changes - condition for EMA calculation
            filter_changes = np.zeros_like(rfilt1, dtype=bool)
            filter_changes[1:] = rfilt1[1:] != rfilt1[:-1]
            filter_changes[0] = True  # Initialize first value
            
            # Apply conditional EMA only when filter changes
            rfilt2 = self.conditional_ema(rfilt1, filter_changes, average_samples)
            hi_band2 = self.conditional_ema(hi_band1, filter_changes, average_samples)
            lo_band2 = self.conditional_ema(lo_band1, filter_changes, average_samples)
            
            return hi_band2, lo_band2, rfilt2
        
        return hi_band1, lo_band1, rfilt1
    
    def calculate_signals(self, 
                         close: np.array, 
                         filter_line: np.array) -> Tuple[np.array, np.array, np.array]:
        """
        Calculate buy/sell signals based on filter - Fixed to match Pine Script exactly
        """
        # Filter direction calculation - matches Pine Script fdir logic
        fdir = np.zeros_like(filter_line)
        fdir_value = 0.0  # Persistent variable like Pine Script
        
        for i in range(1, len(filter_line)):
            if filter_line[i] > filter_line[i-1]:
                fdir_value = 1
            elif filter_line[i] < filter_line[i-1]:
                fdir_value = -1
            # else fdir_value remains unchanged
            fdir[i] = fdir_value
        
        upward = (fdir == 1).astype(int)
        downward = (fdir == -1).astype(int)
        
        # Trading conditions - exact match to Pine Script
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # Handle first bar
        
        long_cond = ((close > filter_line) & (close > prev_close) & (upward > 0)) | \
                   ((close > filter_line) & (close < prev_close) & (upward > 0))
        
        short_cond = ((close < filter_line) & (close < prev_close) & (downward > 0)) | \
                    ((close < filter_line) & (close > prev_close) & (downward > 0))
        
        # CondIni logic - matches Pine Script exactly
        cond_ini = np.zeros_like(close)
        cond_ini_value = 0  # Persistent variable
        
        for i in range(len(close)):
            if long_cond[i]:
                cond_ini_value = 1
            elif short_cond[i]:
                cond_ini_value = -1
            # else cond_ini_value remains unchanged
            cond_ini[i] = cond_ini_value
        
        # Generate buy/sell signals - exact Pine Script logic
        prev_cond_ini = np.roll(cond_ini, 1)
        prev_cond_ini[0] = 0  # Handle first bar
        
        buy_signals = long_cond & (prev_cond_ini == -1)
        sell_signals = short_cond & (prev_cond_ini == 1)
        
        return buy_signals, sell_signals, fdir
    
    def run_filter(self,
                   df: pd.DataFrame,
                   filter_type: str = "Type 1",
                   movement_source: str = "Close",
                   range_quantity: float = 2.618,
                   range_scale: str = "Average Change",
                   range_period: int = 14,
                   smooth_range: bool = True,
                   smooth_period: int = 27,
                   average_filter: bool = True,
                   average_samples: int = 2,
                   point_value: float = 1.0,
                   tick_size: float = 0.01) -> pd.DataFrame:
        """
        Main function to run the Range Filter on OHLC data
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data (columns: 'Open', 'High', 'Low', 'Close')
        filter_type : str
            "Type 1" or "Type 2"
        movement_source : str
            "Close" or "Wicks"
        range_quantity : float
            Range size multiplier
        range_scale : str
            Scale type for range calculation ("Points", "Pips", "Ticks", "% of Price", 
            "ATR", "Average Change", "Standard Deviation", "Absolute")
        range_period : int
            Period for range calculations
        smooth_range : bool
            Whether to smooth the range
        smooth_period : int
            Period for range smoothing
        average_filter : bool
            Whether to average filter changes
        average_samples : int
            Number of changes to average
        point_value : float
            Point value for Points scale
        tick_size : float
            Tick size for Ticks scale
            
        Returns:
        --------
        pd.DataFrame
            Original DataFrame with added Range Filter columns
        """
        
        # Validate input DataFrame
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Extract OHLC data
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        # Determine high/low values based on movement source
        if movement_source == "Wicks":
            h_val = high
            l_val = low
        else:  # Close
            h_val = close
            l_val = close
        
        # Calculate average price for range calculation
        avg_price = (h_val + l_val) / 2
        
        # Calculate range size
        range_size = self.calculate_range_size(
            avg_price, high, low, close, range_scale, 
            range_quantity, range_period, point_value, tick_size
        )
        
        # Calculate Range Filter
        hi_band, lo_band, filter_line = self.range_filter(
            h_val, l_val, range_size, range_period, filter_type,
            smooth_range, smooth_period, average_filter, average_samples
        )
        
        # Calculate signals
        buy_signals, sell_signals, trend_direction = self.calculate_signals(close, filter_line)
        
        # Add results to DataFrame
        result_df['RF_UpperBand'] = hi_band
        result_df['RF_LowerBand'] = lo_band
        result_df['RF_Filter'] = filter_line
        result_df['RF_Trend'] = trend_direction
        result_df['RF_BuySignal'] = buy_signals.astype(int)
        result_df['RF_SellSignal'] = sell_signals.astype(int)
        
        # Store results for potential plotting
        self.results = {
            'df': result_df,
            'filter_line': filter_line,
            'upper_band': hi_band,
            'lower_band': lo_band,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'trend_direction': trend_direction
        }
        
        return result_df

# Example usage:
"""
# Load your OHLC data into a pandas DataFrame
# df = pd.read_csv('your_data.csv')  # Should have columns: Open, High, Low, Close

# Initialize the Range Filter
rf = RangeFilter()

# Run the filter with default parameters (matching Pine Script defaults)
result = rf.run_filter(
    df,
    filter_type="Type 1",
    movement_source="Close", 
    range_quantity=2.618,
    range_scale="Average Change",
    range_period=14,
    smooth_range=True,
    smooth_period=27,
    average_filter=True,
    average_samples=2
)

# Access the results
print(result[['Close', 'RF_Filter', 'RF_BuySignal', 'RF_SellSignal']].tail())

# Plot the results (requires matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.plot(result.index, result['Close'], label='Close', alpha=0.7)
plt.plot(result.index, result['RF_Filter'], label='Range Filter', linewidth=2)
plt.fill_between(result.index, result['RF_LowerBand'], result['RF_UpperBand'], alpha=0.2)

# Mark buy/sell signals
buy_points = result[result['RF_BuySignal'] == 1]
sell_points = result[result['RF_SellSignal'] == 1]

plt.scatter(buy_points.index, buy_points['Close'], color='green', marker='^', s=100, label='Buy Signal')
plt.scatter(sell_points.index, sell_points['Close'], color='red', marker='v', s=100, label='Sell Signal')

plt.legend()
plt.title('Range Filter with Buy/Sell Signals')
plt.show()
"""