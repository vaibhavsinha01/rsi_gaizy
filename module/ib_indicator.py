import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from module.ib_box import calculate_inside_bar_boxes, plot_inside_bar_boxes
from module.ib_box import *

def calculate_heikin_ashi(df):
    """
    Calculate Heikin-Ashi candles from regular OHLC data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'open', 'high', 'low', 'close' columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with Heikin-Ashi candles
    """
    ha_df = df.copy()
    
    # Ensure column names are lowercase
    if not all(col in ha_df.columns for col in ['open', 'high', 'low', 'close']):
        # Try to convert from capitalized column names
        ha_df.rename(columns={
            'Open': 'open', 
            'High': 'high', 
            'Low': 'low', 
            'Close': 'close'
        }, inplace=True, errors='ignore')
    
    # Calculate Heikin-Ashi candles
    ha_df['ha_close'] = (ha_df['open'] + ha_df['high'] + ha_df['low'] + ha_df['close']) / 4
    
    # Initialize first Heikin-Ashi candle
    ha_df.loc[ha_df.index[0], 'ha_open'] = (ha_df.loc[ha_df.index[0], 'open'] + 
                                           ha_df.loc[ha_df.index[0], 'close']) / 2
    
    # Calculate remaining Heikin-Ashi candles
    for i in range(1, len(ha_df)):
        ha_df.loc[ha_df.index[i], 'ha_open'] = (ha_df.loc[ha_df.index[i-1], 'ha_open'] + 
                                               ha_df.loc[ha_df.index[i-1], 'ha_close']) / 2
    
    ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
    ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)
    
    # Replace original OHLC with Heikin-Ashi values
    ha_df['open'] = ha_df['ha_open']
    ha_df['high'] = ha_df['ha_high']
    ha_df['low'] = ha_df['ha_low']
    ha_df['close'] = ha_df['ha_close']
    
    # Drop temporary columns
    ha_df.drop(['ha_open', 'ha_high', 'ha_low', 'ha_close'], axis=1, inplace=True)
    
    return ha_df

def apply_inside_bar_boxes(df, high_low_buffer=0, bar_highlight=True, 
                          show_only_last_box=False, show_break=True, 
                          bg_color='blue', bg_transparency=90, inside_bars_color='orange', 
                          mintick=0.05, plot=False, use_heikin_ashi=True):
    """
    Apply Inside Bar Boxes indicator to OHLC data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'open', 'high', 'low', 'close' columns
    high_low_buffer : float
        Buffer zone around high-low (in mintick units)
    bar_highlight : bool
        Whether to highlight inside bars
    show_only_last_box : bool
        Show only the last box
    show_break : bool
        Show box breakouts
    bg_color : str
        Background color for boxes
    bg_transparency : int
        Box transparency (0-100)
    inside_bars_color : str
        Color for inside bars
    mintick : float
        Minimum tick size
    plot : bool
        Whether to plot the result
    use_heikin_ashi : bool
        Whether to use Heikin-Ashi candles
        
    Returns:
    --------
    tuple
        (box_high Series, box_low Series)
    """
    # Ensure column names are lowercase
    df = df.copy()
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # Try to convert from capitalized column names
        df.rename(columns={
            'Open': 'open', 
            'High': 'high', 
            'Low': 'low', 
            'Close': 'close'
        }, inplace=True, errors='ignore')
    
    # Convert to Heikin-Ashi if requested
    if use_heikin_ashi:
        df = calculate_heikin_ashi(df)
    
    # Calculate inside bar boxes
    result_df, boxes = calculate_inside_bar_boxes(
        df, 
        high_low_buffer=high_low_buffer,
        mintick=mintick,
        bar_highlight=bar_highlight,
        show_only_last_box=show_only_last_box,
        show_break=show_break,
        bg_color=bg_color,
        bg_transparency=bg_transparency,
        inside_bars_color=inside_bars_color
    )
    
    # Plot if requested
    if plot:
        plot_inside_bar_boxes(result_df)
    
    # Extract box_high and box_low as separate Series
    box_high = result_df['BoxHigh']
    box_low = result_df['BoxLow']
    
    # Return only box_high and box_low
    return box_high, box_low

def generate_synthetic_ohlc_data(n_periods=120, base_price=150.0, seed=42):
    """
    Generate synthetic OHLC data using numpy random with seed for reproducibility.
    
    Parameters:
    -----------
    n_periods : int
        Number of periods to generate
    base_price : float
        Starting price level
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with OHLC data
    """
    np.random.seed(seed)
    
    # Generate random walk for closing prices
    returns = np.random.normal(0, 0.02, n_periods)  # 2% daily volatility
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])  # Remove the initial base_price
    
    # Generate OHLC data
    ohlc_data = []
    
    for i in range(n_periods):
        if i == 0:
            open_price = base_price
        else:
            open_price = ohlc_data[i-1]['close']
        
        close_price = prices[i]
        
        # Generate high and low with some randomness
        daily_range = abs(close_price - open_price) + np.random.uniform(0.5, 2.5)
        high_price = max(open_price, close_price) + np.random.uniform(0, daily_range * 0.3)
        low_price = min(open_price, close_price) - np.random.uniform(0, daily_range * 0.3)
        
        ohlc_data.append({
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2)
        })
    
    # Create DataFrame with date index
    dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='D')
    df = pd.DataFrame(ohlc_data, index=dates)
    
    return df

# Example usage
if __name__ == "__main__":
    # Generate synthetic OHLC data
    np.random.seed(42)  # Set seed for reproducible results
    data = generate_synthetic_ohlc_data(n_periods=120, base_price=150.0, seed=42)
    
    print("Generated synthetic OHLC data:")
    print(data.head(10))
    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Apply indicator
    box_high, box_low = apply_inside_bar_boxes(
        data,
        high_low_buffer=0,
        bar_highlight=True,
        show_only_last_box=False,
        show_break=True,
        plot=True,
        use_heikin_ashi=True
    )
    
    # Display the first few rows of box_high and box_low
    print("\nBox High values:")
    print(box_high[box_high.notna()].head())
    
    print("\nBox Low values:")
    print(box_low[box_low.notna()].head())