import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore')

class RangeFilter:
    """Range Filter Implementation - Simplified version for backtesting"""
    
    def __init__(self):
        self.results = {}
        
    def conditional_ema(self, values, condition, period):
        ema_val = np.nan
        result = np.full_like(values, np.nan, dtype=float)
        
        for i in range(len(values)):
            if condition[i] and not np.isnan(values[i]):
                if np.isnan(ema_val):
                    ema_val = values[i]
                else:
                    alpha = 2.0 / (period + 1)
                    ema_val = (values[i] - ema_val) * alpha + ema_val
            result[i] = ema_val
        return result
    
    def true_range(self, high, low, close):
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    def run_filter(self, df, **params):
        """Simplified Range Filter calculation"""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        # Calculate ATR for range size
        tr = self.true_range(high, low, close)
        condition = np.ones_like(tr, dtype=bool)
        atr = self.conditional_ema(tr, condition, params.get('range_period', 14))
        range_size = params.get('range_quantity', 2.618) * atr
        
        # Initialize filter
        rfilt = np.full_like(close, np.nan)
        rfilt[0] = (high[0] + low[0]) / 2
        
        for i in range(1, len(close)):
            if high[i] - range_size[i] > rfilt[i-1]:
                rfilt[i] = high[i] - range_size[i]
            elif low[i] + range_size[i] < rfilt[i-1]:
                rfilt[i] = low[i] + range_size[i]
            else:
                rfilt[i] = rfilt[i-1]
        
        # Calculate signals
        fdir = np.zeros_like(rfilt)
        for i in range(1, len(rfilt)):
            if rfilt[i] > rfilt[i-1]:
                fdir[i] = 1
            elif rfilt[i] < rfilt[i-1]:
                fdir[i] = -1
            else:
                fdir[i] = fdir[i-1]
        
        # Generate buy/sell signals
        long_cond = (close > rfilt) & (fdir == 1)
        short_cond = (close < rfilt) & (fdir == -1)
        
        # Signal transitions - Fixed version
        buy_signals = long_cond & ~np.roll(long_cond, 1)
        buy_signals[0] = long_cond[0]

        sell_signals = short_cond & ~np.roll(short_cond, 1)
        sell_signals[0] = short_cond[0]
        
        result_df = df.copy()
        result_df['RF_Filter'] = rfilt
        result_df['RF_BuySignal'] = buy_signals.astype(int)
        result_df['RF_SellSignal'] = sell_signals.astype(int)
        result_df['RF_Trend'] = fdir
        
        return result_df

def calculate_inside_ib_box(df, high_low_buffer=0.0, mintick=0.05):
    """Simplified IB Box calculation"""
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    df['IsIB'] = False
    df['BoxHigh'] = np.nan
    df['BoxLow'] = np.nan
    df['GreenArrow'] = False
    df['RedArrow'] = False
    
    box_high = np.nan
    box_low = np.nan
    
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        prev_is_ib = df.iloc[i-1]['IsIB'] if i > 1 else False
        
        hp = prev['high'] + high_low_buffer * mintick
        lp = prev['low'] - high_low_buffer * mintick
        
        is_ib = (curr['close'] <= hp and curr['close'] >= lp and 
                 curr['open'] <= hp and curr['open'] >= lp)
        
        df.iat[i, df.columns.get_loc('IsIB')] = is_ib
        
        if is_ib and not prev_is_ib:
            box_high = prev['high']
            box_low = prev['low']
            df.iat[i, df.columns.get_loc('BoxHigh')] = box_high
            df.iat[i, df.columns.get_loc('BoxLow')] = box_low
        elif is_ib and prev_is_ib:
            if not np.isnan(box_high):
                df.iat[i, df.columns.get_loc('BoxHigh')] = box_high
                df.iat[i, df.columns.get_loc('BoxLow')] = box_low
        elif prev_is_ib and not is_ib:
            if not np.isnan(box_high):
                if prev['close'] <= box_high and curr['close'] > box_high:
                    df.iat[i, df.columns.get_loc('GreenArrow')] = True
                elif prev['close'] >= box_low and curr['close'] < box_low:
                    df.iat[i, df.columns.get_loc('RedArrow')] = True
    
    return df

def calculate_heiken_ashi(df):
    """Calculate Heiken-Ashi candlesticks"""
    ha_df = df.copy()
    
    # Initialize first HA candle
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_df['HA_Open'] = (df['Open'] + df['Close']) / 2
    ha_df['HA_High'] = df['High']
    ha_df['HA_Low'] = df['Low']
    
    # Calculate subsequent HA candles
    for i in range(1, len(ha_df)):
        ha_df.iloc[i, ha_df.columns.get_loc('HA_Open')] = (
            ha_df.iloc[i-1]['HA_Open'] + ha_df.iloc[i-1]['HA_Close']
        ) / 2
        
        ha_df.iloc[i, ha_df.columns.get_loc('HA_High')] = max(
            df.iloc[i]['High'], 
            ha_df.iloc[i]['HA_Open'], 
            ha_df.iloc[i]['HA_Close']
        )
        
        ha_df.iloc[i, ha_df.columns.get_loc('HA_Low')] = min(
            df.iloc[i]['Low'], 
            ha_df.iloc[i]['HA_Open'], 
            ha_df.iloc[i]['HA_Close']
        )
    
    return ha_df

class RFIBStrategy(Strategy):
    """
    Range Filter + IB Box Strategy - FIXED VERSION
    """
    
    # Strategy Parameters
    rf_range_quantity = 2.618
    rf_range_period = 14
    rf_smooth_period = 27
    ib_buffer = 0.0
    ib_mintick = 0.05
    use_heiken_ashi = True
    
    # Risk Management - FIXED
    stop_loss_pct = 2.0  # Stop loss percentage
    take_profit_pct = 4.0  # Take profit percentage
    position_size = 0.95  # Position size as fraction of equity
    
    def init(self):
        # Initialize indicators
        self.rf = RangeFilter()
        
        # Manual entry price tracking - FIXED
        self.entry_price = None
        self.position_type = None  # 'long' or 'short'
        
        # Calculate indicators on data
        rf_params = {
            'range_quantity': self.rf_range_quantity,
            'range_period': self.rf_range_period,
            'smooth_period': self.rf_smooth_period
        }
        
        # Prepare data
        df_work = pd.DataFrame({
            'Open': self.data.Open,
            'High': self.data.High,
            'Low': self.data.Low,
            'Close': self.data.Close
        })
        
        # Apply Heiken-Ashi if enabled
        if self.use_heiken_ashi:
            df_work = calculate_heiken_ashi(df_work)
            # Use HA data for IB calculations
            ib_data = pd.DataFrame({
                'open': df_work['HA_Open'],
                'high': df_work['HA_High'],
                'low': df_work['HA_Low'],
                'close': df_work['HA_Close']
            })
        else:
            ib_data = pd.DataFrame({
                'open': df_work['Open'],
                'high': df_work['High'],
                'low': df_work['Low'],
                'close': df_work['Close']
            })
        
        # Calculate Range Filter
        rf_data = self.rf.run_filter(df_work, **rf_params)
        
        # Calculate IB Box
        ib_data = calculate_inside_ib_box(ib_data, self.ib_buffer, self.ib_mintick)
        
        # Store indicators as series for proper indexing - FIXED
        self.rf_buy_signal = pd.Series(rf_data['RF_BuySignal'].values, index=self.data.index)
        self.rf_sell_signal = pd.Series(rf_data['RF_SellSignal'].values, index=self.data.index)
        self.green_arrow = pd.Series(ib_data['GreenArrow'].values, index=self.data.index)
        self.red_arrow = pd.Series(ib_data['RedArrow'].values, index=self.data.index)
        self.rf_filter = pd.Series(rf_data['RF_Filter'].values, index=self.data.index)
        
        # Calculate final signals
        self.final_signals = self._calculate_final_signals()
        
    def _calculate_final_signals(self):
        """Calculate final signals based on RF + IB confirmation - FIXED"""
        signals = pd.Series(0, index=self.data.index)
        
        # RF signal tracking
        last_rf_buy_signal = False
        last_rf_sell_signal = False
        rf_signal_bar = None
        rf_used = False
        
        # Arrow signal tracking
        last_green_arrow = False
        last_red_arrow = False
        arrow_signal_bar = None
        arrow_used = False
        
        for i, idx in enumerate(self.data.index):
            # Get current signals
            current_rf_buy = self.rf_buy_signal.iloc[i] == 1
            current_rf_sell = self.rf_sell_signal.iloc[i] == 1
            current_green_arrow = self.green_arrow.iloc[i]
            current_red_arrow = self.red_arrow.iloc[i]
            
            # Get previous signals
            prev_rf_buy = self.rf_buy_signal.iloc[i-1] if i > 0 else 0
            prev_rf_sell = self.rf_sell_signal.iloc[i-1] if i > 0 else 0
            prev_green_arrow = self.green_arrow.iloc[i-1] if i > 0 else False
            prev_red_arrow = self.red_arrow.iloc[i-1] if i > 0 else False
            
            # Detect new signals
            new_rf_buy = current_rf_buy and prev_rf_buy != 1
            new_rf_sell = current_rf_sell and prev_rf_sell != 1
            new_green_arrow = current_green_arrow and not prev_green_arrow
            new_red_arrow = current_red_arrow and not prev_red_arrow
            
            # Update RF signal tracking
            if new_rf_buy:
                last_rf_buy_signal = True
                last_rf_sell_signal = False
                rf_signal_bar = i
                rf_used = False
            elif new_rf_sell:
                last_rf_sell_signal = True
                last_rf_buy_signal = False
                rf_signal_bar = i
                rf_used = False
            
            # Update Arrow signal tracking
            if new_green_arrow:
                last_green_arrow = True
                last_red_arrow = False
                arrow_signal_bar = i
                arrow_used = False
            elif new_red_arrow:
                last_red_arrow = True
                last_green_arrow = False
                arrow_signal_bar = i
                arrow_used = False
            
            # Reset signals if too old (more than 1 candle)
            if rf_signal_bar is not None and (i - rf_signal_bar) > 1:
                last_rf_buy_signal = False
                last_rf_sell_signal = False
                rf_used = False
            
            if arrow_signal_bar is not None and (i - arrow_signal_bar) > 1:
                last_green_arrow = False
                last_red_arrow = False
                arrow_used = False
            
            signal = 0
            
            # Signal confirmation logic
            # Same candle confirmation
            if new_rf_buy and current_green_arrow and not rf_used:
                signal = 1
                rf_used = True
                last_rf_buy_signal = False
            elif new_rf_sell and current_red_arrow and not rf_used:
                signal = -1
                rf_used = True
                last_rf_sell_signal = False
            
            # Next candle confirmation (RF first)
            elif (last_rf_buy_signal and not rf_used and current_green_arrow and 
                  rf_signal_bar is not None and (i - rf_signal_bar) == 1):
                signal = 1
                rf_used = True
                last_rf_buy_signal = False
            elif (last_rf_sell_signal and not rf_used and current_red_arrow and 
                  rf_signal_bar is not None and (i - rf_signal_bar) == 1):
                signal = -1
                rf_used = True
                last_rf_sell_signal = False
            
            # Next candle confirmation (Arrow first)
            elif (last_green_arrow and not arrow_used and new_rf_buy and 
                  arrow_signal_bar is not None and (i - arrow_signal_bar) == 1):
                signal = 1
                arrow_used = True
                last_green_arrow = False
            elif (last_red_arrow and not arrow_used and new_rf_sell and 
                  arrow_signal_bar is not None and (i - arrow_signal_bar) == 1):
                signal = -1
                arrow_used = True
                last_red_arrow = False
            
            signals.iloc[i] = signal
        
        return signals
    
    def next(self):
        # Get current signal - FIXED indexing
        current_idx = len(self.data) - 1
        if current_idx < len(self.final_signals):
            signal = self.final_signals.iloc[current_idx]
        else:
            signal = 0
        
        current_price = self.data.Close[-1]
        
        # Risk management for existing positions - FIXED
        if self.position:
            if self.entry_price is not None:
                # Calculate percentage change
                if self.position_type == 'long':
                    pct_change = (current_price - self.entry_price) / self.entry_price * 100
                    
                    # Check stop loss (negative change >= stop_loss_pct)
                    if pct_change <= -self.stop_loss_pct:
                        print(f"STOP LOSS triggered: Long position closed at {pct_change:.2f}% loss")
                        self.position.close()
                        self.entry_price = None
                        self.position_type = None
                        return
                    
                    # Check take profit (positive change >= take_profit_pct)
                    elif pct_change >= self.take_profit_pct:
                        print(f"TAKE PROFIT triggered: Long position closed at {pct_change:.2f}% profit")
                        self.position.close()
                        self.entry_price = None
                        self.position_type = None
                        return
                    
                    # Check opposite signal
                    elif signal == -1:
                        print(f"OPPOSITE SIGNAL: Long position closed due to sell signal")
                        self.position.close()
                        self.entry_price = None
                        self.position_type = None
                
                elif self.position_type == 'short':
                    pct_change = (self.entry_price - current_price) / self.entry_price * 100
                    
                    # Check stop loss (negative change >= stop_loss_pct)
                    if pct_change <= -self.stop_loss_pct:
                        print(f"STOP LOSS triggered: Short position closed at {pct_change:.2f}% loss")
                        self.position.close()
                        self.entry_price = None
                        self.position_type = None
                        return
                    
                    # Check take profit (positive change >= take_profit_pct)
                    elif pct_change >= self.take_profit_pct:
                        print(f"TAKE PROFIT triggered: Short position closed at {pct_change:.2f}% profit")
                        self.position.close()
                        self.entry_price = None
                        self.position_type = None
                        return
                    
                    # Check opposite signal
                    elif signal == 1:
                        print(f"OPPOSITE SIGNAL: Short position closed due to buy signal")
                        self.position.close()
                        self.entry_price = None
                        self.position_type = None
        
        # Enter new positions - FIXED
        if not self.position:
            if signal == 1:  # Buy signal
                size = self.position_size
                self.buy(size=size)
                self.entry_price = current_price
                self.position_type = 'long'
                print(f"LONG ENTRY at {current_price:.4f}")
                
            elif signal == -1:  # Sell signal
                size = self.position_size
                self.sell(size=size)
                self.entry_price = current_price
                self.position_type = 'short'
                print(f"SHORT ENTRY at {current_price:.4f}")

def load_eth_data(file_path):
    """
    Load ETH data from CSV file
    Expected columns: close, high, low, open, time, volume
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Print original columns for debugging
        print(f"Original columns: {list(df.columns)}")
        print(f"First few rows of time column:")
        if 'time' in df.columns:
            print(df['time'].head())
        elif 'Time' in df.columns:
            print(df['Time'].head())
        
        # Rename columns to match expected format (capitalize first letter)
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Handle time column conversion more carefully
        time_col = None
        if 'time' in df.columns:
            time_col = 'time'
        elif 'Time' in df.columns:
            time_col = 'Time'
        
        if time_col:
            # Try different datetime conversion methods
            try:
                # First try: assume it's a Unix timestamp (in seconds)
                df[time_col] = pd.to_datetime(df[time_col], unit='s')
            except (ValueError, OSError):
                try:
                    # Second try: assume it's a Unix timestamp (in milliseconds)
                    df[time_col] = pd.to_datetime(df[time_col], unit='ms')
                except (ValueError, OSError):
                    try:
                        # Third try: assume it's already a datetime string
                        df[time_col] = pd.to_datetime(df[time_col])
                    except ValueError:
                        # If all else fails, create a simple datetime index
                        print("Warning: Could not parse time column, creating sequential datetime index")
                        df[time_col] = pd.date_range(start='2020-01-01', periods=len(df), freq='15T')
            
            df.set_index(time_col, inplace=True)
        else:
            # If no time column, create a simple datetime index
            print("Warning: No time column found, creating sequential datetime index")
            df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='15T')
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Sort by index to ensure chronological order
        df = df.sort_index()
        
        print(f"Loaded {len(df)} rows of data")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {list(df.columns)}")
        print(f"Price range: {df['Close'].min():.4f} to {df['Close'].max():.4f}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def run_backtest(data, **strategy_params):
    """
    Run backtest with given parameters - FIXED
    """
    
    # Create a copy of the strategy class to avoid modifying the original
    class TempStrategy(RFIBStrategy):
        pass
    
    # Set strategy parameters
    for param, value in strategy_params.items():
        if hasattr(TempStrategy, param):
            setattr(TempStrategy, param, value)
    
    # Run backtest with more reasonable defaults
    bt = Backtest(
        data, 
        TempStrategy, 
        cash=100000, 
        commission=0.002,  # 0.2% commission
        exclusive_orders=True,  # Prevent overlapping orders
        trade_on_close=False  # Trade on next open, more realistic
    )
    
    return bt.run()

def optimize_parameters(data, param_ranges, maximize='Sharpe Ratio', max_tries=100):
    """
    Optimize strategy parameters - FIXED
    """
    
    bt = Backtest(
        data, 
        RFIBStrategy, 
        cash=100000, 
        commission=0.002,
        exclusive_orders=True,
        trade_on_close=False
    )
    
    try:
        results = bt.optimize(
            **param_ranges,
            maximize=maximize,
            max_tries=max_tries,
            random_state=42,
            constraint=lambda p: p.take_profit_pct > p.stop_loss_pct  # Ensure TP > SL
        )
        return results
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None

def analyze_results(results):
    """
    Analyze and display backtest results - Enhanced
    """
    
    print("="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    # Key metrics
    print(f"Total Return: {results['Return [%]']:.2f}%")
    print(f"Buy & Hold Return: {results['Buy & Hold Return [%]']:.2f}%")
    print(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
    print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
    print(f"Calmar Ratio: {results['Calmar Ratio']:.2f}")
    
    # Trade statistics
    print(f"\nTotal Trades: {results['# Trades']}")
    if results['# Trades'] > 0:
        print(f"Win Rate: {results['Win Rate [%]']:.2f}%")
        print(f"Best Trade: {results['Best Trade [%]']:.2f}%")
        print(f"Worst Trade: {results['Worst Trade [%]']:.2f}%")
        print(f"Avg Trade: {results['Avg. Trade [%]']:.2f}%")
        
        # Duration statistics
        print(f"\nAvg Trade Duration: {results['Avg. Trade Duration']}")
        print(f"Max Trade Duration: {results['Max. Trade Duration']}")
        
        # Exposure
        print(f"\nExposure Time: {results['Exposure Time [%]']:.2f}%")
    else:
        print("No trades executed!")
    
    # Strategy parameters (if available)
    if hasattr(results, '_strategy'):
        print(f"\n" + "="*30)
        print("STRATEGY PARAMETERS")
        print("="*30)
        strategy_attrs = [attr for attr in dir(results._strategy) 
                         if not attr.startswith('_') and not callable(getattr(results._strategy, attr))]
        for attr in strategy_attrs:
            if attr in ['rf_range_quantity', 'rf_range_period', 'stop_loss_pct', 
                       'take_profit_pct', 'position_size', 'use_heiken_ashi']:
                print(f"{attr}: {getattr(results._strategy, attr)}")
    
    return results

def run_example_backtest():
    """
    Example of running a backtest with ETH data - FIXED
    """
    
    # Try to load ETH data first
    # eth_file_path = r"C:\Users\vaibh\OneDrive\Desktop\delta\data\ETHUSD.csv"
    eth_file_path = r"C:\Users\vaibh\OneDrive\Desktop\delta\data\XRPUSD.csv"
    data = load_eth_data(eth_file_path)
    
    if data is None:
        print("Failed to load ETH data!")
        return None
    
    # Run simple backtest with conservative parameters
    print("Running simple backtest...")
    strategy_params = {
        'rf_range_quantity': 2.618,
        'rf_range_period': 14,
        'stop_loss_pct': 2.0,
        'take_profit_pct': 4.0,
        'position_size': 0.95,
        'use_heiken_ashi': True
    }
    
    results = run_backtest(data, **strategy_params)
    analyze_results(results)
    
    # Plot results if possible
    try:
        results.plot(filename='backtest_results.html')
        print("\nBacktest chart saved as 'backtest_results.html'")
    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    return results

def run_example_optimization():
    """
    Example of parameter optimization - FIXED
    """
    
    # Try to load ETH data first
    # eth_file_path = r"C:\Users\vaibh\OneDrive\Desktop\delta\data\ETHUSD.csv"
    eth_file_path = r"C:\Users\vaibh\OneDrive\Desktop\delta\data\XRPUSD.csv"
    data = load_eth_data(eth_file_path)
    
    if data is None:
        print("Failed to load ETH data!")
        return None
    
    # Define parameter ranges - more conservative
    # param_ranges = {
    #     'rf_range_quantity': [1.5, 2.0, 2.618, 3.0],
    #     'rf_range_period': [10, 14, 20],
    #     'stop_loss_pct': [1.0, 1.5, 2.0, 2.5],
    #     'take_profit_pct': [2.0, 3.0, 4.0, 5.0],
    #     'use_heiken_ashi': [True, False]
    # }
    param_ranges = {
    'rf_range_quantity': [1.5, 2.0, 2.618, 3.0],
    'rf_range_period': [10, 14, 20],
    'stop_loss_pct': [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0],
    'take_profit_pct': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.3, 3.5, 4.0],
    'use_heiken_ashi': [True, False]
    }
    
    print("Running parameter optimization...")
    print("This may take a few minutes...")
    
    # Run optimization
    opt_results = optimize_parameters(
        data, 
        param_ranges, 
        maximize='Return [%]',  # Changed to maximize return instead of Sharpe
        max_tries=50
    )
    
    if opt_results is not None:
        try:
            print("\n" + "="*60)
            print("OPTIMIZATION RESULTS")
            print("="*60)
            
            # Display optimized results
            analyze_results(opt_results)
            
            # Plot optimized results
            try:
                opt_results.plot(filename='optimized_backtest_results.html')
                print("\nOptimized backtest chart saved as 'optimized_backtest_results.html'")
            except Exception as e:
                print(f"Could not generate plot: {e}")
            
            return opt_results
            
        except Exception as e:
            print(f"Error processing optimization results: {e}")
            return None
    else:
        print("Optimization failed!")
        return None

if __name__ == "__main__":
    # Run example backtest
    print("Starting backtest analysis...")
    backtest_results = run_example_backtest()
    
    if backtest_results is not None:
        print("\n" + "="*60)
        print("Running optimization...")
        # Run optimization
        optimization_results = run_example_optimization()
    else:
        print("Backtest failed, skipping optimization.")