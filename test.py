from delta_rest_client import DeltaRestClient
import pandas as pd
import numpy as np
from module.rf import RangeFilter
from module.ib_indicator import Box,calculate_inside_bar_boxes
from module.rsi_gaizy import RSIGainzy
import hashlib
import hmac 
import time
import requests
import json

class DeltaExchangeRFGaizy:
    def __init__(self,api_key,api_secret,base_url):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.broker = DeltaRestClient(base_url=base_url,api_key=api_key,api_secret=api_secret)
        self.rf = RangeFilter()
        self.ib = None
        self.Grsi = RSIGainzy()
        
        # Trading state variables
        self.current_position = None  # 'long', 'short', or None
        self.current_order_id = None
        self.stop_loss_hit = False
        self.base_lot_size = 1
        self.current_lot_size = self.base_lot_size
        self.rf_signal_active = False
        self.last_rf_signal_index = -1
        self.stop_loss_x = 100  # Maximum stop loss in points - adjust as needed
        self.trades_from_current_rf = 0  # Track trades per RF signal
        self.awaiting_confirmation = False
        self.confirmation_signal_type = None  # 'rsi_rf', 'rf_ib', 'rsi_ib'
        self.entry_analysis_done = False

    def generate_signature(self,secret,message):
        try:
            message = bytes(message, 'utf-8')
            secret = bytes(secret, 'utf-8')
            hash = hmac.new(secret, message, hashlib.sha256)
            return hash.hexdigest()
        except Exception as e:
            print(f"Error occured in generate_signature: {e}")

    def timestamp_generator(self):
        try:
            import time
            return str(int(time.time()))
        except Exception as e:
            print(f"Error occured in timestamp_generator : {e}")

    def cancel_order(self, order_id="639401895", product_id=84):
        try:
            method = "DELETE"
            path = "/v2/orders"
            url = self.base_url + path

            payload = {
                "id": int(order_id),
                "product_id": int(product_id)
            }
            payload_json = json.dumps(payload, separators=(',', ':'))

            timestamp = str(int(time.time()))
            signature_data = method + timestamp + path + payload_json
            signature = self.generate_signature(self.api_secret, signature_data)

            headers = {
                "api-key": self.api_key,
                "timestamp": timestamp,
                "signature": signature,
                "User-Agent": "python-rest-client",
                "Content-Type": "application/json"
            }

            response = requests.delete(url, headers=headers, data=payload_json)
            print(f"Cancel Order Response Code: {response.status_code}")
            
            if response.status_code == 200:
                self.current_order_id = None
                return True
            return False

        except Exception as e:
            print(f"Error in cancel_order function: {e}")
            return False

    def place_order_market(self, side, size, symbol="BTCUSD"):
        try:
            payload = {
                "product_symbol": symbol,
                "size": int(size),
                "side": side,
                "order_type": "market_order"
            }

            method = 'POST'
            path = '/v2/orders'
            url = self.base_url + path
            timestamp = str(int(time.time()))
            query_string = ''
            payload_json = json.dumps(payload, separators=(',', ':'))
            signature_data = method + timestamp + path + query_string + payload_json
            signature = self.generate_signature(self.api_secret, signature_data)

            headers = {
                'api-key': self.api_key,
                'timestamp': timestamp,
                'signature': signature,
                'User-Agent': 'python-rest-client',
                'Content-Type': 'application/json'
            }

            response = requests.post(url, headers=headers, data=payload_json)
            print(f"Place Order Response Code: {response.status_code}")
            
            if response.status_code == 200:
                order_data = response.json()
                self.current_order_id = order_data.get('id')
                return order_data
            return None

        except Exception as e:
            print(f"Error in place_order_market function: {e}")
            return None

    def place_order_limit(self, side, size, price, symbol="BTCUSD"):
        try:
            payload = {
                "product_symbol": symbol,
                "limit_price": str(price),
                "size": int(size),
                "side": side,
                "order_type": "limit_order",
                "time_in_force": "gtc"
            }

            method = 'POST'
            path = '/v2/orders'
            url = self.base_url + path
            timestamp = str(int(time.time()))
            query_string = ''
            payload_json = json.dumps(payload, separators=(',', ':'))
            signature_data = method + timestamp + path + query_string + payload_json
            signature = self.generate_signature(self.api_secret, signature_data)

            headers = {
                'api-key': self.api_key,
                'timestamp': timestamp,
                'signature': signature,
                'User-Agent': 'python-rest-client',
                'Content-Type': 'application/json'
            }

            response = requests.post(url, headers=headers, data=payload_json)
            print(f"Place Limit Order Response Code: {response.status_code}")
            
            if response.status_code == 200:
                order_data = response.json()
                self.current_order_id = order_data.get('id')
                return order_data
            return None

        except Exception as e:
            print(f"Error in place_order_limit function: {e}")
            return None

    def connect(self):
        try:
            res = self.broker._init_session()
            print(res)
        except Exception as e:
            print(f"Error occured in connection : {e}")

    def fetch_data(self):
        try:
            self.df = self.broker.get_ticker_data(symbol='BTCUSD')
        except Exception as e:
            print(f"Error occured in fetching the data : {e}")

    def calculate_signals(self):
        try:
            self.df.rename(columns={'close':'Close','open':'Open','high':'High','low':'Low','volume':'Volume'},inplace=True)
            self.df = self.Grsi.calculate_signals(self.df,rsi_length=10,smooth_length=3,overbought_level=60,oversold_level=40)
            self.df.rename(columns={'Close':'close','Open':'open','High':'high','Low':'low','Volume':'volume'},inplace=True)
            self.df,_ = calculate_inside_bar_boxes(self.df)
            self.df.rename(columns={'close':'Close','open':'Open','high':'High','low':'Low','volume':'Volume'},inplace=True)
            self.df = self.rf.run_filter(self.df)
            self.df.to_csv('BTCUSD_Indicator.csv')
        except Exception as e:
            print(f"Error occured in calculating the signal : {e}")

    def check_rsi_signal(self, current_idx):
        """Check for RSI buy/sell signals"""
        try:
            if current_idx < 1:
                return None
            
            # Assuming RSI signals are in a column - adjust column name as needed
            # You'll need to check your RSI implementation for the exact signal column
            rsi_col = 'rsi_signal'  # Replace with actual column name
            if rsi_col in self.df.columns:
                if self.df.iloc[current_idx][rsi_col] == 1:  # Buy signal
                    return 'buy'
                elif self.df.iloc[current_idx][rsi_col] == -1:  # Sell signal
                    return 'sell'
            return None
        except Exception as e:
            print(f"Error in check_rsi_signal: {e}")
            return None

    def check_rf_signal(self, current_idx):
        """Check for RF buy/sell signals"""
        try:
            if current_idx < 1:
                return None
                
            # Check the RF signal column - adjust based on your RF implementation
            rf_col = 'rf_signal'  # Replace with actual column name from RF
            if rf_col in self.df.columns:
                current_signal = self.df.iloc[current_idx][rf_col]
                if current_signal == 1:  # Buy signal
                    return 'buy'
                elif current_signal == -1:  # Sell signal
                    return 'sell'
            return None
        except Exception as e:
            print(f"Error in check_rf_signal: {e}")
            return None

    def check_ib_breakout(self, current_idx):
        """Check for Inside Bar breakout"""
        try:
            if current_idx < 1:
                return None
                
            current_row = self.df.iloc[current_idx]
            current_high = current_row['High']
            current_low = current_row['Low']
            
            # Check if there's an active IB box
            if pd.notna(current_row.get('ib_high')) and pd.notna(current_row.get('ib_low')):
                ib_high = current_row['ib_high']
                ib_low = current_row['ib_low']
                
                if current_high > ib_high:
                    return 'upward_breakout'
                elif current_low < ib_low:
                    return 'downward_breakout'
            
            return None
        except Exception as e:
            print(f"Error in check_ib_breakout: {e}")
            return None

    def calculate_stop_loss(self, position_type, current_idx, entry_price):
        """Calculate stop loss levels"""
        try:
            if current_idx < 1:
                return None, None
                
            prev_candle = self.df.iloc[current_idx - 1]
            
            if position_type == 'long':
                # Stop loss at previous candle's low or X points below entry
                prev_low_sl = prev_candle['Low']
                max_sl = entry_price - self.stop_loss_x
                return max(prev_low_sl, max_sl), min(prev_low_sl, max_sl)
                
            elif position_type == 'short':
                # Stop loss at previous candle's high or X points above entry
                prev_high_sl = prev_candle['High']
                max_sl = entry_price + self.stop_loss_x
                return min(prev_high_sl, max_sl), max(prev_high_sl, max_sl)
                
        except Exception as e:
            print(f"Error in calculate_stop_loss: {e}")
            return None, None

    def check_stop_loss_hit(self, current_idx):
        """Check if stop loss is hit"""
        try:
            if not self.current_position or current_idx < 1:
                return False
                
            current_price = self.df.iloc[current_idx]['Close']
            prev_candle = self.df.iloc[current_idx - 1]
            
            if self.current_position == 'long':
                prev_low = prev_candle['Low']
                max_sl = current_price - self.stop_loss_x
                stop_level = max(prev_low, max_sl)
                
                if current_price <= stop_level:
                    return True
                    
            elif self.current_position == 'short':
                prev_high = prev_candle['High']
                max_sl = current_price + self.stop_loss_x
                stop_level = min(prev_high, max_sl)
                
                if current_price >= stop_level:
                    return True
                    
            return False
        except Exception as e:
            print(f"Error in check_stop_loss_hit: {e}")
            return False

    def execute_trade(self, action, current_idx):
        """Execute the actual trade"""
        try:
            current_price = self.df.iloc[current_idx]['Close']
            
            if action in ['long', 'buy']:
                order = self.place_order_market('buy', self.current_lot_size)
                if order:
                    self.current_position = 'long'
                    self.trades_from_current_rf += 1
                    print(f"LONG position opened at {current_price} with size {self.current_lot_size}")
                    return True
                    
            elif action in ['short', 'sell']:
                order = self.place_order_market('sell', self.current_lot_size)
                if order:
                    self.current_position = 'short'
                    self.trades_from_current_rf += 1
                    print(f"SHORT position opened at {current_price} with size {self.current_lot_size}")
                    return True
                    
            return False
        except Exception as e:
            print(f"Error in execute_trade: {e}")
            return False

    def close_position(self, current_idx, reason=""):
        """Close current position"""
        try:
            if not self.current_position:
                return False
                
            current_price = self.df.iloc[current_idx]['Close']
            
            if self.current_position == 'long':
                order = self.place_order_market('sell', self.current_lot_size)
            else:
                order = self.place_order_market('buy', self.current_lot_size)
                
            if order:
                print(f"Position closed at {current_price}. Reason: {reason}")
                
                if reason == "stop_loss":
                    self.stop_loss_hit = True
                    # Double lot size for next entry (maintain multiples of 5)
                    self.current_lot_size = ((self.current_lot_size * 2) // 5) * 5
                    if self.current_lot_size == 0:
                        self.current_lot_size = 5
                        
                self.current_position = None
                self.current_order_id = None
                return True
                
            return False
        except Exception as e:
            print(f"Error in close_position: {e}")
            return False

    def execute_signals(self):
        """Main signal execution logic"""
        try:
            if self.df is None or len(self.df) < 2:
                print("Insufficient data for signal execution")
                return
                
            current_idx = len(self.df) - 1  # Latest candle
            current_row = self.df.iloc[current_idx]
            
            print(f"Executing signals for candle {current_idx}")
            
            # Check if current position needs to be closed due to stop loss
            if self.current_position and self.check_stop_loss_hit(current_idx):
                self.close_position(current_idx, "stop_loss")
                return
                
            # Get current signals
            rsi_signal = self.check_rsi_signal(current_idx)
            rf_signal = self.check_rf_signal(current_idx)
            ib_breakout = self.check_ib_breakout(current_idx)
            
            # Track RF signal changes
            if rf_signal and self.last_rf_signal_index != current_idx:
                self.rf_signal_active = True
                self.last_rf_signal_index = current_idx
                self.trades_from_current_rf = 0
                
            # Skip if already have one trade from current RF signal
            if self.trades_from_current_rf >= 1:
                print("Already have one trade from current RF signal")
                return
                
            # Entry Logic
            if not self.current_position:
                
                # Long Entry Conditions
                if rsi_signal == 'buy':
                    if not self.entry_analysis_done:
                        print("RSI Buy signal detected - Analysis only (no trade)")
                        self.entry_analysis_done = True
                        self.awaiting_confirmation = True
                        self.confirmation_signal_type = 'rsi_rf'
                        return
                        
                    # Condition 3: RSI buy + IB upward breakout
                    if ib_breakout == 'upward_breakout':
                        print("Entry Condition 3: RSI Buy + IB Upward Breakout")
                        self.execute_trade('long', current_idx)
                        self.reset_entry_state()
                        return
                        
                # Condition 1: RSI buy confirmed by RF buy
                if (self.awaiting_confirmation and 
                    self.confirmation_signal_type == 'rsi_rf' and 
                    rf_signal == 'buy'):
                    print("Entry Condition 1: RSI Buy confirmed by RF Buy")
                    self.execute_trade('long', current_idx)
                    self.reset_entry_state()
                    return
                    
                # Condition 2: RF buy + IB upward breakout
                if rf_signal == 'buy' and ib_breakout == 'upward_breakout':
                    print("Entry Condition 2: RF Buy + IB Upward Breakout")
                    self.execute_trade('long', current_idx)
                    return
                    
                # Short Entry Conditions
                if rsi_signal == 'sell':
                    if not self.entry_analysis_done:
                        print("RSI Sell signal detected - Analysis only (no trade)")
                        self.entry_analysis_done = True
                        self.awaiting_confirmation = True
                        self.confirmation_signal_type = 'rsi_rf'
                        return
                        
                    # Condition 3: RSI sell + IB downward breakout
                    if ib_breakout == 'downward_breakout':
                        print("Entry Condition 3: RSI Sell + IB Downward Breakout")
                        self.execute_trade('short', current_idx)
                        self.reset_entry_state()
                        return
                        
                # Condition 1: RSI sell confirmed by RF sell
                if (self.awaiting_confirmation and 
                    self.confirmation_signal_type == 'rsi_rf' and 
                    rf_signal == 'sell'):
                    print("Entry Condition 1: RSI Sell confirmed by RF Sell")
                    self.execute_trade('short', current_idx)
                    self.reset_entry_state()
                    return
                    
                # Condition 2: RF sell + IB downward breakout
                if rf_signal == 'sell' and ib_breakout == 'downward_breakout':
                    print("Entry Condition 2: RF Sell + IB Downward Breakout")
                    self.execute_trade('short', current_idx)
                    return
            
            # Re-entry logic after stop loss
            elif self.stop_loss_hit and not self.current_position:
                # Only allow re-entry based on condition 2 (RF + IB)
                if rf_signal == 'buy' and ib_breakout == 'upward_breakout':
                    print("Re-entry: RF Buy + IB Upward Breakout (after stop loss)")
                    self.execute_trade('long', current_idx)
                    self.stop_loss_hit = False
                    return
                    
                elif rf_signal == 'sell' and ib_breakout == 'downward_breakout':
                    print("Re-entry: RF Sell + IB Downward Breakout (after stop loss)")
                    self.execute_trade('short', current_idx)
                    self.stop_loss_hit = False
                    return
                    
        except Exception as e:
            print(f"Error in execute_signals: {e}")

    def reset_entry_state(self):
        """Reset entry analysis state"""
        self.entry_analysis_done = False
        self.awaiting_confirmation = False
        self.confirmation_signal_type = None

    def run(self):
        try:
            self.connect()
            self.fetch_data()
            self.calculate_signals()
            self.execute_signals()
        except Exception as e:
            print(f"Error occured in run : {e}")
        import time
        time.sleep(10)
    
    def test(self):
        try:
            self.connect()
            # self.place_order_limit()
            self.cancel_order()
        except Exception as e:
            print(f"Error in test : {e}")
        import time
        time.sleep(10)

if __name__ == "__main__":
    while True:
        try:
            api_key = "MZvQgXnmlxSqWnfJRRcvBymh6FFqs5"
            api_secret = "nqERtehNDx4JAyfA8pRiuG3VsCZOor6dF33Ttp8SH7XWObo9D2ZnqVflsoMz"
            base_url = "https://cdn-ind.testnet.deltaex.org"
            Debot = DeltaExchangeRFGaizy(api_key=api_key,api_secret=api_secret,base_url=base_url)
            Debot.run()
            # Debot.test()
        except Exception as e:
            print(f"Error in run : {e}")