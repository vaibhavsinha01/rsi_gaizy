
from delta_rest_client import DeltaRestClient
import pandas as pd
import numpy as np
from module.rf import RangeFilter
from module.ib_indicator import Box,calculate_inside_bar_boxes
from module.rsi_gaizy import RSIGainzy
from module.rsi_buy_sell import RSIBuySellIndicator
import hashlib
import hmac 
import time
import requests
import json
from datetime import datetime, timedelta

class DeltaExchangeRFGaizy:
    def __init__(self,api_key,api_secret,base_url):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.max_stoploss = int(50)
        self.max_profit = int(100)
        self.max_profit_percent = 0.01 # 1 percent
        self.broker = DeltaRestClient(base_url=base_url,api_key=api_key,api_secret=api_secret)
        self.rf = RangeFilter()
        self.bsrsi = RSIBuySellIndicator()
        self.ib = None
        self.Grsi = RSIGainzy()
        self.current_order_id = None
        self.base_leverage = 1
        self.leverage_multiplier = 2
        self.max_leverage = 4
        self.last_price = None
        self.h_pos = 0
        self.flag = 0
        self.base = 1

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

    def get_current_datetime(self):
        """Get current datetime for data fetching"""
        try:
            return datetime.now()
        except Exception as e:
            print(f"Error in get_current_datetime: {e}")
            return None

    def calculate_stoploss(self, entry_price, side, candle_data=None):
        """Calculate stoploss based on 1%, 50 points, or low of 2nd last candle"""
        try:
            stoplosses = []
            
            # 1% stoploss
            if side == "buy":
                sl_1_percent = entry_price * 0.99
                sl_50_points = entry_price - 25
            else:  # sell
                sl_1_percent = entry_price * 1.01
                sl_50_points = entry_price + 25
            
            stoplosses.append(sl_1_percent)
            stoplosses.append(sl_50_points)
            
            # Low of 2nd last candle (if available)
            if candle_data is not None and len(candle_data) >= 2:
                second_last_low = candle_data.iloc[-2]['low']
                if side == "buy":
                    stoplosses.append(second_last_low)
                else:  # sell - use high for sell orders
                    second_last_high = candle_data.iloc[-2]['high']
                    stoplosses.append(second_last_high)
            
            # Choose the most conservative stoploss
            if side == "buy":
                final_sl = max(stoplosses)  # Highest stoploss for buy
                # final_sl = min(stoplosses)
            else:
                final_sl = min(stoplosses)  # Lowest stoploss for sell
                # final_sl = max(stoplosses)
            
            print(f"Calculated stoploss for {side}: {final_sl}")
            return final_sl
            
        except Exception as e:
            print(f"Error in calculate_stoploss: {e}")
            return None

    def calculate_takeprofit(self, entry_price, side):
        """Calculate take profit based on risk-reward ratio"""
        try:
            if side == "buy":
                tp = entry_price * 1.01  # 2% profit
            else:  # sell
                tp = entry_price * 0.99  # 2% profit
            
            print(f"Calculated take profit for {side}: {tp}")
            return tp
            
        except Exception as e:
            print(f"Error in calculate_takeprofit: {e}")
            return None
    
    def leverage_check(self):
        if self.base_leverage<self.max_leverage:
            self.base_leverage = self.base_leverage * self.leverage_multiplier
        else:
            self.base_leverage = 1

    # working
    def cancel_order(self, order_id="639401895", product_id=1699):
        try:
            method = "DELETE"
            path = "/v2/orders"
            url = self.base_url + path

            # Prepare JSON payload with both id and product_id
            payload = {
                "id": int(order_id),
                "product_id": int(product_id)
            }
            payload_json = json.dumps(payload, separators=(',', ':'))

            timestamp = str(int(time.time()))

            # Signature must include method + timestamp + path + payload JSON (no query string)
            signature_data = method + timestamp + path + payload_json
            print(f"Signature Data: {signature_data}")  # Debug

            signature = self.generate_signature(self.api_secret, signature_data)

            headers = {
                "api-key": self.api_key,
                "timestamp": timestamp,
                "signature": signature,
                "User-Agent": "python-rest-client",
                "Content-Type": "application/json"
            }

            # Send DELETE request with JSON body
            response = requests.delete(url, headers=headers, data=payload_json)

            print(f"Response Code: {response.status_code}")
            try:
                print(f"Response Body: {response.json()}")
                return response.json()
            except ValueError:
                print(f"Raw Response Text: {response.text}")
                return None

        except Exception as e:
            print(f"Error in cancel_order function: {e}")
            return None
    
    def place_order_market(self,side="buy",size=1):
        # finally working
        try:
            payload = {
                "product_symbol":"ETHUSD",
                "size": size,
                "side": side,
                "order_type": "market_order",
                "time_in_force":"gtc"
            }

            method = 'POST'
            path = '/v2/orders'
            url = self.base_url + path
            timestamp = str(int(time.time()))
            query_string = ''
            payload_json = json.dumps(payload, separators=(',', ':'))  # compact format
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
            print(f"Normal Response Code: {response.status_code}")
            print(f"Normal Response Body: {response.json()}")
            res = response.json()
            self.current_order_id = res['result']['id']
            if side=="buy":
                self.h_pos = 1
                print(f"self.h_pos is changed to {self.h_pos} in line 204")
            else:
                self.h_pos = -1
                print(f"self.h_pos is changed to {self.h_pos} in line 206")

        except Exception as e:
            print(f"Error in place_order_limit function: {e}")

    def place_order_limit(self,side="buy",size=1):
        # finally working
        try:
            payload = {
                "product_symbol":"ETHUSD",
                "limit_price": "2600",
                "size": size,
                "side": side,
                "order_type": "limit_order",
                "time_in_force":"gtc"
            }

            method = 'POST'
            path = '/v2/orders'
            url = self.base_url + path
            timestamp = str(int(time.time()))
            query_string = ''
            payload_json = json.dumps(payload, separators=(',', ':'))  # compact format
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
            print(f"Response Code: {response.status_code}")
            print(f"Response Body: {response.json()}")

        except Exception as e:
            print(f"Error in place_order_limit function: {e}")
    
    def place_order_bracket_limit(self,limit_price="2560",stop_price="2550",stop_limit_price="2540",take_profit_price="2570",take_profit_limit_price="2560",side="buy",size=1):
        try:
            payload = {
            "product_symbol": "ETHUSD",
            "limit_price":limit_price,
            "size":size,
            "side":side,
            "order_type": "limit_order",
            "time_in_force":"gtc",
            "stop_loss_order": {
                "order_type": "limit_order",
                "stop_price": stop_price,
                "limit_price": stop_limit_price
            },
            "take_profit_order": {
                "order_type": "limit_order",
                "stop_price": take_profit_price,
                "limit_price": take_profit_limit_price
            },
            "bracket_stop_trigger_method": "last_traded_price"
            }

            method = 'POST'
            path = '/v2/orders/bracket'
            url = self.base_url + path
            timestamp = str(int(time.time()))
            query_string = ''
            payload_json = json.dumps(payload, separators=(',', ':'))  # compact format
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
            print(f"Bracket Response Code: {response.status_code}")
            print(f"Bracket Response Body: {response.json()}")

        except Exception as e:
            print(f"Error in place_order_bracket_limit function: {e}")

    def order_status(self, order_id):
        try:
            print(f"current h_pos is {self.h_pos}")
            if self.current_order_id is not None:
                res = self.get_order_status(order_id)
                print(f"response is {res}")
                
                order = res['result']
                
                avg_fill = float(order.get("average_fill_price", 0))
                side = order.get("side")  # 'buy' or 'sell'
                
                # mark_price = self.get_market_price()
                mark_price = self.get_market_price()
                
                # Calculate PnL
                if side == "buy":
                    pnl = (mark_price - avg_fill)
                else:  # 'sell'
                    pnl = (avg_fill - mark_price)
                
                print(f"Average Fill Price: {avg_fill}")
                print(f"Current Market Price: {mark_price}")
                print(f"Take profit is {self.last_tp_price}")
                print(f"Stop loss is {self.last_sl_price}")
                print(f"PNL: {pnl}")
                
                if side == "buy":
                    if self.last_tp_price is not None and mark_price >= self.last_tp_price:
                        print(f"Take Profit hit at {mark_price}")
                        self.flag = 1
                    elif self.last_sl_price is not None and mark_price <= self.last_sl_price:
                        print(f"Stop Loss hit at {mark_price}")
                        self.flag = -1
                else:  
                    if self.last_tp_price is not None and mark_price <= self.last_tp_price:
                        print(f"Take Profit hit at {mark_price}")
                        self.flag = 1
                        # TP hit logic
                    elif self.last_sl_price is not None and mark_price >= self.last_sl_price:
                        print(f"Stop Loss hit at {mark_price}")
                        self.flag = -1

                if self.flag == 1:
                    # self.h_pos = 0
                    print(f"self.flag is {self.flag} the position is resetted to {self.h_pos} in line 338")
                elif self.flag == -1:
                    # self.h_pos = 0
                    print(f"self.flag is {self.flag} the position is resetted to {self.h_pos} in line 341")
                    self.leverage_check()
                else:
                    print("Order is still open")

        except Exception as e:
            print(f"Error in order_status function : {e}")

    def get_market_price(self):
        last = self.df.iloc[-1]
        return last['close']
    

    def get_usd_balance(self):
        """Get USD/USDT balance specifically"""
        try:
            method = "GET"
            path = "/v2/wallet/balances"
            url = self.base_url + path
            timestamp = str(int(time.time()))
            query_string = ''
            signature_data = method + timestamp + path + query_string
            signature = self.generate_signature(self.api_secret, signature_data)

            headers = {
                "api-key": self.api_key,
                "timestamp": timestamp,
                "signature": signature,
                "User-Agent": "python-rest-client"
            }

            response = requests.get(url, headers=headers)
            wallet_data = response.json()
            print(wallet_data['result'][0]['balance'])
            
            return wallet_data['result'][0]['balance']
            
        except Exception as e:
            print(f"Error getting USD balance: {e}")
            return None

    def get_order_status(self, order_id):
        """Get order status"""
        try:
            method = "GET"
            path = f"/v2/orders/{order_id}"
            url = self.base_url + path
            timestamp = str(int(time.time()))
            query_string = ''
            signature_data = method + timestamp + path + query_string
            signature = self.generate_signature(self.api_secret, signature_data)

            headers = {
                "api-key": self.api_key,
                "timestamp": timestamp,
                "signature": signature,
                "User-Agent": "python-rest-client"
            }

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            return None

        except Exception as e:
            print(f"Error in get_order_status: {e}")
            return None

    def connect(self):
        try:
            res  = self.broker._init_session() # i think from here on the session is established
            print(res)
        except Exception as e:
            print(f"Error occured in connection : {e}")

    def fetch_data(self):
        try:
            # Add datetime info to the data fetching
            current_time = self.get_current_datetime()
            print(f"Fetching data at: {current_time}")
            
            self.df = self.broker.get_ticker_data(symbol='ETHUSD') # current time / other things need to be accounted for
            
            self.df['Timestamp'] = pd.to_datetime(self.df['time'],unit='s')
            self.df.sort_values(by='Timestamp',ascending=False,inplace=True)
            self.df = self.df.iloc[::-1].reset_index(drop=True)
            self.df.to_csv("ETHUSD_Indicator.csv")
                
        except Exception as e:
            print(f"Error occured in fetching the data : {e}")

    def rsi(self, window=10,ema_window=10):
        try:
            delta = self.df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

            rs = gain / loss
            self.df['rsi'] = 100 - (100 / (1 + rs))
            self.df['rsi'] = self.df['rsi'].ewm(span=ema_window, adjust=False).mean()
            return self.df['rsi']
        except Exception as e:
            print(f'Error in rsi function : {e}')
    
    def set_leverage_delta(self, value, product_id):
        """
        Set leverage for a specific product on Delta Exchange manually using HTTP POST.

        Args:
            value (int or str): Leverage to set (e.g., 10).
            product_id (int or str): The Delta Exchange product ID (e.g., 27 for ETHUSD).
        """
        import time
        import json
        import requests

        try:
            method = "POST"
            path = f"/v2/products/{product_id}/orders/leverage"
            url = self.base_url + path
            timestamp = str(int(time.time()))

            # JSON body
            payload = {
                "leverage": str(value)
            }
            payload_json = json.dumps(payload, separators=(',', ':'))

            # Signature: method + timestamp + path + body
            signature_data = method + timestamp + path + payload_json
            signature = self.generate_signature(self.api_secret, signature_data)

            # Headers
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'api-key': self.api_key,
                'timestamp': timestamp,
                'signature': signature,
                'User-Agent': 'custom-python-client/1.0'
            }

            # POST request
            response = requests.post(url, headers=headers, data=payload_json)

            # Print response
            print(f"Leverage status: {response.status_code}")
            try:
                response_data = response.json()
                print("Leverage response:", response_data)
            except ValueError:
                print("Leverage response: Non-JSON response received.")
                return None

            # Success check
            if response.status_code == 200 and response_data.get("success", False):
                return response_data
            else:
                print("Failed to set leverage.")
                return None

        except Exception as e:
            print(f"Error in set_leverage_delta: {e}")
            return None


    def calculate_heiken_ashi(self):
        try:
            print("heiken-ashi is being calculated.")
            # Ensure required columns exist
            required_cols = {'Open', 'High', 'Low', 'Close'}
            if not required_cols.issubset(self.df.columns):
                raise ValueError(f"DataFrame must contain the following columns: {required_cols}")

            # Create new columns for Heiken-Ashi
            self.df['HA_Close'] = (self.df['Open'] + self.df['High'] + self.df['Low'] + self.df['Close']) / 4
            self.df['HA_Open'] = 0.0
            self.df['HA_High'] = 0.0
            self.df['HA_Low'] = 0.0

            # Initialize first HA_Open
            self.df.at[self.df.index[0], 'HA_Open'] = (self.df.at[self.df.index[0], 'Open'] + self.df.at[self.df.index[0], 'Close']) / 2

            # Calculate remaining Heiken-Ashi values
            for i in range(1, len(self.df)):
                prev_ha_open = self.df.at[self.df.index[i - 1], 'HA_Open']
                prev_ha_close = self.df.at[self.df.index[i - 1], 'HA_Close']

                self.df.at[self.df.index[i], 'HA_Open'] = (prev_ha_open + prev_ha_close) / 2
                self.df.at[self.df.index[i], 'HA_High'] = max(
                    self.df.at[self.df.index[i], 'High'],
                    self.df.at[self.df.index[i], 'HA_Open'],
                    self.df.at[self.df.index[i], 'HA_Close']
                )
                self.df.at[self.df.index[i], 'HA_Low'] = min(
                    self.df.at[self.df.index[i], 'Low'],
                    self.df.at[self.df.index[i], 'HA_Open'],
                    self.df.at[self.df.index[i], 'HA_Close']
                )

            # Replace original OHLC columns with Heiken-Ashi values
            self.df.drop(columns=['Open', 'High', 'Low', 'Close'], inplace=True)
            self.df.rename(columns={
                'HA_Open': 'Open',
                'HA_High': 'High',
                'HA_Low': 'Low',
                'HA_Close': 'Close'
            }, inplace=True)
        except Exception as e:
            print(f"Error in heiken-ashi calculation : {e}")

    def calculate_signals(self):
        try:
            self.df.rename(columns={'close':'Close','open':'Open','high':'High','low':'Low','volume':'Volume'},inplace=True)
            # self.calculate_heiken_ashi()
            # self.df = self.Grsi.calculate_signals(self.df,rsi_length=10,smooth_length=3,overbought_level=60,oversold_level=40)
            self.rsi()
            self.df = self.rf.run_filter(self.df)
            self.df['gaizy_color'] = self.Grsi.calculate_signals(df=self.df)
            self.df['rsi'],self.df['rsi_buy'],self.df['rsi_sell'] = self.bsrsi.generate_signals(self.df['Close'])
            self.df.rename(columns={'Close':'close','Open':'open','High':'high','Low':'low','Volume':'volume'},inplace=True)
            self.df,_ = calculate_inside_bar_boxes(self.df)
            columns_to_drop = [
                'RF_UpperBand', 'RF_LowerBand', 'RF_Filter', 'RF_Trend',
                'IsIB', 'BoxHigh', 'BoxLow', 'BarColor'
            ]
            self.df = self.df.drop(columns=columns_to_drop)
            # print(self.df)
            print(self.df.tail(1))
            self.df.to_csv('ETHUSD_Indicator.csv')
        except Exception as e:
            print(f"Error occured in calculating the signal : {e}")

    def execute_signals(self):
        try:
            self.df = self.df.tail(600)
            self.df['Signal_Final'] = 0

            # Initialize signal tracking variables
            last_rf_buy_signal = False
            last_rf_sell_signal = False
            rf_signal_candle = -1
            rf_used = False

            pending_rsi_buy = False
            pending_rsi_sell = False
            rsi_signal_candle = -1

            # Track used RSI_Gaizy lines to ensure only one trade per color line
            used_gaizy_green = False
            used_gaizy_red = False
            used_gaizy_pink = False

            for i in range(len(self.df)):
                row = self.df.iloc[i]
                prev_row = self.df.iloc[i - 1] if i > 0 else None

                # Detect new RF signals (transition from 0 to 1)
                current_rf_buy = row['RF_BuySignal'] == 1
                current_rf_sell = row['RF_SellSignal'] == 1

                new_rf_buy = current_rf_buy and (prev_row is None or prev_row['RF_BuySignal'] != 1)
                new_rf_sell = current_rf_sell and (prev_row is None or prev_row['RF_SellSignal'] != 1)

                # Update RF signal tracking
                if new_rf_buy:
                    last_rf_buy_signal = True
                    last_rf_sell_signal = False
                    rf_signal_candle = i
                    rf_used = False
                elif new_rf_sell:
                    last_rf_sell_signal = True
                    last_rf_buy_signal = False
                    rf_signal_candle = i
                    rf_used = False

                # Detect RSI_Gaizy color changes
                current_gaizy = row['gaizy_color']
                gaizy_changed = prev_row is not None and prev_row['gaizy_color'] != current_gaizy

                # Reset color usage flags when new color appears
                if gaizy_changed:
                    if current_gaizy in ['bright_green', 'dark_green']:
                        used_gaizy_green = False
                    elif current_gaizy in ['red']:
                        used_gaizy_red = False
                    elif current_gaizy in ['pink']:
                        used_gaizy_pink = False

                signal = 0  # Default

                # === 1. RSI_Gaizy Integration + IB_box ===
                # Each RSI_Gaizy color line can trigger only one trade
                if current_gaizy in ['bright_green', 'dark_green'] and not used_gaizy_green:
                    # Green line → Triggers Green Box trade only
                    if row['GreenArrow'] == 1:
                        signal = 1
                        used_gaizy_green = True
                elif current_gaizy == 'red' and not used_gaizy_red:
                    # Red line → Triggers Red Box trade only
                    if row['RedArrow'] == 1:
                        signal = -1
                        used_gaizy_red = True
                elif current_gaizy == 'pink' and not used_gaizy_pink:
                    # Pink strong sell → Triggers Red Box trade only
                    if row['RedArrow'] == 1:
                        signal = -1
                        used_gaizy_pink = True
                elif current_gaizy == 'black':
                    # Black signal → Take trade based on IB box
                    if row['GreenArrow'] == 1:
                        signal = 1
                    elif row['RedArrow'] == 1:
                        signal = -1

                # === 2. Range Filter (RF) + IB_Box Confirmation ===
                # Only one trade per RF signal is allowed
                
                # Condition 1: RF and IB box signals in same candle
                if new_rf_buy and row['GreenArrow'] == 1 and not rf_used:
                    signal = 2
                    rf_used = True
                    last_rf_buy_signal = False
                elif new_rf_sell and row['RedArrow'] == 1 and not rf_used:
                    signal = -2
                    rf_used = True
                    last_rf_sell_signal = False
                
                # Condition 2: RF signal, then IB box signal in immediate next candle
                elif last_rf_buy_signal and not rf_used and row['GreenArrow'] == 1 and (i - rf_signal_candle) == 1:
                    signal = 2
                    rf_used = True
                    last_rf_buy_signal = False
                elif last_rf_sell_signal and not rf_used and row['RedArrow'] == 1 and (i - rf_signal_candle) == 1:
                    signal = -2
                    rf_used = True
                    last_rf_sell_signal = False

                # === 3. RSI Buy/sell + RF Logic ===
                # RSI signal and RF confirmation (same candle or next candle)
                
                # Track RSI signals
                if row['rsi_buy'] == 1:
                    pending_rsi_buy = True
                    rsi_signal_candle = i
                elif row['rsi_sell'] == 1:
                    pending_rsi_sell = True
                    rsi_signal_candle = i

                # Check for RF confirmation after RSI signal
                if pending_rsi_buy and current_rf_buy and (i - rsi_signal_candle) <= 1 and not rf_used:
                    signal = 3
                    pending_rsi_buy = False
                    rf_used = True
                    last_rf_buy_signal = False
                elif pending_rsi_sell and current_rf_sell and (i - rsi_signal_candle) <= 1 and not rf_used:
                    signal = -3
                    pending_rsi_sell = False
                    rf_used = True
                    last_rf_sell_signal = False

                # Reset pending RSI signals if too much time has passed (more than 2 candles)
                if pending_rsi_buy and (i - rsi_signal_candle) > 2:
                    pending_rsi_buy = False
                if pending_rsi_sell and (i - rsi_signal_candle) > 2:
                    pending_rsi_sell = False

                # Assign the final signal
                self.df.iat[i, self.df.columns.get_loc('Signal_Final')] = signal

            self.df.to_csv('ETHUSD_Final.csv')

            # === Execute Latest Signal ===
            # last_candle = self.df.iloc[-1]
            last_candle = self.df.iloc[-1]
            last_signal = last_candle['Signal_Final']
            print(f"the current order id is {self.current_order_id}")
            print(f"the last signal is {last_signal}")

            if last_signal != 0 and self.h_pos == 0:
                print(f"a new order would be placed since last signal is {last_signal} and current position is {self.h_pos}")
                current_price = float(last_candle['close'])
                self.last_price = current_price
                side = "buy" if last_signal > 0 else "sell"
                if side == "buy":
                    self.h_pos = 1
                    print(f"self.h_pos changed to {self.h_pos} in line 659")
                elif side == "sell":
                    self.h_pos = -1
                    print(f"self.h_pos changed to {self.h_pos} in line 661")

                sl_price = self.calculate_stoploss(current_price, side, self.df)
                tp_price = self.calculate_takeprofit(current_price, side)
                self.last_sl_price = sl_price
                self.last_tp_price = tp_price

                if sl_price is not None and tp_price is not None:
                    stop_limit = sl_price + 10 if side == "buy" else sl_price + 10
                    tp_limit = tp_price - 10 if side == "buy" else tp_price + 10
                    self.set_leverage_delta(value=self.base_leverage,product_id="1699")
                    self.leverage_check()
                    self.place_order_market(side=side, size=self.base)
                    # import time
                    # time.sleep(1) # sleep for 1 seconds

                    self.place_order_bracket_limit(
                        limit_price=str(current_price),
                        stop_price=str(sl_price),
                        take_profit_price=str(tp_price),
                        stop_limit_price=str(stop_limit),
                        take_profit_limit_price=str(tp_limit),
                        side=side,
                        size=self.base
                    )

                # self.dynamic_order_check()

        except Exception as e:
            print(f"Error occurred in execution: {e}")
            import traceback
            traceback.print_exc()
   
    def dynamic_order_check(self):
        """Dynamic position management with signal validity check"""
        try:
            count = 0
            while count < 3:
                try:
                    print("Function in try block")
                    
                    # Refresh data to check current conditions
                    self.fetch_data()
                    self.calculate_signals()
                    self.execute_signals()
                    
                    # last_candle = self.df.iloc[-1]
                    last_candle = self.df.iloc[-1]
                    print(last_candle)
                    if last_candle['Signal_Final'] != 0:
                        print("Signal still valid, keeping order")
                    elif last_candle['Signal_Final'] == 0:
                        print("Signal conditions no longer met, cancelling order")
                        if self.current_order_id:
                            self.cancel_order(order_id=str(self.current_order_id),product_id="1699")
                            print(f"self.h_pos resetted to 0 in line 712")
                            self.h_pos = 0
                        break
                    
                    # Check order status
                    if self.current_order_id:
                        self.order_status(self.current_order_id)
                    
                    count += 1
                    time.sleep(10)
                    
                except Exception as e:
                    print(f"Error in dynamic order_check loop : {e}")
                    count += 1
            
            # If order not filled after 3 attempts, convert to bracket order
            if count >= 3 and self.current_order_id:
                try:
                    self.order_status(order_id=self.current_order_id)
                except Exception as e:
                    print(f"error in dynamic order check after 3 counts {e}")
        except Exception as e:
            print(f"Error in dynamic order check function : {e}")
    
    def run(self):
        try:
            if self.h_pos == 0:
                print(f"Starting bot at: {self.get_current_datetime()}")
                self.connect()
                self.fetch_data()
                self.calculate_signals()
                self.execute_signals()
                if self.current_order_id:
                    self.dynamic_order_check()
            else:
                print(f"Currently holding a position")
                self.connect()
                self.order_status(self.current_order_id)
                if self.current_order_id:
                    self.dynamic_order_check()
        except Exception as e:
            print(f"Error occured in run : {e}")
        import time
        time.sleep(10)
    
    def test(self):
        try:
            import time
            time.sleep(5)
            self.connect()
            self.get_usd_balance()
            pass
        except Exception as e:
            print(f"Exception in test run {e}")

if __name__ == "__main__":
    api_key = "MZvQgXnmlxSqWnfJRRcvBymh6FFqs5"
    api_secret = "nqERtehNDx4JAyfA8pRiuG3VsCZOor6dF33Ttp8SH7XWObo9D2ZnqVflsoMz"
    base_url = "https://cdn-ind.testnet.deltaex.org"
    Debot = DeltaExchangeRFGaizy(api_key=api_key,api_secret=api_secret,base_url=base_url)
    while True:
        try:
            # Debot.run()
            Debot.test()
            # Debot.test2()
        except Exception as e:
            print(f"Error in run : {e}")