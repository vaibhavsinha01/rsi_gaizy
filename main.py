
from delta_rest_client import DeltaRestClient
import pandas as pd
import numpy as np
from module.rf import RangeFilter
from module.ib_indicator import calculate_inside_ib_box
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
        self.base_leverage = 2
        self.leverage_multiplier = 2
        self.max_leverage = 4
        self.last_price = None
        self.flag = 0
        self.base = 1
        self.account_balance = None
        self.min_lot = 0.01
        self.heikan_choice = 1
        self.last_trade_status = None
        self.previous_position_size = 0
        self.trade_entry_price = None
        self.cooldown_time = 1800

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
                sl_50_points = entry_price - 10
            else:  # sell
                sl_1_percent = entry_price * 1.01
                sl_50_points = entry_price + 10
            
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
    import time
    import json
    import requests

    def cancel_order(self, order_id, product_id):
        try:
            print(f"order id is {int(order_id)} and product_id is {int(product_id)}")
            payload = {
                "id":int(order_id),
                "product_id":int(product_id)
            }

            method = 'DELETE'
            path = '/v2/orders'
            url = self.base_url + path
            timestamp = str(int(time.time()))
            query_string = ''
            payload_json = json.dumps(payload, separators=(',', ':'))  # compact format
            signature_data = method + timestamp + path + query_string + payload_json
            signature = self.generate_signature(self.api_secret, signature_data)

            # Headers
            headers = {
                "api-key": self.api_key,
                "timestamp": timestamp,
                "signature": signature,
                "User-Agent": "python-rest-client",
                "Content-Type": "application/json"
            }

            # Send DELETE request with JSON body
            response = requests.delete(url, headers=headers, data=payload_json)

            # Output
            print("Response Code:", response.status_code)
            try:
                response_json = response.json()
                print("Response Body:", response_json)
                return response_json
            except ValueError:
                print("Raw Response Text:", response.text)
                return None

        except Exception as e:
            print("Error in cancel_order function:", e)
            return None
    
    def cancel_all_orders(self, product_id=None, contract_types=None,cancel_limit_orders=False, cancel_stop_orders=False, cancel_reduce_only_orders=False):
        try:
            path = '/v2/orders/all'
            url = self.base_url + path
            method = 'DELETE'
            timestamp = str(int(time.time()))
            query_string = ''

            payload = {}
            if product_id is not None:
                payload["product_id"] = int(product_id)
            if contract_types is not None:
                # contract_types is comma separated string e.g. "perpetual_futures,put_options"
                payload["contract_types"] = contract_types
            payload["cancel_limit_orders"] = cancel_limit_orders
            payload["cancel_stop_orders"] = cancel_stop_orders
            payload["cancel_reduce_only_orders"] = cancel_reduce_only_orders

            payload_json = json.dumps(payload, separators=(',', ':'))
            signature_data = method + timestamp + path + query_string + payload_json
            signature = self.generate_signature(self.api_secret, signature_data)

            headers = {
                "api-key": self.api_key,
                "timestamp": timestamp,
                "signature": signature,
                "User-Agent": "python-rest-client",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            response = requests.delete(url, headers=headers, data=payload_json)

            print("Response Code:", response.status_code)
            try:
                response_json = response.json()
                print("Response Body:", response_json)
                return response_json
            except ValueError:
                print("Raw Response Text:", response.text)
                return None

        except Exception as e:
            print("Error in cancel_all_orders function:", e)
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
                    print(f"self.flag is {self.flag} in line 338")
                elif self.flag == -1:
                    print(f"self.flag is {self.flag} in line 341")
                    self.leverage_check()
                else:
                    print("Order is still open")

        except Exception as e:
            print(f"Error in order_status function : {e}")

    def get_market_price(self):
        last = self.df.iloc[-1]
        return last['close']
    
    def get_active_positions_status(self):
        """Get active positions from Delta Exchange for ETHUSD (product_id: 1699)"""
        try:
            method = "GET"
            path = "/v2/positions"
            url = self.base_url + path
            timestamp = str(int(time.time()))
            
            # Build query parameters for BTCUSD
            params = {'product_id': '1699'}
            
            # Build query string
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            
            # Create signature data - ADD '?' before query_string for GET requests with params
            signature_data = method + timestamp + path + '?' + query_string
            signature = self.generate_signature(self.api_secret, signature_data)
            
            headers = {
                "api-key": self.api_key,
                "timestamp": timestamp,
                "signature": signature,
                "User-Agent": "python-rest-client"
            }
            
            # Make the request with query parameters
            response = requests.get(url, headers=headers, params=params)
            
            positions_data = response.json()
            
            # Print the full response for debugging
            print("Active Positions Response:", positions_data)
            print(f"Current size of positions is: {positions_data['result']['size']}")

            if int(abs(positions_data['result']['size']))>0:
                return False
            else:
                return True
                
        except Exception as e:
            print(f"Error getting active positions: {e}")
            return None

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
        
    def get_base_margin_size(self):
        try:
            if hasattr(self,'df'):
                if self.df is not None:
                    self.account_balance = float(self.get_usd_balance())
                    self.base_price = float(self.get_market_price()) * self.min_lot
                    self.margin = self.account_balance/self.base_price
                    self.base = self.margin/2 # use something like kelly's criteria
                    print(f"You can trade for base_size {int(self.margin)} without using any leverage and {int(self.margin*self.base_leverage)} with base leverage")
            else:
                self.df = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\delta\data\ETHUSD_Final_main.csv")
                self.account_balance = float(self.get_usd_balance())
                self.base_price = float(self.get_market_price()) * self.min_lot
                self.margin = self.account_balance/self.base_price
                self.base = self.margin/2  # use something like the kelly's criteria's part
                print(f"You can trade for base_size {int(self.margin)} without using any leverage and {int(self.margin*self.base_leverage)} with base leverage")
        except Exception as e:
            print(f"Exception in get_base_margin_size")

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
            self.df.to_csv("data/ETHUSD_Indicator_main.csv")
                
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
            if self.heikan_choice == 1:
                self.calculate_heiken_ashi()
            self.df = self.rf.run_filter(self.df)
            # self.df['gaizy_color'] = self.Grsi.calculate_signals(df=self.df)
            self.df['rsi'],self.df['rsi_buy'],self.df['rsi_sell'] = self.bsrsi.generate_signals(self.df['Close'])
            self.df.rename(columns={'Close':'close','Open':'open','High':'high','Low':'low','Volume':'volume'},inplace=True)
            self.df['gaizy_color'] = self.Grsi.calculate_gainzy_colors(df=self.df)
            # self.df,_ = calculate_inside_bar_boxes(self.df)
            self.df = calculate_inside_ib_box(self.df)
            columns_to_drop = [
                'RF_UpperBand', 'RF_LowerBand', 'RF_Filter', 'RF_Trend',
                'IsIB', 'BoxHigh', 'BoxLow', 'BarColor','rsi'
            ]
            self.df = self.df.drop(columns=columns_to_drop)
            # print(self.df)
            print(self.df.tail(1))
            # self.df.to_csv('ETHUSD_Indicator.csv')
            self.df.to_csv("data/ETHUSD_Indicator_main.csv")
            current_position_size = self.get_current_position_size()  # New method needed
            
            # Detect if position just closed (was non-zero, now zero)
            if self.previous_position_size != 0 and current_position_size == 0:
                print("Position closed - checking win/loss status")
                self.check_last_trade_result()
                self.adjust_leverage_based_on_result()
            
            # Update previous position size for next iteration
            self.previous_position_size = current_position_size
            
            # Only proceed with new trades if no active position
            if current_position_size != 0:
                print("Active position exists, skipping new signals")
                return
                
            self.df = self.df.tail(200)
            self.df['Signal_Final'] = 0

            # Initialize signal tracking variables
            last_rf_buy_signal = False
            last_rf_sell_signal = False
            rf_signal_candle = -1
            rf_used = False

            # Initialize Arrow signal tracking variables (added from second code)
            last_green_arrow = False
            last_red_arrow = False
            arrow_signal_candle = -1
            arrow_used = False

            pending_rsi_buy = False
            pending_rsi_sell = False
            rsi_signal_candle = -1

            # Track used RSI_Gaizy lines to ensure only one trade per color line
            used_gaizy_green = False
            used_gaizy_red = False
            used_gaizy_pink = False
            used_gaizy_black = False  # Added this line
            used_gaizy_blue = False

            for i in range(len(self.df)):
                row = self.df.iloc[i]
                prev_row = self.df.iloc[i - 1] if i > 0 else None

                # Detect new RF signals (transition from 0 to 1)
                current_rf_buy = row['RF_BuySignal'] == 1
                current_rf_sell = row['RF_SellSignal'] == 1

                new_rf_buy = current_rf_buy and (prev_row is None or prev_row['RF_BuySignal'] != 1)
                new_rf_sell = current_rf_sell and (prev_row is None or prev_row['RF_SellSignal'] != 1)

                # Detect new Arrow signals (added from second code)
                new_green_arrow = row['GreenArrow'] == 1 and (prev_row is None or prev_row['GreenArrow'] != 1)
                new_red_arrow = row['RedArrow'] == 1 and (prev_row is None or prev_row['RedArrow'] != 1)

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

                # Update Arrow signal tracking (added from second code)
                if new_green_arrow:
                    last_green_arrow = True
                    last_red_arrow = False
                    arrow_signal_candle = i
                    arrow_used = False
                elif new_red_arrow:
                    last_red_arrow = True
                    last_green_arrow = False
                    arrow_signal_candle = i
                    arrow_used = False

                # Reset RF signals if they're older than 1 candle and not used
                if (i - rf_signal_candle) > 1 and (last_rf_buy_signal or last_rf_sell_signal):
                    last_rf_buy_signal = False
                    last_rf_sell_signal = False
                    rf_used = False

                # Reset Arrow signals if they're older than 1 candle and not used
                if (i - arrow_signal_candle) > 1 and (last_green_arrow or last_red_arrow):
                    last_green_arrow = False
                    last_red_arrow = False
                    arrow_used = False

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
                    elif current_gaizy == 'blue':
                        used_gaizy_blue = False
                    elif current_gaizy == 'black':  # Added this condition
                        used_gaizy_black = False

                signal = 0  # Default

                # === PRIORITY 1: Range Filter (RF) + IB_Box Confirmation ===
                # This gets highest priority to ensure it's fulfilled
                
                # Scenario A: RF signal first, then Arrow signal
                # Condition A1: RF and Arrow signals in same candle
                if new_rf_buy and row['GreenArrow'] == 1 and not rf_used:
                    signal = 2  # RF + IB_Box buy signal
                    rf_used = True
                    last_rf_buy_signal = False
                    # print(f"RF + IB_Box BUY signal triggered at candle {i} (same candle)")
                elif new_rf_sell and row['RedArrow'] == 1 and not rf_used:
                    signal = -2  # RF + IB_Box sell signal
                    rf_used = True
                    last_rf_sell_signal = False
                    # print(f"RF + IB_Box SELL signal triggered at candle {i} (same candle)")
                
                # Condition A2: RF signal, then Arrow signal in immediate next candle
                elif last_rf_buy_signal and not rf_used and row['GreenArrow'] == 1 and (i - rf_signal_candle) == 1:
                    signal = 2  # RF + IB_Box buy signal
                    rf_used = True
                    last_rf_buy_signal = False
                    # print(f"RF + IB_Box BUY signal triggered at candle {i} (RF at {rf_signal_candle}, Arrow at {i})")
                elif last_rf_sell_signal and not rf_used and row['RedArrow'] == 1 and (i - rf_signal_candle) == 1:
                    signal = -2  # RF + IB_Box sell signal
                    rf_used = True
                    last_rf_sell_signal = False
                    # print(f"RF + IB_Box SELL signal triggered at candle {i} (RF at {rf_signal_candle}, Arrow at {i})")

                # Scenario B: Arrow signal first, then RF signal
                # Condition B1: Arrow signal, then RF signal in immediate next candle
                elif last_green_arrow and not arrow_used and new_rf_buy and (i - arrow_signal_candle) == 1:
                    signal = 2  # RF + IB_Box buy signal
                    arrow_used = True
                    last_green_arrow = False
                    # print(f"RF + IB_Box BUY signal triggered at candle {i} (Arrow at {arrow_signal_candle}, RF at {i})")
                elif last_red_arrow and not arrow_used and new_rf_sell and (i - arrow_signal_candle) == 1:
                    signal = -2  # RF + IB_Box sell signal
                    arrow_used = True
                    last_red_arrow = False
                    # print(f"RF + IB_Box SELL signal triggered at candle {i} (Arrow at {arrow_signal_candle}, RF at {i})")

                # === PRIORITY 2: RSI_Gaizy Integration + IB_box ===
                # Only execute if no RF + IB_Box signal was triggered
                elif signal == 0:
                    # Each RSI_Gaizy color line can trigger only one trade
                    if current_gaizy in ['light_green', 'green'] and not used_gaizy_green:
                        # Green line → Triggers Green Box trade only
                        if row['GreenArrow'] == 1:
                            signal = 1
                            used_gaizy_green = True
                            # print(f"RSI_Gaizy GREEN + IB_Box signal triggered at candle {i}")
                    elif current_gaizy == 'red' and not used_gaizy_red:
                        # Red line → Triggers Red Box trade only
                        if row['RedArrow'] == 1:
                            signal = -1
                            used_gaizy_red = True
                            # print(f"RSI_Gaizy RED + IB_Box signal triggered at candle {i}")
                    elif current_gaizy == 'pink' and not used_gaizy_pink:
                        # Pink strong sell → Triggers Red Box trade only
                        if row['RedArrow'] == 1:
                            signal = -1
                            used_gaizy_pink = True
                            # print(f"RSI_Gaizy PINK + IB_Box signal triggered at candle {i}")
                    elif current_gaizy == 'blue' and not used_gaizy_blue:
                        if row['RedArrow'] == 1:
                            signal = 1
                            used_gaizy_blue = True
                        elif row['GreenArrow'] == 1:
                            signal = -1
                            used_gaizy_blue = False
                    elif current_gaizy == 'black' and not used_gaizy_black:  # Added usage check
                        # Black signal → Take trade based on IB box
                        if row['GreenArrow'] == 1:
                            signal = 1
                            used_gaizy_black = True  # Mark as used
                            # print(f"RSI_Gaizy BLACK + Green IB_Box signal triggered at candle {i}")
                        elif row['RedArrow'] == 1:
                            signal = -1
                            used_gaizy_black = True  # Mark as used
                            # print(f"RSI_Gaizy BLACK + Red IB_Box signal triggered at candle {i}")

                # Mark RSI_Gaizy colors as used when ANY signal is triggered (including RF + IB_Box)
                if signal != 0:
                    if current_gaizy in ['bright_green', 'dark_green']:
                        used_gaizy_green = True
                    elif current_gaizy == 'red':
                        used_gaizy_red = True
                    elif current_gaizy == 'pink':
                        used_gaizy_pink = True
                    elif current_gaizy == 'blue':
                        used_gaizy_blue = True
                    elif current_gaizy == 'black':
                        used_gaizy_black = True

                # === PRIORITY 3: RSI Buy/sell + RF Logic ===
                # Only execute if no higher priority signal was triggered
                if signal == 0:
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
                        # print(f"RSI + RF BUY signal triggered at candle {i}")
                    elif pending_rsi_sell and current_rf_sell and (i - rsi_signal_candle) <= 1 and not rf_used:
                        signal = -3
                        pending_rsi_sell = False
                        rf_used = True
                        last_rf_sell_signal = False
                        # print(f"RSI + RF SELL signal triggered at candle {i}")

                    # Reset pending RSI signals if too much time has passed (more than 2 candles)
                    if pending_rsi_buy and (i - rsi_signal_candle) > 2:
                        pending_rsi_buy = False
                    if pending_rsi_sell and (i - rsi_signal_candle) > 2:
                        pending_rsi_sell = False

                # Assign the final signal
                self.df.iat[i, self.df.columns.get_loc('Signal_Final')] = signal

            # self.df.to_csv('ETHUSD_Final.csv')
            self.df.to_csv("data/ETHUSD_Final_main.csv")
        except Exception as e:
            print(f"Error occured in calculating the signal : {e}")
    
    def get_current_position_size(self):
        """Get current position size (modify from existing get_active_positions)"""
        try:
            method = "GET"
            path = "/v2/positions"
            url = self.base_url + path
            timestamp = str(int(time.time()))
            
            params = {'product_id': '1699'}
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature_data = method + timestamp + path + '?' + query_string
            signature = self.generate_signature(self.api_secret, signature_data)
            
            headers = {
                "api-key": self.api_key,
                "timestamp": timestamp,
                "signature": signature,
                "User-Agent": "python-rest-client"
            }
            
            response = requests.get(url, headers=headers, params=params)
            positions_data = response.json()
            
            return int(abs(float(positions_data['result']['size'])))
            
        except Exception as e:
            print(f"Error getting position size: {e}")
            return 0

    def check_last_trade_result(self):
        """Check if last trade was win or loss based on current price vs TP/SL"""
        try:
            if self.last_tp_price is None or self.last_sl_price is None or self.trade_entry_price is None:
                print("Missing price data for win/loss calculation")
                return
                
            current_price = self.get_market_price()
            
            # Calculate distances to TP and SL
            distance_to_tp = abs(current_price - self.last_tp_price)
            distance_to_sl = abs(current_price - self.last_sl_price)
            
            # Determine win/loss based on which is closer
            if distance_to_tp < distance_to_sl:
                self.last_trade_status = 1  # Win
                print(f"Last trade was a WIN. Current price {current_price} closer to TP {self.last_tp_price}")
            else:
                self.last_trade_status = 0  # Loss
                print(f"Last trade was a LOSS. Current price {current_price} closer to SL {self.last_sl_price}")
                
        except Exception as e:
            print(f"Error in check_last_trade_result: {e}")

    def adjust_leverage_based_on_result(self):
        """Adjust leverage based on last trade result"""
        try:
            if self.last_trade_status == 1:  # Win
                print("Last trade was profitable - keeping same leverage")
                # Keep current leverage unchanged
            elif self.last_trade_status == 0:  # Loss
                print("Last trade was loss - doubling leverage")
                if self.base_leverage < self.max_leverage:
                    self.base_leverage = self.base_leverage * 2
                    print(f"Leverage increased to {self.base_leverage}")
                else:
                    print(f"Max leverage {self.max_leverage} reached, resetting to base")
                    self.base_leverage = 2  # Reset to base leverage
                    
        except Exception as e:
            print(f"Error in adjust_leverage_based_on_result: {e}")

    def execute_signals(self):
        try:
            # === Execute Latest Signal ===
            last_candle = self.df.iloc[-1]
            last_signal = last_candle['Signal_Final']
            print(f"the current order id is {self.current_order_id}")
            print(f"the last signal is {last_signal}")

            if last_signal != 0:
                print(f"a new order would be placed since last signal is {last_signal}")
                current_price = float(last_candle['close'])
                self.last_price = current_price
                self.trade_entry_price = current_price  # Store entry price for win/loss calculation
                side = "buy" if last_signal > 0 else "sell"

                sl_price = self.calculate_stoploss(current_price, side, self.df)
                tp_price = self.calculate_takeprofit(current_price, side)
                self.last_sl_price = sl_price
                self.last_tp_price = tp_price

                if sl_price is not None and tp_price is not None:
                    stop_limit = sl_price + 10 if side == "buy" else sl_price + 10
                    tp_limit = tp_price - 10 if side == "buy" else tp_price + 10
                    self.set_leverage_delta(value=self.base_leverage,product_id="1699")
                    # self.leverage_check()
                    self.get_base_margin_size() # updates the self.base size
                    self.place_order_market(side=side, size=int(self.base))
                    # import time
                    # time.sleep(1) # sleep for 1 seconds

                    self.place_order_bracket_limit(
                        limit_price=str(current_price),
                        stop_price=str(sl_price),
                        take_profit_price=str(tp_price),
                        stop_limit_price=str(stop_limit),
                        take_profit_limit_price=str(tp_limit),
                        side=side,
                        size=int(self.base)
                    )

        except Exception as e:
            print(f"Error occurred in execution: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        try:
            if self.get_active_positions_status():
                print(f"Starting bot at: {self.get_current_datetime()}")
                self.connect()
                self.fetch_data()
                self.calculate_signals()
                last_candle = self.df.iloc[-1]
                last_signal = last_candle['Signal_Final']
                import time
                if last_signal != 0:
                    self.execute_signals()
                    print(f"trade executed sleeping for {self.cooldown_time}")
                    time.sleep(self.cooldown_time)
                else:
                    time.sleep(5)
                import time
                time.sleep(10)

        except Exception as e:
            print(f"Error occured in run : {e}")
    
    def test(self):
        try:
            import time
            time.sleep(5)
            self.connect()
            # self.cancel_order(order_id=640925488,product_id=1699)
            self.cancel_all_orders(product_id=1699)
            # self.get_usd_balance()
            # self.get_base_margin_size()
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
            Debot.run()
        except Exception as e:
            print(f"Error in run : {e}")