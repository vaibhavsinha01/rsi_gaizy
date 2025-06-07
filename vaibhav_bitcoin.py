
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
        self.base_leverage = 100
        self.leverage_multiplier = 1
        self.max_leverage = 10
        self.last_price = None
        self.flag = 0
        self.base = 1
        self.account_balance = None
        self.min_lot = 0.001
        self.heikan_choice = 1

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

    def calculate_stoploss(self,entry_price,side):
        try:
            if side == "buy":
                # sl = entry_price * 0.99
                sl = entry_price - 150
            else:
                # sl = entry_price * 1.01
                sl = entry_price + 150
            print(f"Calculated stop loss for {side}: {sl}")
            return sl
        
        except Exception as e:
            print(f"Error occured in calculate_stoploss : {e}")

    def calculate_takeprofit(self, entry_price, side):
        """Calculate take profit based on risk-reward ratio"""
        try:
            if side == "buy":
                # tp = entry_price * 1.01  # 2% profit
                tp = entry_price + 300
            else:  # sell
                # tp = entry_price * 0.99  # 2% profit
                tp = entry_price - 300
            
            print(f"Calculated take profit for {side}: {tp}")
            return tp
            
        except Exception as e:
            print(f"Error in calculate_takeprofit: {e}")
            return None
    
    def leverage_check(self):
        if self.base_leverage<self.max_leverage:
            # self.base_leverage = self.base_leverage * self.leverage_multiplier
            self.base_leverage = 10
        else:
            self.base_leverage = 10

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
                "product_symbol":"BTCUSD",
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
                "product_symbol":"BTCUSD",
                "limit_price": "100000",
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
    
    def place_order_bracket_limit(self,limit_price="100000",stop_price="99000",stop_limit_price="98000",take_profit_price="102000",take_profit_limit_price="101000",side="buy",size=1):
        try:
            payload = {
            "product_symbol": "BTCUSD",
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
                    print(f"self.flag is {self.flag}")
                elif self.flag == -1:
                    print(f"self.flag is {self.flag}")
                    self.leverage_check()
                else:
                    print("Order is still open")

        except Exception as e:
            print(f"Error in order_status function : {e}")

    def get_market_price(self):
        last = self.df.iloc[-1]
        return last['close']
    
    def get_active_orders(self):
        """Get active orders from Delta Exchange"""
        try:
            method = "GET"
            path = "/v2/orders"
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
            orders_data = response.json()
            
            # Print the full response for debugging
            print("Active Orders Response:", orders_data)
            
            # Return the orders data
            if 'result' in orders_data:
                return orders_data['result']
            else:
                return orders_data
                
        except Exception as e:
            print(f"Error getting active orders: {e}")
            return None
        
    def get_active_positions_bitcoin(self):
        """Get active positions from Delta Exchange for BTCUSD (product_id: 84)"""
        try:
            method = "GET"
            path = "/v2/positions"
            url = self.base_url + path
            timestamp = str(int(time.time()))
            
            # Build query parameters for BTCUSD
            params = {'product_id': '84'}
            
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
        
    def get_active_positions(self):
        """Get active positions from Delta Exchange for ETHUSD (product_id: 1699)"""
        try:
            method = "GET"
            path = "/v2/positions"
            url = self.base_url + path
            timestamp = str(int(time.time()))
            
            # Build query parameters for ETHUSD
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
            # print("Active Positions Response:", positions_data)
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
                    self.base = self.margin*self.leverage_multiplier/2 # use something like kelly's criteria
                    print(f"You can trade for base_size {int(self.margin)} without using any leverage and {int(self.margin*self.base_leverage)} with base leverage")
            else:
                self.df = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\delta\BTCUSD_Final.csv")
                self.account_balance = float(self.get_usd_balance())
                self.base_price = float(self.get_market_price()) * self.min_lot
                self.margin = self.account_balance/self.base_price
                self.base = self.margin*self.base_leverage/2  # use something like the kelly's criteria's part
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
            
            self.df = self.broker.get_ticker_data(symbol='BTCUSD') # current time / other things need to be accounted for
            
            self.df['Timestamp'] = pd.to_datetime(self.df['time'],unit='s')
            self.df.sort_values(by='Timestamp',ascending=False,inplace=True)
            self.df = self.df.iloc[::-1].reset_index(drop=True)
            self.df.to_csv("BTCUSD_Indicator.csv")
                
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
            product_id (int or str): The Delta Exchange product ID (e.g., 84 for BTCUSD).
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
            self.df.rename(columns={'Close':'close','Open':'open','High':'high','Low':'low','Volume':'volume'},inplace=True)
            self.df = calculate_inside_ib_box(self.df)
            columns_to_drop = [
                'RF_UpperBand', 'RF_LowerBand', 'RF_Filter', 'RF_Trend',
                'IsIB', 'BoxHigh', 'BoxLow', 'BarColor'
            ]
            self.df = self.df.drop(columns=columns_to_drop)
            # print(self.df)
            print(self.df.tail(1))
            self.df.to_csv('BTCUSD_Indicator.csv')
        except Exception as e:
            print(f"Error occured in calculating the signal : {e}")

    def execute_signals(self):
        try:
            self.df = self.df.tail(3000)
            self.df['Signal_Final'] = 0

            # Initialize RF signal tracking variables
            last_rf_buy_signal = False
            last_rf_sell_signal = False
            rf_signal_candle = -1
            rf_used = False

            # Initialize Arrow signal tracking variables
            last_green_arrow = False
            last_red_arrow = False
            arrow_signal_candle = -1
            arrow_used = False

            for i in range(len(self.df)):
                row = self.df.iloc[i]
                prev_row = self.df.iloc[i - 1] if i > 0 else None

                # Detect new RF signals (transition from 0 to 1)
                current_rf_buy = row['RF_BuySignal'] == 1
                current_rf_sell = row['RF_SellSignal'] == 1

                new_rf_buy = current_rf_buy and (prev_row is None or prev_row['RF_BuySignal'] != 1)
                new_rf_sell = current_rf_sell and (prev_row is None or prev_row['RF_SellSignal'] != 1)

                # Detect new Arrow signals
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

                # Update Arrow signal tracking
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
                elif (i - rf_signal_candle) > 1 and (last_rf_buy_signal or last_rf_sell_signal):
                    last_rf_buy_signal = False
                    last_rf_sell_signal = False
                    rf_used = False

                # Reset Arrow signals if they're older than 1 candle and not used
                elif (i - arrow_signal_candle) > 1 and (last_green_arrow or last_red_arrow):
                    last_green_arrow = False
                    last_red_arrow = False
                    arrow_used = False

                signal = 0  # Default

                # === Range Filter (RF) + IB_Box Confirmation ===
                # Allow signals within 1 candle of each other in BOTH directions
                
                # Scenario A: RF signal first, then Arrow signal
                # Condition A1: RF and Arrow signals in same candle
                if new_rf_buy and row['GreenArrow'] == 1 and not rf_used:
                    signal = 1
                    rf_used = True
                    last_rf_buy_signal = False
                elif new_rf_sell and row['RedArrow'] == 1 and not rf_used:
                    signal = -1
                    rf_used = True
                    last_rf_sell_signal = False
                
                # Condition A2: RF signal, then Arrow signal in immediate next candle
                elif last_rf_buy_signal and not rf_used and row['GreenArrow'] == 1 and (i - rf_signal_candle) == 1:
                    signal = 1
                    rf_used = True
                    last_rf_buy_signal = False
                elif last_rf_sell_signal and not rf_used and row['RedArrow'] == 1 and (i - rf_signal_candle) == 1:
                    signal = -1
                    rf_used = True
                    last_rf_sell_signal = False

                # Scenario B: Arrow signal first, then RF signal
                # Condition B1: Arrow signal, then RF signal in immediate next candle
                elif last_green_arrow and not arrow_used and new_rf_buy and (i - arrow_signal_candle) == 1:
                    signal = 1
                    arrow_used = True
                    last_green_arrow = False
                elif last_red_arrow and not arrow_used and new_rf_sell and (i - arrow_signal_candle) == 1:
                    signal = -1
                    arrow_used = True
                    last_red_arrow = False

                # Assign the final signal
                self.df.iat[i, self.df.columns.get_loc('Signal_Final')] = signal

            self.df.to_csv('BTCUSD_Final.csv')

            # === Execute Latest Signal ===
            # last_candle = self.df.iloc[-1]
            last_candle = self.df.iloc[-1]
            last_signal = last_candle['Signal_Final']
            print(f"the current order id is {self.current_order_id}")
            print(f"the last signal is {last_signal}")

            if last_signal != 0 and self.get_active_positions_bitcoin():
                print(f"a new order would be placed since last signal is {last_signal}")
                print(f"active position status is {self.get_active_positions_bitcoin()}")
                import time
                time.sleep(10)
                current_price = float(last_candle['close'])
                self.last_price = current_price
                side = "buy" if last_signal > 0 else "sell"

                sl_price = self.calculate_stoploss(current_price, side)
                tp_price = self.calculate_takeprofit(current_price, side)

                self.last_sl_price = sl_price
                self.last_tp_price = tp_price

                if sl_price is not None and tp_price is not None:
                    stop_limit = sl_price + 10 if side == "buy" else sl_price + 10
                    tp_limit = tp_price - 10 if side == "buy" else tp_price + 10
                    self.set_leverage_delta(value=self.base_leverage,product_id="84")
                    self.leverage_check()
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

    def test(self):
        try:
            self.connect()
            # self.get_active_orders()
            # self.get_active_positions_bitcoin()
            self.place_order_market(side="buy",size=1)
            exit(0)
        except Exception as e:
            print(f"Error in test run : {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        try:
            print(f"Starting bot at: {self.get_current_datetime()}")
            self.connect()
            # if self.get_active_positions(): # true means size 0 no position
            if self.get_active_positions_bitcoin():
                print(f"Current status of get_active_positions is {self.get_active_positions_bitcoin()}")
                self.fetch_data()
                self.calculate_signals()
                self.execute_signals()
                import time
                time.sleep(5)
            else:
                print(f"Status of the position being closed is {self.get_active_positions_bitcoin()}")
                import time
                time.sleep(10)

        except Exception as e:
            print(f"Error occured in run : {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    api_key = "ToSrsEfM1Uk9uWXB4XE8AuWBSxRNwd"
    api_secret = "WurI06CuZU6QGU0lqq8L7UPQSHbMjsgiOj1Kz3kmtQAUHbREAJgjYyXZUa1s"
    # base_url = "https://cdn-ind.testnet.deltaex.org"
    base_url = "https://api.india.delta.exchange"
    Debot = DeltaExchangeRFGaizy(api_key=api_key,api_secret=api_secret,base_url=base_url)
    while True:
        try:
            Debot.run()
            # Debot.test()
        except Exception as e:
            print(f"Error in run : {e}")