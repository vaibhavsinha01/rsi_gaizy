import numpy as np
import pandas as pd
from binance.client import Client

class BrokerBinance:
    def __init__(self,api_key,api_secret,base_url):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.Client = Client(api_key=api_key,api_secret=api_secret,testnet=True)
        self.Client.API_TESTNET_URL = self.base_url

    def generate_signature(self):
        pass

    def generate_timestamp(self):
        pass

    def connect(self):
        pass
    
    def fetch_data(self):
        pass

    def place_market_order(self):
        pass

    def place_bracket_order(self):
        pass

    def cancel_order(self):
        pass

    def get_order_status(self):
        pass

    def set_leverage(self):
        pass

    def run(self):
        pass

if __name__ == "__main__":
    bb = BrokerBinance(api_key="kFKTjhE7BXfnO29MK2EeHFrkhGc7QDVCHYIyrj6MvAUiVns7swe4abgq6LHcmaMR",api_secret="RCYkn1a1CSiARylQiCDwytCEkirCOzNTLDgeZSCl8e2sIBsiL0EuoSDVCPuIXkAa",base_url="https://testnet.binance.vision/api")
    bb.run()