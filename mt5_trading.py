import MetaTrader5 as mt5
import pandas as pd
from trading_gui import TradingGUI

class MT5Trading:
    def __init__(self, account, password, server):
        self.account = account
        self.password = password
        self.server = server
        self.symbol = "EURUSD"
        self.initialize_mt5()
    
    def initialize_mt5(self):
        if not mt5.initialize():
            print(f"MT5 Initialization failed, error: {mt5.last_error()}")
            quit()
        if not mt5.login(self.account, self.password, self.server):
            print(f"Login failed, error: {mt5.last_error()}")
            mt5.shutdown()
            quit()
        if not mt5.symbol_select(self.symbol, True):
            print(f"Failed to enable {self.symbol}, exiting.")
            mt5.shutdown()
            quit()
    
    def place_order(self, order_type, volume):
        symbol_info = mt5.symbol_info(self.symbol)
        point = symbol_info.point
        ask_price = mt5.symbol_info_tick(self.symbol).ask
        bid_price = mt5.symbol_info_tick(self.symbol).bid
        sl_pips = max(25, symbol_info.trade_stops_level / 10)
        tp_pips = max(20, symbol_info.trade_stops_level / 10)

        order_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "magic": 234000,
            "order": 0,
            "symbol": self.symbol,
            "volume": volume,
            "deviation": 10,
            "type_filling": mt5.ORDER_FILLING_FOK,
            "type_time": mt5.ORDER_TIME_GTC,
            "expiration": 0,
            "comment": "python script",
        }
        
        if order_type == "buy":
            order_request.update({
                "type": mt5.ORDER_TYPE_BUY,
                "price": ask_price,
                "sl": ask_price - sl_pips * point,
                "tp": ask_price + tp_pips * point,
            })
        elif order_type == "sell":
            order_request.update({
                "type": mt5.ORDER_TYPE_SELL,
                "price": bid_price,
                "sl": bid_price + sl_pips * point,
                "tp": bid_price - tp_pips * point,
            })

        result = mt5.order_send(order_request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return f"Order failed, error code: {result.retcode}"
        return f"✅ Order placed successfully! Order ID: {result.order}"

if __name__ == "__main__":
    account = 90145790
    password = "5qMgTx*v"
    server = "MetaQuotes-Demo"
    app = TradingGUI(MT5Trading(account, password, server))
    app.run()
