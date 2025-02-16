import MetaTrader5 as mt5
import pandas as pd

# Initialize and Login
account = 90145790
password = "5qMgTx*v"
server = "MetaQuotes-Demo"

if not mt5.initialize():
    print(f"MT5 Initialization failed, error: {mt5.last_error()}")
    quit()

if not mt5.login(account, password, server):
    print(f"Login failed, error: {mt5.last_error()}")
    mt5.shutdown()
    quit()

# Ensure EURUSD is available
symbol = "EURUSD"
if not mt5.symbol_select(symbol, True):
    print(f"Failed to enable {symbol}, exiting.")
    mt5.shutdown()
    quit()

# Get Symbol Info
symbol_info = mt5.symbol_info(symbol)
point = symbol_info.point
ask_price = mt5.symbol_info_tick(symbol).ask
bid_price = mt5.symbol_info_tick(symbol).bid

# Ensure valid SL/TP
sl_pips = max(10, symbol_info.trade_stops_level / 10)
tp_pips = max(20, symbol_info.trade_stops_level / 10)

# Ensure valid volume
min_vol = symbol_info.volume_min

if symbol_info is None:
    print("Symbol EURUSD not found!")
    mt5.shutdown()
    quit()

# Check if EURUSD is actually tradeable
if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
    print("EURUSD is disabled for trading on this account!")
    mt5.shutdown()
    quit()

# Prepare trade request
def buy():
    symbol_info = mt5.symbol_info(symbol)
    point = symbol_info.point
    ask_price = mt5.symbol_info_tick(symbol).ask
    buy_request = {
        "action": mt5.TRADE_ACTION_DEAL,  # Trade operation type (instant execution)
        "magic": 234000,  # Expert Advisor ID (magic number)
        "order": 0,  # Order ticket (0 for new market orders)
        "symbol": symbol,  # Trade symbol
        "volume": 1.0, #min_vol,  # Requested volume in lots
        "price": ask_price,  # Current market price
        "stoplimit": 0.0,  # StopLimit level (used only for stop-limit orders)
        "sl": ask_price - sl_pips * point,  # Stop Loss level
        "tp": ask_price + tp_pips * point,  # Take Profit level
        "deviation": 10,  # Maximum price deviation
        "type": mt5.ORDER_TYPE_BUY,  # Order type (BUY)
        "type_filling": mt5.ORDER_FILLING_FOK,  # Order execution type
        "type_time": mt5.ORDER_TIME_GTC,  # Order duration (Good Till Canceled)
        "expiration": 0,  # Order expiration (only used for ORDER_TIME_SPECIFIED)
        "comment": "python script",  # Order comment
    }
    return buy_request
def sell():
    symbol_info = mt5.symbol_info(symbol)
    point = symbol_info.point
    bid_price = mt5.symbol_info_tick(symbol).bid
    sell_request = {
        "action": mt5.TRADE_ACTION_DEAL,  # Trade operation type (instant execution)
        "magic": 234000,  # Expert Advisor ID (magic number)
        "order": 0,  # Order ticket (0 for new market orders)
        "symbol": symbol,  # Trade symbol
        "volume": 1.0, #min_vol,  # Requested volume in lots
        "price": bid_price,  # Current market price
        "stoplimit": 0.0,  # StopLimit level (used only for stop-limit orders)
        "sl": bid_price + sl_pips * point,  # Stop Loss level
        "tp": bid_price - tp_pips * point,  # Take Profit level
        "deviation": 10,  # Maximum price deviation
        "type": mt5.ORDER_TYPE_SELL,  # Order type (BUY)
        "type_filling": mt5.ORDER_FILLING_FOK,  # Order execution type
        "type_time": mt5.ORDER_TIME_GTC,  # Order duration (Good Till Canceled)
        "expiration": 0,  # Order expiration (only used for ORDER_TIME_SPECIFIED)
        "comment": "python script",  # Order comment
    }
    return sell_request

request = buy()   #change this to buy() or sell()

# Send order
result = mt5.order_send(request)

# Check order result
print("\nOrder Response:")
if result is None:
    print("Order send failed, no response from server.")
elif result.retcode != mt5.TRADE_RETCODE_DONE:
    print(f"Order failed, error code: {result.retcode}")
else:
    print(f"✅ Order placed successfully! Order ID: {result.order}")


#data
timeframe = mt5.TIMEFRAME_M1
currency_pairs = ["EURUSD", "EURCAD", "GBPUSD", "GBPJPY", "USDJPY", "USDCAD", "AUDCAD", "AUDJPY", "AUDUSD"]

# Fetch data
candles = {}
for pair in currency_pairs:
    data = mt5.copy_rates_from_pos(pair, timeframe, 0, 100) #curreny pair symbol, 1 minute timeframe, current candle to 100 candles previous
    if data is not None:
        candles[pair] = pd.DataFrame(data)
        candles[pair]["time"] = pd.to_datetime(candles[pair]["time"], unit="s")  # Convert time to readable format
print(candles["EURUSD"].head())
#can print the most recent couple candles data
#print(candles["EURUSD"]["tick_volume"].iloc[0])  # Last (most recent) candle's volume for EURUSD
#can use "high" "open" "close" "low" "time" "tick_volume"
#test

mt5.shutdown()
