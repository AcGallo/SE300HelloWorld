import MetaTrader5 as mt5
import pandas as pd
import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Initialize Variables
global risk
global orders

# Initialize and Login
account = 10005657095
password = "@aAiQ3Uv"
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

account_info = mt5.account_info()

# Check if account info is available
if account_info is None:
    print("Failed to retrieve account info!")
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
def buy(order):
    symbol_info = mt5.symbol_info(symbol)
    point = symbol_info.point
    ask_price = mt5.symbol_info_tick(symbol).ask
    buy_request = {
        "action": mt5.TRADE_ACTION_DEAL,  # Trade operation type (instant execution)
        "magic": 234000,  # Expert Advisor ID (magic number)
        "order": 0,  # Order ticket (0 for new market orders)
        "symbol": symbol,  # Trade symbol
        "volume": order, #min_vol,  # Requested volume in lots
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
def sell(order):
    symbol_info = mt5.symbol_info(symbol)
    point = symbol_info.point
    bid_price = mt5.symbol_info_tick(symbol).bid
    sell_request = {
        "action": mt5.TRADE_ACTION_DEAL,  # Trade operation type (instant execution)
        "magic": 234000,  # Expert Advisor ID (magic number)
        "order": 0,  # Order ticket (0 for new market orders)
        "symbol": symbol,  # Trade symbol
        "volume": order, #min_vol,  # Requested volume in lots
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

def place_trade():
    #can only use one of the following at a time
    risk = 0.1 #in percentages #0.2 seems like the cap
    orders = 10 #in number of orders

    #current account balance used for risk calculations
    balance = account_info.balance

    if(risk > 0.0):
        orders = (balance * (risk/100))/sl_pips

    valid_volume = round(max(symbol_info.volume_min, round(orders / symbol_info.volume_step) * symbol_info.volume_step, min(symbol_info.volume_max, round(orders / symbol_info.volume_step) * symbol_info.volume_step)), 1)
    request = buy(valid_volume)   #change this to buy() or sell() #number must have 1 decimal place ex. 10.0

    # Send order
    result = mt5.order_send(request)

    # Check order result
    print("\nOrder Response:")
    if result is None:
        print("Order send failed, no response from server.")
        print(mt5.last_error())
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed, error code: {result.retcode}")
        print(mt5.last_error())
    else:
        print(f"✅ Order placed successfully! Order ID: {result.order}")
        print(mt5.last_error())

#data
timeframe = mt5.TIMEFRAME_M1
currency_pairs = ["EURUSD", "EURCAD", "GBPUSD", "GBPJPY", "USDJPY", "USDCAD", "AUDCAD", "AUDJPY", "AUDUSD"]

# Fetch data
def getData():
    all_data = []

    for pair in currency_pairs:
        data = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_M1, 0, 1)  # 100 most recent 1-minute candles
        if data is not None and len(data) > 0:
            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["time"], unit="s")  # Convert timestamp to readable format

            # Extract OHLC + tick volume as a list of lists and extend to flatten
            all_data.extend(df[["open", "high", "low", "close", "tick_volume"]].values.tolist())

    return all_data


def standard_data():
    # Get data and convert to NumPy array
    data = getData()  
    data_array = np.array(data)

    # Apply standardization
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data_array)
    return data_standardized


#get standardized data
standardized_data = standard_data()

#place a trade
place_trade()

mt5.shutdown()
