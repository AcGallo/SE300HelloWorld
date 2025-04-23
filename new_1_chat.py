import os
import gym
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import time
import logging
from datetime import datetime

# Read settings file and convert the first two lines to floats
with open('settings.txt', 'r') as file:
    content = file.readlines()

# Expecting settings.txt to have two values:
# First line: risk percentage, Second line: number of entry orders
risk = float(content[0].strip()) if float(content[0].strip()) > 0 else 0.0
entry_orders = float(content[1].strip()) if float(content[1].strip()) > 0 else 0.0

np.set_printoptions(threshold=np.inf, linewidth=200) 

# Initialize and Login
account = 91395010
password = "T_X5YbZz"
server = "MetaQuotes-Demo"

while not mt5.initialize():
    print(f"MT5 Re-initialization failed, error: {mt5.last_error()}")
    time.sleep(60)  # Wait and retry
    mt5.initialize()

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
if account_info is None:
    print("Failed to retrieve account info!")
    mt5.shutdown()
    quit()

symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    print("Symbol EURUSD not found!")
    mt5.shutdown()
    quit()

# Ensure EURUSD is tradeable
if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
    print("EURUSD is disabled for trading on this account!")
    mt5.shutdown()
    quit()

# Trade parameters
point = symbol_info.point
ask_price = mt5.symbol_info_tick(symbol).ask
bid_price = mt5.symbol_info_tick(symbol).bid
sl_pips = max(30, symbol_info.trade_stops_level / 10)
tp_pips = max(30, symbol_info.trade_stops_level / 10)
min_vol = symbol_info.volume_min

# Trade request functions
def buy(order):
    try:
        ask_price = mt5.symbol_info_tick(symbol).ask
        buy_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "magic": 234000,
            "order": 0,
            "symbol": symbol,
            "volume": order,
            "price": ask_price,
            "stoplimit": 0.0,
            "sl": ask_price - sl_pips * point,
            "tp": ask_price + tp_pips * point,
            "deviation": 10,
            "type": mt5.ORDER_TYPE_BUY,
            "type_filling": mt5.ORDER_FILLING_FOK,
            "type_time": mt5.ORDER_TIME_GTC,
            "expiration": 0,
            "comment": "python script",
        }
        return buy_request
    except Exception as e:
        print(f"Error in buy(): {e}")
        return None

def sell(order):
    try:
        bid_price = mt5.symbol_info_tick(symbol).bid
        sell_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "magic": 234000,
            "order": 0,
            "symbol": symbol,
            "volume": order,
            "price": bid_price,
            "stoplimit": 0.0,
            "sl": bid_price + sl_pips * point,
            "tp": bid_price - tp_pips * point,
            "deviation": 10,
            "type": mt5.ORDER_TYPE_SELL,
            "type_filling": mt5.ORDER_FILLING_FOK,
            "type_time": mt5.ORDER_TIME_GTC,
            "expiration": 0,
            "comment": "python script",
        }
        return sell_request
    except Exception as e:
        print(f"Error in sell(): {e}")
        return None

def place_trade(action):
    try:
        # Refresh account info for up-to-date balance
        current_account_info = mt5.account_info()
        balance = current_account_info.balance if current_account_info else 0
        orders = 0
        if risk > 0.0:
            # Calculate orders based on risk and stop loss in pips
            orders = (balance * (risk / 100)) / sl_pips          
        if entry_orders > 0:
            orders = entry_orders

        # Compute the volume as the rounded value based on the volume step,
        # and then clamp between the volume_min and volume_max.
        computed_volume = round(orders / symbol_info.volume_step) * symbol_info.volume_step
        valid_volume = round(min(max(computed_volume, symbol_info.volume_min), symbol_info.volume_max), 2)
        
        if action == "buy":  # Buy
            request = buy(valid_volume)
        elif action == "sell":  # Sell
            request = sell(valid_volume)
        else:
            print("Invalid action. Use 'buy' or 'sell'.")
            return None

        if request:
            result = mt5.order_send(request)

            if result is None:
                print("Order send failed, no response from server.")
                print(mt5.last_error())
                return None
            elif result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Order failed, error code: {result.retcode}")
                print(mt5.last_error())
                return None
            else:
                print(f"✅ Order placed successfully! Order ID: {result.order}")
                return result.order  # Return the order ID
    except Exception as e:
        print(f"Error in place_trade(): {e}")
        return None

# Data fetching functions
timeframe = mt5.TIMEFRAME_M1
currency_pairs = ["EURUSD"]

def getData():
    try:
        # Fetch raw data for the specified symbol and timeframe
        data = mt5.copy_rates_from_pos(symbol, timeframe, 0, 60)  # Fetch the last 60 candles
        
        if data is None or len(data) == 0:
            print("⚠️ No data retrieved from MT5.")
            return np.zeros((60, 5))  # Return a default observation of shape (60, 5)

        # Extract Open, High, Low, Close, and Volume
        simplified_data = []
        for candle in data:
            simplified_data.append([
                candle['open'],   # Open price
                candle['high'],   # High price
                candle['low'],    # Low price
                candle['close'],  # Close price
                candle['tick_volume']  # Volume
            ])

        # Convert the list to a NumPy array
        simplified_data = np.array(simplified_data, dtype=np.float32)

        # Ensure the array has the correct shape (60, 5)
        if simplified_data.shape != (60, 5):
            print("⚠️ Data shape is incorrect. Returning default observation.")
            return np.zeros((60, 5))  # Return a default observation of shape (60, 5)

        return simplified_data

    except Exception as e:
        print(f"⚠️ Error in getData(): {e}")
        return np.zeros((60, 5))  # Return a default observation of shape (60, 5)
    
data = getData()

# Machine learning model environment
class ForexEnv(gym.Env):
    def __init__(self):
        super(ForexEnv, self).__init__()
        
        self.entry_price = None
        self.data = np.zeros((60, 5), dtype=np.float32)  # Initialize with zeros
        self.current_step = 0
        self.max_steps = 60 * 24  # Max steps per episode (24 hours)
        self.balance = mt5.account_info().balance if mt5.account_info() else 10000  # Fallback balance
        self.position = 0  
        self.trade_count = 0
        self.last_trade_time = time.time()

        # Observation space: 60 candles, 5 features each
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(60, 5),
            dtype=np.float32
        )
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

    def reset(self):
        try:
            # Get fresh market data
            self.data = getData()
            
            # Handle empty/invalid data
            if self.data is None or len(self.data) == 0:
                print("⚠️ No data available, using zeros")
                self.data = np.zeros((60, 5), dtype=np.float32)
            
            # Ensure correct shape (60, 5)
            if self.data.shape != (60, 5):
                print(f"⚠️ Reshaping data from {self.data.shape} to (60, 5)")
                if len(self.data) < 60:
                    # Pad with zeros if insufficient data
                    padding = np.zeros((60 - len(self.data), 5), dtype=np.float32)
                    self.data = np.vstack([self.data, padding])
                elif len(self.data) > 60:
                    # Take most recent 60 candles if too much data
                    self.data = self.data[-60:]
                self.data = self.data.reshape(60, 5)

            # Reset environment state
            self.current_step = 0
            self.position = 0
            self.trade_count = 0
            self.balance = mt5.account_info().balance if mt5.account_info() else self.balance
            
            # Return a copy to prevent modification of internal state
            obs = self.data[-60:] if len(self.data) >= 60 else np.zeros((60, 5))
            return obs.reshape(1, 60, 5)
            
        except Exception as e:
            print(f"⚠️ Error in reset(): {e}")
            return np.zeros((60, 5), dtype=np.float32)

    def update_data(self, new_data):
        """Update the environment's data buffer"""
        try:
            if new_data is not None and len(new_data) > 0:
                self.data = new_data[-60:]  # Keep only last 60 candles
        except Exception as e:
            print(f"⚠️ Error in update_data(): {e}")

    def step(self, action):
        try:
            reward = 0
            done = False
            info = {}

            # Validate action is an integer among 0,1,2
            if not isinstance(action, int) or action not in [0, 1, 2]:
                print(f"⚠️ Invalid action: {action}")
                return self.data.copy().reshape(1, 60, 5), reward, done, info

            current_candle = self.data[self.current_step % len(self.data)]
            current_price = current_candle[3]  # Use the close price

            # Execute trade actions only if no position is currently open
            if action == 1 and self.position == 0:  # Buy
                order_id = place_trade("buy")
                if order_id:
                    self.position = 1
                    self.entry_price = current_price
                    self.trade_count += 1
                    self.last_trade_time = time.time()
                    wait_for_trade_close(order_id)
                    reward = calculate_reward(order_id)
                else:
                    reward = -0.1

            elif action == 2 and self.position == 0:  # Sell
                order_id = place_trade("sell")
                if order_id:
                    self.position = -1
                    self.entry_price = current_price
                    self.trade_count += 1
                    self.last_trade_time = time.time()
                    wait_for_trade_close(order_id)
                    reward = calculate_reward(order_id)
                else:
                    reward = -0.1

            else:  # Hold action or if already in a position
                reward = 0.001
                time.sleep(60)  # Wait a minute

            self.current_step += 1
            
            # Termination condition checks: end of data, max steps, or long inactivity
            if (self.current_step >= len(self.data) or 
                self.current_step >= self.max_steps or
                (time.time() - self.last_trade_time > 3600)):
                done = True

            # Prepare observation ensuring it is (1, 60, 5)
            obs = self.data[-60:] if len(self.data) >= 60 else np.zeros((60, 5))
            obs = obs.reshape(1, 60, 5)

            self.balance = mt5.account_info().balance if mt5.account_info() else self.balance
            
            return obs, reward, done, info

        except Exception as e:
            print(f"⚠️ Error in step(): {e}")
            return np.zeros((1, 60, 5), dtype=np.float32), 0, True, {}

def get_observation(env):
    """
    Safely gets the current observation with shape (1, 60, 5)
    """
    current_data = env.envs[0].data
    if len(current_data) < 60:
        padding = np.zeros((60 - len(current_data), 5), dtype=np.float32)
        obs = np.vstack([padding, current_data])
    else:
        obs = current_data[-60:]
    obs = obs.reshape(1, 60, 5)
    return obs

# Trade status, waiting, and reward functions
def check_trade_status(order_id):
    """
    Checks the status of a trade in MT5 using the order ID.
    Returns "open", "closed", or "error".
    """
    if not mt5.initialize():
        print("❌ MT5 Initialization Failed")
        return "error"

    positions = mt5.positions_get()
    if positions is None:
        print("⚠️ No active positions found.")
        return "closed"

    for position in positions:
        if position.ticket == order_id:
            return "open"
    
    return "closed"

def wait_for_trade_close(order_id):
    """Waits until the trade with order_id is closed before proceeding."""
    while True:
        status = check_trade_status(order_id)
        if status == "closed":
            print(f"✅ Trade {order_id} has closed.")
            break
        elif status == "error":
            print(f"❌ Error checking trade {order_id}, retrying...")
        time.sleep(5)

def calculate_reward(order_id):
    """
    Fetches trade result from MetaTrader and calculates the reward.
    """
    history = mt5.history_deals_get(position=order_id)
    
    if history is None or len(history) == 0:
        print(f"⚠️ Trade {order_id} not found in history!")
        return 0

    entry_price = None
    exit_price = None
    volume = None
    trade_type = None  # 0 = buy, 1 = sell

    for deal in history:
        if deal.entry == mt5.DEAL_ENTRY_IN:
            entry_price = deal.price
            volume = deal.volume
            trade_type = 0 if deal.type == mt5.DEAL_TYPE_BUY else 1
        elif deal.entry == mt5.DEAL_ENTRY_OUT:
            exit_price = deal.price

    if entry_price is None or exit_price is None or volume is None:
        print(f"⚠️ Incomplete trade data for order {order_id}.")
        return 0

    if trade_type == 0:  # Buy trade
        profit = (exit_price - entry_price) * volume
        reward = profit / volume if volume != 0 else 0
    elif trade_type == 1:  # Sell trade
        profit = (entry_price - exit_price) * volume
        reward = profit / volume if volume != 0 else 0
    else:
        print(f"⚠️ Invalid trade type for order {order_id}.")
        return 0

    print(f"✅ Trade {order_id} closed. Reward: {reward}")
    return reward

# Initialize environment and model
raw_env = ForexEnv()
env = DummyVecEnv([lambda: raw_env])
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

model_path = "forex_trading_bot.zip"
if os.path.exists(model_path):
    model = PPO.load(model_path)
    print("✅ Loaded existing model!")
else:
    model = PPO("MlpPolicy", env, verbose=1)
    print("🚀 No saved model found, starting new training.")

logging.basicConfig(filename="trading_bot.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

last_trade_time = time.time()

# Main training and trading loop
total_steps = 0
episode = 0
previous_data = getData()

# Training parameters
train_interval = 100   # Train every 100 steps
train_batch_size = 1000  # Train for 1000 timesteps each time
save_interval = 100    # Save the model every 100 steps

while True:
    try:
        new_data = getData()
        
        if np.array_equal(new_data, previous_data):
            print(f"Step {total_steps}: No new data, waiting...")
            time.sleep(10)
            continue
        else:
            previous_data = new_data
            env.envs[0].update_data(new_data)
            
            if total_steps == 0:
                obs = env.reset()
            else:
                obs = get_observation(env)
            
            if obs.ndim != 3 or obs.shape != (1, 60, 5):
                print(f"⚠️ Invalid observation shape: {obs.shape}. Resetting environment...")
                obs = env.reset()
                
            if total_steps > 0 and total_steps % train_interval == 0:
                print(f"🚀 Training model at step {total_steps}...")
                model.learn(total_timesteps=train_batch_size, reset_num_timesteps=False)
                print(f"✅ Training completed at step {total_steps}.")

            if total_steps % save_interval == 0:
                print(f"💾 Saving model at step {total_steps}...")
                model.save("forex_trading_bot.zip")
                model.save(f"forex_trading_bot_step_{total_steps}.zip")
                print(f"💾 Model saved at step {total_steps}.")

            print(f"Step {total_steps}: Observation shape: {obs.shape}")
            print(f"Step {total_steps}: Observation content: {obs}")
            
            # Use .item() to extract scalar action from the prediction array
            action, _ = model.predict(obs)
            action = int(np.array(action).item())
            print(f"Step {total_steps}: Predicted Action {action}")

            obs, reward, done, _ = env.step(action)
            print(f"Step {total_steps}: Action {action}, Reward {reward}")

            logging.info(f"Step {total_steps}: Action {action}, Reward {reward}, Episode {episode}, Trade Count {env.envs[0].trade_count}")

            total_steps += 1

            if done:
                episode += 1
                print(f"✅ Episode {episode} completed at step {total_steps}. Resetting environment...")
                obs = env.reset()

    except Exception as e:
        print(f"⚠️ Step {total_steps}: Error in loop: {e}")
        logging.error(f"Step {total_steps}: Error in loop: {e}")
        break

mt5.shutdown()
