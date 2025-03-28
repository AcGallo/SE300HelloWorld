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

np.set_printoptions(threshold=np.inf, linewidth=200) 

# Initialize Variables
risk = 0.1  # In percentage
orders = 10  # Number of orders

# Initialize and Login
account = 10005657095
password = "@aAiQ3Uv"
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
sl_pips = max(50, symbol_info.trade_stops_level / 10)
tp_pips = max(50, symbol_info.trade_stops_level / 10)
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
        balance = account_info.balance
        if risk > 0.0:
            orders = (balance * (risk / 100)) / sl_pips
        
        valid_volume = round(
            max(symbol_info.volume_min, round(orders / symbol_info.volume_step) * symbol_info.volume_step,
                min(symbol_info.volume_max, round(orders / symbol_info.volume_step) * symbol_info.volume_step)), 2)
        
        if action == "buy":  # Buy
            request = buy(valid_volume)
        elif action == "sell":  # Sell
            request = sell(valid_volume)
        else:
            print("Invalid action. Use 1 (buy) or 2 (sell).")
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
            return [[0.0, 0.0, 0.0, 0.0, 0.0]]  # Return a default observation

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
        return simplified_data

    except Exception as e:
        print(f"⚠️ Error in getData(): {e}")
        return [[0.0, 0.0, 0.0, 0.0, 0.0]]  # Return a default observation
    
data = getData()

# Machine learning model
class ForexEnv(gym.Env):
    def __init__(self):
        super(ForexEnv, self).__init__()
        
        self.entry_price = None
        self.data = []  # Initialize empty data
        self.current_step = 0
        self.max_steps = 60  # Maximum number of steps per episode
        self.balance = account_info.balance
        self.position = 0  
        self.trade_count = 0  # Track the number of trades

        # Define observation space (5 features: Open, High, Low, Close, Volume)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(60,5),  # Shape of the observation (20 candles, 5 features)
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell

    def reset(self):
        """Reset the environment and start from step 0"""
        self.data = getData()  # Fetch new data
        if not self.data:
            print("Error: No data available for reset.")
            return np.zeros((60, 5), dtype=np.float32)  # Return a default observation

        self.current_step = 0
        self.balance = account_info.balance
        self.position = 0
        self.trade_count = 0  # Reset trade count

        # Ensure we have at least 10 candles
        if len(self.data) < 60:
            print("⚠️ Not enough data for 10 candles. Padding with zeros.")
            padding = np.zeros((60 - len(self.data), 5), dtype=np.float32)
            self.data = np.vstack([padding, self.data])

        # Return the last 60 candles as the initial observation
        observation = np.array(self.data[self.current_step:self.current_step + 60], dtype=np.float32)
        return observation

    def update_data(self, new_data):
        """Update the environment's internal state with new data"""
        self.data = new_data
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1  # Reset to the last valid index

    def step(self, action):
        try:
            reward = 0
            done = False

            # Validate action
            if action not in [0, 1, 2]:  # 0=Hold, 1=Buy, 2=Sell
                print(f"⚠️ Invalid action: {action}. Use 0 (hold), 1 (buy), or 2 (sell).")
                return np.zeros((60, 5), dtype=np.float32), reward, done, {}

            # Get current price (assuming the first feature is the price)
            if self.current_step >= len(self.data):
                return np.zeros((60, 5), dtype=np.float32), reward, done, {}  # Return default observation if out of bounds
            current_price = self.data[self.current_step][0]

            # Handle actions
            if action == 1:  # Buy
                if self.position == 0:  # No open position
                    # Place a real BUY trade in MT5
                    order_id = place_trade("buy")
                    if order_id:
                        self.position = 1  # Open buy position
                        self.entry_price = current_price  # Store entry price
                        self.trade_count += 1  # Increment trade count
                        print(f"✅ Step {self.current_step}: Opened BUY position at price: {self.entry_price}")
                        logging.info(f"Step {self.current_step}: Opened BUY position at price: {self.entry_price}")

                        # Wait for the trade to close and calculate reward
                        wait_for_trade_close(order_id)
                        reward = calculate_reward(order_id)  # Pass only order_id
                    else:
                        print(f"⚠️ Step {self.current_step}: Failed to place BUY trade.")
                        logging.warning(f"Step {self.current_step}: Failed to place BUY trade.")

            elif action == 2:  # Sell
                if self.position == 0:  # No open position
                    # Place a real SELL trade in MT5
                    order_id = place_trade("sell")
                    if order_id:
                        self.position = -1  # Open sell position
                        self.entry_price = current_price  # Store entry price
                        self.trade_count += 1  # Increment trade count
                        print(f"✅ Step {self.current_step}: Opened SELL position at price: {self.entry_price}")
                        logging.info(f"Step {self.current_step}: Opened SELL position at price: {self.entry_price}")

                        # Wait for the trade to close and calculate reward
                        wait_for_trade_close(order_id)
                        reward = calculate_reward(order_id)  # Pass only order_id
                    else:
                        print(f"⚠️ Step {self.current_step}: Failed to place SELL trade.")
                        logging.warning(f"Step {self.current_step}: Failed to place SELL trade.")
            elif action == 0:  # Hold
                reward = 0.00001  # Small reward for holding
                logging.info(f"Step {self.current_step}: Holding...")
                time.sleep(60)

            # Move to the next step
            self.current_step += 1

            # Check if the episode is done
            if self.current_step >= len(self.data) - 1:
                done = True
                print(f"✅ Step {self.current_step}: Episode completed.")
                logging.info(f"Step {self.current_step}: Episode completed.")

            # Ensure self.current_step does not exceed the length of self.data
            if self.current_step >= len(self.data):
                self.current_step = len(self.data) - 1  # Reset to the last valid index

            # Get the new observation (last 60 candles)
            observation = np.array(self.data[self.current_step:self.current_step + 60], dtype=np.float32)

            # Debugging: Log the observation shape and content
            print(f"Step {self.current_step}: Observation shape: {observation.shape}")
            print(f"Step {self.current_step}: Observation content: {observation}")

            # Return the new state, reward, done flag, and additional info
            return observation, reward, done, {}

        except Exception as e:
            print(f"⚠️ Step {self.current_step}: Error in step(): {e}")
            logging.error(f"Step {self.current_step}: Error in step(): {e}")
            return np.zeros((60, 5), dtype=np.float32), 0, True, {}           

env = DummyVecEnv([lambda: ForexEnv()])
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# Loading saved model
model_path = "forex_trading_bot.zip"
if os.path.exists(model_path):
    model = PPO.load(model_path)  # Load saved model
    print("✅ Loaded existing model!")
else:
    model = PPO("MlpPolicy", env, verbose=1)  # Start fresh
    print("🚀 No saved model found, starting new training.")

# Configure logging
logging.basicConfig(filename="trading_bot.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

last_trade_time = time.time()  # Track last trade time

def check_trade_status(order_id):
    """
    Checks the status of a trade in MT5 using the order ID.

    Returns:
        - "open" if the trade is still active.
        - "closed" if the trade has been completed.
        - "error" if the trade ID is not found.
    """
    if not mt5.initialize():
        print("❌ MT5 Initialization Failed")
        return "error"

    # Retrieve all active orders
    positions = mt5.positions_get()
    if positions is None:
        print("⚠️ No active positions found.")
        return "closed"

    # Check if our order is still active
    for position in positions:
        if position.ticket == order_id:
            return "open"  # Trade is still active
    
    return "closed"  # If not found, assume trade has been closed

def wait_for_trade_close(order_id):
    """Waits until the trade with order_id is closed before proceeding."""
    while True:
        status = check_trade_status(order_id)
        if status == "closed":
            print(f"✅ Trade {order_id} has closed.")
            break  # Exit the loop when the trade is closed
        elif status == "error":
            print(f"❌ Error checking trade {order_id}, retrying...")
        
        time.sleep(5)  # Check every 5 seconds

def calculate_reward(order_id):
    """
    Fetches trade result from MetaTrader and calculates the reward manually.
    """
    # Get trade history (executed trades)
    history = mt5.history_deals_get(position=order_id)
    
    if history is None or len(history) == 0:
        print(f"⚠️ Trade {order_id} not found in history!")
        return 0  # Neutral reward if no history is found

    # Initialize variables to store trade details
    entry_price = None
    exit_price = None
    volume = None
    trade_type = None  # 0 = buy, 1 = sell

    # Loop through the deals to find entry and exit prices
    for deal in history:
        if deal.entry == mt5.DEAL_ENTRY_IN:  # Entry deal
            entry_price = deal.price
            volume = deal.volume
            trade_type = 0 if deal.type == mt5.DEAL_TYPE_BUY else 1  # 0 = buy, 1 = sell
        elif deal.entry == mt5.DEAL_ENTRY_OUT:  # Exit deal
            exit_price = deal.price

    # Check if all required data is available
    if entry_price is None or exit_price is None or volume is None:
        print(f"⚠️ Incomplete trade data for order {order_id}.")
        return 0  # Neutral reward if data is incomplete

    # Calculate profit manually
    if trade_type == 0:  # Buy trade
        profit = (exit_price - entry_price) * volume
        # Normalize reward based on profit and volume
        reward = profit / volume if volume != 0 else 0
    elif trade_type == 1:  # Sell trade
        profit = (entry_price - exit_price) * volume
        # Normalize reward based on profit and volume
        reward = profit / volume if volume != 0 else 0
    else:
        print(f"⚠️ Invalid trade type for order {order_id}.")
        return 0  # Neutral reward if trade type is invalid

    print(f"✅ Trade {order_id} closed. Reward: {reward}")
    return reward  # Return the calculated reward

open_trades = {}  # Format: {order_id: {"action": action, "step": step}}

# Main loop with model training, logging, and periodic saving
total_steps = 0  # Global step counter
episode = 0  # Episode counter
previous_data = getData()

# Training parameters
train_interval = 100  # Train every 100 steps
train_batch_size = 10  # Train for 10 timesteps each time
save_interval = 1000  # Save the model every 1,000 steps

while True:
    try:
        new_data = getData()
        
        if new_data == previous_data:  
            print(f"Step {total_steps}: No new data, waiting...")
            time.sleep(10)
            continue
        else:
            previous_data = new_data
            # Update the environment's internal state with new data
            env.envs[0].update_data(new_data)
            
            # Reset the environment if it's the first step
            if total_steps == 0:
                obs = env.reset()
            else:
                obs = np.array(env.envs[0].data[env.envs[0].current_step:env.envs[0].current_step + 10], dtype=np.float32)
            
            # Ensure the observation is a 2D array
            if obs.ndim != 2 or obs.shape != (10, 5):
                print(f"⚠️ Invalid observation shape: {obs.shape}. Resetting environment...")
                obs = env.reset()
            
            # Debugging: Log the observation shape and content
            print(f"Step {total_steps}: Observation shape: {obs.shape}")
            print(f"Step {total_steps}: Observation content: {obs}")
            
            # Predict action (0=Hold, 1=Buy, 2=Sell)
            action, _ = model.predict(obs)  # Pass the observation directly (NumPy array)
            print(f"Step {total_steps}: Predicted Action {action}")

            # Pass action to the environment and place trades
            obs, reward, done, _ = env.step(action)  
            print(f"Step {total_steps}: Action {action}, Reward {reward}")

            # Log the step details
            logging.info(f"Step {total_steps}: Action {action}, Reward {reward}, Episode {episode}, Trade Count {env.envs[0].trade_count}")

            # Train the model periodically
            if total_steps > 0 and total_steps % train_interval == 0:  # Start training after step 0
                print(f"🚀 Training model at step {total_steps}...")
                model.learn(total_timesteps=train_batch_size, reset_num_timesteps=False)
                print(f"✅ Training completed at step {total_steps}.")

            # Save the model periodically
            if total_steps % save_interval == 0:
                print(f"💾 Saving model at step {total_steps}...")
                model.save(f"forex_trading_bot_step_{total_steps}.zip")
                print(f"💾 Model saved at step {total_steps}.")

            total_steps += 1

            # Check if the episode is done
            if done:
                episode += 1
                print(f"✅ Episode {episode} completed at step {total_steps}. Resetting environment...")
                obs = env.reset()

            # Stop after a certain number of steps (for testing)
            if total_steps >= 1440:
                print("🛑 Stopping after 1440 steps.")
                break

    except Exception as e:
        print(f"⚠️ Step {total_steps}: Error in loop: {e}")
        logging.error(f"Step {total_steps}: Error in loop: {e}")
        break  # Exit the loop on critical errors

# Shutdown MT5
mt5.shutdown