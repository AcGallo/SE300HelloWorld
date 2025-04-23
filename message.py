# ==== PART 1: Imports & CSV Loader ====
import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import os

def load_history(path):
    df = pd.read_csv(path)

    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'tick_volume']
    df = df[['open', 'high', 'low', 'close', 'tick_volume']]

    df_std = df.std()
    df_std[df_std == 0] = 1e-8
    df = (df - df.mean()) / df_std
    df = df.dropna()

    print(f"[INFO] Loaded history from {path}, shape: {df.shape}")
    return df.values.astype(np.float32)

# ==== PART 2: Custom Forex Gym Environment ====
class ForexEnv(gym.Env):
    def __init__(self, history):
        super(ForexEnv, self).__init__()
        self.full_history = history
        self.max_steps = 200
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(60, 5), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.reset()

    def reset(self):
        self.start_index = np.random.randint(60, len(self.full_history) - self.max_steps)
        self.current_step = 0
        self.position = 0
        self.entry_price = None
        self.data = self.full_history[self.start_index - 60 : self.start_index]
        return self.data.reshape(1, 60, 5)

    def step(self, action):
        done = False
        reward = 0
        current_price = self.full_history[self.start_index + self.current_step][3]

        if action == 1 and self.position == 0:
            self.entry_price = current_price
            self.position = 1

        elif action == 2 and self.position == 0:
            self.entry_price = current_price
            self.position = -1

        elif action == 0 and self.position != 0:
            exit_price = current_price
            reward = (exit_price - self.entry_price) if self.position == 1 else (self.entry_price - exit_price)
            reward = np.clip(reward * 1000, -1, 1)
            self.position = 0
            self.entry_price = None

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        obs = self.full_history[self.start_index + self.current_step - 60 : self.start_index + self.current_step]
        return obs.reshape(1, 60, 5), reward, done, {}

# ==== PART 3: PPO Training ====
def train_agent():
    with open("settings.txt", "r") as file:
        lines = file.readlines()
        if len(lines) < 3:
            raise ValueError("CSV filename missing in settings.txt. Please select one from the GUI.")
        csv_name = lines[2].strip()

    if not os.path.exists(csv_name):
        raise FileNotFoundError(f"[ERROR] CSV file not found: {csv_name}")

    history = load_history(csv_name)
    assert not np.isnan(history).any(), "[ERROR] Dataset contains NaNs!"

    env = DummyVecEnv([lambda: ForexEnv(history)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)

    model.save("forex_trading_bot.zip")
    print("[INFO] Training complete. Model saved to 'forex_trading_bot.zip'")

# ==== PART 4: Evaluate PPO Model + Plot ====
def evaluate_agent(episodes=5):
    with open("settings.txt", "r") as file:
        lines = file.readlines()
        csv_name = lines[2].strip()

    history = load_history(csv_name)
    env = ForexEnv(history)

    assert os.path.exists("forex_trading_bot.zip"), "[ERROR] Model not found. Train it first."
    model = PPO.load("forex_trading_bot.zip")

    rewards = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        print(f"[EVAL] Episode {ep+1}: Reward = {total_reward:.4f}")

    print(f"\n[SUMMARY] Avg Reward over {episodes} episodes: {np.mean(rewards):.4f}")
    plot_evaluation_rewards(rewards)

def plot_evaluation_rewards(rewards):
    episodes = list(range(1, len(rewards) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, marker='o', linestyle='-', color='dodgerblue', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Evaluation Performance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("evaluation_rewards.png")
    plt.show()

# ==== PART 5: Entry ====
if __name__ == "__main__":
    train_agent()
    evaluate_agent(episodes=5)
