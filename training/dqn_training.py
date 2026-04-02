"""
DQN Training Script for NCD Prevention Environment
Includes extensive hyperparameter tuning with 10+ configurations
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from environment.custom_env import NCDPreventionEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import json
from datetime import datetime


class DQNTrainer:
    """DQN Training with hyperparameter tuning"""
    
    def __init__(self, env_id: str = "NCDPrevention-v0"):
        self.env_id = env_id
        self.results = []
        self.models = {}
        
    def train_dqn(self, 
                  learning_rate: float,
                  gamma: float,
                  buffer_size: int,
                  batch_size: int,
                  exploration_fraction: float,
                  exploration_final_eps: float,
                  total_timesteps: int = 50000,
                  run_name: str = "default") -> dict:
        """
        Train DQN model with given hyperparameters
        
        Args:
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            exploration_fraction: Fraction of total timesteps for exploration
            exploration_final_eps: Final exploration epsilon
            total_timesteps: Total timesteps to train
            run_name: Name for this run
        
        Returns:
            dict with training results and metrics
        """
        print(f"\n{'='*60}")
        print(f"Training DQN - Run: {run_name}")
        print(f"{'='*60}")
        print(f"LR: {learning_rate}, Gamma: {gamma}, Buffer: {buffer_size}")
        print(f"Batch: {batch_size}, Exploration: {exploration_fraction}")
        
        # Create environment
        env = NCDPreventionEnv(max_steps=52)
        
        # Train DQN
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            verbose=1,
            device="cpu"
        )
        
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate trained model
        eval_env = NCDPreventionEnv(max_steps=52)
        mean_reward, std_reward = self._evaluate_model(model, eval_env, num_episodes=10)
        
        # Store results
        result = {
            "run": run_name,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "exploration_fraction": exploration_fraction,
            "exploration_final_eps": exploration_final_eps,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "total_timesteps": total_timesteps
        }
        
        self.results.append(result)
        self.models[run_name] = model
        
        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Save model
        os.makedirs("models/dqn", exist_ok=True)
        model.save(f"models/dqn/dqn_{run_name}")
        
        env.close()
        eval_env.close()
        
        return result
    
    def _evaluate_model(self, model, env, num_episodes: int = 10) -> tuple:
        """
        Evaluate model performance
        
        Returns:
            (mean_reward, std_reward)
        """
        episode_rewards = []
        
        for _ in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards)
    
    def run_hyperparameter_sweep(self) -> pd.DataFrame:
        """Run 10+ different hyperparameter configurations"""
        
        configurations = [
            # Run 1: Low learning rate, high gamma
            {
                "learning_rate": 1e-4,
                "gamma": 0.99,
                "buffer_size": 50000,
                "batch_size": 32,
                "exploration_fraction": 0.1,
                "exploration_final_eps": 0.05,
                "run_name": "dqn_run_1_low_lr_high_gamma"
            },
            # Run 2: High learning rate, low gamma
            {
                "learning_rate": 1e-3,
                "gamma": 0.95,
                "buffer_size": 50000,
                "batch_size": 32,
                "exploration_fraction": 0.2,
                "exploration_final_eps": 0.1,
                "run_name": "dqn_run_2_high_lr_low_gamma"
            },
            # Run 3: Medium learning rate, medium gamma
            {
                "learning_rate": 5e-4,
                "gamma": 0.97,
                "buffer_size": 100000,
                "batch_size": 64,
                "exploration_fraction": 0.15,
                "exploration_final_eps": 0.05,
                "run_name": "dqn_run_3_medium"
            },
            # Run 4: Large buffer size
            {
                "learning_rate": 1e-4,
                "gamma": 0.99,
                "buffer_size": 200000,
                "batch_size": 128,
                "exploration_fraction": 0.1,
                "exploration_final_eps": 0.02,
                "run_name": "dqn_run_4_large_buffer"
            },
            # Run 5: Small buffer size
            {
                "learning_rate": 5e-4,
                "gamma": 0.95,
                "buffer_size": 10000,
                "batch_size": 16,
                "exploration_fraction": 0.3,
                "exploration_final_eps": 0.1,
                "run_name": "dqn_run_5_small_buffer"
            },
            # Run 6: Large batch size
            {
                "learning_rate": 1e-4,
                "gamma": 0.98,
                "buffer_size": 100000,
                "batch_size": 256,
                "exploration_fraction": 0.15,
                "exploration_final_eps": 0.05,
                "run_name": "dqn_run_6_large_batch"
            },
            # Run 7: Small batch size
            {
                "learning_rate": 5e-4,
                "gamma": 0.96,
                "buffer_size": 50000,
                "batch_size": 8,
                "exploration_fraction": 0.2,
                "exploration_final_eps": 0.08,
                "run_name": "dqn_run_7_small_batch"
            },
            # Run 8: High exploration
            {
                "learning_rate": 1e-4,
                "gamma": 0.97,
                "buffer_size": 50000,
                "batch_size": 32,
                "exploration_fraction": 0.5,
                "exploration_final_eps": 0.2,
                "run_name": "dqn_run_8_high_exploration"
            },
            # Run 9: Low exploration
            {
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "buffer_size": 100000,
                "batch_size": 64,
                "exploration_fraction": 0.05,
                "exploration_final_eps": 0.01,
                "run_name": "dqn_run_9_low_exploration"
            },
            # Run 10: Balanced & optimized
            {
                "learning_rate": 3e-4,
                "gamma": 0.98,
                "buffer_size": 75000,
                "batch_size": 48,
                "exploration_fraction": 0.15,
                "exploration_final_eps": 0.05,
                "run_name": "dqn_run_10_balanced"
            },
        ]
        
        for config in configurations:
            self.train_dqn(**config)
        
        # Save results to CSV
        results_df = pd.DataFrame(self.results)
        results_df.to_csv("models/dqn/dqn_results.csv", index=False)
        
        print(f"\n{'='*60}")
        print("DQN Hyperparameter Tuning Results Summary")
        print(f"{'='*60}")
        print(results_df.to_string())
        
        return results_df


def main():
    """Main training function"""
    trainer = DQNTrainer()
    results_df = trainer.run_hyperparameter_sweep()
    
    # Find best performing model
    best_idx = results_df["mean_reward"].idxmax()
    best_result = results_df.loc[best_idx]
    print(f"\n{'='*60}")
    print("Best DQN Model:")
    print(f"{'='*60}")
    print(best_result)
    
    return trainer, results_df


if __name__ == "__main__":
    trainer, results_df = main()
