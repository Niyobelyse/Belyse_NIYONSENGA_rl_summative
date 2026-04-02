"""
Policy Gradient Training Scripts: REINFORCE and PPO
Includes extensive hyperparameter tuning with 10+ configurations each
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
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
warnings.filterwarnings('ignore')


class PolicyGradientTrainer:
    """Policy Gradient Training with hyperparameter tuning - REINFORCE (via A2C) and PPO"""
    
    def __init__(self):
        self.reinforce_results = []
        self.ppo_results = []
        self.models_reinforce = {}
        self.models_ppo = {}
    
    # ==================== REINFORCE Training ====================
    # Note: Pure REINFORCE not available in SB3, using A2C with n_steps=1 for REINFORCE-like behavior
    
    def train_reinforce(self,
                       learning_rate: float,
                       gamma: float,
                       entropy_coef: float,
                       value_fn_coef: float,
                       n_steps: int = 1,
                       total_timesteps: int = 50000,
                       run_name: str = "default") -> dict:
        """
        Train REINFORCE-like agent (using A2C with n_steps=1)
        
        Args:
            learning_rate: Learning rate
            gamma: Discount factor
            entropy_coef: Entropy coefficient for exploration
            value_fn_coef: Value function loss coefficient
            n_steps: Number of steps (1 for REINFORCE)
            total_timesteps: Total training timesteps
            run_name: Run identifier
        
        Returns:
            dict with training results
        """
        print(f"\n{'='*60}")
        print(f"Training REINFORCE - Run: {run_name}")
        print(f"{'='*60}")
        print(f"LR: {learning_rate}, Gamma: {gamma}, Entropy: {entropy_coef}")
        
        env = NCDPreventionEnv(max_steps=52)
        
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            ent_coef=entropy_coef,
            vf_coef=value_fn_coef,
            n_steps=n_steps,
            verbose=1,
            device="cpu"
        )
        
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate
        eval_env = NCDPreventionEnv(max_steps=52)
        mean_reward, std_reward = self._evaluate_model(model, eval_env, num_episodes=10)
        
        result = {
            "run": run_name,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "entropy_coef": entropy_coef,
            "value_fn_coef": value_fn_coef,
            "n_steps": n_steps,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "total_timesteps": total_timesteps
        }
        
        self.reinforce_results.append(result)
        self.models_reinforce[run_name] = model
        
        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        os.makedirs("models/pg", exist_ok=True)
        model.save(f"models/pg/reinforce_{run_name}")
        
        env.close()
        eval_env.close()
        
        return result
    
    # ==================== PPO Training ====================
    
    def train_ppo(self,
                  learning_rate: float,
                  gamma: float,
                  clip_range: float,
                  ent_coef: float,
                  vf_coef: float,
                  n_steps: int = 2048,
                  batch_size: int = 64,
                  n_epochs: int = 10,
                  total_timesteps: int = 50000,
                  run_name: str = "default") -> dict:
        """
        Train PPO agent
        
        Args:
            learning_rate: Learning rate
            gamma: Discount factor
            clip_range: PPO clipping range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            n_steps: Number of steps per batch
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            total_timesteps: Total training timesteps
            run_name: Run identifier
        
        Returns:
            dict with training results
        """
        print(f"\n{'='*60}")
        print(f"Training PPO - Run: {run_name}")
        print(f"{'='*60}")
        print(f"LR: {learning_rate}, Gamma: {gamma}, Clip: {clip_range}")
        print(f"Entropy: {ent_coef}, VF Coef: {vf_coef}")
        
        env = NCDPreventionEnv(max_steps=52)
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            n_steps=min(n_steps, 1000),  # Limit to episode size
            batch_size=batch_size,
            n_epochs=n_epochs,
            verbose=1,
            device="cpu"
        )
        
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate
        eval_env = NCDPreventionEnv(max_steps=52)
        mean_reward, std_reward = self._evaluate_model(model, eval_env, num_episodes=10)
        
        result = {
            "run": run_name,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "clip_range": clip_range,
            "entropy_coef": ent_coef,
            "vf_coef": vf_coef,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "total_timesteps": total_timesteps
        }
        
        self.ppo_results.append(result)
        self.models_ppo[run_name] = model
        
        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        os.makedirs("models/pg", exist_ok=True)
        model.save(f"models/pg/ppo_{run_name}")
        
        env.close()
        eval_env.close()
        
        return result
    
    def _evaluate_model(self, model, env, num_episodes: int = 10) -> tuple:
        """Evaluate model performance"""
        episode_rewards = []
        
        for _ in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action) if isinstance(action, np.ndarray) else action
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards)
    
    def run_reinforce_sweep(self) -> pd.DataFrame:
        """Run 10+ REINFORCE configurations"""
        
        configurations = [
            {"learning_rate": 1e-3, "gamma": 0.99, "entropy_coef": 0.01, 
             "value_fn_coef": 0.5, "run_name": "reinforce_run_1_low_lr"},
            
            {"learning_rate": 5e-3, "gamma": 0.95, "entropy_coef": 0.05, 
             "value_fn_coef": 0.5, "run_name": "reinforce_run_2_high_lr"},
            
            {"learning_rate": 3e-3, "gamma": 0.97, "entropy_coef": 0.01, 
             "value_fn_coef": 1.0, "run_name": "reinforce_run_3_medium"},
            
            {"learning_rate": 1e-3, "gamma": 0.99, "entropy_coef": 0.05, 
             "value_fn_coef": 0.5, "run_name": "reinforce_run_4_high_entropy"},
            
            {"learning_rate": 5e-3, "gamma": 0.90, "entropy_coef": 0.01, 
             "value_fn_coef": 0.5, "run_name": "reinforce_run_5_low_gamma"},
            
            {"learning_rate": 2e-3, "gamma": 0.98, "entropy_coef": 0.02, 
             "value_fn_coef": 2.0, "run_name": "reinforce_run_6_high_vf"},
            
            {"learning_rate": 1e-3, "gamma": 0.96, "entropy_coef": 0.01, 
             "value_fn_coef": 0.5, "run_name": "reinforce_run_7_balanced"},
            
            {"learning_rate": 8e-3, "gamma": 0.94, "entropy_coef": 0.03, 
             "value_fn_coef": 0.5, "run_name": "reinforce_run_8_exploration"},
            
            {"learning_rate": 5e-4, "gamma": 0.99, "entropy_coef": 0.01, 
             "value_fn_coef": 1.0, "run_name": "reinforce_run_9_low_lr_stable"},
            
            {"learning_rate": 3e-3, "gamma": 0.97, "entropy_coef": 0.02, 
             "value_fn_coef": 1.0, "run_name": "reinforce_run_10_tuned"},
        ]
        
        for config in configurations:
            self.train_reinforce(**config)
        
        results_df = pd.DataFrame(self.reinforce_results)
        results_df.to_csv("models/pg/reinforce_results.csv", index=False)
        
        print(f"\n{'='*60}")
        print("REINFORCE Hyperparameter Tuning Results Summary")
        print(f"{'='*60}")
        print(results_df.to_string())
        
        return results_df
    
    def run_ppo_sweep(self) -> pd.DataFrame:
        """Run 10+ PPO configurations"""
        
        configurations = [
            {"learning_rate": 1e-4, "gamma": 0.99, "clip_range": 0.2, 
             "ent_coef": 0.01, "vf_coef": 0.5, "n_epochs": 10, "run_name": "ppo_run_1"},
            
            {"learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.2, 
             "ent_coef": 0.01, "vf_coef": 0.5, "n_epochs": 10, "run_name": "ppo_run_2"},
            
            {"learning_rate": 5e-4, "gamma": 0.99, "clip_range": 0.2, 
             "ent_coef": 0.01, "vf_coef": 0.5, "n_epochs": 10, "run_name": "ppo_run_3"},
            
            {"learning_rate": 1e-4, "gamma": 0.95, "clip_range": 0.2, 
             "ent_coef": 0.01, "vf_coef": 0.5, "n_epochs": 10, "run_name": "ppo_run_4"},
            
            {"learning_rate": 1e-4, "gamma": 0.99, "clip_range": 0.3, 
             "ent_coef": 0.01, "vf_coef": 0.5, "n_epochs": 10, "run_name": "ppo_run_5"},
            
            {"learning_rate": 1e-4, "gamma": 0.99, "clip_range": 0.1, 
             "ent_coef": 0.01, "vf_coef": 0.5, "n_epochs": 10, "run_name": "ppo_run_6"},
            
            {"learning_rate": 1e-4, "gamma": 0.99, "clip_range": 0.2, 
             "ent_coef": 0.05, "vf_coef": 0.5, "n_epochs": 10, "run_name": "ppo_run_7"},
            
            {"learning_rate": 1e-4, "gamma": 0.99, "clip_range": 0.2, 
             "ent_coef": 0.01, "vf_coef": 1.0, "n_epochs": 10, "run_name": "ppo_run_8"},
            
            {"learning_rate": 1e-4, "gamma": 0.99, "clip_range": 0.2, 
             "ent_coef": 0.01, "vf_coef": 0.5, "n_epochs": 20, "run_name": "ppo_run_9"},
            
            {"learning_rate": 2e-4, "gamma": 0.97, "clip_range": 0.2, 
             "ent_coef": 0.02, "vf_coef": 0.7, "n_epochs": 15, "run_name": "ppo_run_10"},
        ]
        
        for config in configurations:
            self.train_ppo(**config)
        
        results_df = pd.DataFrame(self.ppo_results)
        results_df.to_csv("models/pg/ppo_results.csv", index=False)
        
        print(f"\n{'='*60}")
        print("PPO Hyperparameter Tuning Results Summary")
        print(f"{'='*60}")
        print(results_df.to_string())
        
        return results_df


def main():
    """Main training function"""
    trainer = PolicyGradientTrainer()
    
    print("\n" + "="*60)
    print("REINFORCE Training")
    print("="*60)
    reinforce_results = trainer.run_reinforce_sweep()
    
    print("\n" + "="*60)
    print("PPO Training")
    print("="*60)
    ppo_results = trainer.run_ppo_sweep()
    
    # Summary
    print(f"\n{'='*60}")
    print("BEST MODELS SUMMARY")
    print(f"{'='*60}")
    
    best_reinforce_idx = reinforce_results["mean_reward"].idxmax()
    best_reinforce = reinforce_results.loc[best_reinforce_idx]
    print(f"\nBest REINFORCE: {best_reinforce['run']}")
    print(f"Mean Reward: {best_reinforce['mean_reward']:.2f}")
    
    best_ppo_idx = ppo_results["mean_reward"].idxmax()
    best_ppo = ppo_results.loc[best_ppo_idx]
    print(f"\nBest PPO: {best_ppo['run']}")
    print(f"Mean Reward: {best_ppo['mean_reward']:.2f}")
    
    return trainer, reinforce_results, ppo_results


if __name__ == "__main__":
    trainer, reinforce_results, ppo_results = main()
