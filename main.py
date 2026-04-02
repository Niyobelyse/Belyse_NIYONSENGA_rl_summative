"""
Main Entry Point for NCD Prevention RL Agent
Runs best-performing trained models with visualization
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from environment.custom_env import NCDPreventionEnv
from environment.rendering import HealthEnvironmentVisualizer
from stable_baselines3 import DQN, PPO, A2C
import warnings
warnings.filterwarnings('ignore')


class NCDPreventionAgent:
    """Main agent runner for demonstration and testing"""
    
    def __init__(self):
        self.env = NCDPreventionEnv(max_steps=52)
        self.visualizer = HealthEnvironmentVisualizer()
        
    def run_random_agent(self, num_episodes: int = 3):
        """Run agent with random actions for baseline"""
        print("\n" + "="*60)
        print("Running Random Agent (Baseline)")
        print("="*60)
        
        episode_rewards = []
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            
            print(f"\nEpisode {ep+1}/{num_episodes}")
            print(f"Initial Disease Risk: {obs[9]:.2f}")
            
            step_count = 0
            while step_count < self.env.max_steps:
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                step_count += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            print(f"Final Disease Risk: {obs[9]:.2f}")
            print(f"Episode Reward: {episode_reward:.2f}")
        
        print(f"\nAverage Reward (Random): {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        return episode_rewards
    
    def run_trained_agent(self, model_path: str, model_type: str = "dqn", 
                         num_episodes: int = 3, visualize: bool = False):
        """
        Run trained agent
        
        Args:
            model_path: Path to saved model
            model_type: Type of model ("dqn", "ppo", "reinforce")
            num_episodes: Number of episodes to run
            visualize: Whether to visualize with Pygame
        """
        print("\n" + "="*60)
        print(f"Running Trained {model_type.upper()} Agent")
        print("="*60)
        
        # Load model
        if model_type.lower() == "dqn":
            model = DQN.load(model_path)
        elif model_type.lower() == "ppo":
            model = PPO.load(model_path)
        elif model_type.lower() == "reinforce":
            model = A2C.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        episode_rewards = []
        episode_returns = []
        risk_reductions = []
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            initial_risk = obs[9]
            total_risk_reduction = 0
            
            print(f"\nEpisode {ep+1}/{num_episodes}")
            print(f"Initial Patient State:")
            print(f"  Age: {int(obs[0]/100*80)}, BMI: {obs[1]/100*40:.1f}")
            print(f"  Disease Risk: {initial_risk:.2f}")
            
            step_count = 0
            while step_count < self.env.max_steps:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action) if isinstance(action, np.ndarray) else action
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                total_risk_reduction += info.get("risk_reduction", 0)
                step_count += 1
                
                if step_count % 10 == 0:
                    print(f"  Week {step_count}: Risk={obs[9]:.2f}, Action={info['action']}")
                
                if terminated or truncated:
                    break
            
            final_risk = obs[9]
            risk_reduction_percent = (initial_risk - final_risk) / initial_risk * 100
            
            episode_rewards.append(episode_reward)
            episode_returns.append(final_risk)
            risk_reductions.append(risk_reduction_percent)
            
            print(f"Final Disease Risk: {final_risk:.2f}")
            print(f"Risk Reduction: {risk_reduction_percent:.1f}%")
            print(f"Episode Reward: {episode_reward:.2f}")
        
        print(f"\n{'='*60}")
        print("Performance Summary")
        print(f"{'='*60}")
        print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average Final Risk: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
        print(f"Average Risk Reduction: {np.mean(risk_reductions):.1f}% ± {np.std(risk_reductions):.1f}%")
        
        if visualize:
            print("\nStarting Pygame Visualization...")
            self.visualizer.visualize_trained_agent(self.env, model, num_steps=52)
        
        return episode_rewards, episode_returns, risk_reductions
    
    def run_random_visualization(self):
        """Visualize random agent with Pygame"""
        print("\nStarting Random Agent Visualization...")
        self.visualizer.visualize_random_actions(self.env, num_steps=52)
    
    def compare_algorithms(self):
        """Compare performance of all trained algorithms"""
        print("\n" + "="*60)
        print("Algorithm Comparison")
        print("="*60)
        
        # Load best models
        dqn_model = DQN.load("models/dqn/dqn_dqn_run_8_high_exploration")
        ppo_model = PPO.load("models/pg/ppo_ppo_run_9")
        reinforce_model = A2C.load("models/pg/reinforce_reinforce_run_9_low_lr_stable")
        
        results = {}
        
        for name, model in [("DQN", dqn_model), ("PPO", ppo_model), ("REINFORCE", reinforce_model)]:
            rewards = []
            for _ in range(5):
                obs, _ = self.env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action) if isinstance(action, np.ndarray) else action
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                
                rewards.append(episode_reward)
            
            results[name] = {
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards)
            }
            print(f"{name}: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        
        return results
    
    def close(self):
        """Clean up resources"""
        self.env.close()
        self.visualizer.close()


def main():
    parser = argparse.ArgumentParser(description="NCD Prevention RL Agent")
    parser.add_argument("--mode", type=str, choices=["random", "dqn", "ppo", "reinforce", "compare"],
                       default="random", help="Mode to run agent")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--visualize", action="store_true", help="Enable Pygame visualization")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    
    args = parser.parse_args()
    
    agent = NCDPreventionAgent()
    
    try:
        if args.mode == "random":
            if args.visualize:
                agent.run_random_visualization()
            else:
                agent.run_random_agent(num_episodes=args.episodes)
        
        elif args.mode == "dqn":
            model_path = args.model_path or "models/dqn/dqn_dqn_run_8_high_exploration"
            agent.run_trained_agent(model_path, "dqn", args.episodes, args.visualize)
        
        elif args.mode == "ppo":
            model_path = args.model_path or "models/pg/ppo_ppo_run_9"
            agent.run_trained_agent(model_path, "ppo", args.episodes, args.visualize)
        
        elif args.mode == "reinforce":
            model_path = args.model_path or "models/pg/reinforce_reinforce_run_9_low_lr_stable"
            agent.run_trained_agent(model_path, "reinforce", args.episodes, args.visualize)
        
        elif args.mode == "compare":
            agent.compare_algorithms()
    
    finally:
        agent.close()


if __name__ == "__main__":
    main()
