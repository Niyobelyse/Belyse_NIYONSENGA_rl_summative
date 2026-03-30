"""
Visualization and Rendering for NCD Prevention Environment using Pygame
"""

import pygame
import numpy as np
from typing import Tuple, Dict, Optional
import math


class HealthEnvironmentVisualizer:
    """
    Pygame-based visualization for NCD Prevention Environment
    Displays patient health metrics, agent actions, and environment state
    """
    
    def __init__(self, width: int = 1200, height: int = 800):
        """Initialize visualizer"""
        pygame.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption("NCD Prevention RL Agent - Healthcare Intervention")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 28)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Colors
        self.BG_COLOR = (240, 240, 240)
        self.TEXT_COLOR = (30, 30, 30)
        self.ACCENT_COLOR = (70, 130, 180)
        self.GOOD_COLOR = (34, 139, 34)  # Forest green
        self.WARNING_COLOR = (255, 165, 0)  # Orange
        self.DANGER_COLOR = (220, 20, 60)  # Crimson
        
    def close(self):
        """Close visualizer"""
        pygame.quit()
    
    def _map_risk_to_color(self, risk: float) -> Tuple[int, int, int]:
        """Map disease risk to color gradient"""
        # Green (low risk) -> Yellow (medium) -> Red (high risk)
        if risk < 33:
            # Green to Yellow
            r = int(34 + (255 - 34) * (risk / 33))
            g = 139
            b = 34
        elif risk < 66:
            # Yellow to Orange
            r = 255
            g = int(165 - (165 - 69) * ((risk - 33) / 33))
            b = 0
        else:
            # Orange to Red
            r = 255
            g = int(69 - 49 * ((risk - 66) / 34))
            b = 0
        
        return (int(r), int(g), int(b))
    
    def visualize_random_actions(self, env, num_steps: int = 52):
        """
        Visualize agent taking random actions in environment
        Used to demonstrate environment mechanics without trained model
        """
        obs, info = env.reset()
        
        for step in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            self.display.fill(self.BG_COLOR)
            self._draw_state(obs, info, action, step, num_steps, is_random=True)
            
            pygame.display.flip()
            self.clock.tick(2)  # 2 FPS for viewing
            
            if terminated or truncated:
                break
        
        pygame.time.wait(2000)
    
    def visualize_trained_agent(self, env, model, num_steps: int = 52):
        """
        Visualize trained agent taking actions in environment
        """
        obs, info = env.reset()
        
        for step in range(num_steps):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action) if isinstance(action, np.ndarray) else action
            obs, reward, terminated, truncated, info = env.step(action)
            
            self.display.fill(self.BG_COLOR)
            self._draw_state(obs, info, int(action), step, num_steps, is_random=False)
            
            pygame.display.flip()
            self.clock.tick(2)  # 2 FPS for viewing
            
            if terminated or truncated:
                break
        
        pygame.time.wait(2000)
    
    def _draw_state(self, obs: np.ndarray, info: Dict, action: int, 
                   current_step: int, total_steps: int, is_random: bool = False):
        """Draw current environment state"""
        margin = 20
        
        # Title
        mode_text = "Random Actions" if is_random else "Trained Agent"
        title = self.font_large.render(
            f"NCD Prevention RL Environment - {mode_text}",
            True, self.ACCENT_COLOR
        )
        self.display.blit(title, (margin, margin))
        
        # Progress bar
        progress_y = 60
        progress_width = self.width - 2 * margin
        progress_height = 20
        pygame.draw.rect(self.display, (200, 200, 200), 
                        (margin, progress_y, progress_width, progress_height))
        progress = (current_step / total_steps) * progress_width
        pygame.draw.rect(self.display, self.ACCENT_COLOR,
                        (margin, progress_y, progress, progress_height))
        
        progress_text = self.font_small.render(f"Week {current_step + 1}/{total_steps}", 
                                              True, self.TEXT_COLOR)
        self.display.blit(progress_text, (margin + 10, progress_y + 2))
        
        # Patient Metrics Panel
        metrics_y = 100
        self._draw_metrics_panel(obs, metrics_y)
        
        # Risk Gauge
        gauge_y = 300
        self._draw_risk_gauge(obs[9], gauge_y)
        
        # Action Panel
        action_y = 500
        self._draw_action_panel(action, action_y, info)
        
        # Info Panel
        info_y = 650
        self._draw_info_panel(info, current_step, info_y)
    
    def _draw_metrics_panel(self, obs: np.ndarray, y: int):
        """Draw patient health metrics"""
        margin = 20
        x = margin
        
        label = self.font_medium.render("Patient Health Metrics:", True, self.TEXT_COLOR)
        self.display.blit(label, (x, y))
        
        # Denormalize observations
        age = int(obs[0] / 100 * 80)
        bmi = round(obs[1] / 100 * 40, 1)
        systolic_bp = int(obs[2] / 100 * 180)
        diastolic_bp = int(obs[3] / 100 * 120)
        glucose = int(obs[4] / 100 * 200)
        cholesterol = int(obs[5] / 100 * 300)
        exercise = round(obs[6], 1)
        diet = round(obs[7], 1)
        stress = round(obs[8], 1)
        
        metrics = [
            f"Age: {age} years",
            f"BMI: {bmi} kg/m²",
            f"Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg",
            f"Glucose: {glucose} mg/dL",
            f"Cholesterol: {cholesterol} mg/dL",
            f"Exercise Level: {exercise:.1f}/100",
            f"Diet Quality: {diet:.1f}/100",
            f"Stress Level: {stress:.1f}/100"
        ]
        
        y_offset = y + 30
        for metric in metrics:
            text = self.font_small.render(metric, True, self.TEXT_COLOR)
            self.display.blit(text, (x + 20, y_offset))
            y_offset += 22
    
    def _draw_risk_gauge(self, risk: float, y: int):
        """Draw disease risk gauge visualization"""
        margin = 20
        x = margin + 500
        
        label = self.font_medium.render("Disease Risk Score:", True, self.TEXT_COLOR)
        self.display.blit(label, (x, y))
        
        # Gauge background
        gauge_width = 250
        gauge_height = 40
        gauge_x = x
        gauge_y = y + 30
        
        pygame.draw.rect(self.display, (220, 220, 220),
                        (gauge_x, gauge_y, gauge_width, gauge_height))
        pygame.draw.rect(self.display, self.TEXT_COLOR,
                        (gauge_x, gauge_y, gauge_width, gauge_height), 2)
        
        # Risk fill
        risk_fill = int((risk / 100) * gauge_width)
        risk_color = self._map_risk_to_color(risk)
        pygame.draw.rect(self.display, risk_color,
                        (gauge_x, gauge_y, risk_fill, gauge_height))
        
        # Risk value text
        risk_text = self.font_medium.render(f"{risk:.1f}/100", True, self.TEXT_COLOR)
        self.display.blit(risk_text, (gauge_x + 270, gauge_y + 5))
        
        # Risk level indicator
        if risk < 33:
            level = "LOW RISK ✓"
            color = self.GOOD_COLOR
        elif risk < 66:
            level = "MEDIUM RISK !"
            color = self.WARNING_COLOR
        else:
            level = "HIGH RISK ✗"
            color = self.DANGER_COLOR
        
        level_text = self.font_small.render(level, True, color)
        self.display.blit(level_text, (gauge_x, gauge_y + 50))
    
    def _draw_action_panel(self, action: int, y: int, info: Dict):
        """Draw current action taken by agent"""
        margin = 20
        x = margin
        
        label = self.font_medium.render("Agent Action:", True, self.TEXT_COLOR)
        self.display.blit(label, (x, y))
        
        action_name = info.get("action", "Unknown")
        action_text = self.font_medium.render(f"→ {action_name}", 
                                             True, self.ACCENT_COLOR)
        self.display.blit(action_text, (x + 20, y + 30))
        
        # Reward info
        reward_info = f"Risk Reduction: {info.get('risk_reduction', 0):.2f} | "
        reward_info += f"Cost: {info.get('cost', 0):.2f}"
        reward_text = self.font_small.render(reward_info, True, self.TEXT_COLOR)
        self.display.blit(reward_text, (x + 20, y + 60))
    
    def _draw_info_panel(self, info: Dict, step: int, y: int):
        """Draw episode information"""
        margin = 20
        x = margin
        
        episode_reward = info.get("episode_reward", 0)
        risk = info.get("disease_risk", 0)
        
        info_text = f"Step: {step} | Episode Reward: {episode_reward:.2f} | Disease Risk: {risk:.1f}"
        text = self.font_small.render(info_text, True, self.TEXT_COLOR)
        self.display.blit(text, (x, y))
    
    def create_observation_summary(self, obs: np.ndarray) -> str:
        """Create human-readable summary of observations"""
        age = int(obs[0] / 100 * 80)
        bmi = round(obs[1] / 100 * 40, 1)
        systolic_bp = int(obs[2] / 100 * 180)
        diastolic_bp = int(obs[3] / 100 * 120)
        glucose = int(obs[4] / 100 * 200)
        cholesterol = int(obs[5] / 100 * 300)
        
        summary = f"Patient: {age}y, BMI:{bmi}, BP:{systolic_bp}/{diastolic_bp}, "
        summary += f"Glucose:{glucose}, Chol:{cholesterol}"
        return summary
