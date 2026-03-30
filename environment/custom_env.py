"""
Custom Gymnasium Environment for NCD (Non-Communicable Disease) Risk Management
Agent learns to recommend personalized interventions to reduce patient disease risk
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any


class NCDPreventionEnv(gym.Env):
    """
    Healthcare Environment for Treatment Recommendation
    
    Goal: Agent recommends interventions to minimize disease risk
    State: Patient health metrics (age, BMI, blood pressure, glucose, cholesterol, etc.)
    Action: Intervention recommendation (exercise, diet, medication, consultation, etc.)
    Reward: Risk reduction - intervention costs + early detection bonus
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    # Action space
    ACTIONS = {
        0: "No Intervention",
        1: "Exercise Program (Low)",
        2: "Exercise Program (High)",
        3: "Dietary Intervention (Moderate)",
        4: "Dietary Intervention (Strict)",
        5: "Preventive Medication",
        6: "Stress Management Program",
        7: "Comprehensive Lifestyle Intervention",
        8: "Medical Consultation"
    }
    
    # Intervention costs (for reward calculation)
    ACTION_COSTS = {
        0: 0,      # No intervention
        1: 50,     # Low exercise
        2: 100,    # High exercise
        3: 75,     # Moderate diet
        4: 120,    # Strict diet
        5: 200,    # Medication
        6: 150,    # Stress management
        7: 300,    # Comprehensive
        8: 250     # Medical consultation
    }
    
    def __init__(self, render_mode: str = None, max_steps: int = 52):
        """
        Initialize the NCD Prevention Environment
        
        Args:
            render_mode: "human" for GUI, "rgb_array" for image output
            max_steps: Number of weeks in simulation (default: 52 weeks = 1 year)
        """
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: 9 discrete interventions
        self.action_space = spaces.Discrete(9)
        
        # Observation space: 10 continuous health metrics
        # [age, bmi, systolic_bp, diastolic_bp, glucose, cholesterol, 
        #  exercise_level, diet_quality, stress_level, disease_risk_score]
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(10,),
            dtype=np.float32
        )
        
        # Patient state (persistent across episodes)
        self.patient_state = None
        self.initial_patient_state = None
        self.episode_rewards = 0
        self.disease_progression_counter = 0
        
        # Rendering
        self.window = None
        self.clock = None
        
    def _init_patient(self) -> np.ndarray:
        """Initialize a random patient with health metrics"""
        # Age: 30-80 years (normalize to 0-100)
        age = np.random.uniform(30, 80)
        
        # BMI: 18-40 (normalize to 0-100)
        bmi = np.random.uniform(18, 40)
        
        # Blood pressure: systolic 90-180, diastolic 60-120
        systolic_bp = np.random.uniform(90, 180)
        diastolic_bp = np.random.uniform(60, 120)
        
        # Glucose: 70-200 mg/dL (normalize to 0-100)
        glucose = np.random.uniform(70, 200)
        
        # Cholesterol: 120-300 mg/dL (normalize to 0-100)
        cholesterol = np.random.uniform(120, 300)
        
        # Exercise level: 0-100 (weekly minutes normalized)
        exercise_level = np.random.uniform(20, 100)
        
        # Diet quality: 0-100 (subjective score)
        diet_quality = np.random.uniform(20, 80)
        
        # Stress level: 0-100 (subjective score)
        stress_level = np.random.uniform(20, 80)
        
        # Disease risk score: computed from health metrics
        disease_risk = self._compute_disease_risk(
            age, bmi, systolic_bp, diastolic_bp, glucose, 
            cholesterol, exercise_level, diet_quality, stress_level
        )
        
        state = np.array([
            age / 80 * 100,           # Normalize age to 0-100
            bmi / 40 * 100,           # Normalize BMI to 0-100
            systolic_bp / 180 * 100,  # Normalize BP to 0-100
            diastolic_bp / 120 * 100,
            glucose / 200 * 100,      # Normalize glucose to 0-100
            cholesterol / 300 * 100,  # Normalize cholesterol to 0-100
            exercise_level,
            diet_quality,
            stress_level,
            disease_risk
        ], dtype=np.float32)
        
        return state
    
    def _compute_disease_risk(self, age, bmi, systolic_bp, diastolic_bp, 
                              glucose, cholesterol, exercise, diet, stress) -> float:
        """
        Compute disease risk score based on health metrics
        Uses weighted sum of risk factors
        """
        # Normalize inputs to 0-1
        age_norm = age / 80
        bmi_norm = abs(bmi - 25) / 15  # Risk increases away from 25
        bp_norm = (systolic_bp + diastolic_bp) / 300
        glucose_norm = (glucose - 100) / 100 if glucose > 100 else (100 - glucose) / 100
        chol_norm = (cholesterol - 150) / 150 if cholesterol > 150 else (150 - cholesterol) / 150
        exercise_norm = 1 - (exercise / 100)  # Less exercise = more risk
        diet_norm = 1 - (diet / 100)  # Poor diet = more risk
        stress_norm = stress / 100
        
        # Weighted risk calculation
        risk = (
            0.15 * age_norm +
            0.15 * bmi_norm +
            0.20 * bp_norm +
            0.15 * glucose_norm +
            0.15 * chol_norm +
            0.10 * exercise_norm +
            0.05 * diet_norm +
            0.05 * stress_norm
        )
        
        return np.clip(risk * 100, 0, 100)
    
    def _apply_intervention(self, action: int, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply intervention and update patient state
        
        Returns:
            new_state: Updated patient state
            risk_reduction: How much disease risk was reduced
        """
        new_state = state.copy()
        old_risk = state[9]
        
        # Unpack current metrics
        age = state[0] / 100 * 80
        bmi = state[1] / 100 * 40
        systolic_bp = state[2] / 100 * 180
        diastolic_bp = state[3] / 100 * 120
        glucose = state[4] / 100 * 200
        cholesterol = state[5] / 100 * 300
        exercise = state[6]
        diet = state[7]
        stress = state[8]
        
        # Apply intervention effects with stochasticity
        if action == 0:  # No intervention
            pass
        
        elif action == 1:  # Low exercise
            exercise = min(100, exercise + np.random.uniform(5, 15))
            stress = max(0, stress - np.random.uniform(2, 5))
            
        elif action == 2:  # High exercise
            exercise = min(100, exercise + np.random.uniform(15, 30))
            stress = max(0, stress - np.random.uniform(5, 10))
            
        elif action == 3:  # Moderate diet
            glucose = glucose - np.random.uniform(5, 15)
            cholesterol = cholesterol - np.random.uniform(5, 15)
            diet = min(100, diet + np.random.uniform(5, 15))
            
        elif action == 4:  # Strict diet
            glucose = glucose - np.random.uniform(15, 30)
            cholesterol = cholesterol - np.random.uniform(15, 30)
            diet = min(100, diet + np.random.uniform(15, 25))
            
        elif action == 5:  # Preventive medication
            systolic_bp = systolic_bp - np.random.uniform(10, 20)
            diastolic_bp = diastolic_bp - np.random.uniform(5, 15)
            glucose = glucose - np.random.uniform(10, 20)
            cholesterol = cholesterol - np.random.uniform(10, 20)
            
        elif action == 6:  # Stress management
            stress = max(0, stress - np.random.uniform(15, 30))
            diet = min(100, diet + np.random.uniform(5, 10))
            
        elif action == 7:  # Comprehensive intervention
            exercise = min(100, exercise + np.random.uniform(15, 25))
            diet = min(100, diet + np.random.uniform(10, 20))
            stress = max(0, stress - np.random.uniform(10, 20))
            glucose = glucose - np.random.uniform(10, 20)
            cholesterol = cholesterol - np.random.uniform(10, 20)
            
        elif action == 8:  # Medical consultation
            systolic_bp = systolic_bp - np.random.uniform(5, 15)
            diastolic_bp = diastolic_bp - np.random.uniform(3, 10)
            glucose = glucose - np.random.uniform(5, 15)
            cholesterol = cholesterol - np.random.uniform(5, 15)
        
        # Apply natural disease progression (without intervention, risk increases with time)
        natural_progression = self.current_step * 0.3
        age += 52 / 52  # Age increases by 1 year per 52 weeks
        bmi += np.random.uniform(-1, 1)  # Natural BMI fluctuation
        
        # Constrain values to realistic ranges
        bmi = np.clip(bmi, 15, 45)
        glucose = np.clip(glucose, 60, 250)
        cholesterol = np.clip(cholesterol, 100, 350)
        systolic_bp = np.clip(systolic_bp, 80, 200)
        diastolic_bp = np.clip(diastolic_bp, 50, 140)
        exercise = np.clip(exercise, 0, 100)
        diet = np.clip(diet, 0, 100)
        stress = np.clip(stress, 0, 100)
        
        # Compute new risk
        new_risk = self._compute_disease_risk(
            age, bmi, systolic_bp, diastolic_bp, glucose,
            cholesterol, exercise, diet, stress
        ) + natural_progression
        new_risk = np.clip(new_risk, 0, 100)
        
        # Update state
        new_state[0] = age / 80 * 100
        new_state[1] = bmi / 40 * 100
        new_state[2] = systolic_bp / 180 * 100
        new_state[3] = diastolic_bp / 120 * 100
        new_state[4] = glucose / 200 * 100
        new_state[5] = cholesterol / 300 * 100
        new_state[6] = exercise
        new_state[7] = diet
        new_state[8] = stress
        new_state[9] = new_risk
        
        risk_reduction = old_risk - new_risk
        
        return new_state, risk_reduction
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_rewards = 0
        self.disease_progression_counter = 0
        
        self.patient_state = self._init_patient()
        self.initial_patient_state = self.patient_state.copy()
        
        return self.patient_state.copy(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Convert numpy array to int if needed (common with SB3 models)
        action = int(action) if isinstance(action, np.ndarray) else action
        
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        self.current_step += 1
        
        # Apply intervention
        new_state, risk_reduction = self._apply_intervention(action, self.patient_state)
        
        # Compute reward
        cost = self.ACTION_COSTS[action] / 500  # Normalize cost to reward scale
        early_detection_bonus = 0
        
        # Bonus for keeping risk low
        if new_state[9] < 30:
            early_detection_bonus = 5
        elif new_state[9] < 50:
            early_detection_bonus = 2
        
        reward = (risk_reduction - cost + early_detection_bonus) / 10
        
        self.patient_state = new_state.copy()
        self.episode_rewards += reward
        
        # Terminal conditions
        terminated = self.current_step >= self.max_steps
        truncated = new_state[9] >= 95  # Disease reaches critical level
        
        info = {
            "disease_risk": float(new_state[9]),
            "action": self.ACTIONS[action],
            "risk_reduction": float(risk_reduction),
            "cost": float(cost),
            "episode_reward": float(self.episode_rewards)
        }
        
        return new_state.copy(), reward, terminated, truncated, info
    
    def render(self):
        """Render environment state"""
        if self.render_mode == "human":
            print(f"\n--- Week {self.current_step} ---")
            print(f"Patient State: {self.patient_state}")
            print(f"Disease Risk: {self.patient_state[9]:.2f}")
            print(f"Episode Reward: {self.episode_rewards:.2f}")
    
    def close(self):
        """Clean up resources"""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
