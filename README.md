# NCD Prevention RL Agent - Healthcare Policy Learning

##  Project Overview

This project implements a **Reinforcement Learning solution for Non-Communicable Disease (NCD) prevention** through **intelligent treatment recommendation**. The agent learns to recommend personalized interventions (exercise programs, dietary changes, medications, stress management, etc.) to minimize patient disease risk.

**Problem Statement**: Non-communicable diseases (diabetes, hypertension, cardiovascular disease, etc.) are among the leading causes of mortality globally. Early intervention and personalized prevention strategies can significantly reduce prevalence and severity. This RL agent learns optimal intervention policies to minimize disease risk progression.

**Mission Alignment**: Reduce the impact of NCDs by developing ML models for early detection and prevention, helping people understand health risks sooner and making healthcare management easier.

---

##  Environment Description

### Agent
The agent represents a healthcare decision system that observes patient health metrics and recommends interventions to minimize disease risk. The agent operates over a 52-week (1-year) simulation period with weekly decision points.

### Action Space (Discrete - 9 Actions)
```
0: No Intervention
1: Exercise Program (Low Intensity)
2: Exercise Program (High Intensity)
3: Dietary Intervention (Moderate)
4: Dietary Intervention (Strict)
5: Preventive Medication
6: Stress Management Program
7: Comprehensive Lifestyle Intervention
8: Medical Consultation
```

### Observation Space (10 Continuous Features)
```
[age, bmi, systolic_bp, diastolic_bp, glucose, cholesterol, 
 exercise_level, diet_quality, stress_level, disease_risk_score]
```

All observations normalized to [0, 100] for stable learning.

### Reward Function
```
reward = -risk_reduction_weight * (new_risk - old_risk) 
         - cost_weight * intervention_cost
         + early_detection_bonus
```

**Components**:
- **Risk Reduction**: Negative reward for increasing disease risk
- **Intervention Costs**: Different actions have different costs (no intervention: $0, comprehensive: $300)
- **Early Detection Bonus**: Rewards keeping risk below thresholds

### Terminal Conditions
- Episode ends at 52 weeks (calendar year)
- Early termination if disease risk exceeds 95% (critical stage)

---

## Algorithms Implemented

### 1. **DQN (Deep Q-Network)** - Value-Based Learning
- Neural network approximates Q-values
- Experience replay buffer for sample efficiency
- Target network for training stability
- **10 hyperparameter configurations tested**

**Best DQN Configuration**:
- Learning Rate: 3e-4
- Gamma: 0.98
- Buffer Size: 75,000
- Batch Size: 48

### 2. **REINFORCE** - Policy Gradient Method
- Monte Carlo policy gradients
- Implemented via A2C with n_steps=1
- Entropy regularization for exploration
- **10 hyperparameter configurations tested**

**Best REINFORCE Configuration**:
- Learning Rate: 3e-3
- Gamma: 0.97
- Entropy Coefficient: 0.02
- Value Function Coefficient: 1.0

### 3. **PPO (Proximal Policy Optimization)** - Policy Gradient Method
- Clipped objective for sample efficiency
- Multiple epochs per data batch
- Trust region optimization
- **10 hyperparameter configurations tested**

**Best PPO Configuration**:
- Learning Rate: 2e-4
- Gamma: 0.97
- Clip Range: 0.2
- Entropy Coefficient: 0.02
- Value Function Coefficient: 0.7
- Epochs: 15

---

##  Project Structure

```
reinforcement_learning/
├── environment/
│   ├── custom_env.py          # NCDPreventionEnv Gymnasium implementation
│   ├── rendering.py            # Pygame visualization components
│   └── __init__.py
├── training/
│   ├── dqn_training.py         # DQN training script (10+ runs)
│   ├── pg_training.py          # REINFORCE & PPO training (10+ runs each)
│   └── __init__.py
├── models/
│   ├── dqn/                    # Saved DQN models & results
│   │   ├── dqn_*.zip
│   │   └── dqn_results.csv
│   └── pg/                     # Saved PG models & results
│       ├── ppo_*.zip
│       ├── reinforce_*.zip
│       ├── ppo_results.csv
│       └── reinforce_results.csv
├── main.py                     # Entry point for running best model
├── requirements.txt            # Project dependencies
├── README.md                   # This file
└── .git/                       # Git repository
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Installation Steps

1. **Clone the repository** (or create locally):
```bash
git clone https://github.com/reinforcement_learning.git
cd reinforcement_learning
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

---

## Training & Evaluation

### Train All Models (Full Hyperparameter Sweep)

```bash
# Train DQN models (10+ runs)
python training/dqn_training.py

# Train REINFORCE & PPO models (10+ runs each)
python training/pg_training.py
```

This will:
- Train multiple configurations for each algorithm
- Save all trained models to `models/` directory
- Generate CSV files with hyperparameter results

### Run Best-Performing Agent

```bash
# Run best pretrained models (requires models to exist)
python main.py --mode "dqn" --episodes 3
python main.py --mode "ppo" --episodes 3
python main.py --mode "reinforce" --episodes 3

# Compare all algorithms
python main.py --mode compare

# Visualize with Pygame
python main.py --mode "dqn" --visualize
```

### Random Baseline

```bash
# Run random agent (no learning)
python main.py --mode random --episodes 5
python main.py --mode random --visualize
```

---

## Performance Metrics & Analysis

### Evaluation Metrics
- **Mean Episode Reward**: Total discounted reward per episode
- **Disease Risk Score**: Final patient risk (lower is better, 0-100)
- **Risk Reduction %**: Percentage improvement from initial to final risk
- **Algorithm Stability**: Standard deviation of rewards across episodes

### Hyperparameter Tuning Results

#### DQN Results (10 Runs)
| Run | Learning Rate | Gamma | Buffer Size | Batch Size | Mean Reward |
|-----|---------------|-------|-------------|-----------|------------|
| 1   | 1e-4          | 0.99  | 50000       | 32        | -2.45      |
| 2   | 1e-3          | 0.95  | 50000       | 32        | -2.89      |
| 3   | 5e-4          | 0.97  | 100000      | 64        | -1.82      |
| ... | ...           | ...   | ...         | ...       | ...        |
| 10  | 3e-4          | 0.98  | 75000       | 48        | **-1.42**  |

#### REINFORCE Results (10 Runs)
| Run | Learning Rate | Gamma | Entropy | Value Coef | Mean Reward |
|-----|---------------|-------|---------|-----------|------------|
| 1   | 1e-3          | 0.99  | 0.01    | 0.5       | -2.12      |
| ...| ...           | ...   | ...     | ...       | ...        |
| 10  | 3e-3          | 0.97  | 0.02    | 1.0       | **-1.55**  |

#### PPO Results (10 Runs)
| Run | Learning Rate | Gamma | Clip | Entropy | Epochs | Mean Reward |
|-----|---------------|-------|------|---------|--------|------------|
| 1   | 1e-4          | 0.99  | 0.2  | 0.01    | 10     | -1.98      |
| ...| ...           | ...   | ...  | ...     | ...    | ...        |
| 10  | 2e-4          | 0.97  | 0.2  | 0.02    | 15     | **-1.28**  |

---

## Video Demonstration

A demonstration video showing:
1. **Problem Statement**: NCD prevention through intelligent intervention
2. **Agent Behavior**: How the agent recommends interventions over time
3. **Reward Structure**: How different actions affect disease risk
4. **Objective**: Minimize patient disease risk through learned policy
5. **Simulation**: GUI visualization with terminal output showing:
   - Patient health metrics
   - Weekly interventions
   - Disease risk trajectory
   - Agent performance metrics

**Video Link**: [To be added - recorded with camera on, full screen share]

---

## Key Insights & Discussion

### Algorithm Comparison

1. **DQN (Value-Based)**
   - Best for: Discrete action spaces
   - Strengths: Sample efficient, convergent to optimal policy
   - Weaknesses: High variance in early training, can be unstable

2. **REINFORCE (Policy Gradient)**
   - Best for: Exploration requirements
   - Strengths: Directly optimize policy, good for stochastic policies
   - Weaknesses: High variance, slower convergence

3. **PPO (Policy Gradient)**
   - Best for: Stability and performance trade-off
   - Strengths: More stable than REINFORCE, easier to tune
   - Weaknesses: More samples required than DQN

### Performance Analysis

**Best Performing Algorithm**: PPO (Mean Reward: **-1.28**)
- Achieved lowest disease risk in simulation
- Most stable training across hyperparameter variations
- Good balance between sample efficiency and stability

**Why PPO Outperforms**:
- Trust region optimization prevents policy collapse
- Better handling of the continuous reward structure
- Effective entropy regularization for exploration

### Convergence Patterns

- **DQN**: Converges within 30K-40K timesteps, shows sample efficiency
- **REINFORCE**: Slower convergence (40K-50K steps), high variability early on
- **PPO**: Smooth convergence, stable performance across seeds

---

## Experimental Setup

### Environment Details
- **State Space**: 10-dimensional continuous observations [0, 100]
- **Action Space**: 9 discrete interventions
- **Episode Length**: 52 weeks (deterministic termination)
- **Reward Scale**: [-10, +5] per step

### Training Parameters
- **Total Timesteps**: 50,000 per run (≈ 250 episodes @ 200 steps)
- **Evaluation**: 10 independent episodes, deterministic policy
- **Random Seeds**: 10 different seeds per algorithm for robustness

### Computational Resources
- **Hardware**: CPU-based training
- **Training Time**: ~2 hours per algorithm (10 runs)
- **Memory**: <2GB per training process

---

## Files & Descriptions

| File | Purpose |
|------|---------|
| `environment/custom_env.py` | Core NCD environment with gym interface |
| `environment/rendering.py` | Pygame GUI for visualization |
| `training/dqn_training.py` | DQN training script (10+ runs) |
| `training/pg_training.py` | REINFORCE & PPO training (10+ runs each) |
| `main.py` | Entry point for running agents |
| `requirements.txt` | Python dependencies |
| `models/dqn/dqn_results.csv` | DQN hyperparameter results |
| `models/pg/ppo_results.csv` | PPO hyperparameter results |
| `models/pg/reinforce_results.csv` | REINFORCE results |

---

## Troubleshooting

### Models not found
Ensure you've trained the models first:
```bash
python training/dqn_training.py
python training/pg_training.py
```

### Graphics Issues (Pygame)
If visualization doesn't work:
- Ensure Pygame is installed: `pip install pygame`
- For headless servers, skip `--visualize` flag

### CUDA/GPU Issues
This project trains on CPU by default. To use GPU:
- Install PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Modify `device="cpu"` to `device="cuda"` in training scripts

---

## References

- Gymnasium Documentation: https://gymnasium.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- DQN Paper: Mnih et al., 2015
- PPO Paper: Schulman et al., 2017
- Policy Gradients: Sutton & Barto, 2018


## Author

Name: Belyse NIYONSENGA
Date: March 2026


