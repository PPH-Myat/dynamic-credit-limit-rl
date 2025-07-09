# Risk-Sensitive Credit Strategy Optimization with Reinforcement Learning

This repository contains the final project for QF624 - Machine Learning and Financial Applications, completed by Group 2 (MQF 2024): CHANG WEN YU, LI SINUAN, HONG YANG, PHOO PYAE HSU MYAT, ZHANG YIN LIANG.

## Objective

To design a reinforcement learning (RL) system that optimizes credit card limit adjustments by balancing profitability and risk. The project integrates supervised learning for behavioral segmentation and deep RL for sequential decision-making in a simulated financial environment.

---

## Project Overview

### 1. Balance Class Classification (Phase I)
- **Input**: Historical transaction-level features (e.g., utilization rate, payment rate, missed payments).
- **Target**: Balance Class (High, Low/Moderate, No balance).
- **Models Used**: Logistic Regression, Random Forest, XGBoost, LightGBM.
- **Result**: Best model (LightGBM/XGBoost) achieved ~90% accuracy and macro F1 ~0.86.

### 2. Environment Simulation (Phase II)
- **Customer State**: Balance class, payment/utilization history, interest rate, credit limit features.
- **Actions**: Maintain or increase credit limit by 20%.
- **Reward Function**:
 Reward = 3 * INT * BAL * (1 - PD) 
         - PD * LGD * [BAL_3 + CCF * (L_P - BAL_3)]
  where PD is class-based (0.01 or 0.05), LGD = 0.5, CCF = 0.8.

### 3. Reinforcement Learning Models
- **Double Q-learning (Tabular)**: Validated basic agent behavior; unstable due to large state space.
- **Deep Q-Network (DQN)**:
  - Fully connected neural network with two hidden layers (128 units each).
  - Experience replay, target network, and ε-greedy exploration.
  - Trained over 200–500 episodes.
  - Reward trajectory volatile but upper bound improves over time.
  - γ = 0.99 yields conservative policy behavior.

### 4. Strategy Evaluation
| Strategy         | Performance Summary                                           |
|------------------|---------------------------------------------------------------|
| DQN Policy        | Among top; adaptive and risk-aware                            |
| Random            | Inconsistent; lacks learning or structure                     |
| Always Maintain   | Competitive baseline; avoids provisioning risks               |
| Always Increase   | Poor performance; triggers high provisioning penalties        |

---

## Dataset

Derived from the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) competition:
- `credit_card_balance.csv`
- `application_train.csv`
- `bureau.csv`
- `previous_application.csv`

Time-sliced into three windows:
- **Training**: Months -9 to -7
- **Validation**: Months -6 to -4
- **Prediction**: Months -3 to -1

---

## Future Improvements

- Train personalized PD models instead of fixed class-based PD.
- Adopt Expected Credit Loss (ECL) formulation for reward design.
- Simulate sequential customer behavior using Markov state transitions.
- Expand reward function to include long-term customer value and engagement.
- Improve training efficiency: batch environments, multi-step returns, parallelization.
- Enhance transparency: SHAP/LIME explanations, heatmap visualization, API deployment.

