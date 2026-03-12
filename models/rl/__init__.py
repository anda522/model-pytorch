"""
强化学习算法手撕实现

包含:
- DQN (Deep Q-Network) 及变体: Double DQN, Dueling DQN
- Policy Gradient: REINFORCE, REINFORCE with Baseline
- Actor-Critic: A2C, PPO
- GRPO (Group Relative Policy Optimization): 无需Critic的策略优化算法
- 经验回放: ReplayBuffer, PrioritizedReplayBuffer
- 网络结构: QNetwork, DuelingQNetwork, PolicyNetwork, ValueNetwork, ActorCriticNetwork, GRPONetwork
"""

from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .network import (
    QNetwork,
    DuelingQNetwork,
    PolicyNetwork,
    ValueNetwork,
    ActorCriticNetwork
)
from .dqn import DQNAgent
from .policy_gradient import REINFORCEAgent
from .actor_critic import A2CAgent, PPOAgent
from .grpo import GRPOAgent, GRPOConfig, GRPONetwork, GRPOTrainer
from .example_env import SimpleEnv, CartPoleWrapper

__all__ = [
    # Buffers
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    # Networks
    'QNetwork',
    'DuelingQNetwork',
    'PolicyNetwork',
    'ValueNetwork',
    'ActorCriticNetwork',
    'GRPONetwork',
    # Agents
    'DQNAgent',
    'REINFORCEAgent',
    'A2CAgent',
    'PPOAgent',
    'GRPOAgent',
    # GRPO
    'GRPOConfig',
    'GRPOTrainer',
    # Environments
    'SimpleEnv',
    'CartPoleWrapper',
]
