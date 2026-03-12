"""
测试所有强化学习算法的脚本
运行方式: python -m models.rl.test_all (从 model-pytorch 目录)
或者: python test_all.py (在 rl 目录下)
"""
import sys
import os

# 确保可以正确导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from models.rl.example_env import SimpleEnv
from models.rl.dqn import DQNAgent
from models.rl.policy_gradient import REINFORCEAgent
from models.rl.actor_critic import A2CAgent, PPOAgent
from models.rl.grpo import GRPOAgent, GRPOConfig
from models.rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from models.rl.network import QNetwork, DuelingQNetwork, PolicyNetwork, ValueNetwork, ActorCriticNetwork


def test_replay_buffer():
    """测试经验回放缓冲区"""
    print("\n" + "=" * 50)
    print("Testing ReplayBuffer...")
    print("=" * 50)
    
    buffer = ReplayBuffer(capacity=100)
    for i in range(10):
        state = np.zeros(4)
        state[i % 4] = 1.0
        buffer.push(state, i % 2, float(i), np.zeros(4), False)
    
    print(f"Buffer size: {len(buffer)}")
    states, actions, rewards, next_states, dones = buffer.sample(5)
    print(f"Sampled batch shapes: states={states.shape}, actions={actions.shape}")
    print("ReplayBuffer test PASSED!")


def test_prioritized_replay_buffer():
    """测试优先级经验回放"""
    print("\n" + "=" * 50)
    print("Testing PrioritizedReplayBuffer...")
    print("=" * 50)
    
    buffer = PrioritizedReplayBuffer(capacity=100)
    for i in range(10):
        state = np.zeros(4)
        state[i % 4] = 1.0
        buffer.push(state, i % 2, float(i), np.zeros(4), False, priority=float(i+1))
    
    print(f"Buffer size: {len(buffer)}")
    states, actions, rewards, next_states, dones, indices, weights = buffer.sample(5)
    print(f"Sampled batch shapes: states={states.shape}, weights={weights.shape}")
    print("PrioritizedReplayBuffer test PASSED!")


def test_networks():
    """测试网络结构"""
    print("\n" + "=" * 50)
    print("Testing Networks...")
    print("=" * 50)
    
    batch_size = 4
    state_dim = 8
    action_dim = 2
    hidden_dim = 32
    
    x = torch.randn(batch_size, state_dim)
    
    # QNetwork
    q_net = QNetwork(state_dim, action_dim, hidden_dim)
    q_values = q_net(x)
    print(f"QNetwork output shape: {q_values.shape}")
    assert q_values.shape == (batch_size, action_dim)
    
    # DuelingQNetwork
    dueling_q_net = DuelingQNetwork(state_dim, action_dim, hidden_dim)
    q_values = dueling_q_net(x)
    print(f"DuelingQNetwork output shape: {q_values.shape}")
    assert q_values.shape == (batch_size, action_dim)
    
    # PolicyNetwork
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
    probs = policy_net(x)
    print(f"PolicyNetwork output shape: {probs.shape}")
    assert probs.shape == (batch_size, action_dim)
    
    # ValueNetwork
    value_net = ValueNetwork(state_dim, hidden_dim)
    values = value_net(x)
    print(f"ValueNetwork output shape: {values.shape}")
    assert values.shape == (batch_size, 1)
    
    # ActorCriticNetwork
    ac_net = ActorCriticNetwork(state_dim, action_dim, hidden_dim)
    probs, values = ac_net(x)
    print(f"ActorCriticNetwork output shapes: probs={probs.shape}, values={values.shape}")
    assert probs.shape == (batch_size, action_dim)
    assert values.shape == (batch_size, 1)
    
    print("All network tests PASSED!")


def test_dqn():
    """测试DQN算法"""
    print("\n" + "=" * 50)
    print("Testing DQN Agent...")
    print("=" * 50)
    
    env = SimpleEnv(size=5)
    agent = DQNAgent(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_dim=32,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=1000,
        batch_size=32,
        double_dqn=True
    )
    
    num_episodes = 50
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
    
    avg_reward = np.mean(rewards_history[-10:])
    print(f"DQN - Episodes: {num_episodes}, Last 10 avg reward: {avg_reward:.2f}")
    print("DQN test PASSED!")
    return agent, rewards_history


def test_reinforce():
    """测试REINFORCE算法"""
    print("\n" + "=" * 50)
    print("Testing REINFORCE Agent...")
    print("=" * 50)
    
    env = SimpleEnv(size=5)
    agent = REINFORCEAgent(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_dim=32,
        lr=1e-2,
        gamma=0.99,
        use_baseline=True
    )
    
    num_episodes = 50
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            state = next_state
        
        agent.update()
        rewards_history.append(sum(agent.saved_rewards) if agent.saved_rewards else 0)
    
    avg_reward = np.mean(rewards_history[-10:])
    print(f"REINFORCE - Episodes: {num_episodes}, Last 10 avg reward: {avg_reward:.2f}")
    print("REINFORCE test PASSED!")
    return agent, rewards_history


def test_a2c():
    """测试A2C算法"""
    print("\n" + "=" * 50)
    print("Testing A2C Agent...")
    print("=" * 50)
    
    env = SimpleEnv(size=5)
    agent = A2CAgent(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_dim=32,
        lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    num_episodes = 50
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward_done(reward, done)
            state = next_state
            total_reward += reward
            
            if done:
                agent.update(next_state)
        
        rewards_history.append(total_reward)
    
    avg_reward = np.mean(rewards_history[-10:])
    print(f"A2C - Episodes: {num_episodes}, Last 10 avg reward: {avg_reward:.2f}")
    print("A2C test PASSED!")
    return agent, rewards_history


def test_ppo():
    """测试PPO算法"""
    print("\n" + "=" * 50)
    print("Testing PPO Agent...")
    print("=" * 50)
    
    env = SimpleEnv(size=5)
    agent = PPOAgent(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_dim=32,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        ppo_epochs=4,
        mini_batch_size=16
    )
    
    num_episodes = 50
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward_done(reward, done)
            state = next_state
            total_reward += reward
            
            if done:
                agent.update(next_state)
        
        rewards_history.append(total_reward)
    
    avg_reward = np.mean(rewards_history[-10:])
    print(f"PPO - Episodes: {num_episodes}, Last 10 avg reward: {avg_reward:.2f}")
    print("PPO test PASSED!")
    return agent, rewards_history


def test_grpo():
    """测试GRPO算法"""
    print("\n" + "=" * 50)
    print("Testing GRPO Agent...")
    print("=" * 50)
    
    env = SimpleEnv(size=5)
    config = GRPOConfig(
        hidden_dim=32,
        lr=1e-3,
        group_size=4,
        clip_epsilon=0.2,
        kl_coef=0.1,
        entropy_coef=0.01,
        ppo_epochs=2
    )
    agent = GRPOAgent(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        config=config
    )
    
    num_episodes = 30
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        # 简化GRPO测试：每个episode收集group_size个样本
        for _ in range(agent.group_size):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.add_to_group(state, action, log_prob, reward)
            total_reward += reward
            state = next_state
            if done:
                state = env.reset()
        
        # 更新策略
        agent.update()
        rewards_history.append(total_reward)
    
    avg_reward = np.mean(rewards_history[-10:])
    print(f"GRPO - Episodes: {num_episodes}, Last 10 avg reward: {avg_reward:.2f}")
    print("GRPO test PASSED!")
    return agent, rewards_history


def main():
    """运行所有测试"""
    print("=" * 60)
    print("   强化学习算法测试套件")
    print("=" * 60)
    
    # 测试基础组件
    test_replay_buffer()
    test_prioritized_replay_buffer()
    test_networks()
    
    # 测试各个算法
    test_dqn()
    test_reinforce()
    test_a2c()
    test_ppo()
    test_grpo()
    
    print("\n" + "=" * 60)
    print("   所有测试通过! (All tests PASSED!)")
    print("=" * 60)


if __name__ == "__main__":
    main()
