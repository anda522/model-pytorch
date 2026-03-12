"""
DQN (Deep Q-Network) 及其变体实现
包含: DQN, Double DQN, Dueling DQN
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional
from .network import QNetwork, DuelingQNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN智能体，支持Double DQN和Dueling DQN"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        double_dqn: bool = True,
        dueling: bool = False,
        device: str = "cpu"
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.device = device
        self.update_count = 0
        
        # 网络初始化
        NetClass = DuelingQNetwork if dueling else QNetwork
        self.q_network = NetClass(state_dim, action_dim, hidden_dim).to(device)
        self.target_network = NetClass(state_dim, action_dim, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """epsilon-greedy策略选择动作"""
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """从replay buffer采样并更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: 用主网络选动作，目标网络评估
                best_actions = self.q_network(next_states).argmax(dim=1)
                next_q = self.target_network(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                # 标准DQN
                next_q = self.target_network(next_states).max(dim=1)[0]
            
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失并更新
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
