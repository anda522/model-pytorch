"""
Actor-Critic 算法实现
包含: A2C (Advantage Actor-Critic)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional, Tuple
from .network import ActorCriticNetwork


class A2CAgent:
    """Advantage Actor-Critic (A2C) 智能体"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Actor-Critic网络
        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 存储数据
        self.saved_states: List[np.ndarray] = []
        self.saved_actions: List[int] = []
        self.saved_log_probs: List[float] = []
        self.saved_values: List[float] = []
        self.saved_rewards: List[float] = []
        self.saved_dones: List[bool] = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """根据策略采样动作，返回(action, value)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs, value = self.network(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action).item()
        
        self.saved_states.append(state)
        self.saved_actions.append(action.item())
        self.saved_log_probs.append(log_prob)
        self.saved_values.append(value.item())
        
        return action.item(), value.item()
    
    def store_reward_done(self, reward: float, done: bool):
        """存储奖励和done标志"""
        self.saved_rewards.append(reward)
        self.saved_dones.append(done)
    
    def _compute_gae(self, next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算广义优势估计 (GAE)"""
        returns = []
        advantages = []
        gae = 0
        
        # 反向计算
        for t in reversed(range(len(self.saved_rewards))):
            if t == len(self.saved_rewards) - 1:
                next_val = next_value
            else:
                next_val = self.saved_values[t + 1]
            
            delta = self.saved_rewards[t] + self.gamma * next_val * (1 - self.saved_dones[t]) - self.saved_values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.saved_dones[t]) * gae
            
            returns.insert(0, gae + self.saved_values[t])
            advantages.insert(0, gae)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        
        # 标准化优势
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self, next_state: Optional[np.ndarray] = None) -> Optional[Tuple[float, float, float]]:
        """更新网络参数，返回(policy_loss, value_loss, entropy)"""
        if len(self.saved_rewards) == 0:
            return None
        
        # 获取下一个状态的价值
        if next_state is not None:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value = self.network(next_state_tensor)
                next_value = next_value.item()
        else:
            next_value = 0.0
        
        # 计算returns和advantages
        returns, advantages = self._compute_gae(next_value)
        
        # 前向传播
        states = torch.FloatTensor(np.array(self.saved_states)).to(self.device)
        actions = torch.LongTensor(self.saved_actions).to(self.device)
        
        probs, values = self.network(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # 计算损失
        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # 总损失
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # 清空数据
        self.saved_states.clear()
        self.saved_actions.clear()
        self.saved_log_probs.clear()
        self.saved_values.clear()
        self.saved_rewards.clear()
        self.saved_dones.clear()
        
        return policy_loss.item(), value_loss.item(), entropy.item()
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class PPOAgent:
    """Proximal Policy Optimization (PPO) 智能体"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        device: str = "cpu"
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.device = device
        
        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 存储数据
        self.saved_states: List[np.ndarray] = []
        self.saved_actions: List[int] = []
        self.saved_log_probs: List[float] = []
        self.saved_values: List[float] = []
        self.saved_rewards: List[float] = []
        self.saved_dones: List[bool] = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """采样动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs, value = self.network(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action).item()
        
        self.saved_states.append(state)
        self.saved_actions.append(action.item())
        self.saved_log_probs.append(log_prob)
        self.saved_values.append(value.item())
        
        return action.item(), value.item()
    
    def store_reward_done(self, reward: float, done: bool):
        self.saved_rewards.append(reward)
        self.saved_dones.append(done)
    
    def _compute_gae(self, next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算GAE"""
        returns = []
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.saved_rewards))):
            if t == len(self.saved_rewards) - 1:
                next_val = next_value
            else:
                next_val = self.saved_values[t + 1]
            
            delta = self.saved_rewards[t] + self.gamma * next_val * (1 - self.saved_dones[t]) - self.saved_values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.saved_dones[t]) * gae
            
            returns.insert(0, gae + self.saved_values[t])
            advantages.insert(0, gae)
        
        return (
            torch.tensor(returns, dtype=torch.float32, device=self.device),
            torch.tensor(advantages, dtype=torch.float32, device=self.device)
        )
    
    def update(self, next_state: Optional[np.ndarray] = None) -> Optional[float]:
        """PPO更新"""
        if len(self.saved_rewards) == 0:
            return None
        
        # 获取下一个状态的价值
        if next_state is not None:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value = self.network(next_state_tensor)
                next_value = next_value.item()
        else:
            next_value = 0.0
        
        # 计算returns和advantages
        returns, advantages = self._compute_gae(next_value)
        
        # 准备数据
        states = torch.FloatTensor(np.array(self.saved_states)).to(self.device)
        actions = torch.LongTensor(self.saved_actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.saved_log_probs).to(self.device)
        
        total_loss = 0.0
        
        # PPO多轮更新
        for _ in range(self.ppo_epochs):
            # 创建mini-batch
            indices = np.arange(len(self.saved_rewards))
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]
                
                # 前向传播
                probs, values = self.network(states[mb_indices])
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(actions[mb_indices])
                entropy = dist.entropy().mean()
                
                # 计算ratio
                ratio = torch.exp(log_probs - old_log_probs[mb_indices])
                
                # PPO-Clip损失
                surr1 = ratio * advantages[mb_indices]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[mb_indices]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = nn.MSELoss()(values.squeeze(), returns[mb_indices])
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                total_loss += loss.item()
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # 清空数据
        self.saved_states.clear()
        self.saved_actions.clear()
        self.saved_log_probs.clear()
        self.saved_values.clear()
        self.saved_rewards.clear()
        self.saved_dones.clear()
        
        return total_loss / (self.ppo_epochs * (len(indices) // self.mini_batch_size + 1))
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
