"""
Policy Gradient 算法实现
包含: REINFORCE, REINFORCE with Baseline
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional
from .network import PolicyNetwork, ValueNetwork


class REINFORCEAgent:
    """REINFORCE策略梯度智能体"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        use_baseline: bool = True,
        device: str = "cpu"
    ):
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.device = device
        
        # 策略网络
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 可选的baseline网络
        if use_baseline:
            self.baseline = ValueNetwork(state_dim, hidden_dim).to(device)
            self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=lr)
        
        # 存储一个episode的数据
        self.saved_log_probs: List[torch.Tensor] = []
        self.saved_states: List[np.ndarray] = []
        self.saved_rewards: List[float] = []
    
    def select_action(self, state: np.ndarray) -> int:
        """根据策略采样动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        self.saved_log_probs.append(log_prob)
        self.saved_states.append(state)
        
        return action.item()
    
    def store_reward(self, reward: float):
        """存储奖励"""
        self.saved_rewards.append(reward)
    
    def _compute_returns(self) -> torch.Tensor:
        """计算折扣回报 G_t"""
        returns = []
        G = 0
        for r in reversed(self.saved_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # 标准化
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update(self) -> Optional[float]:
        """更新策略网络"""
        if len(self.saved_rewards) == 0:
            return None
        
        returns = self._compute_returns()
        log_probs = torch.cat(self.saved_log_probs)
        
        if self.use_baseline:
            # 计算baseline (状态价值)
            states = torch.FloatTensor(np.array(self.saved_states)).to(self.device)
            with torch.no_grad():
                baseline = self.baseline(states).squeeze()
            advantages = returns - baseline
            
            # 更新baseline网络
            baseline_pred = self.baseline(states).squeeze()
            baseline_loss = nn.MSELoss()(baseline_pred, returns)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()
        else:
            advantages = returns
        
        # 策略梯度损失
        policy_loss = -(log_probs * advantages).mean()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # 清空数据
        self.saved_log_probs.clear()
        self.saved_states.clear()
        self.saved_rewards.clear()
        
        return policy_loss.item()
    
    def save(self, path: str):
        """保存模型"""
        save_dict = {'policy': self.policy.state_dict()}
        if self.use_baseline:
            save_dict['baseline'] = self.baseline.state_dict()
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        if self.use_baseline and 'baseline' in checkpoint:
            self.baseline.load_state_dict(checkpoint['baseline'])
