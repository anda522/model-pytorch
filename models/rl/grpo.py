"""
GRPO (Group Relative Policy Optimization) 算法实现

GRPO 是一种基于组的相对策略优化算法，特别适用于 LLM 微调场景。
主要特点：
1. 不需要单独的价值函数 (Critic)
2. 使用组内相对优势来估计优势值
3. 通过多次采样同一提示词来构建组
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class GRPOConfig:
    """GRPO 配置类"""
    # 网络参数
    hidden_dim: int = 256
    # 学习率
    lr: float = 1e-5
    # GRPO 特有参数
    group_size: int = 4  # 每组采样数量
    clip_epsilon: float = 0.2  # PPO裁剪参数
    kl_coef: float = 0.1  # KL散度系数
    entropy_coef: float = 0.01  # 熵系数
    # 优化参数
    gamma: float = 1.0  # 折扣因子 (通常为1.0用于LLM)
    max_grad_norm: float = 1.0
    ppo_epochs: int = 2  # PPO更新轮数
    # 其他
    device: str = "cpu"
    use_reference_model: bool = True  # 是否使用参考模型


class GRPONetwork(nn.Module):
    """GRPO 策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回动作 logits"""
        return self.net(state)
    
    def get_probs(self, state: torch.Tensor) -> torch.Tensor:
        """返回动作概率"""
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def get_log_probs(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """返回指定动作的log概率"""
        logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
    
    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """返回策略熵"""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1)


class GRPOAgent:
    """
    Group Relative Policy Optimization (GRPO) 智能体
    
    GRPO 的核心思想是：
    1. 对于同一个状态/提示词，采样多个响应 (group_size 个)
    2. 计算每个响应的奖励
    3. 使用组内相对优势 (相对于组平均) 来估计优势值
    4. 这样避免了训练一个单独的价值函数
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[GRPOConfig] = None,
        **kwargs
    ):
        # 合并配置
        if config is None:
            config = GRPOConfig(**kwargs)
        self.config = config
        
        self.device = config.device
        self.group_size = config.group_size
        self.clip_epsilon = config.clip_epsilon
        self.kl_coef = config.kl_coef
        self.entropy_coef = config.entropy_coef
        self.gamma = config.gamma
        self.max_grad_norm = config.max_grad_norm
        self.ppo_epochs = config.ppo_epochs
        
        # 策略网络
        self.policy = GRPONetwork(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        
        # 参考网络 (冻结，用于KL散度计算)
        self.use_reference_model = config.use_reference_model
        if self.use_reference_model:
            self.reference_policy = GRPONetwork(state_dim, action_dim, config.hidden_dim).to(self.device)
            self.reference_policy.load_state_dict(self.policy.state_dict())
            self.reference_policy.eval()
            for param in self.reference_policy.parameters():
                param.requires_grad = False
        
        # 经验存储
        self.groups: List[Dict[str, Any]] = []
        self.current_group: List[Dict[str, Any]] = []
    
    def select_action(self, state: np.ndarray) -> int:
        """采样动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.policy(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).item()
        
        return action.item(), log_prob
    
    def add_to_group(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        sequence_length: int = 1
    ):
        """
        添加一个样本到当前组
        
        Args:
            state: 状态/提示词
            action: 动作/响应
            log_prob: 动作的log概率
            reward: 奖励值
            sequence_length: 序列长度 (用于长度归一化)
        """
        self.current_group.append({
            'state': state,
            'action': action,
            'log_prob': log_prob,
            'reward': reward,
            'sequence_length': sequence_length
        })
        
        # 当组大小达到阈值时，保存组
        if len(self.current_group) >= self.group_size:
            self.groups.append(self.current_group.copy())
            self.current_group.clear()
    
    def compute_group_advantages(self, group_rewards: List[float]) -> torch.Tensor:
        """
        计算组内相对优势
        
        GRPO 核心公式:
        advantage_i = (reward_i - mean(rewards)) / std(rewards)
        
        这避免了训练单独的价值函数
        """
        rewards = torch.tensor(group_rewards, dtype=torch.float32, device=self.device)
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        advantages = (rewards - mean) / std
        return advantages
    
    def compute_kl_divergence(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        计算当前策略与参考策略之间的KL散度
        
        KL(π_ref || π) = Σ π_ref(a|s) * log(π_ref(a|s) / π(a|s))
        """
        if not self.use_reference_model:
            return torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            ref_logits = self.reference_policy(state)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_log_prob = ref_log_probs.gather(dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
        
        cur_logits = self.policy(state)
        cur_log_probs = F.log_softmax(cur_logits, dim=-1)
        cur_log_prob = cur_log_probs.gather(dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
        
        # KL divergence
        kl = ref_log_prob - cur_log_prob
        return kl
    
    def update(self) -> Optional[Dict[str, float]]:
        """
        GRPO 更新
        
        Returns:
            包含损失信息的字典，如果没有数据则返回 None
        """
        if len(self.groups) == 0:
            return None
        
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        # 对每组进行更新
        for group in self.groups:
            # 提取组数据
            states = np.array([s['state'] for s in group])
            actions = np.array([s['action'] for s in group])
            old_log_probs = np.array([s['log_prob'] for s in group])
            rewards = [s['reward'] for s in group]
            
            # 转换为张量
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
            
            # 计算组内相对优势
            advantages = self.compute_group_advantages(rewards)
            
            # PPO 多轮更新
            for _ in range(self.ppo_epochs):
                # 计算当前 log_prob 和熵
                logits = self.policy(states_tensor)
                log_probs = F.log_softmax(logits, dim=-1)
                cur_log_probs = log_probs.gather(dim=-1, index=actions_tensor.unsqueeze(-1)).squeeze(-1)
                
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                
                # 计算重要性采样比率
                ratio = torch.exp(cur_log_probs - old_log_probs_tensor)
                
                # PPO-Clip 损失
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # KL 散度损失
                kl_loss = self.compute_kl_divergence(states_tensor, actions_tensor).mean()
                
                # 总损失
                loss = policy_loss + self.kl_coef * kl_loss - self.entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_kl_loss += kl_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        # 清空组数据
        self.groups.clear()
        
        if num_updates == 0:
            return None
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'kl_loss': total_kl_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
    def update_reference_model(self):
        """更新参考模型为当前策略"""
        if self.use_reference_model:
            self.reference_policy.load_state_dict(self.policy.state_dict())
    
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """获取动作概率分布"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy.get_probs(state_tensor)
        return probs.cpu().numpy().flatten()
    
    def save(self, path: str):
        """保存模型"""
        save_dict = {
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        if self.use_reference_model:
            save_dict['reference_policy'] = self.reference_policy.state_dict()
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.use_reference_model and 'reference_policy' in checkpoint:
            self.reference_policy.load_state_dict(checkpoint['reference_policy'])


class GRPOTrainer:
    """
    GRPO 训练器
    
    提供完整的训练循环，包括：
    - 环境交互
    - 组采样
    - 奖励计算
    - 策略更新
    """
    
    def __init__(
        self,
        agent: GRPOAgent,
        env,
        reward_fn=None
    ):
        """
        Args:
            agent: GRPO 智能体
            env: 环境 (需要符合 OpenAI Gym 接口)
            reward_fn: 自定义奖励函数 (可选)
        """
        self.agent = agent
        self.env = env
        self.reward_fn = reward_fn
        
        # 训练统计
        self.episode_rewards = []
        self.losses = []
    
    def collect_group(self, state: np.ndarray) -> Tuple[List[Dict], float]:
        """
        从同一状态收集一个组的样本
        
        Args:
            state: 初始状态
            
        Returns:
            (组样本列表, 总奖励)
        """
        group_samples = []
        total_reward = 0.0
        
        for _ in range(self.agent.group_size):
            # 重置到相同状态 (如果环境支持)
            if hasattr(self.env, 'set_state'):
                self.env.set_state(state)
            
            action, log_prob = self.agent.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            
            if self.reward_fn is not None:
                reward = self.reward_fn(state, action, next_state, info)
            
            group_samples.append({
                'state': state,
                'action': action,
                'log_prob': log_prob,
                'reward': reward
            })
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        return group_samples, total_reward
    
    def train_episode(self) -> Dict[str, float]:
        """训练一个 episode"""
        state = self.env.reset()
        episode_reward = 0.0
        
        while True:
            # 收集一组样本
            group_samples, group_reward = self.collect_group(state)
            episode_reward += group_reward
            
            # 添加到智能体的组存储
            for sample in group_samples:
                self.agent.add_to_group(
                    sample['state'],
                    sample['action'],
                    sample['log_prob'],
                    sample['reward']
                )
            
            # 检查是否结束
            if len(group_samples) > 0:
                # 获取最后一个样本的状态
                # 这里简化处理，实际可能需要更复杂的状态管理
                pass
            
            # 更新策略
            loss_info = self.agent.update()
            if loss_info is not None:
                self.losses.append(loss_info)
            
            # 简单的终止条件
            if episode_reward > 1000 or len(self.agent.groups) > 100:
                break
        
        self.episode_rewards.append(episode_reward)
        
        return {
            'episode_reward': episode_reward,
            'loss_info': self.losses[-1] if self.losses else None
        }
    
    def train(
        self,
        num_episodes: int,
        log_interval: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, List]:
        """
        训练多个 episodes
        
        Args:
            num_episodes: 训练的 episode 数量
            log_interval: 日志输出间隔
            save_path: 模型保存路径 (可选)
            
        Returns:
            训练统计信息
        """
        for episode in range(num_episodes):
            result = self.train_episode()
            
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Average Reward: {avg_reward:.2f}")
                if result['loss_info']:
                    print(f"  Policy Loss: {result['loss_info']['policy_loss']:.4f}")
                    print(f"  KL Loss: {result['loss_info']['kl_loss']:.4f}")
        
        if save_path:
            self.agent.save(save_path)
            print(f"Model saved to {save_path}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }
