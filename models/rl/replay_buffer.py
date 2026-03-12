"""
经验回放缓冲区
"""
import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """标准的经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加一条经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """随机采样一个批次"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """优先级经验回放 (PER)"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        super().__init__(capacity)
        self.alpha = alpha
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, priority: float = 1.0):
        """添加经验，priority越大越重要"""
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority ** self.alpha)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """优先级采样，返回 (数据, 索引, 重要性权重)"""
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # 归一化
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32)
        )
    
    def update_priorities(self, indices, priorities):
        """更新采样数据的优先级"""
        for i, p in zip(indices, priorities):
            self.priorities[i] = p ** self.alpha
