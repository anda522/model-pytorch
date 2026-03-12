"""
简单的示例环境，用于测试RL算法
"""
import numpy as np
from typing import Tuple, Optional


class SimpleEnv:
    """一个简单的网格世界环境"""
    
    def __init__(self, size: int = 5):
        self.size = size
        self.agent_pos = 0
        self.goal_pos = size - 1
        self.max_steps = size * 2
        self.steps = 0
        
        # 状态维度: one-hot编码的位置
        self.observation_dim = size
        # 动作: 0=左, 1=右
        self.action_dim = 2
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.agent_pos = 0
        self.steps = 0
        return self._get_obs()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行一步动作"""
        self.steps += 1
        
        # 移动
        if action == 0:  # 左
            self.agent_pos = max(0, self.agent_pos - 1)
        else:  # 右
            self.agent_pos = min(self.size - 1, self.agent_pos + 1)
        
        # 计算奖励
        if self.agent_pos == self.goal_pos:
            reward = 1.0
            done = True
        else:
            reward = -0.01  # 每步小惩罚
            done = self.steps >= self.max_steps
        
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self) -> np.ndarray:
        """获取one-hot编码的观测"""
        obs = np.zeros(self.observation_dim, dtype=np.float32)
        obs[self.agent_pos] = 1.0
        return obs
    
    def render(self):
        """打印环境状态"""
        grid = ['.' for _ in range(self.size)]
        grid[self.agent_pos] = 'A'
        grid[self.goal_pos] = 'G'
        print(' '.join(grid))


class CartPoleWrapper:
    """CartPole环境包装器（如果gym可用）"""
    
    def __init__(self):
        try:
            import gym
            self.env = gym.make('CartPole-v1')
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: gym not installed, using SimpleEnv instead")
            self.env = SimpleEnv(size=10)
    
    @property
    def observation_dim(self) -> int:
        if hasattr(self.env, 'observation_space'):
            return self.env.observation_space.shape[0]
        return self.env.observation_dim
    
    @property
    def action_dim(self) -> int:
        if hasattr(self.env, 'action_space'):
            return self.env.action_space.n
        return self.env.action_dim
    
    def reset(self) -> np.ndarray:
        result = self.env.reset()
        return result if isinstance(result, np.ndarray) else result[0]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        result = self.env.step(action)
        if len(result) == 4:
            return result
        return result[0], result[1], result[2], {}
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


def train_dqn_example():
    """DQN训练示例"""
    from dqn import DQNAgent
    
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
    
    num_episodes = 200
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, rewards_history


def train_reinforce_example():
    """REINFORCE训练示例"""
    from policy_gradient import REINFORCEAgent
    
    env = SimpleEnv(size=5)
    agent = REINFORCEAgent(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_dim=32,
        lr=1e-2,
        gamma=0.99,
        use_baseline=True
    )
    
    num_episodes = 200
    rewards_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            state = next_state
        
        loss = agent.update()
        rewards_history.append(sum(agent.saved_rewards) if agent.saved_rewards else 0)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
    
    return agent, rewards_history


def train_a2c_example():
    """A2C训练示例"""
    from actor_critic import A2CAgent
    
    env = SimpleEnv(size=5)
    agent = A2CAgent(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_dim=32,
        lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    num_episodes = 200
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
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
    
    return agent, rewards_history


if __name__ == "__main__":
    print("=" * 50)
    print("Training DQN...")
    print("=" * 50)
    train_dqn_example()
    
    print("\n" + "=" * 50)
    print("Training REINFORCE...")
    print("=" * 50)
    train_reinforce_example()
    
    print("\n" + "=" * 50)
    print("Training A2C...")
    print("=" * 50)
    train_a2c_example()
