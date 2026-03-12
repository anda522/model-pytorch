# 强化学习算法实现

本目录包含多种强化学习算法的手写实现，所有算法均使用 PyTorch 框架实现。

## 目录结构

```
rl/
├── __init__.py           # 模块初始化，导出所有类和函数
├── network.py            # 神经网络结构定义
├── replay_buffer.py      # 经验回放缓冲区
├── dqn.py                # DQN 及其变体算法
├── policy_gradient.py    # 策略梯度算法 (REINFORCE)
├── actor_critic.py       # Actor-Critic 算法 (A2C, PPO)
├── grpo.py               # GRPO 算法 (用于 LLM 微调)
├── example_env.py        # 示例环境
├── test_all.py           # 测试脚本
└── README.md             # 本文件
```

## 文件功能说明

### 1. `network.py` - 神经网络结构

定义了强化学习中常用的网络结构：

| 类名 | 描述 |
|------|------|
| `QNetwork` | 标准 Q 网络，用于 DQN |
| `DuelingQNetwork` | Dueling DQN 网络，将 Q(s,a) 分解为 V(s) 和 A(s,a) |
| `PolicyNetwork` | 策略网络，输出动作概率分布 |
| `ValueNetwork` | 价值网络，输出状态价值 V(s) |
| `ActorCriticNetwork` | Actor-Critic 共享网络，同时输出策略和价值 |

### 2. `replay_buffer.py` - 经验回放缓冲区

| 类名 | 描述 |
|------|------|
| `ReplayBuffer` | 标准经验回放缓冲区，支持随机采样 |
| `PrioritizedReplayBuffer` | 优先级经验回放 (PER)，重要样本被采样概率更高 |

### 3. `dqn.py` - DQN 算法

**DQNAgent** 类实现了 Deep Q-Network 算法，支持以下变体：

- **标准 DQN**: 使用目标网络稳定训练
- **Double DQN**: 减少过估计问题
- **Dueling DQN**: 分离状态价值和优势函数

主要参数：
```python
DQNAgent(
    state_dim,           # 状态维度
    action_dim,          # 动作维度
    hidden_dim=128,      # 隐藏层维度
    lr=1e-3,             # 学习率
    gamma=0.99,          # 折扣因子
    epsilon_start=1.0,   # 初始探索率
    epsilon_end=0.01,    # 最终探索率
    epsilon_decay=0.995, # 探索率衰减
    buffer_size=10000,   # 经验回放容量
    batch_size=64,       # 批大小
    double_dqn=True,     # 是否使用 Double DQN
    dueling=False        # 是否使用 Dueling 网络
)
```

### 4. `policy_gradient.py` - 策略梯度算法

**REINFORCEAgent** 类实现了 REINFORCE 算法：

- 基础 REINFORCE
- REINFORCE with Baseline (使用状态价值作为基线)

主要参数：
```python
REINFORCEAgent(
    state_dim,         # 状态维度
    action_dim,        # 动作维度
    hidden_dim=128,    # 隐藏层维度
    lr=1e-3,           # 学习率
    gamma=0.99,        # 折扣因子
    use_baseline=True  # 是否使用 baseline
)
```

### 5. `actor_critic.py` - Actor-Critic 算法

实现了两种 Actor-Critic 算法：

#### A2CAgent (Advantage Actor-Critic)
```python
A2CAgent(
    state_dim,           # 状态维度
    action_dim,          # 动作维度
    hidden_dim=128,      # 隐藏层维度
    lr=1e-3,             # 学习率
    gamma=0.99,          # 折扣因子
    gae_lambda=0.95,     # GAE lambda 参数
    entropy_coef=0.01,   # 熵正则化系数
    value_coef=0.5       # 价值损失系数
)
```

#### PPOAgent (Proximal Policy Optimization)
```python
PPOAgent(
    state_dim,            # 状态维度
    action_dim,           # 动作维度
    hidden_dim=128,       # 隐藏层维度
    lr=3e-4,              # 学习率
    gamma=0.99,           # 折扣因子
    gae_lambda=0.95,      # GAE lambda 参数
    clip_epsilon=0.2,     # PPO 裁剪参数
    entropy_coef=0.01,    # 熵正则化系数
    ppo_epochs=4,         # PPO 更新轮数
    mini_batch_size=64    # mini-batch 大小
)
```

### 6. `grpo.py` - GRPO 算法

**GRPOAgent** (Group Relative Policy Optimization) 是一种专为 LLM 微调设计的算法：

特点：
- 不需要单独的价值函数 (Critic)
- 使用组内相对优势估计优势值
- 通过多次采样同一提示词构建组

```python
config = GRPOConfig(
    hidden_dim=256,        # 隐藏层维度
    lr=1e-5,               # 学习率
    group_size=4,          # 每组采样数量
    clip_epsilon=0.2,      # PPO 裁剪参数
    kl_coef=0.1,           # KL 散度系数
    entropy_coef=0.01,     # 熵系数
    ppo_epochs=2           # PPO 更新轮数
)
agent = GRPOAgent(state_dim, action_dim, config=config)
```

### 7. `example_env.py` - 示例环境

提供了测试用的简单环境：

| 类名 | 描述 |
|------|------|
| `SimpleEnv` | 简单网格世界环境，智能体需要从起点走到终点 |
| `CartPoleWrapper` | CartPole 环境包装器（需要安装 gym） |

## 快速开始

### 环境要求

```bash
pip install torch numpy
```

可选（用于 CartPole 环境）：
```bash
pip install gym
```

### 运行测试

**方法一：从 model-pytorch 目录运行**
```bash
cd model-pytorch
python -m models.rl.test_all
```

**方法二：从 rl 目录运行**
```bash
cd model-pytorch/models/rl
python test_all.py
```

### 测试输出示例

```
============================================================
   强化学习算法测试套件
============================================================

==================================================
Testing ReplayBuffer...
==================================================
Buffer size: 10
Sampled batch shapes: states=(5, 4), actions=(5,)
ReplayBuffer test PASSED!

...

==================================================
Testing DQN Agent...
==================================================
DQN - Episodes: 50, Last 10 avg reward: 0.96
DQN test PASSED!

...

============================================================
   所有测试通过! (All tests PASSED!)
============================================================
```

## 使用示例

### DQN 训练示例

```python
from models.rl.dqn import DQNAgent
from models.rl.example_env import SimpleEnv

# 创建环境和智能体
env = SimpleEnv(size=5)
agent = DQNAgent(
    state_dim=env.observation_dim,
    action_dim=env.action_dim,
    hidden_dim=32,
    double_dqn=True
)

# 训练循环
for episode in range(100):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.update()
        state = next_state
```

### PPO 训练示例

```python
from models.rl.actor_critic import PPOAgent
from models.rl.example_env import SimpleEnv

env = SimpleEnv(size=5)
agent = PPOAgent(
    state_dim=env.observation_dim,
    action_dim=env.action_dim,
    hidden_dim=32,
    clip_epsilon=0.2
)

for episode in range(100):
    state = env.reset()
    done = False
    
    while not done:
        action, value = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_reward_done(reward, done)
        state = next_state
        
        if done:
            agent.update(next_state)
```

### GRPO 训练示例

```python
from models.rl.grpo import GRPOAgent, GRPOConfig
from models.rl.example_env import SimpleEnv

env = SimpleEnv(size=5)
config = GRPOConfig(
    hidden_dim=32,
    lr=1e-3,
    group_size=4
)
agent = GRPOAgent(
    state_dim=env.observation_dim,
    action_dim=env.action_dim,
    config=config
)

for episode in range(100):
    state = env.reset()
    
    # 收集一组样本
    for _ in range(agent.group_size):
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.add_to_group(state, action, log_prob, reward)
        state = next_state
        if done:
            state = env.reset()
    
    # 更新策略
    agent.update()
```

## 算法对比

| 算法 | 类型 | 特点 | 适用场景 |
|------|------|------|----------|
| DQN | 值函数方法 | 使用经验回放和目标网络 | 离散动作空间 |
| Double DQN | 值函数方法 | 解决 Q 值过估计问题 | 离散动作空间 |
| Dueling DQN | 值函数方法 | 分离状态价值和优势函数 | 离散动作空间 |
| REINFORCE | 策略梯度 | 蒙特卡洛方法，无偏但方差大 | 简单任务 |
| REINFORCE+Baseline | 策略梯度 | 减少方差 | 简单任务 |
| A2C | Actor-Critic | 在线策略，使用 GAE | 连续/离散动作空间 |
| PPO | Actor-Critic | 裁剪目标函数，稳定训练 | 连续/离散动作空间 |
| GRPO | 策略优化 | 无需 Critic，组内相对优势 | LLM 微调 |

## 关键技术

### 广义优势估计 (GAE)
A2C 和 PPO 使用 GAE 计算优势函数，平衡偏差和方差：
```
A_t = δ_t + (γλ)δ_{t+1} + ... + (γλ)^{T-t+1}δ_{T-1}
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

### PPO 裁剪目标
PPO 通过裁剪目标函数防止策略更新过大：
```
L = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
```
其中 r_t = π(a|s) / π_old(a|s)

### GRPO 组内相对优势
GRPO 不使用 Critic，而是通过组内样本计算相对优势：
```
advantage_i = (reward_i - mean(rewards)) / std(rewards)
```

## 注意事项

1. **超参数调优**: 不同算法对超参数敏感，建议根据任务调整学习率、熵系数等
2. **奖励设计**: 合理设计奖励函数对训练效果至关重要
3. **探索与利用**: 适当调整探索参数（如 DQN 的 epsilon）以平衡探索和利用
4. **网络结构**: 可根据任务复杂度调整隐藏层维度和层数

## 参考文献

- [DQN] Mnih et al. "Human-level control through deep reinforcement learning" (2015)
- [Double DQN] van Hasselt et al. "Deep Reinforcement Learning with Double Q-learning" (2016)
- [Dueling DQN] Wang et al. "Dueling Network Architectures for Deep Reinforcement Learning" (2016)
- [PPO] Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
- [GAE] Schulman et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
- [GRPO] Group Relative Policy Optimization for LLM Fine-tuning
