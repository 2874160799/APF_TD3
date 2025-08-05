# APF_TD3 API Documentation

This document provides comprehensive API documentation for the APF_TD3 library, including all public classes, methods, functions, and their usage patterns.

## Table of Contents

- [Core Classes](#core-classes)
  - [APF_TD3_Agent](#apf_td3_agent)
  - [Environment](#environment)
  - [APF_Controller](#apf_controller)
  - [TD3_Network](#td3_network)
- [Utility Functions](#utility-functions)
- [Configuration](#configuration)
- [Exceptions](#exceptions)
- [Type Definitions](#type-definitions)

## Core Classes

### APF_TD3_Agent

The main agent class that implements the hybrid APF-TD3 algorithm for autonomous navigation.

#### Class Definition

```python
class APF_TD3_Agent:
    """
    APF_TD3 Agent for autonomous navigation combining Artificial Potential Fields
    with Twin Delayed Deep Deterministic Policy Gradient.
    
    This agent learns to navigate in complex environments by combining the reactive
    nature of APF with the strategic learning capabilities of TD3.
    """
```

#### Constructor

```python
def __init__(
    self,
    state_dim: int,
    action_dim: int,
    max_action: float,
    lr: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_freq: int = 2,
    buffer_size: int = 1000000,
    batch_size: int = 256,
    device: str = "auto"
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state_dim` | `int` | Required | Dimension of the observation space |
| `action_dim` | `int` | Required | Dimension of the action space |
| `max_action` | `float` | Required | Maximum absolute value of actions |
| `lr` | `float` | `3e-4` | Learning rate for both actor and critic networks |
| `gamma` | `float` | `0.99` | Discount factor for future rewards |
| `tau` | `float` | `0.005` | Soft update coefficient for target networks |
| `policy_noise` | `float` | `0.2` | Standard deviation of target policy smoothing noise |
| `noise_clip` | `float` | `0.5` | Clipping range for target policy noise |
| `policy_freq` | `int` | `2` | Frequency of delayed policy updates |
| `buffer_size` | `int` | `1000000` | Maximum size of the replay buffer |
| `batch_size` | `int` | `256` | Batch size for training |
| `device` | `str` | `"auto"` | Device for computation ("cpu", "cuda", or "auto") |

**Raises:**
- `ValueError`: If state_dim or action_dim <= 0
- `ValueError`: If max_action <= 0
- `RuntimeError`: If CUDA is requested but not available

#### Methods

##### `select_action(state: np.ndarray, add_noise: bool = False) -> np.ndarray`

Selects an action for the given state using the current policy.

**Parameters:**
- `state` (`np.ndarray`): Current environment state with shape `(state_dim,)`
- `add_noise` (`bool`, optional): Whether to add exploration noise. Default: `False`

**Returns:**
- `np.ndarray`: Action vector with shape `(action_dim,)`, clipped to `[-max_action, max_action]`

**Example:**
```python
agent = APF_TD3_Agent(state_dim=6, action_dim=2, max_action=1.0)
state = np.array([0.1, 0.2, 0.0, 0.0, 0.8, 0.9])
action = agent.select_action(state)
print(f"Selected action: {action}")  # e.g., [0.23, -0.15]
```

##### `train(env: Environment, episodes: int = 1000, save_freq: int = 100, eval_freq: int = 50) -> Dict[str, List[float]]`

Trains the agent in the specified environment.

**Parameters:**
- `env` (`Environment`): Training environment
- `episodes` (`int`, optional): Number of training episodes. Default: `1000`
- `save_freq` (`int`, optional): Frequency of model saving (in episodes). Default: `100`
- `eval_freq` (`int`, optional): Frequency of evaluation (in episodes). Default: `50`

**Returns:**
- `Dict[str, List[float]]`: Training metrics including:
  - `"episode_rewards"`: List of total rewards per episode
  - `"episode_lengths"`: List of episode lengths
  - `"actor_losses"`: List of actor network losses
  - `"critic_losses"`: List of critic network losses
  - `"eval_rewards"`: List of evaluation rewards
  - `"success_rates"`: List of success rates during evaluation

**Example:**
```python
env = Environment(width=50, height=50)
agent = APF_TD3_Agent(state_dim=env.state_dim, action_dim=env.action_dim, max_action=1.0)

metrics = agent.train(env, episodes=2000, save_freq=200, eval_freq=100)

# Plot training progress
import matplotlib.pyplot as plt
plt.plot(metrics["episode_rewards"])
plt.title("Training Progress")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
```

##### `save(filepath: str, save_buffer: bool = False) -> None`

Saves the trained model to disk.

**Parameters:**
- `filepath` (`str`): Path where to save the model (should end with `.pth`)
- `save_buffer` (`bool`, optional): Whether to save the replay buffer. Default: `False`

**Raises:**
- `IOError`: If the file cannot be written
- `OSError`: If the directory doesn't exist

**Example:**
```python
agent.save("models/apf_td3_checkpoint.pth")
agent.save("models/apf_td3_with_buffer.pth", save_buffer=True)
```

##### `load(filepath: str, load_buffer: bool = False) -> None`

Loads a trained model from disk.

**Parameters:**
- `filepath` (`str`): Path to the saved model file
- `load_buffer` (`bool`, optional): Whether to load the replay buffer. Default: `False`

**Raises:**
- `FileNotFoundError`: If the model file doesn't exist
- `RuntimeError`: If the model architecture doesn't match

**Example:**
```python
agent = APF_TD3_Agent(state_dim=6, action_dim=2, max_action=1.0)
agent.load("models/pretrained_agent.pth")
```

##### `update(replay_buffer: ReplayBuffer, iterations: int = 1) -> Dict[str, float]`

Performs training updates using samples from the replay buffer.

**Parameters:**
- `replay_buffer` (`ReplayBuffer`): Buffer containing training experiences
- `iterations` (`int`, optional): Number of update iterations. Default: `1`

**Returns:**
- `Dict[str, float]`: Update metrics including actor and critic losses

**Example:**
```python
# This method is typically called internally during training
buffer = ReplayBuffer(state_dim=6, action_dim=2, max_size=100000)
# ... fill buffer with experiences ...
losses = agent.update(buffer, iterations=10)
print(f"Actor loss: {losses['actor_loss']:.4f}")
```

##### `set_apf_params(k_att: float, k_rep: float, rho_0: float) -> None`

Updates the APF controller parameters.

**Parameters:**
- `k_att` (`float`): Attractive force gain
- `k_rep` (`float`): Repulsive force gain  
- `rho_0` (`float`): Influence distance of obstacles

**Example:**
```python
# Increase obstacle avoidance strength
agent.set_apf_params(k_att=1.0, k_rep=15.0, rho_0=10.0)
```

### Environment

Simulation environment for training and testing navigation agents.

#### Class Definition

```python
class Environment:
    """
    2D navigation environment with obstacles for training APF_TD3 agents.
    
    The environment provides a continuous 2D space with circular obstacles,
    start and goal positions, and realistic physics simulation.
    """
```

#### Constructor

```python
def __init__(
    self,
    width: float,
    height: float,
    obstacles: List[Tuple[float, float, float]] = None,
    goal: Tuple[float, float] = None,
    start: Tuple[float, float] = None,
    dt: float = 0.1,
    max_episode_steps: int = 1000,
    goal_threshold: float = 2.0,
    collision_threshold: float = 1.0,
    max_velocity: float = 2.0,
    velocity_decay: float = 0.95
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | `float` | Required | Environment width in units |
| `height` | `float` | Required | Environment height in units |
| `obstacles` | `List[Tuple[float, float, float]]` | `None` | List of obstacles as (x, y, radius) |
| `goal` | `Tuple[float, float]` | `None` | Goal position (x, y). Random if None |
| `start` | `Tuple[float, float]` | `None` | Start position (x, y). Random if None |
| `dt` | `float` | `0.1` | Time step for simulation |
| `max_episode_steps` | `int` | `1000` | Maximum steps per episode |
| `goal_threshold` | `float` | `2.0` | Distance threshold for goal reaching |
| `collision_threshold` | `float` | `1.0` | Distance threshold for collision |
| `max_velocity` | `float` | `2.0` | Maximum agent velocity |
| `velocity_decay` | `float` | `0.95` | Velocity decay factor per step |

#### Properties

##### `state_dim: int`
Dimension of the state space (typically 6: position, velocity, goal relative position).

##### `action_dim: int`
Dimension of the action space (typically 2: acceleration in x and y).

##### `max_action: float`
Maximum absolute value for actions (acceleration limits).

#### Methods

##### `reset(start: Tuple[float, float] = None, goal: Tuple[float, float] = None) -> np.ndarray`

Resets the environment to initial conditions.

**Parameters:**
- `start` (`Tuple[float, float]`, optional): Override start position
- `goal` (`Tuple[float, float]`, optional): Override goal position

**Returns:**
- `np.ndarray`: Initial state observation

**Example:**
```python
env = Environment(width=50, height=50, obstacles=[(25, 25, 5)])
initial_state = env.reset(start=(5, 5), goal=(45, 45))
print(f"Initial state: {initial_state}")
```

##### `step(action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]`

Executes one simulation step with the given action.

**Parameters:**
- `action` (`np.ndarray`): Action to execute with shape `(action_dim,)`

**Returns:**
- `Tuple[np.ndarray, float, bool, Dict[str, Any]]`: 
  - Next state observation
  - Reward for the action
  - Whether episode is done
  - Info dictionary with additional metrics

**Example:**
```python
action = np.array([0.5, -0.3])  # Accelerate right and slightly down
next_state, reward, done, info = env.step(action)

print(f"Reward: {reward:.3f}")
print(f"Distance to goal: {info['distance_to_goal']:.2f}")
if done:
    print(f"Episode finished: {info['termination_reason']}")
```

##### `render(mode: str = 'human', save_path: str = None) -> Optional[np.ndarray]`

Renders the current environment state.

**Parameters:**
- `mode` (`str`, optional): Rendering mode. Options: `'human'`, `'rgb_array'`. Default: `'human'`
- `save_path` (`str`, optional): Path to save the rendered image

**Returns:**
- `Optional[np.ndarray]`: RGB image array if mode is `'rgb_array'`, None otherwise

**Example:**
```python
# Display environment
env.render()

# Save screenshot
env.render(save_path="episode_frame.png")

# Get RGB array for processing
rgb_array = env.render(mode='rgb_array')
print(f"Image shape: {rgb_array.shape}")
```

##### `add_obstacle(x: float, y: float, radius: float) -> None`

Dynamically adds an obstacle to the environment.

**Parameters:**
- `x` (`float`): X coordinate of obstacle center
- `y` (`float`): Y coordinate of obstacle center  
- `radius` (`float`): Obstacle radius

**Example:**
```python
# Add obstacle during episode
env.add_obstacle(30, 20, 4)
```

##### `remove_obstacle(index: int) -> None`

Removes an obstacle by index.

**Parameters:**
- `index` (`int`): Index of obstacle to remove

**Raises:**
- `IndexError`: If index is out of range

##### `get_state() -> np.ndarray`

Returns the current state observation.

**Returns:**
- `np.ndarray`: Current state vector

##### `is_collision(position: Tuple[float, float]) -> bool`

Checks if a position collides with any obstacle or boundary.

**Parameters:**
- `position` (`Tuple[float, float]`): Position to check

**Returns:**
- `bool`: True if collision detected

### APF_Controller

Artificial Potential Field controller for reactive navigation.

#### Class Definition

```python
class APF_Controller:
    """
    Artificial Potential Field controller implementing attractive and repulsive forces
    for reactive obstacle avoidance and goal seeking.
    """
```

#### Constructor

```python
def __init__(
    self,
    k_att: float = 1.0,
    k_rep: float = 10.0,
    rho_0: float = 5.0,
    max_force: float = 10.0,
    goal_threshold: float = 1.0
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_att` | `float` | `1.0` | Attractive force gain coefficient |
| `k_rep` | `float` | `10.0` | Repulsive force gain coefficient |
| `rho_0` | `float` | `5.0` | Influence distance of obstacles |
| `max_force` | `float` | `10.0` | Maximum force magnitude |
| `goal_threshold` | `float` | `1.0` | Distance threshold for goal attraction |

#### Methods

##### `compute_force(position: np.ndarray, velocity: np.ndarray, goal: np.ndarray, obstacles: List[np.ndarray]) -> np.ndarray`

Computes the total APF force at the given position.

**Parameters:**
- `position` (`np.ndarray`): Current position [x, y]
- `velocity` (`np.ndarray`): Current velocity [vx, vy]
- `goal` (`np.ndarray`): Goal position [x, y]
- `obstacles` (`List[np.ndarray]`): List of obstacle positions and radii [[x, y, r], ...]

**Returns:**
- `np.ndarray`: Total force vector [fx, fy]

**Example:**
```python
apf = APF_Controller(k_att=1.0, k_rep=15.0, rho_0=8.0)

position = np.array([10.0, 10.0])
velocity = np.array([1.0, 0.5])
goal = np.array([40.0, 40.0])
obstacles = [np.array([20.0, 15.0, 3.0]), np.array([25.0, 25.0, 4.0])]

force = apf.compute_force(position, velocity, goal, obstacles)
print(f"APF force: {force}")
```

##### `attractive_force(position: np.ndarray, goal: np.ndarray) -> np.ndarray`

Computes attractive force toward the goal.

**Parameters:**
- `position` (`np.ndarray`): Current position
- `goal` (`np.ndarray`): Goal position

**Returns:**
- `np.ndarray`: Attractive force vector

##### `repulsive_force(position: np.ndarray, obstacles: List[np.ndarray]) -> np.ndarray`

Computes repulsive forces from obstacles.

**Parameters:**
- `position` (`np.ndarray`): Current position
- `obstacles` (`List[np.ndarray]`): List of obstacles

**Returns:**
- `np.ndarray`: Total repulsive force vector

### TD3_Network

Neural network components for the TD3 algorithm.

#### Actor Network

```python
class Actor(nn.Module):
    """
    Actor network for TD3 that outputs continuous actions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dims: List[int] = [256, 256]
    ) -> None
```

#### Critic Network

```python
class Critic(nn.Module):
    """
    Twin critic networks for TD3 that estimate Q-values.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ) -> None
```

## Utility Functions

### `visualize_trajectory(trajectory: List[np.ndarray], environment: Environment, save_path: str = None) -> None`

Visualizes an agent's trajectory in the environment.

**Parameters:**
- `trajectory` (`List[np.ndarray]`): List of positions along the trajectory
- `environment` (`Environment`): Environment for context
- `save_path` (`str`, optional): Path to save the visualization

**Example:**
```python
trajectory = []
state = env.reset()
while not done:
    action = agent.select_action(state)
    state, reward, done, info = env.step(action)
    trajectory.append(env.agent_position.copy())

visualize_trajectory(trajectory, env, save_path="trajectory.png")
```

### `evaluate_performance(agent: APF_TD3_Agent, env: Environment, episodes: int = 100, render: bool = False) -> Dict[str, float]`

Evaluates agent performance over multiple episodes.

**Parameters:**
- `agent` (`APF_TD3_Agent`): Agent to evaluate
- `env` (`Environment`): Evaluation environment
- `episodes` (`int`, optional): Number of evaluation episodes. Default: `100`
- `render` (`bool`, optional): Whether to render episodes. Default: `False`

**Returns:**
- `Dict[str, float]`: Performance metrics:
  - `"success_rate"`: Fraction of episodes reaching goal
  - `"avg_reward"`: Average episode reward
  - `"avg_path_length"`: Average path length
  - `"avg_episode_length"`: Average episode duration
  - `"collision_rate"`: Fraction of episodes ending in collision

### `load_config(config_path: str) -> Dict[str, Any]`

Loads configuration from YAML file.

**Parameters:**
- `config_path` (`str`): Path to configuration file

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

### `create_environment_from_config(config: Dict[str, Any]) -> Environment`

Creates environment from configuration dictionary.

**Parameters:**
- `config` (`Dict[str, Any]`): Configuration dictionary

**Returns:**
- `Environment`: Configured environment instance

## Configuration

### Configuration Schema

```yaml
# Environment configuration
environment:
  width: float                    # Environment width
  height: float                   # Environment height
  dt: float                      # Time step
  max_episode_steps: int         # Maximum steps per episode
  goal_threshold: float          # Goal reaching threshold
  collision_threshold: float     # Collision detection threshold
  max_velocity: float            # Maximum agent velocity
  velocity_decay: float          # Velocity decay factor

# Obstacle definitions
obstacles:
  - x: float                     # X coordinate
    y: float                     # Y coordinate  
    radius: float                # Obstacle radius

# Agent configuration
agent:
  start_position: [float, float] # Starting position [x, y]
  goal_position: [float, float]  # Goal position [x, y]
  max_action: float              # Maximum action magnitude

# APF parameters
apf_parameters:
  k_attractive: float            # Attractive force gain
  k_repulsive: float             # Repulsive force gain
  influence_distance: float      # Obstacle influence distance
  max_force: float               # Maximum force magnitude

# TD3 parameters  
td3_parameters:
  learning_rate: float           # Learning rate
  gamma: float                   # Discount factor
  tau: float                     # Target network update rate
  policy_noise: float            # Target policy noise
  noise_clip: float              # Noise clipping range
  policy_freq: int               # Policy update frequency
  batch_size: int                # Training batch size
  buffer_size: int               # Replay buffer size

# Training configuration
training:
  episodes: int                  # Number of training episodes
  save_freq: int                 # Model saving frequency
  eval_freq: int                 # Evaluation frequency
  log_freq: int                  # Logging frequency
```

## Exceptions

### `APF_TD3_Error`

Base exception class for APF_TD3 specific errors.

### `EnvironmentError` 

Raised when environment configuration or state is invalid.

### `AgentError`

Raised when agent configuration or operation fails.

### `NetworkError`

Raised when neural network operations fail.

## Type Definitions

```python
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np

# Basic types
Position = Tuple[float, float]
Obstacle = Tuple[float, float, float]  # (x, y, radius)
State = np.ndarray
Action = np.ndarray
Reward = float

# Configuration types
EnvironmentConfig = Dict[str, Any]
AgentConfig = Dict[str, Any]
TrainingConfig = Dict[str, Any]

# Metrics types
TrainingMetrics = Dict[str, List[float]]
EvaluationMetrics = Dict[str, float]
```

## Version Information

- **API Version**: 1.0.0
- **Minimum Python**: 3.8
- **PyTorch Version**: 1.9.0+
- **NumPy Version**: 1.19.0+

## Changelog

### Version 1.0.0
- Initial API release
- Core APF_TD3_Agent implementation
- Environment simulation
- Basic visualization utilities
- Configuration system
- Comprehensive documentation