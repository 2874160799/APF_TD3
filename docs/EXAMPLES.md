# APF_TD3 Examples and Usage Guide

This document provides comprehensive examples and usage patterns for the APF_TD3 library, from basic navigation tasks to advanced applications.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Examples](#basic-examples)
- [Advanced Examples](#advanced-examples)
- [Integration Examples](#integration-examples)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Installation and Setup

```python
# Install the library
pip install apf-td3

# Import required modules
import numpy as np
import matplotlib.pyplot as plt
from apf_td3 import APF_TD3_Agent, Environment, APF_Controller
from apf_td3.utils import visualize_trajectory, evaluate_performance
```

### Your First Navigation Agent

```python
# Create a simple environment
env = Environment(
    width=30,
    height=30,
    obstacles=[(15, 15, 3)],  # One obstacle at center
    start=(5, 5),
    goal=(25, 25)
)

# Create and train an agent
agent = APF_TD3_Agent(
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    max_action=1.0
)

# Quick training
print("Training agent...")
metrics = agent.train(env, episodes=500)

# Test the trained agent
state = env.reset()
trajectory = [env.agent_position.copy()]

while not env.done:
    action = agent.select_action(state)
    state, reward, done, info = env.step(action)
    trajectory.append(env.agent_position.copy())

# Visualize results
visualize_trajectory(trajectory, env)
plt.title("First Navigation Result")
plt.show()
```

## Basic Examples

### Example 1: Simple Obstacle Avoidance

```python
import numpy as np
from apf_td3 import Environment, APF_TD3_Agent

def simple_obstacle_avoidance():
    """
    Basic example with a single obstacle between start and goal.
    """
    # Create environment
    env = Environment(
        width=40,
        height=40,
        obstacles=[(20, 20, 5)],  # Single obstacle
        start=(5, 5),
        goal=(35, 35),
        dt=0.1
    )
    
    # Initialize agent
    agent = APF_TD3_Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=2.0,
        lr=1e-3
    )
    
    # Training
    print("Training for simple obstacle avoidance...")
    metrics = agent.train(env, episodes=1000)
    
    # Save trained model
    agent.save("models/simple_obstacle_agent.pth")
    
    # Evaluation
    performance = evaluate_performance(agent, env, episodes=50)
    print(f"Success rate: {performance['success_rate']:.2%}")
    print(f"Average path length: {performance['avg_path_length']:.2f}")
    
    return agent, env, metrics

# Run the example
agent, env, metrics = simple_obstacle_avoidance()
```

### Example 2: Multiple Obstacles Navigation

```python
def multiple_obstacles_navigation():
    """
    Navigation through multiple obstacles of different sizes.
    """
    # Create complex environment
    obstacles = [
        (15, 10, 3),   # Small obstacle
        (25, 20, 4),   # Medium obstacle  
        (10, 25, 2),   # Small obstacle
        (30, 30, 5),   # Large obstacle
        (35, 15, 3)    # Small obstacle
    ]
    
    env = Environment(
        width=50,
        height=40,
        obstacles=obstacles,
        start=(5, 5),
        goal=(45, 35),
        max_episode_steps=1500
    )
    
    # Agent with adjusted parameters for complex environment
    agent = APF_TD3_Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=1.5,
        lr=3e-4,
        gamma=0.99,
        tau=0.005
    )
    
    # Configure APF parameters for better obstacle avoidance
    agent.set_apf_params(k_att=1.0, k_rep=20.0, rho_0=8.0)
    
    print("Training for multiple obstacles navigation...")
    metrics = agent.train(env, episodes=2000, save_freq=250, eval_freq=100)
    
    # Visualize training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(metrics['success_rates'])
    plt.title('Success Rate')
    plt.xlabel('Evaluation')
    plt.ylabel('Success Rate')
    
    plt.subplot(1, 3, 3)
    env.render()
    plt.title('Environment Layout')
    
    plt.tight_layout()
    plt.show()
    
    return agent, env

agent, env = multiple_obstacles_navigation()
```

### Example 3: Dynamic Environment

```python
def dynamic_environment_example():
    """
    Example with moving obstacles and changing goals.
    """
    class DynamicEnvironment(Environment):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.obstacle_velocities = [
                np.array([0.1, 0.05]),   # Obstacle 1 velocity
                np.array([-0.08, 0.1])   # Obstacle 2 velocity
            ]
        
        def step(self, action):
            # Move obstacles
            for i, vel in enumerate(self.obstacle_velocities):
                if i < len(self.obstacles):
                    self.obstacles[i] = (
                        self.obstacles[i][0] + vel[0],
                        self.obstacles[i][1] + vel[1],
                        self.obstacles[i][2]
                    )
                    
                    # Bounce off walls
                    if self.obstacles[i][0] <= 0 or self.obstacles[i][0] >= self.width:
                        self.obstacle_velocities[i][0] *= -1
                    if self.obstacles[i][1] <= 0 or self.obstacles[i][1] >= self.height:
                        self.obstacle_velocities[i][1] *= -1
            
            return super().step(action)
    
    # Create dynamic environment
    env = DynamicEnvironment(
        width=60,
        height=40,
        obstacles=[(20, 20, 4), (40, 15, 3)],
        start=(5, 5),
        goal=(55, 35),
        dt=0.1
    )
    
    # Agent configured for dynamic environments
    agent = APF_TD3_Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=2.0,
        lr=5e-4,
        policy_noise=0.3,  # Higher noise for exploration
        buffer_size=500000
    )
    
    print("Training in dynamic environment...")
    metrics = agent.train(env, episodes=3000, eval_freq=150)
    
    # Test with visualization
    state = env.reset()
    trajectory = []
    
    for step in range(500):
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
        trajectory.append(env.agent_position.copy())
        
        if step % 50 == 0:
            env.render()
            plt.title(f"Dynamic Navigation - Step {step}")
            plt.pause(0.1)
        
        if done:
            break
    
    return agent, env

# Uncomment to run dynamic example
# agent, env = dynamic_environment_example()
```

## Advanced Examples

### Example 4: Custom Reward Function

```python
class CustomRewardEnvironment(Environment):
    """
    Environment with custom reward function for specific behaviors.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_efficiency_weight = 0.1
        self.smoothness_weight = 0.05
        self.previous_action = np.zeros(self.action_dim)
    
    def step(self, action):
        state, reward, done, info = super().step(action)
        
        # Add path efficiency reward
        direct_distance = np.linalg.norm(np.array(self.goal) - np.array(self.start))
        current_distance = info['distance_to_goal']
        efficiency_reward = self.path_efficiency_weight * (1.0 - current_distance / direct_distance)
        
        # Add smoothness reward (penalize abrupt changes)
        action_change = np.linalg.norm(action - self.previous_action)
        smoothness_reward = -self.smoothness_weight * action_change
        
        # Update total reward
        reward += efficiency_reward + smoothness_reward
        
        # Store for next step
        self.previous_action = action.copy()
        
        # Add custom info
        info['efficiency_reward'] = efficiency_reward
        info['smoothness_reward'] = smoothness_reward
        
        return state, reward, done, info

def custom_reward_example():
    """
    Training with custom reward function for smoother paths.
    """
    env = CustomRewardEnvironment(
        width=50,
        height=50,
        obstacles=[(20, 15, 4), (30, 35, 5), (35, 20, 3)],
        start=(5, 5),
        goal=(45, 45)
    )
    
    agent = APF_TD3_Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=1.0,
        lr=2e-4,
        gamma=0.95
    )
    
    print("Training with custom reward function...")
    metrics = agent.train(env, episodes=2500)
    
    # Compare with standard environment
    standard_env = Environment(
        width=50, height=50,
        obstacles=[(20, 15, 4), (30, 35, 5), (35, 20, 3)],
        start=(5, 5), goal=(45, 45)
    )
    
    standard_agent = APF_TD3_Agent(
        state_dim=standard_env.state_dim,
        action_dim=standard_env.action_dim,
        max_action=1.0
    )
    
    standard_metrics = standard_agent.train(standard_env, episodes=2500)
    
    # Visualize comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['episode_rewards'], label='Custom Reward')
    plt.plot(standard_metrics['episode_rewards'], label='Standard Reward')
    plt.title('Training Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Test both agents
    custom_perf = evaluate_performance(agent, env, episodes=100)
    standard_perf = evaluate_performance(standard_agent, standard_env, episodes=100)
    
    metrics_names = ['success_rate', 'avg_path_length', 'avg_episode_length']
    custom_values = [custom_perf[m] for m in metrics_names]
    standard_values = [standard_perf[m] for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.bar(x - width/2, custom_values, width, label='Custom Reward')
    plt.bar(x + width/2, standard_values, width, label='Standard Reward')
    plt.xticks(x, metrics_names)
    plt.title('Performance Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return agent, env

# Run custom reward example
agent, env = custom_reward_example()
```

### Example 5: Multi-Goal Navigation

```python
def multi_goal_navigation():
    """
    Agent that learns to navigate to multiple sequential goals.
    """
    class MultiGoalEnvironment(Environment):
        def __init__(self, goals_sequence, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.goals_sequence = goals_sequence
            self.current_goal_idx = 0
            self.goals_reached = 0
            
        def reset(self, **kwargs):
            self.current_goal_idx = 0
            self.goals_reached = 0
            self.goal = self.goals_sequence[0]
            return super().reset(**kwargs)
        
        def step(self, action):
            state, reward, done, info = super().step(action)
            
            # Check if current goal is reached
            if info['distance_to_goal'] < self.goal_threshold:
                self.goals_reached += 1
                self.current_goal_idx += 1
                
                if self.current_goal_idx < len(self.goals_sequence):
                    # Move to next goal
                    self.goal = self.goals_sequence[self.current_goal_idx]
                    reward += 100  # Bonus for reaching intermediate goal
                    done = False
                    info['goal_reached'] = True
                else:
                    # All goals reached
                    reward += 500  # Large bonus for completing sequence
                    done = True
                    info['all_goals_reached'] = True
            
            info['goals_reached'] = self.goals_reached
            info['current_goal'] = self.goal
            
            return state, reward, done, info
    
    # Define goal sequence
    goals = [(15, 15), (35, 15), (35, 35), (15, 35), (25, 25)]
    
    env = MultiGoalEnvironment(
        goals_sequence=goals,
        width=50,
        height=50,
        obstacles=[(20, 20, 3), (30, 30, 4)],
        start=(5, 5),
        max_episode_steps=2000
    )
    
    agent = APF_TD3_Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=1.5,
        lr=1e-3,
        gamma=0.98,
        buffer_size=2000000
    )
    
    print("Training multi-goal navigation...")
    metrics = agent.train(env, episodes=3000, eval_freq=200)
    
    # Test the trained agent
    state = env.reset()
    trajectory = []
    goal_positions = []
    
    for step in range(2000):
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
        
        trajectory.append(env.agent_position.copy())
        goal_positions.append(info['current_goal'])
        
        if done:
            print(f"Episode finished at step {step}")
            print(f"Goals reached: {info['goals_reached']}/{len(goals)}")
            break
    
    # Visualize multi-goal trajectory
    plt.figure(figsize=(10, 8))
    
    # Plot environment
    env.render()
    
    # Plot trajectory with color gradient
    trajectory = np.array(trajectory)
    for i in range(len(trajectory)-1):
        color_intensity = i / len(trajectory)
        plt.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], 
                color=plt.cm.viridis(color_intensity), linewidth=2)
    
    # Plot goals sequence
    for i, goal in enumerate(goals):
        plt.plot(goal[0], goal[1], 'r*', markersize=15, 
                label=f'Goal {i+1}' if i == 0 else "")
        plt.text(goal[0]+1, goal[1]+1, f'{i+1}', fontsize=12, fontweight='bold')
    
    plt.title('Multi-Goal Navigation Trajectory')
    plt.legend()
    plt.show()
    
    return agent, env

# Run multi-goal example
agent, env = multi_goal_navigation()
```

## Integration Examples

### Example 6: ROS Integration

```python
#!/usr/bin/env python3
"""
ROS integration example for APF_TD3 agent.
Requires: ros-noetic-desktop-full, geometry_msgs, sensor_msgs
"""

import rospy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np
from apf_td3 import APF_TD3_Agent

class APF_TD3_ROSNode:
    def __init__(self):
        rospy.init_node('apf_td3_navigator')
        
        # Load trained agent
        self.agent = APF_TD3_Agent(state_dim=6, action_dim=2, max_action=1.0)
        self.agent.load("models/trained_agent.pth")
        
        # ROS publishers and subscribers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
        # State variables
        self.current_pose = np.array([0.0, 0.0])
        self.current_velocity = np.array([0.0, 0.0])
        self.goal_pose = np.array([5.0, 5.0])
        self.laser_data = None
        
        # Control parameters
        self.max_linear_vel = 0.5
        self.max_angular_vel = 1.0
        
        rospy.loginfo("APF_TD3 ROS Node initialized")
    
    def laser_callback(self, msg):
        """Process laser scan data for obstacle detection."""
        self.laser_data = msg
        
        # Convert laser scan to obstacle positions (simplified)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        
        # Filter valid readings
        valid_idx = (ranges > msg.range_min) & (ranges < msg.range_max)
        valid_ranges = ranges[valid_idx]
        valid_angles = angles[valid_idx]
        
        # Convert to Cartesian coordinates relative to robot
        obstacle_x = valid_ranges * np.cos(valid_angles)
        obstacle_y = valid_ranges * np.sin(valid_angles)
        
        # Update state and get action
        self.update_and_control()
    
    def goal_callback(self, msg):
        """Update goal position from ROS message."""
        self.goal_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y
        ])
        rospy.loginfo(f"New goal received: {self.goal_pose}")
    
    def update_and_control(self):
        """Update agent state and publish control commands."""
        if self.laser_data is None:
            return
        
        # Construct state vector
        relative_goal = self.goal_pose - self.current_pose
        state = np.concatenate([
            self.current_pose,
            self.current_velocity,
            relative_goal
        ])
        
        # Get action from agent
        action = self.agent.select_action(state)
        
        # Convert action to ROS Twist message
        cmd_msg = Twist()
        cmd_msg.linear.x = np.clip(action[0], -self.max_linear_vel, self.max_linear_vel)
        cmd_msg.angular.z = np.clip(action[1], -self.max_angular_vel, self.max_angular_vel)
        
        # Publish command
        self.cmd_pub.publish(cmd_msg)
    
    def run(self):
        """Main control loop."""
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            self.update_and_control()
            rate.sleep()

if __name__ == '__main__':
    try:
        node = APF_TD3_ROSNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
```

### Example 7: OpenAI Gym Integration

```python
import gym
from gym import spaces
import numpy as np
from apf_td3 import Environment as APF_Environment, APF_TD3_Agent

class APF_TD3_GymWrapper(gym.Env):
    """
    OpenAI Gym wrapper for APF_TD3 environments.
    """
    
    def __init__(self, width=50, height=50, obstacles=None, **kwargs):
        super().__init__()
        
        # Initialize APF environment
        self.apf_env = APF_Environment(
            width=width,
            height=height,
            obstacles=obstacles or [],
            **kwargs
        )
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-self.apf_env.max_action,
            high=self.apf_env.max_action,
            shape=(self.apf_env.action_dim,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.apf_env.state_dim,),
            dtype=np.float32
        )
    
    def reset(self):
        return self.apf_env.reset()
    
    def step(self, action):
        state, reward, done, info = self.apf_env.step(action)
        return state, reward, done, info
    
    def render(self, mode='human'):
        return self.apf_env.render(mode=mode)
    
    def close(self):
        pass

# Example usage with stable-baselines3
def stable_baselines_example():
    """
    Example using APF_TD3 environment with stable-baselines3.
    """
    try:
        from stable_baselines3 import TD3
        from stable_baselines3.common.env_util import make_vec_env
        
        # Create environment
        def make_env():
            return APF_TD3_GymWrapper(
                width=40,
                height=40,
                obstacles=[(20, 20, 4), (30, 15, 3)]
            )
        
        # Create vectorized environment
        env = make_vec_env(make_env, n_envs=4)
        
        # Create stable-baselines3 TD3 agent
        model = TD3("MlpPolicy", env, verbose=1)
        
        # Train the agent
        print("Training with stable-baselines3 TD3...")
        model.learn(total_timesteps=100000)
        
        # Save the model
        model.save("sb3_td3_apf")
        
        # Test the trained model
        env = make_env()
        obs = env.reset()
        
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            
            if done:
                obs = env.reset()
        
        env.close()
        
    except ImportError:
        print("stable-baselines3 not installed. Install with: pip install stable-baselines3")

# Run stable-baselines example
# stable_baselines_example()
```

## Performance Optimization

### Example 8: Hyperparameter Optimization

```python
import optuna
from apf_td3 import APF_TD3_Agent, Environment
from apf_td3.utils import evaluate_performance

def optimize_hyperparameters():
    """
    Use Optuna for hyperparameter optimization.
    """
    
    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        gamma = trial.suggest_float('gamma', 0.9, 0.999)
        tau = trial.suggest_float('tau', 0.001, 0.01)
        policy_noise = trial.suggest_float('policy_noise', 0.1, 0.5)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        
        # APF parameters
        k_att = trial.suggest_float('k_att', 0.5, 2.0)
        k_rep = trial.suggest_float('k_rep', 5.0, 25.0)
        rho_0 = trial.suggest_float('rho_0', 3.0, 12.0)
        
        # Create environment
        env = Environment(
            width=40,
            height=40,
            obstacles=[(15, 15, 3), (25, 25, 4), (30, 10, 2)],
            start=(5, 5),
            goal=(35, 35)
        )
        
        # Create agent with suggested parameters
        agent = APF_TD3_Agent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            max_action=1.0,
            lr=lr,
            gamma=gamma,
            tau=tau,
            policy_noise=policy_noise,
            batch_size=batch_size
        )
        
        # Set APF parameters
        agent.set_apf_params(k_att=k_att, k_rep=k_rep, rho_0=rho_0)
        
        # Quick training
        metrics = agent.train(env, episodes=800)
        
        # Evaluate performance
        performance = evaluate_performance(agent, env, episodes=20)
        
        # Return objective value (success rate)
        return performance['success_rate']
    
    # Create study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    # Print best parameters
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print(f"Best success rate: {study.best_value:.3f}")
    
    # Visualize optimization
    try:
        import plotly
        
        # Plot optimization history
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.show()
        
        # Plot parameter importances
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.show()
        
    except ImportError:
        print("Install plotly for visualization: pip install plotly")
    
    return study.best_params

# Run hyperparameter optimization
# best_params = optimize_hyperparameters()
```

### Example 9: Parallel Training

```python
import multiprocessing as mp
import numpy as np
from apf_td3 import APF_TD3_Agent, Environment

def parallel_training_worker(worker_id, shared_params, results_queue):
    """
    Worker function for parallel training.
    """
    # Create environment for this worker
    np.random.seed(worker_id)  # Different seed for each worker
    
    env = Environment(
        width=50,
        height=50,
        obstacles=[(np.random.uniform(10, 40), np.random.uniform(10, 40), 
                   np.random.uniform(2, 5)) for _ in range(3)],
        start=(5, 5),
        goal=(45, 45)
    )
    
    # Create agent
    agent = APF_TD3_Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=1.0,
        **shared_params
    )
    
    # Train agent
    print(f"Worker {worker_id} starting training...")
    metrics = agent.train(env, episodes=1000)
    
    # Evaluate performance
    performance = evaluate_performance(agent, env, episodes=50)
    
    # Save model
    agent.save(f"models/parallel_agent_{worker_id}.pth")
    
    # Return results
    results_queue.put({
        'worker_id': worker_id,
        'performance': performance,
        'final_reward': np.mean(metrics['episode_rewards'][-100:])
    })

def run_parallel_training():
    """
    Run parallel training with multiple workers.
    """
    # Shared hyperparameters
    shared_params = {
        'lr': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'batch_size': 256
    }
    
    # Number of parallel workers
    num_workers = mp.cpu_count() // 2
    print(f"Starting parallel training with {num_workers} workers...")
    
    # Create processes
    processes = []
    results_queue = mp.Queue()
    
    for i in range(num_workers):
        p = mp.Process(
            target=parallel_training_worker,
            args=(i, shared_params, results_queue)
        )
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    for _ in range(num_workers):
        results.append(results_queue.get())
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Find best performing agent
    best_result = max(results, key=lambda x: x['performance']['success_rate'])
    print(f"Best agent: Worker {best_result['worker_id']}")
    print(f"Success rate: {best_result['performance']['success_rate']:.3f}")
    
    return results

# Run parallel training
# results = run_parallel_training()
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Agent Not Learning

```python
def debug_training_issues():
    """
    Debug common training issues.
    """
    env = Environment(width=30, height=30, obstacles=[(15, 15, 3)])
    agent = APF_TD3_Agent(state_dim=env.state_dim, action_dim=env.action_dim, max_action=1.0)
    
    # Monitor training closely
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    
    for episode in range(100):
        state = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.replay_buffer.add(state, action, next_state, reward, done)
            
            episode_reward += reward
            state = next_state
            
            # Update agent
            if len(agent.replay_buffer) > agent.batch_size:
                losses = agent.update(agent.replay_buffer)
                if losses:
                    actor_losses.append(losses.get('actor_loss', 0))
                    critic_losses.append(losses.get('critic_loss', 0))
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Print diagnostics
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_actor_loss = np.mean(actor_losses[-50:]) if actor_losses else 0
            avg_critic_loss = np.mean(critic_losses[-50:]) if critic_losses else 0
            
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                  f"Actor Loss = {avg_actor_loss:.4f}, Critic Loss = {avg_critic_loss:.4f}")
            
            # Check for common issues
            if avg_reward < -500:
                print("WARNING: Very low rewards - check reward function")
            if avg_actor_loss > 100:
                print("WARNING: High actor loss - consider reducing learning rate")
            if len(set(episode_rewards[-10:])) == 1:
                print("WARNING: Rewards not changing - check exploration")

# Run debugging
# debug_training_issues()
```

#### Issue 2: Performance Monitoring

```python
def monitor_agent_performance():
    """
    Comprehensive performance monitoring during training.
    """
    import time
    import matplotlib.pyplot as plt
    from collections import deque
    
    env = Environment(width=40, height=40, obstacles=[(20, 20, 4)])
    agent = APF_TD3_Agent(state_dim=env.state_dim, action_dim=env.action_dim, max_action=1.0)
    
    # Monitoring variables
    episode_rewards = deque(maxlen=100)
    success_rates = deque(maxlen=20)
    training_times = deque(maxlen=100)
    
    # Real-time plotting setup
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for episode in range(1000):
        start_time = time.time()
        
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 1000:
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, info = env.step(action)
            
            agent.replay_buffer.add(state, action, next_state, reward, done)
            episode_reward += reward
            state = next_state
            steps += 1
            
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update(agent.replay_buffer)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        training_times.append(time.time() - start_time)
        
        # Evaluate success rate every 50 episodes
        if episode % 50 == 0 and episode > 0:
            success_count = 0
            for _ in range(10):
                test_state = env.reset()
                for _ in range(1000):
                    test_action = agent.select_action(test_state, add_noise=False)
                    test_state, _, test_done, test_info = env.step(test_action)
                    if test_done and test_info.get('success', False):
                        success_count += 1
                        break
            success_rates.append(success_count / 10.0)
        
        # Update plots every 25 episodes
        if episode % 25 == 0:
            axes[0, 0].clear()
            axes[0, 0].plot(list(episode_rewards))
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            
            axes[0, 1].clear()
            if success_rates:
                axes[0, 1].plot(list(success_rates))
                axes[0, 1].set_title('Success Rate')
                axes[0, 1].set_xlabel('Evaluation')
                axes[0, 1].set_ylabel('Success Rate')
            
            axes[1, 0].clear()
            axes[1, 0].plot(list(training_times))
            axes[1, 0].set_title('Training Time per Episode')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Time (s)')
            
            axes[1, 1].clear()
            if len(episode_rewards) > 1:
                moving_avg = np.convolve(list(episode_rewards), 
                                       np.ones(min(20, len(episode_rewards)))/min(20, len(episode_rewards)), 
                                       mode='valid')
                axes[1, 1].plot(moving_avg)
                axes[1, 1].set_title('Moving Average Reward')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Avg Reward')
            
            plt.tight_layout()
            plt.pause(0.01)
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(list(episode_rewards)[-50:])
            avg_time = np.mean(list(training_times)[-50:])
            current_success = success_rates[-1] if success_rates else 0
            
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                  f"Success Rate = {current_success:.2f}, "
                  f"Avg Time = {avg_time:.2f}s")
    
    plt.ioff()
    plt.show()

# Run performance monitoring
# monitor_agent_performance()
```

This comprehensive examples documentation provides practical guidance for using the APF_TD3 library in various scenarios, from basic navigation to advanced applications with custom environments, ROS integration, and performance optimization techniques.