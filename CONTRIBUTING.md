# Contributing to APF_TD3

Thank you for your interest in contributing to APF_TD3! This document provides guidelines and information for contributors to help maintain code quality and ensure smooth collaboration.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Performance Considerations](#performance-considerations)
- [Submitting Changes](#submitting-changes)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.8 or higher
- Git installed and configured
- Basic understanding of reinforcement learning concepts
- Familiarity with PyTorch
- Knowledge of artificial potential fields (helpful but not required)

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fixing issues in existing code
- **Feature additions**: New algorithms, environments, or utilities
- **Performance improvements**: Optimizations and efficiency enhancements
- **Documentation**: Improving docs, examples, and tutorials
- **Testing**: Adding tests or improving test coverage
- **Examples**: New use cases and integration examples

## Development Setup

### 1. Fork and Clone the Repository

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/APF_TD3.git
cd APF_TD3

# Add the original repository as upstream
git remote add upstream https://github.com/original-owner/APF_TD3.git
```

### 2. Create a Virtual Environment

```bash
# Create and activate virtual environment
python -m venv apf_td3_dev
source apf_td3_dev/bin/activate  # On Windows: apf_td3_dev\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Development Dependencies

```bash
# Install the package in development mode with all dependencies
pip install -e ".[dev]"

# Or install dependencies manually
pip install -e .
pip install pytest pytest-cov black flake8 mypy pre-commit sphinx
pip install matplotlib seaborn plotly optuna
```

### 4. Set Up Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks on all files (optional)
pre-commit run --all-files
```

### 5. Verify Installation

```bash
# Run tests to ensure everything is working
python -m pytest tests/

# Check code formatting
black --check apf_td3/
flake8 apf_td3/
mypy apf_td3/
```

## Contribution Workflow

### 1. Create a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create a new branch for your feature
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clean, well-documented code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest tests/ --cov=apf_td3 --cov-report=html

# Run specific test files
python -m pytest tests/test_agent.py -v
```

### 4. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "Add feature: brief description of changes

- Detailed description of what was added/changed
- Any breaking changes or important notes
- Fixes #issue_number (if applicable)"
```

### 5. Push and Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create a pull request on GitHub
# Include a detailed description of your changes
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Import organization**: Use `isort` for consistent import ordering
- **Docstrings**: Google-style docstrings for all public functions and classes

### Code Formatting

We use automated tools for consistent formatting:

```bash
# Format code with Black
black apf_td3/ tests/ examples/

# Sort imports
isort apf_td3/ tests/ examples/

# Check linting
flake8 apf_td3/ tests/

# Type checking
mypy apf_td3/
```

### Naming Conventions

- **Classes**: PascalCase (`APF_TD3_Agent`, `Environment`)
- **Functions/Methods**: snake_case (`select_action`, `compute_force`)
- **Variables**: snake_case (`state_dim`, `max_action`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_EPISODES`, `DEFAULT_LR`)
- **Private methods**: Leading underscore (`_update_networks`)

### Example Code Structure

```python
"""Module docstring describing the module's purpose."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from apf_td3.base import BaseAgent


class ExampleAgent(BaseAgent):
    """
    Example agent class demonstrating coding standards.
    
    This class shows how to structure code according to our guidelines,
    including proper docstrings, type hints, and error handling.
    
    Args:
        state_dim: Dimension of the state space.
        action_dim: Dimension of the action space.
        learning_rate: Learning rate for the optimizer.
        device: Device for computation ('cpu' or 'cuda').
        
    Attributes:
        state_dim: Stored state dimension.
        action_dim: Stored action dimension.
        device: Computation device.
        
    Example:
        >>> agent = ExampleAgent(state_dim=6, action_dim=2)
        >>> action = agent.select_action(state)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        device: str = "auto"
    ) -> None:
        super().__init__()
        
        # Validate inputs
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = self._setup_device(device)
        
        # Initialize networks
        self._build_networks()
        
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action based on the current state.
        
        Args:
            state: Current environment state with shape (state_dim,).
            
        Returns:
            Selected action with shape (action_dim,).
            
        Raises:
            ValueError: If state has incorrect shape.
            
        Example:
            >>> state = np.array([1.0, 2.0, 3.0])
            >>> action = agent.select_action(state)
        """
        if state.shape != (self.state_dim,):
            raise ValueError(
                f"Expected state shape ({self.state_dim},), "
                f"got {state.shape}"
            )
            
        # Implementation details...
        action = self._compute_action(state)
        return action
        
    def _compute_action(self, state: np.ndarray) -> np.ndarray:
        """Private method for action computation."""
        # Implementation...
        pass
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
```

## Testing Guidelines

### Test Structure

Tests are organized in the `tests/` directory:

```
tests/
├── __init__.py
├── test_agent.py          # Agent-related tests
├── test_environment.py    # Environment tests
├── test_apf.py           # APF controller tests
├── test_utils.py         # Utility function tests
├── integration/          # Integration tests
│   ├── test_training.py
│   └── test_ros_integration.py
└── fixtures/             # Test fixtures and data
    ├── __init__.py
    └── sample_environments.py
```

### Writing Tests

Use `pytest` for all tests:

```python
"""Test module for APF_TD3_Agent."""

import numpy as np
import pytest
import torch

from apf_td3 import APF_TD3_Agent, Environment


class TestAPF_TD3_Agent:
    """Test cases for APF_TD3_Agent class."""
    
    @pytest.fixture
    def simple_env(self):
        """Create a simple test environment."""
        return Environment(width=20, height=20, obstacles=[(10, 10, 2)])
    
    @pytest.fixture
    def agent(self, simple_env):
        """Create a test agent."""
        return APF_TD3_Agent(
            state_dim=simple_env.state_dim,
            action_dim=simple_env.action_dim,
            max_action=1.0
        )
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.state_dim == 6
        assert agent.action_dim == 2
        assert agent.max_action == 1.0
        
    def test_select_action_shape(self, agent):
        """Test action selection returns correct shape."""
        state = np.random.randn(agent.state_dim)
        action = agent.select_action(state)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (agent.action_dim,)
        assert np.all(np.abs(action) <= agent.max_action)
        
    def test_invalid_state_dimension(self, agent):
        """Test error handling for invalid state dimensions."""
        invalid_state = np.random.randn(5)  # Wrong dimension
        
        with pytest.raises(ValueError, match="Expected state shape"):
            agent.select_action(invalid_state)
            
    def test_training_basic(self, agent, simple_env):
        """Test basic training functionality."""
        metrics = agent.train(simple_env, episodes=10)
        
        assert 'episode_rewards' in metrics
        assert len(metrics['episode_rewards']) == 10
        assert all(isinstance(r, (int, float)) for r in metrics['episode_rewards'])
        
    @pytest.mark.slow
    def test_training_convergence(self, agent, simple_env):
        """Test that agent shows learning progress (slow test)."""
        metrics = agent.train(simple_env, episodes=100)
        
        # Check that performance improves
        early_rewards = np.mean(metrics['episode_rewards'][:20])
        late_rewards = np.mean(metrics['episode_rewards'][-20:])
        
        assert late_rewards > early_rewards, "Agent should show learning progress"
        
    def test_save_load(self, agent, tmp_path):
        """Test model saving and loading."""
        # Train briefly
        env = Environment(width=10, height=10)
        agent.train(env, episodes=5)
        
        # Save model
        save_path = tmp_path / "test_agent.pth"
        agent.save(str(save_path))
        assert save_path.exists()
        
        # Create new agent and load
        new_agent = APF_TD3_Agent(
            state_dim=agent.state_dim,
            action_dim=agent.action_dim,
            max_action=agent.max_action
        )
        new_agent.load(str(save_path))
        
        # Test that loaded agent produces same actions
        state = np.random.randn(agent.state_dim)
        action1 = agent.select_action(state)
        action2 = new_agent.select_action(state)
        
        np.testing.assert_allclose(action1, action2, rtol=1e-5)


# Parametrized tests for different configurations
@pytest.mark.parametrize("state_dim,action_dim", [
    (4, 2),
    (6, 2),
    (8, 3),
])
def test_agent_different_dimensions(state_dim, action_dim):
    """Test agent with different state/action dimensions."""
    agent = APF_TD3_Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0
    )
    
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    
    assert action.shape == (action_dim,)


# Integration test
def test_full_training_pipeline():
    """Test complete training pipeline."""
    env = Environment(width=30, height=30, obstacles=[(15, 15, 3)])
    agent = APF_TD3_Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=1.0
    )
    
    # Train
    metrics = agent.train(env, episodes=20)
    
    # Test trained agent
    state = env.reset()
    total_reward = 0
    
    for _ in range(100):
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    # Should achieve some reasonable performance
    assert total_reward > -1000, f"Poor performance: {total_reward}"
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=apf_td3 --cov-report=html

# Run specific test file
python -m pytest tests/test_agent.py -v

# Run specific test
python -m pytest tests/test_agent.py::TestAPF_TD3_Agent::test_initialization -v

# Run only fast tests (skip slow ones)
python -m pytest tests/ -m "not slow"

# Run tests in parallel
python -m pytest tests/ -n auto
```

## Documentation Guidelines

### Docstring Format

Use Google-style docstrings:

```python
def compute_force(
    self,
    position: np.ndarray,
    goal: np.ndarray,
    obstacles: List[np.ndarray]
) -> np.ndarray:
    """
    Compute the artificial potential field force.
    
    This method calculates both attractive forces toward the goal and
    repulsive forces away from obstacles, combining them into a total
    force vector for navigation.
    
    Args:
        position: Current position as [x, y] coordinates.
        goal: Goal position as [x, y] coordinates.
        obstacles: List of obstacles, each as [x, y, radius].
        
    Returns:
        Total force vector as [fx, fy].
        
    Raises:
        ValueError: If position or goal have incorrect dimensions.
        
    Example:
        >>> controller = APF_Controller()
        >>> pos = np.array([0.0, 0.0])
        >>> goal = np.array([10.0, 10.0])
        >>> obstacles = [np.array([5.0, 5.0, 2.0])]
        >>> force = controller.compute_force(pos, goal, obstacles)
        >>> print(f"Force: {force}")
        Force: [0.8 0.8]
        
    Note:
        The force magnitude is automatically clipped to max_force to
        prevent numerical instabilities.
    """
```

### Type Hints

Use comprehensive type hints:

```python
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

def train(
    self,
    env: Environment,
    episodes: int = 1000,
    save_freq: Optional[int] = None,
    eval_freq: int = 100
) -> Dict[str, List[float]]:
    """Training method with full type hints."""
    pass

# For complex types, define type aliases
StateDict = Dict[str, Union[np.ndarray, float, int]]
TrainingMetrics = Dict[str, List[float]]
```

### Adding Examples

When adding new features, include examples in `examples/`:

```python
"""
Example: Custom Environment Creation

This example demonstrates how to create a custom environment
by extending the base Environment class.
"""

import numpy as np
from apf_td3 import Environment, APF_TD3_Agent


class CustomEnvironment(Environment):
    """Custom environment with special reward function."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_penalty = 0.01
        
    def step(self, action):
        state, reward, done, info = super().step(action)
        
        # Add custom step penalty
        reward -= self.step_penalty
        
        # Add distance-based reward shaping
        distance_reward = -0.1 * info['distance_to_goal']
        reward += distance_reward
        
        return state, reward, done, info


def main():
    """Run the custom environment example."""
    # Create custom environment
    env = CustomEnvironment(
        width=40,
        height=40,
        obstacles=[(20, 20, 4)],
        start=(5, 5),
        goal=(35, 35)
    )
    
    # Train agent
    agent = APF_TD3_Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=1.0
    )
    
    print("Training with custom environment...")
    metrics = agent.train(env, episodes=1000)
    
    # Show results
    print(f"Final average reward: {np.mean(metrics['episode_rewards'][-100:]):.2f}")


if __name__ == "__main__":
    main()
```

## Performance Considerations

### Computational Efficiency

- **Vectorization**: Use NumPy vectorized operations when possible
- **Memory Management**: Be mindful of memory usage in replay buffers
- **GPU Utilization**: Ensure proper GPU usage for neural networks
- **Batch Processing**: Process multiple samples together when possible

### Profiling Code

```python
# Use cProfile for performance analysis
python -m cProfile -o profile_output.prof examples/basic_navigation.py

# Use line_profiler for line-by-line analysis
@profile
def slow_function():
    # Function to profile
    pass

kernprof -l -v script.py
```

### Memory Optimization

```python
# Example of memory-efficient replay buffer
class EfficientReplayBuffer:
    def __init__(self, max_size: int, state_dim: int, action_dim: int):
        # Pre-allocate arrays
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        # ... other arrays
        
    def add(self, state, action, next_state, reward, done):
        # Use in-place operations
        idx = self.ptr % self.max_size
        self.states[idx] = state
        self.actions[idx] = action
        # ... store other values
```

## Submitting Changes

### Pull Request Checklist

Before submitting a pull request, ensure:

- [ ] Code follows the style guidelines
- [ ] All tests pass
- [ ] New functionality includes tests
- [ ] Documentation is updated
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages are descriptive
- [ ] Pre-commit hooks pass

### Pull Request Template

```markdown
## Description

Brief description of the changes and their purpose.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have added integration tests if applicable

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published

## Additional Notes

Any additional information, concerns, or questions about the changes.
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and style checks
2. **Code Review**: Maintainers review code for quality and correctness
3. **Discussion**: Address feedback and make necessary changes
4. **Approval**: Once approved, changes are merged

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers get started
- Focus on the technical aspects of contributions

### Communication

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and general topics
- **Pull Requests**: Keep discussions focused on the specific changes

### Getting Help

If you need help:

1. Check the documentation and examples
2. Search existing issues and discussions
3. Create a new issue with a clear description
4. Join community discussions

### Recognition

Contributors are recognized in:

- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- GitHub contributor statistics

Thank you for contributing to APF_TD3! Your efforts help make this project better for everyone.