# AI Mastery Course - Day 89: Reinforcement Learning

**Imagine that...** you're training a master chef who learns to create perfect dishes not through recipes, but by tasting countless variations, remembering what worked, and gradually refining their technique through trial, error, and reward. This is the essence of reinforcement learning - an AI approach where agents learn optimal behavior through interaction with their environment, receiving feedback in the form of rewards and penalties.

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand the fundamental concepts of reinforcement learning and its key components
- Implement Q-learning algorithms to solve decision-making problems
- Build and train Deep Q-Networks (DQN) for complex state spaces
- Apply Actor-Critic methods for continuous action spaces
- Design multi-agent environments where multiple learners interact

---

## 1. Q-Learning and Policy Gradients

### The Foundation: Q-Learning

Think of Q-learning as teaching a chef to rate every possible cooking decision. The "Q-table" is like a master cookbook that records the value of each action in every situation.

```python
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Initialize the Q-learning agent
        
        Args:
            learning_rate (float): How fast the agent learns (alpha)
            discount_factor (float): How much future rewards matter (gamma)
            epsilon (float): Exploration rate for epsilon-greedy policy
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.training_rewards = []
    
    def get_action(self, state, available_actions, training=True):
        """
        Choose an action using epsilon-greedy policy
        Like a chef deciding whether to try a new technique or stick with what works
        """
        if training and random.random() < self.epsilon:
            # Explore: try a random action (experiment with new ingredients)
            return random.choice(available_actions)
        else:
            # Exploit: choose the best known action (use proven techniques)
            q_values = [self.q_table[state][action] for action in available_actions]
            max_q = max(q_values)
            # Handle ties by randomly choosing among best actions
            best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, next_actions):
        """
        Update Q-values using the Q-learning formula
        Like updating the chef's knowledge after tasting the result
        """
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # Maximum Q-value for next state (best future option)
        if next_actions:
            next_max_q = max([self.q_table[next_state][a] for a in next_actions])
        else:
            next_max_q = 0  # Terminal state
        
        # Q-learning update rule: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q

# Simple Grid World Environment (like a kitchen layout)
class GridWorld:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height
        self.reset()
        
        # Define rewards (like ingredients and hazards in a kitchen)
        self.rewards = {
            (4, 4): 10,  # Goal (perfect dish)
            (2, 2): -5,  # Obstacle (burnt ingredient)
            (1, 3): -5,  # Another obstacle
        }
    
    def reset(self):
        """Reset to starting position"""
        self.agent_pos = (0, 0)  # Start at top-left (prep station)
        return self.agent_pos
    
    def get_available_actions(self):
        """Get valid moves from current position"""
        actions = []
        x, y = self.agent_pos
        
        if x > 0: actions.append('left')
        if x < self.width - 1: actions.append('right')
        if y > 0: actions.append('up')
        if y < self.height - 1: actions.append('down')
        
        return actions
    
    def step(self, action):
        """Take an action and return new state, reward, done"""
        x, y = self.agent_pos
        
        # Move based on action
        if action == 'up' and y > 0:
            y -= 1
        elif action == 'down' and y < self.height - 1:
            y += 1
        elif action == 'left' and x > 0:
            x -= 1
        elif action == 'right' and x < self.width - 1:
            x += 1
        
        self.agent_pos = (x, y)
        
        # Calculate reward
        reward = self.rewards.get(self.agent_pos, -0.1)  # Small penalty for each step
        done = (self.agent_pos == (4, 4))  # Reached the goal
        
        return self.agent_pos, reward, done

# Training the Q-learning agent
def train_q_learning(episodes=1000):
    """Train the agent like teaching a chef through practice"""
    env = GridWorld()
    agent = QLearningAgent()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100  # Prevent infinite loops
        
        while steps < max_steps:
            # Get available actions and choose one
            available_actions = env.get_available_actions()
            action = agent.get_action(state, available_actions)
            
            # Take action and observe result
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Update Q-table
            next_actions = env.get_available_actions()
            agent.update(state, action, reward, next_state, next_actions)
            
            state = next_state
            steps += 1
            
            if done:
                break
        
        agent.training_rewards.append(total_reward)
        
        # Decay exploration rate (chef becomes more confident over time)
        if episode % 100 == 0:
            agent.epsilon = max(0.01, agent.epsilon * 0.95)
            print(f"Episode {episode}, Average Reward: {np.mean(agent.training_rewards[-100:]):.2f}")
    
    return agent, env

# Run the training
trained_agent, environment = train_q_learning(episodes=1000)
```

**Syntax Explanation:**
- `defaultdict(lambda: defaultdict(float))`: Creates a nested dictionary that automatically initializes missing keys with default values
- `random.choice(available_actions)`: Selects a random element from the list for exploration
- `max([self.q_table[next_state][a] for a in next_actions])`: List comprehension to find maximum Q-value among available actions
- The Q-learning update formula implements the Bellman equation for optimal value functions

### Policy Gradients: Direct Policy Learning

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """
    Neural network that learns a policy directly
    Like a chef's intuition network that maps situations to actions
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        """Forward pass through the network"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class REINFORCEAgent:
    """
    REINFORCE algorithm implementation
    Learns by adjusting action probabilities based on episode outcomes
    """
    def __init__(self, state_size, action_size, learning_rate=0.01):
        self.policy_net = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Storage for episode data
        self.log_probs = []
        self.rewards = []
    
    def get_action(self, state):
        """Select action based on current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_net(state_tensor)
        
        # Create probability distribution and sample action
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Store log probability for later update
        self.log_probs.append(dist.log_prob(action))
        
        return action.item()
    
    def update_policy(self):
        """Update policy based on collected episode data"""
        # Calculate discounted rewards
        discounted_rewards = []
        running_reward = 0
        
        # Work backwards through rewards (like reflecting on a cooking session)
        for reward in reversed(self.rewards):
            running_reward = reward + 0.99 * running_reward
            discounted_rewards.insert(0, running_reward)
        
        # Normalize rewards
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        
        # Update policy network
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.log_probs.clear()
        self.rewards.clear()
```

**Syntax Explanation:**
- `nn.Module`: Base class for all neural network modules in PyTorch
- `F.softmax(self.fc3(x), dim=-1)`: Applies softmax activation along the last dimension to get action probabilities
- `Categorical(action_probs)`: Creates a categorical probability distribution for sampling discrete actions
- `dist.log_prob(action)`: Computes log probability of the sampled action for gradient computation

---

## 2. Deep Q-Networks (DQN)

When the environment becomes too complex for a simple table (like a professional kitchen with thousands of ingredient combinations), we need Deep Q-Networks.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

class DQN(nn.Module):
    """
    Deep Q-Network: Neural network approximation of Q-function
    Like a chef's deep knowledge network that handles complex ingredient combinations
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """
    Experience replay buffer
    Like a chef's memory of past cooking experiences
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store an experience"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.BoolTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    DQN Agent with experience replay and target network
    """
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Neural networks
        self.q_network = DQN(state_size, 64, action_size)
        self.target_network = DQN(state_size, 64, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Example usage with a more complex environment
class ComplexEnvironment:
    """
    A more complex environment requiring DQN
    Like a high-end restaurant kitchen with many variables
    """
    def __init__(self):
        self.state_size = 8  # Multiple state variables
        self.action_size = 4  # Four possible actions
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Initialize state with multiple variables
        # (position, velocity, temperature, ingredients_available, etc.)
        self.state = np.random.normal(0, 1, self.state_size)
        return self.state.copy()
    
    def step(self, action):
        """Take action and return next state, reward, done"""
        # Simulate environment dynamics
        noise = np.random.normal(0, 0.1, self.state_size)
        
        # Action effects (simplified)
        if action == 0:  # Move left
            self.state[0] -= 0.1
        elif action == 1:  # Move right
            self.state[0] += 0.1
        elif action == 2:  # Increase heat
            self.state[2] += 0.2
        elif action == 3:  # Decrease heat
            self.state[2] -= 0.2
        
        # Add noise and clip values
        self.state += noise
        self.state = np.clip(self.state, -3, 3)
        
        # Calculate reward (achieve balance in all state variables)
        reward = -np.sum(np.abs(self.state))  # Reward for balanced state
        
        # Episode ends if we achieve good balance or after too many steps
        done = np.sum(np.abs(self.state)) < 2.0
        
        return self.state.copy(), reward, done

# Training function for DQN
def train_dqn(episodes=1000):
    """Train DQN agent"""
    env = ComplexEnvironment()
    agent = DQNAgent(env.state_size, env.action_size)
    
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 200
        
        while steps < max_steps:
            action = agent.act(state, training=True)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        
        # Update target network periodically
        if episode % 100 == 0:
            agent.update_target_network()
            print(f"Episode {episode}, Average Score: {np.mean(scores[-100:]):.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, env

# Train the DQN agent
dqn_agent, dqn_env = train_dqn(episodes=500)
```

**Syntax Explanation:**
- `deque(maxlen=capacity)`: Creates a double-ended queue with maximum length for efficient memory management
- `torch.FloatTensor(states)`: Converts numpy arrays to PyTorch tensors
- `.gather(1, actions.unsqueeze(1))`: Gathers Q-values for taken actions along dimension 1
- `next_q_values.max(1)[0].detach()`: Gets maximum Q-values and detaches from computation graph
- `~dones`: Boolean NOT operation to mask terminal states

---

## 3. Actor-Critic Methods

Actor-Critic combines the best of both worlds: direct policy learning (Actor) with value function estimation (Critic), like having both a creative chef (Actor) and a food critic (Critic) working together.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class Actor(nn.Module):
    """
    Actor network: learns the policy (what actions to take)
    Like the creative chef who decides what to cook
    """
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # For continuous actions: output mean and log standard deviation
        self.mean_layer = nn.Linear(hidden_size, action_size)
        self.log_std_layer = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = torch.tanh(self.mean_layer(x))  # Bounded actions
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)  # Clamp for stability
        
        return mean, log_std
    
    def get_action(self, state):
        """Sample action from policy"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Create normal distribution and sample
        normal_dist = Normal(mean, std)
        action = normal_dist.sample()
        log_prob = normal_dist.log_prob(action)
        
        return action, log_prob

class Critic(nn.Module):
    """
    Critic network: learns the value function (how good is this state)
    Like the food critic who evaluates the current situation
    """
    def __init__(self, state_size, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_layer(x)
        return value

class ActorCriticAgent:
    """
    Actor-Critic agent combining policy and value learning
    """
    def __init__(self, state_size, action_size, lr_actor=0.0003, lr_critic=0.001):
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = 0.99  # Discount factor
        
        # Storage for episode data
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def get_action(self, state):
        """Get action from actor and value from critic"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action from actor
        action, log_prob = self.actor.get_action(state_tensor)
        
        # Get value from critic
        value = self.critic(state_tensor)
        
        return action.squeeze().numpy(), log_prob.squeeze(), value.squeeze()
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """Store transition data"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self):
        """Update actor and critic networks"""
        # Convert to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.FloatTensor(self.actions)
        log_probs = torch.stack(self.log_probs)
        rewards = torch.FloatTensor(self.rewards)
        values = torch.stack(self.values)
        dones = torch.BoolTensor(self.dones)
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        
        # Calculate discounted returns
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        
        # Calculate advantages (returns - values)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor loss (policy gradient with advantage)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss (value function approximation)
        critic_loss = F.mse_loss(values, returns.detach())
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Clear episode data
        self.clear_memory()
        
        return actor_loss.item(), critic_loss.item()
    
    def clear_memory(self):
        """Clear stored transitions"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

# Continuous action environment
class ContinuousEnvironment:
    """
    Environment with continuous action space
    Like controlling cooking temperature and timing precisely
    """
    def __init__(self):
        self.state_size = 4
        self.action_size = 2  # Two continuous actions
        self.reset()
    
    def reset(self):
        """Reset to initial state"""
        self.state = np.random.uniform(-1, 1, self.state_size)
        self.steps = 0
        return self.state.copy()
    
    def step(self, action):
        """Take continuous action"""
        # Clip actions to reasonable range
        action = np.clip(action, -1, 1)
        
        # Simple dynamics: actions affect state directly
        self.state[:2] += action * 0.1
        self.state[2:] += np.random.normal(0, 0.05, 2)  # Some randomness
        
        # Clip state values
        self.state = np.clip(self.state, -2, 2)
        
        # Reward: negative distance from origin (want balanced cooking)
        reward = -np.sum(self.state**2)
        
        self.steps += 1
        done = self.steps >= 200 or reward > -0.1
        
        return self.state.copy(), reward, done

# Training function
def train_actor_critic(episodes=1000):
    """Train Actor-Critic agent"""
    env = ContinuousEnvironment()
    agent = ActorCriticAgent(env.state_size, env.action_size)
    
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            agent.store_transition(state, action, log_prob, reward, value, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update networks
        actor_loss, critic_loss = agent.update()
        scores.append(total_reward)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Avg Score: {avg_score:.2f}, "
                  f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
    
    return agent, env

# Train the agent
ac_agent, ac_env = train_actor_critic(episodes=500)
```

**Syntax Explanation:**
- `Normal(mean, std)`: Creates a normal (Gaussian) probability distribution for continuous actions
- `torch.clamp(log_std, -20, 2)`: Clamps values between -20 and 2 for numerical stability
- `torch.nn.utils.clip_grad_norm_()`: Clips gradient norms to prevent exploding gradients
- `.detach()`: Removes tensor from computation graph to prevent gradient flow

---

## 4. Multi-Agent Environments

In a bustling restaurant kitchen, multiple chefs must coordinate their actions. Multi-agent RL deals with scenarios where multiple learning agents interact.

```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class MultiAgentEnvironment:
    """
    Multi-agent environment where agents must cooperate
    Like a kitchen where multiple chefs work together on different stations
    """
    def __init__(self, num_agents=3, grid_size=8):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.action_size = 5  # up, down, left, right, stay
        self.state_size = 4   # x, y, has_ingredient, goal_distance
        
        # Define cooking stations and ingredients
        self.stations = [(1, 1), (6, 1), (3, 6)]  # Prep stations
        self.ingredients = [(2, 3), (5, 2), (4, 5)]  # Ingredient locations
        self.final_station = (4, 4)  # Where final dish is assembled
        
        self.reset()
    
    def reset(self):
        """Reset environment with agents at different starting positions"""
        self.agents_pos = [(0, i) for i in range(self.num_agents)]
        self.agents_have_ingredient = [False] * self.num_agents
        self.ingredients_collected = 0
        self.steps = 0
        
        return self.get_states()
    
    def get_states(self):
        """Get observation for each agent"""
        states = []
        for i in range(self.num_agents):
            x, y = self.agents_pos[i]
            has_ingredient = float(self.agents_have_ingredient[i])
            
            # Distance to nearest uncollected ingredient or final station
            if not self.agents_have_ingredient[i]:
                # Find nearest ingredient
                distances = [abs(x - ix) + abs(y - iy) for ix, iy in self.ingredients]
                goal_distance = min(distances) / self.grid_size
            else:
                # Distance to final station
                fx, fy = self.final_station
                goal_distance = (abs(x - fx) + abs(y - fy)) / self.grid_size
            
            state = [x / self.grid_size, y / self.grid_size, has_ingredient, goal_distance]
            states.append(state)
        
        return states
    
    def step(self, actions):
        """Execute actions for all agents simultaneously"""
        rewards = [0] * self.num_agents
        
        # Move agents
        for i, action in enumerate(actions):
            x, y = self.agents_pos[i]
            
            if action == 0:  # up
                y = max(0, y - 1)
            elif action == 1:  # down
                y = min(self.grid_size - 1, y + 1)
            elif action == 2:  # left
                x = max(0, x - 1)
            elif action == 3:  # right
                x = min(self.grid_size - 1, x + 1)
            # action == 4 is stay (no movement)
            
            self.agents_pos[i] = (x, y)
        
        # Check for ingredient collection
        for i in range(self.num_agents):
            if not self.agents_have_ingredient[i]:
                for j, ingredient_pos in enumerate(self.ingredients):
                    if self.agents_pos[i] == ingredient_pos:
                        self.agents_have_ingredient[i] = True
                        rewards[i] += 10  # Reward for collecting ingredient
                        # Remove ingredient (it's been collected)
                        self.ingredients[j] = (-1, -1)  # Mark as collected
        
        # Check for final assembly (cooperation reward)
        agents_at_final = sum(1 for pos in self.agents_pos if pos == self.final_station)
        ingredients_ready = sum(self.agents_have_ingredient)
        
        if agents_at_final >= 2 and ingredients_ready >= 2:
            # Bonus for cooperation
            cooperation_reward = 20
            for i in range(self.num_agents):
                rewards[i] += cooperation_reward
        
        # Small penalty for each step (encourage efficiency)
        for i in range(self.num_agents):
            rewards[i] -= 0.1
        
        self.steps += 1
        done = self.steps >= 100 or (ingredients_ready >= 2 and agents_at_final >= 2)
        
        return self.get_states(), rewards, done

class IndependentDQNAgent:
    """
    Independent DQN agent for multi-agent learning
    Each chef learns independently but observes the shared kitchen
    """
    def __init__(self, state_size, action_size, agent_id, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Simple neural network
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Train the network"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_multi_agent(episodes=1000):
    """Train multiple agents to cooperate"""
    env = MultiAgentEnvironment(num_agents=3)
    agents = [IndependentDQNAgent(env.state_size, env.action_size, i) for i in range(env.num_agents)]
    
    episode_rewards = []
    
    for episode in range(episodes):
        states = env.reset()
        total_rewards = [0] * env.num_agents
        
        while True:
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(agents):
                action = agent.act(states[i], training=True)
                actions.append(action)
            
            # Execute actions
            next_states, rewards, done = env.step(actions)
            
            # Store experiences and update
            for i, agent in enumerate(agents):
                agent.remember(states[i], actions[i], rewards[i], next_states[i], done)
                agent.replay()
                total_rewards[i] += rewards[i]
            
            states = next_states
            
            if done:
                break
        
        episode_rewards.append(sum(total_rewards))
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Team Reward: {avg_reward:.2f}")
            print(f"Agent Epsilons: {[f'{agent.epsilon:.3f}' for agent in agents]}")
    
    return agents, env

# Train multi-agent system
multi_agents, multi_env = train_multi_agent(episodes=800)

print("Multi-agent training completed!")
```

**Syntax Explanation:**
- `sum(1 for pos in self.agents_pos if pos == self.final_station)`: Generator expression with conditional counting
- `zip(*batch)`: Unpacks list of tuples into separate lists (transpose operation)
- `torch.LongTensor(actions)`: Converts to long tensor for indexing operations
- List comprehensions `[f'{agent.epsilon:.3f}' for agent in agents]`: Creates formatted list of epsilon values

---

## Final Project: Comprehensive RL Trading Agent

Now let's combine all concepts into a sophisticated trading agent that learns to make buy/sell decisions in a simulated market - like a master chef managing an entire restaurant operation.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random

class TradingEnvironment:
    """
    Stock trading environment that combines all RL concepts
    Like managing a restaurant's supply chain, inventory, and daily operations
    """
    def __init__(self, data_length=1000):
        # Generate synthetic market data
        np.random.seed(42)
        self.prices = self.generate_market_data(data_length)
        self.initial_balance = 10000
        self.reset()
    
    def generate_market_data(self, length):
        """Generate realistic market price data"""
        prices = [100]  # Starting price
        for _ in range(length - 1):
            # Random walk with trend and volatility
            trend = np.random.normal(0.0005, 0.002)  # Small positive trend
            volatility = np.random.normal(0, 0.02)   # Daily volatility
            change = trend + volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # Prevent negative prices
        return np.array(prices)
    
    def reset(self):
        """Reset trading environment"""
        self.current_step = 50  # Start with some history
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.trades = []
        
        return self.get_state()
    
    def get_state(self):
        """
        Get current state representation
        Includes price history, technical indicators, and portfolio state
        """
        # Price history (last 10 days normalized)
        price_history = self.prices[self.current_step-10:self.current_step]
        price_history = (price_history - price_history.mean()) / (price_history.std() + 1e-8)
        
        # Technical indicators
        current_price = self.prices[self.current_step]
        sma_5 = np.mean(self.prices[self.current_step-5:self.current_step])
        sma_20 = np.mean(self.prices[self.current_step-20:self.current_step])
        
        price_to_sma5 = current_price / sma_5 - 1
        price_to_sma20 = current_price / sma_20 - 1
        sma_ratio = sma_5 / sma_20 - 1
        
        # Portfolio state
        portfolio_ratio = self.shares_held * current_price / self.balance if self.balance > 0 else 0
        cash_ratio = self.balance / self.initial_balance
        
        # Combine all features
        state = np.concatenate([
            price_history,
            [price_to_sma5, price_to_sma20, sma_ratio, portfolio_ratio, cash_ratio]
        ])
        
        return state
    
    def step(self, action):
        """
        Execute trading action
        Actions: 0 = Hold, 1 = Buy, 2 = Sell
        """
        current_price = self.prices[self.current_step]
        reward = 0
        
        if action == 1:  # Buy
            # Buy as many shares as possible with available balance
            if self.balance > current_price:
                shares_to_buy = int(self.balance * 0.1 / current_price)  # Use 10% of balance
                if shares_to_buy > 0:
                    self.shares_held += shares_to_buy
                    self.balance -= shares_to_buy * current_price
                    self.trades.append(('BUY', self.current_step, current_price, shares_to_buy))
                    reward = -0.001  # Small transaction cost
        
        elif action == 2:  # Sell
            # Sell 10% of holdings or all if less than 10 shares
            if self.shares_held > 0:
                shares_to_sell = max(1, int(self.shares_held * 0.1))
                shares_to_sell = min(shares_to_sell, self.shares_held)
                
                self.shares_held -= shares_to_sell
                self.balance += shares_to_sell * current_price
                self.trades.append(('SELL', self.current_step, current_price, shares_to_sell))
                reward = -0.001  # Small transaction cost
        
        # Calculate new net worth
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Reward based on net worth change
        if self.net_worth > self.max_net_worth:
            reward += (self.net_worth - self.max_net_worth) / self.initial_balance
            self.max_net_worth = self.net_worth
        
        # Move to next time step
        self.current_step += 1
        
        # Episode ends if we reach end of data or lose too much money
        done = (self.current_step >= len(self.prices) - 1) or (self.net_worth < self.initial_balance * 0.5)
        
        return self.get_state(), reward, done
    
    def get_performance_stats(self):
        """Calculate trading performance statistics"""
        total_return = (self.net_worth - self.initial_balance) / self.initial_balance
        return {
            'total_return': total_return,
            'final_balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth,
            'num_trades': len(self.trades)
        }

class AdvancedTradingAgent:
    """
    Advanced trading agent using Actor-Critic with experience replay
    Combines the best of all RL methods we've learned
    """
    def __init__(self, state_size, action_size=3, hidden_size=128):
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0005)
        
        # Experience replay buffer
        self.memory = deque(maxlen=5000)
        self.batch_size = 64
        
        # Training parameters
        self.gamma = 0.95
        self.entropy_weight = 0.01
        
    def get_action(self, state, training=True):
        """Get action from policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
        
        if training:
            # Sample from probability distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()
            
            return action.item(), log_prob.item(), entropy.item(), value.item()
        else:
            # Choose best action for evaluation
            return action_probs.argmax().item(), 0, 0, value.item()
    
    def remember(self, state, action, reward, next_state, done, log_prob, entropy, value):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done, log_prob, entropy, value))
    
    def learn(self):
        """Learn from batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0, 0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, log_probs, entropies, values = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        old_log_probs = torch.FloatTensor(log_probs)
        old_entropies = torch.FloatTensor(entropies)
        old_values = torch.FloatTensor(values)
        
        # Calculate returns and advantages
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze()
            returns = rewards + self.gamma * next_values * ~dones
            advantages = returns - old_values
        
        # Current policy evaluation
        current_action_probs = self.actor(states)
        current_values = self.critic(states).squeeze()
        
        # Calculate losses
        action_dist = torch.distributions.Categorical(current_action_probs)
        new_log_probs = action_dist.log_prob(actions)
        entropy_loss = action_dist.entropy().mean()
        
        # Actor loss (policy gradient with entropy regularization)
        ratio = torch.exp(new_log_probs - old_log_probs)
        actor_loss = -(ratio * advantages.detach()).mean() - self.entropy_weight * entropy_loss
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_values, returns.detach())
        
        # Update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

def train_trading_agent(episodes=500):
    """Train the comprehensive trading agent"""
    env = TradingEnvironment(data_length=2000)
    state_size = len(env.get_state())
    agent = AdvancedTradingAgent(state_size)
    
    episode_returns = []
    episode_lengths = []
    
    print("Training Advanced Trading Agent...")
    print("=" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        actor_losses = []
        critic_losses = []
        
        while True:
            # Get action from agent
            action, log_prob, entropy, value = agent.get_action(state, training=True)
            
            # Execute action
            next_state, reward, done = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done, log_prob, entropy, value)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Learn from experience
            if len(agent.memory) >= agent.batch_size and episode_length % 10 == 0:
                actor_loss, critic_loss = agent.learn()
                if actor_loss != 0:  # Only record if learning occurred
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
            
            if done:
                break
        
        # Record episode statistics
        performance = env.get_performance_stats()
        episode_returns.append(performance['total_return'])
        episode_lengths.append(episode_length)
        
        # Print progress
        if episode % 50 == 0:
            avg_return = np.mean(episode_returns[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
            avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
            
            print(f"Episode {episode:3d} | "
                  f"Avg Return: {avg_return:6.2%} | "
                  f"Avg Length: {avg_length:5.1f} | "
                  f"Actor Loss: {avg_actor_loss:7.4f} | "
                  f"Critic Loss: {avg_critic_loss:7.4f}")
            print(f"            | "
                  f"Final Net Worth: ${performance['net_worth']:8.2f} | "
                  f"Trades: {performance['num_trades']:3d}")
    
    return agent, env, episode_returns

# Train the comprehensive agent
print("Starting comprehensive reinforcement learning training...")
final_agent, final_env, returns_history = train_trading_agent(episodes=300)

# Evaluate the trained agent
def evaluate_agent(agent, env, episodes=10):
    """Evaluate trained agent performance"""
    evaluation_returns = []
    
    for _ in range(episodes):
        state = env.reset()
        while True:
            action, _, _, _ = agent.get_action(state, training=False)
            state, _, done = env.step(action)
            if done:
                break
        
        performance = env.get_performance_stats()
        evaluation_returns.append(performance['total_return'])
    
    return evaluation_returns

# Evaluate the final agent
eval_returns = evaluate_agent(final_agent, final_env, episodes=20)
print("\n" + "=" * 50)
print("FINAL EVALUATION RESULTS")
print("=" * 50)
print(f"Average Return: {np.mean(eval_returns):.2%}")
print(f"Best Return: {np.max(eval_returns):.2%}")
print(f"Worst Return: {np.min(eval_returns):.2%}")
print(f"Standard Deviation: {np.std(eval_returns):.2%}")
print(f"Win Rate: {sum(r > 0 for r in eval_returns)/len(eval_returns):.1%}")
```

**Syntax Explanation:**
- `nn.BatchNorm1d()`: Batch normalization layer that normalizes inputs across the batch dimension
- `nn.Dropout(0.2)`: Regularization technique that randomly zeros 20% of input units during training
- `torch.distributions.Categorical()`: Creates a categorical distribution for discrete action sampling
- `torch.exp(new_log_probs - old_log_probs)`: Calculates probability ratio for policy gradient methods
- `zip(*batch)`: Unpacks batch of experiences into separate arrays for each component

---

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

class TicTacToeEnvironment:
    """
    Like a master chef's kitchen setup, this environment provides the playing field
    where our AI agent will learn to make winning moves
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the game board - like clearing the countertop for a fresh cooking session"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for AI, -1 for opponent
        self.done = False
        return self.get_state()
    
    def get_state(self):
        """Get current board state - like checking all ingredients on the counter"""
        return self.board.flatten()
    
    def get_valid_actions(self):
        """Get available moves - like checking which cooking stations are free"""
        return [i for i in range(9) if self.board.flatten()[i] == 0]
    
    def step(self, action):
        """Make a move - like placing an ingredient in the perfect spot"""
        if action not in self.get_valid_actions():
            return self.get_state(), -10, True  # Invalid move penalty
        
        # Convert action to board position
        row, col = action // 3, action % 3
        self.board[row, col] = self.current_player
        
        reward = self.calculate_reward()
        self.done = self.is_game_over()
        
        # Switch players if game not over
        if not self.done:
            self.current_player *= -1
            # Make random opponent move
            if self.get_valid_actions():
                opp_action = random.choice(self.get_valid_actions())
                opp_row, opp_col = opp_action // 3, opp_action % 3
                self.board[opp_row, opp_col] = self.current_player
                self.current_player *= -1
                self.done = self.is_game_over()
                reward += self.calculate_reward()
        
        return self.get_state(), reward, self.done
    
    def calculate_reward(self):
        """Calculate reward - like tasting the dish to see how well it turned out"""
        winner = self.check_winner()
        if winner == 1:  # AI wins
            return 100
        elif winner == -1:  # AI loses
            return -100
        elif self.is_board_full():  # Draw
            return 10
        else:
            return 0  # Game continues
    
    def check_winner(self):
        """Check for winner - like a head chef inspecting the final dish"""
        # Check rows, columns, and diagonals
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:  # Row
                return self.board[i, 0]
            if abs(sum(self.board[:, i])) == 3:  # Column
                return self.board[0, i]
        
        # Diagonals
        if abs(sum([self.board[i, i] for i in range(3)])) == 3:
            return self.board[1, 1]
        if abs(sum([self.board[i, 2-i] for i in range(3)])) == 3:
            return self.board[1, 1]
        
        return 0
    
    def is_board_full(self):
        """Check if board is full - like checking if all burners are occupied"""
        return len(self.get_valid_actions()) == 0
    
    def is_game_over(self):
        """Check if game is over - like determining if the meal service is complete"""
        return self.check_winner() != 0 or self.is_board_full()

class DQN(nn.Module):
    """
    Deep Q-Network - like a sous chef's brain that learns the best cooking techniques
    through experience and practice
    """
    def __init__(self, input_size=9, hidden_size=128, output_size=9):
        super(DQN, self).__init__()
        # Neural network layers - like different skill levels in cooking
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """Forward pass - like following a recipe step by step"""
        x = F.relu(self.fc1(x))  # First layer activation
        x = F.relu(self.fc2(x))  # Second layer activation
        x = self.fc3(x)          # Output layer (Q-values)
        return x

class DQNAgent:
    """
    The AI agent - like a chef-in-training who learns optimal cooking strategies
    through trial, error, and memory of past successes
    """
    def __init__(self, state_size=9, action_size=9, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.epsilon = 1.0        # Exploration rate - like trying new recipes
        self.epsilon_min = 0.01   # Minimum exploration
        self.epsilon_decay = 0.995 # Decay rate
        
        # Neural networks - main and target (for stability)
        self.q_network = DQN(state_size, 128, action_size)
        self.target_network = DQN(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.update_target_network()
        
    def update_target_network(self):
        """Update target network - like updating the master recipe book"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience - like writing down what worked in the recipe book"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions):
        """Choose action - like a chef deciding which technique to use"""
        if np.random.rand() <= self.epsilon:
            # Explore - try random valid action (like experimenting with flavors)
            return random.choice(valid_actions)
        
        # Exploit - use learned knowledge
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        
        # Mask invalid actions by setting their Q-values very low
        masked_q_values = q_values.clone()
        for i in range(self.action_size):
            if i not in valid_actions:
                masked_q_values[0][i] = -float('inf')
        
        return masked_q_values.argmax().item()
    
    def replay(self, batch_size=32):
        """Learn from past experiences - like reviewing successful dishes"""
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        # Compute loss and update
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(episodes=2000):
    """
    Train the agent - like a culinary school where the student chef
    practices thousands of dishes to master the craft
    """
    env = TicTacToeEnvironment()
    agent = DQNAgent()
    
    scores = []
    win_rate = []
    wins = 0
    
    print("🍳 Starting AI Chef Training...")
    print("Like a dedicated sous chef learning the art of perfect timing and strategy!")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            action = agent.act(state, valid_actions)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                if env.check_winner() == 1:  # AI won
                    wins += 1
                break
        
        # Train the agent
        if len(agent.memory) > 32:
            agent.replay(32)
        
        # Update target network every 100 episodes
        if episode % 100 == 0:
            agent.update_target_network()
        
        scores.append(total_reward)
        
        # Calculate win rate over last 100 games
        if episode >= 99:
            recent_wins = sum(1 for i in range(max(0, episode-99), episode+1) 
                            if scores[i] > 50)  # Win gives reward > 50
            win_rate.append(recent_wins)
        
        # Progress updates
        if episode % 200 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            current_win_rate = win_rate[-1] if win_rate else 0
            print(f"Episode {episode}, Avg Score: {avg_score:.2f}, "
                  f"Win Rate: {current_win_rate}%, Epsilon: {agent.epsilon:.3f}")
    
    return agent, scores, win_rate

def play_against_agent(agent):
    """
    Play against the trained agent - like challenging the master chef
    to a cooking competition!
    """
    env = TicTacToeEnvironment()
    
    print("\n🎮 Time to challenge our trained AI Chef!")
    print("You are 'O' (-1), AI is 'X' (1)")
    
    state = env.reset()
    env.current_player = -1  # Human starts
    
    while not env.done:
        # Display board
        display_board = env.board.copy()
        display_board[display_board == 1] = 'X'
        display_board[display_board == -1] = 'O'
        display_board[display_board == 0] = ' '
        
        print("\nCurrent Board:")
        for i in range(3):
            print(f" {display_board[i,0]} | {display_board[i,1]} | {display_board[i,2]} ")
            if i < 2:
                print("-----------")
        
        if env.current_player == -1:  # Human turn
            valid_moves = env.get_valid_actions()
            print(f"\nValid moves: {valid_moves}")
            
            try:
                move = int(input("Enter your move (0-8): "))
                if move not in valid_moves:
                    print("Invalid move! Try again.")
                    continue
            except ValueError:
                print("Please enter a number!")
                continue
            
            row, col = move // 3, move % 3
            env.board[row, col] = -1
            env.current_player = 1
            
        else:  # AI turn
            valid_actions = env.get_valid_actions()
            if valid_actions:
                action = agent.act(state, valid_actions)
                row, col = action // 3, action % 3
                env.board[row, col] = 1
                env.current_player = -1
                print(f"\nAI Chef plays position {action}")
        
        state = env.get_state()
        env.done = env.is_game_over()
    
    # Final board
    display_board = env.board.copy()
    display_board[display_board == 1] = 'X'
    display_board[display_board == -1] = 'O'
    display_board[display_board == 0] = ' '
    
    print("\nFinal Board:")
    for i in range(3):
        print(f" {display_board[i,0]} | {display_board[i,1]} | {display_board[i,2]} ")
        if i < 2:
            print("-----------")
    
    winner = env.check_winner()
    if winner == 1:
        print("\n🤖 AI Chef wins! The student has become the master!")
    elif winner == -1:
        print("\n🎉 You won! Even master chefs can learn from their guests!")
    else:
        print("\n🤝 It's a draw! A perfectly balanced meal!")

def visualize_training_progress(scores, win_rate):
    """
    Visualize training progress - like tracking a chef's improvement over time
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot scores
    ax1.plot(scores)
    ax1.set_title('Training Scores Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    
    # Plot win rate
    if win_rate:
        ax2.plot(range(99, 99 + len(win_rate)), win_rate)
        ax2.set_title('Win Rate Over Time (Last 100 Games)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Win Rate (%)')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    print("🚀 Welcome to the AI Game Chef Academy!")
    print("Training a master chef to play the perfect game...")
    
    # Train the agent
    trained_agent, training_scores, training_win_rate = train_agent(2000)
    
    print(f"\n🎓 Training Complete!")
    print(f"Final win rate: {training_win_rate[-1] if training_win_rate else 0}%")
    
    # Visualize progress
    visualize_training_progress(training_scores, training_win_rate)
    
    # Play against the agent
    while True:
        play_choice = input("\nWould you like to play against the AI? (y/n): ").lower()
        if play_choice == 'y':
            play_against_agent(trained_agent)
        elif play_choice == 'n':
            print("Thanks for visiting the AI Game Chef Academy! 👋")
            break
        else:
            print("Please enter 'y' or 'n'")

## Assignment: Custom Multi-Agent Restaurant Kitchen

**Objective:** Design and implement a multi-agent reinforcement learning system where three specialized chef agents must collaborate to run an efficient restaurant kitchen.

**Requirements:**

1. **Environment Design:**
   - Create a 10x10 grid kitchen with different stations: prep (3), cooking (2), plating (1), and serving (1)
   - Implement order queue system with different dish types requiring specific sequences
   - Add time pressure mechanics where orders expire if not completed within time limits

2. **Agent Specialization:**
   - **Prep Chef:** Specializes in ingredient preparation and inventory management
   - **Line Cook:** Focuses on cooking processes and heat management  
   - **Expediter:** Handles plating, quality control, and order coordination

3. **Implementation Tasks:**
   - Implement cooperative reward structure where agents succeed together or fail together
   - Add communication mechanism between agents (shared message passing)
   - Create dynamic difficulty scaling based on restaurant "rush hours"
   - Implement customer satisfaction metrics based on order completion time and quality

4. **Performance Metrics:**
   - Orders completed per episode
   - Average customer wait time
   - Coordination efficiency (how well agents work together)
   - Resource utilization rates

5. **Deliverables:**
   - Complete working multi-agent environment (300+ lines of code)
   - Training script with performance visualization
   - Analysis report comparing independent learning vs. centralized training approaches
   - Demonstration showing trained agents handling a busy dinner rush scenario

**Evaluation Criteria:**
- Code quality and documentation (25%)
- Environment complexity and realism (25%)
- Agent coordination and emergent behaviors (25%)
- Performance analysis and insights (25%)

The assignment should demonstrate mastery of multi-agent coordination, environment design, and the ability to balance competition and cooperation in RL systems - much like master chefs who must coordinate seamlessly to deliver exceptional dining experiences under pressure.

---

## Key Concepts Summary

Throughout this reinforcement learning journey, we've explored how agents learn optimal behavior through interaction, much like chefs perfecting their craft through experience:

**Q-Learning** taught us the fundamentals of value-based learning, where agents build knowledge tables of state-action values through trial and error.

**Deep Q-Networks** extended this concept to handle complex environments using neural networks as function approximators, enabling learning in high-dimensional spaces.

**Actor-Critic Methods** combined the best of policy-based and value-based approaches, with actors making decisions while critics evaluate their worth.

**Multi-Agent Systems** showed us how multiple learners can coexist and cooperate, creating emergent behaviors through interaction.

Our final trading agent project demonstrated how these concepts integrate into sophisticated real-world applications, combining experience replay, policy gradients, and value estimation in a unified framework.

The key insight is that reinforcement learning mirrors how we naturally learn complex skills - through practice, feedback, and gradual refinement of our decision-making processes. Whether training AI agents or master chefs, the principles remain the same: learn from experience, balance exploration with exploitation, and continuously adapt to achieve optimal performance.