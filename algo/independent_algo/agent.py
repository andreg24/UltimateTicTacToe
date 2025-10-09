from abc import ABC, abstractmethod
import random
import numpy as np

import torch
import torch.nn as nn

from utils.board_utils import relative_to_absolute

class SimplePolicy(nn.Module):
    """Network taking in input observations and outputting masked probability distributions for each possible action"""

    def __init__(self, net = None, epsilon=0.1, dropout_p = 0.2):
        super(SimplePolicy, self).__init__()
        self.epsilon = epsilon
        self.dropout_p = dropout_p

        if net is None:
            self.net = nn.Sequential(
                nn.Conv2d(4, 2, 5, padding=3),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.LazyConv2d(2, 5, padding=2),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.LazyConv2d(3, 5, padding=1),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.LazyConv2d(1, 3, padding=1),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(81),
            )
        else:
            self.net = net

        self.softmax = nn.Softmax(1)
    
    def forward(self, state: torch.Tensor):
        """state should be tensor of shape (B, 4, 9, 9)"""
        action_mask = state[:, 2, :, :].reshape(state.size(0), -1).bool().clone()
        logits = self.net(state) # (B, 81)
        masked_logits = logits.masked_fill(~action_mask, float('-inf'))
        probs = self.softmax(masked_logits)

        if self.epsilon > 0:
            unif_probs = torch.zeros_like(probs)
            unif_probs [action_mask] = 1.0
            unif_probs = unif_probs / unif_probs.sum(dim=1, keepdim=True)
            probs = (1 - self.epsilon) * probs + self.epsilon * unif_probs

        return probs

class BaseAgent(ABC):

    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def pick_action(self, state: dict) -> dict:
        """Takes in input a dictionary and outputs a dictionary with required key 'action'"""
        pass

class EnvRandomAgent:
    """Agent that picks actions uniformly at random"""

    def __init__(self, name, action_mask_enabled=True):
        self.name = name
        self.action_mask_enabled = action_mask_enabled
    
    def pick_action(self, env):
        if self.action_mask_enabled:
            action_mask = env.last()[0]['action_mask']
            return env.action_space(self.name).sample(action_mask)
        else:
            return env.action_space(self.name).sample()

class RandomAgent(BaseAgent):
    """Agent that picks actions uniformly at random"""

    def __init__(self, name, action_mask_enabled=True):
        super().__init__(name)
        self.action_mask_enabled = action_mask_enabled
    
    def pick_action(self, state):
        if not self.action_mask_enabled:
            action = np.random.randint(0, 81)
        else:
            mask = state['action_mask']
            valid_actions = np.where(mask == 1)[0]
            action = np.random.choice(valid_actions)
        return {'action': action}

class ManualAgent(BaseAgent):
    """Agent that picks actions uniformly at random"""

    def __init__(self, name):
        super().__init__(name)
    
    def pick_action(self, state):
        action = input("insert position: ")
        super_pos, sub_pos = action.split(' ')
        super_pos = int(super_pos)
        sub_pos = int(sub_pos)
        action = relative_to_absolute(super_pos, sub_pos)
        return {'action': action}

def state_to_tensor(state, turn_enabled=True, dtype=torch.float32, device=torch.device("cpu")):
    # consider adding a 4th channel for the turn number
    board_tensor = torch.tensor(state['observation'])
    action_mask_tensor = torch.tensor(state['action_mask'].reshape(9, 9)).unsqueeze(0)
    if turn_enabled:
        turn_tensor = torch.ones(1, 9, 9) * state['turn']
    
    if turn_enabled:
        state_tensor = torch.cat((board_tensor, action_mask_tensor, turn_tensor))
    else:
        state_tensor = torch.cat((board_tensor, action_mask_tensor))
    
    state_tensor = state_tensor.unsqueeze(0)
    state_tensor = state_tensor.to(dtype=dtype, device=device)

    return state_tensor

class NeuralAgent(BaseAgent):
    """Agent with a neural network evaluating policy(action/state)"""

    def __init__(
        self, 
        name,
        epsilon = 0.1,
        learning_power = 2,
        exploration_power = 6,
        policy_net = None,
        optimizer = None,
        device = None,
        mode = 'train'
    ):
        super().__init__(name)
        self.policy_net = policy_net if policy_net is not None else SimplePolicy(epsilon=epsilon**exploration_power)
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(self.policy_net.parameters(), lr=epsilon**learning_power)
        self.device = device if device is not None else torch.device('cpu')
        self.mode = mode

    def pick_action(self, state: torch.Tensor):
        """
        Obs is a (B, 3, 9, 9) tensor, where 
        the first two channels are the players moves on the board
        and the last one is the action mask for the board
        """

        state_tensor = state_to_tensor(state, device=self.device)

        probs = self.policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return {'action': action, 'log_prob': log_prob, 'probs': probs}

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 