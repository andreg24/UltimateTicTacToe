import torch
import torch.nn as nn

from utils.board_utils import relative_to_absolute

class SimplePolicy(nn.Module):
    """Network taking in input observations and outputting masked probability distributions for each possible action"""

    def __init__(self, net = None):
        super(SimplePolicy, self).__init__()

        if net is None:
            self.net = nn.Sequential(
                nn.Conv2d(3, 3, 7, padding=3),
                nn.ReLU(),
                nn.Conv2d(3, 3, 5, padding=2),
                nn.ReLU(),
                nn.Conv2d(3, 3, 3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(243, 81),
            )
        else:
            self.net = net

        self.softmax = nn.Softmax(1)
    
    def forward(self, state: torch.Tensor):
        """state should be tensor of shape (B, 3, 9, 9)"""
        action_mask = state[:, 2, :, :].reshape(state.size(0), -1).bool()
        logits = self.net(state) # (B, 81)
        masked_logits = logits.masked_fill(~action_mask, float('-inf'))
        probs = self.softmax(masked_logits)
        return probs

class RandomAgent:
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

class ManualAgent:
    """Agent that picks actions uniformly at random"""

    def __init__(self, name, action_mask_enabled=True):
        self.name = name
        self.action_mask_enabled = action_mask_enabled
    
    def pick_action(self, env):
        action = input("insert position: ")
        super_pos, sub_pos = action.split(' ')
        super_pos = int(super_pos)
        sub_pos = int(sub_pos)
        action = relative_to_absolute(super_pos, sub_pos)
        return action


class NeuralAgent:
    """Agent with a neural network evaluating policy(action/state)"""

    def __init__(
        self, 
        name, 
        policy_net = None, 
        optimizer = None,
        device = None,
        mode = 'train'
    ):
        self.name = name
        self.policy_net = policy_net if policy_net is not None else SimplePolicy()
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.device = device if device is not None else torch.device('cpu')
        self.mode = mode

    def pick_action(self, obs: torch.Tensor):
        """
        Obs is a (B, 3, 9, 9) tensor, where 
        the first two channels are the players moves on the board
        and the last one is the action mask for the board
        """

        probs = self.policy_net(obs.to(self.device))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
