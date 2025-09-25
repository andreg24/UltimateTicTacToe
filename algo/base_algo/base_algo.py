import torch
import torch.nn as nn

class SimplePolicy(nn.Module):
    """Network taking in input observations and outputting masked probability distributions for each possible action"""

    def __init__(self, net = None):
        super(SimplePolicy, self).__init__()

        if net is None:
            self.net = nn.Sequential(
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

class NeuralAgent:
    """TODO"""

    def __init__(
        self, 
        name, 
        policy_net, 
        optimizer,
        device,
        mode = 'train'
    ):
        self.name = name
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.device = device
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