from __future__ import annotations

from ultimatetictactoe import ultimatetictactoe

# import torch
from torch import nn

import random

env = ultimatetictactoe.env(render_mode="human")
env.reset(42)
c = 0
for agent in env.agent_iter():
    print(c)
    c += 1
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        mask = observation["action_mask"]
        action = env.action_space(agent).sample(mask)

    env.step(action)
