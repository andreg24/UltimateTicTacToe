"""Small script to run a random Super Tic Tac Toe game"""
from __future__ import annotations
from ultimatetictactoe import ultimatetictactoe

env = ultimatetictactoe.env(render_mode="human")
env.reset(42)
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        mask = observation["action_mask"]
        action = env.action_space(agent).sample(mask)
    env.step(action)
