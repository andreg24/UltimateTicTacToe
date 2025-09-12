from __future__ import annotations

import os
from os import path as os_path

from enum import Enum
import random

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.classic.tictactoe.board import TTT_GAME_NOT_OVER, TTT_TIE, Board
from pettingzoo.utils import AgentSelector, wrappers

import supertictactoe
from board import SuperTicTacToeBoard

env = supertictactoe.env(render_mode="human")
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