from __future__ import annotations

import os

import random

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

from board import SuperTicTacToeBoard, Status
from utils.render_utils import get_image, get_font

SCREEN_HEIGHT = 500

BOARD_IMG = "board.jpg"
CROSS_IMG = "cross.jpg"
CIRCLE_IMG = "circle.jpg"

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "supertictactoe_v3",
        "is_parallelizable": False,
        "render_fps": 5,
    }

    def __init__(
        self, render_mode: str | None = None, screen_height: int | None = SCREEN_HEIGHT
    ):
        super().__init__()
        EzPickle.__init__(self, render_mode, screen_height)
        self.board = SuperTicTacToeBoard()

        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {i: spaces.Discrete(81) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(9, 9, 2), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(81,), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.render_mode = render_mode
        self.screen_height = screen_height
        self.screen = None

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def observe(self, agent):
        board_vals = np.array(self.board.squares).reshape(9, 9) # TO DO
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2

        observation = np.empty((9, 9, 2), dtype=np.int8)
        # this will give a copy of the board that is 1 for player 1's
        # marks and zero for every other square, whether empty or not.
        observation[:, :, 0] = np.equal(board_vals, cur_player + 1)
        observation[:, :, 1] = np.equal(board_vals, opp_player + 1)

        action_mask = self._get_mask(agent)

        return {"observation": observation, "action_mask": action_mask}

    def _get_mask(self, agent):
        action_mask = np.zeros(81, dtype=np.int8)

        # Per the documentation, the mask of any agent other than the
        # currently selected one is all zeros.
        if agent == self.agent_selection:
            for i in self.board.legal_moves():
                action_mask[i] = 1

        return action_mask

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # action in this case is a value from 0 to 8 indicating position to move on tictactoe board
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        self.board.play_turn(self.agents.index(self.agent_selection), action)

        status = self.board.game_status()
        if status != Status.GAME_NOT_OVER:
            if status == Status.TIE:
                pass
            else:
                winner = status.value - 1  # either TTT_PLAYER1_WIN or TTT_PLAYER2_WIN
                loser = winner ^ 1  # 0 -> 1; 1 -> 0
                self.rewards[self.agents[winner]] += 1
                self.rewards[self.agents[loser]] -= 1

            # once either play wins or there is a draw, game over, both players are done
            self.terminations = {i: True for i in self.agents}
            self._accumulate_rewards()

        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):
        self.board.reset()

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        if self.render_mode is not None and self.screen is None:
            pygame.init()

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.screen_height, self.screen_height)
            )
            pygame.display.set_caption("Super Tic-Tac-Toe")
        elif self.render_mode == "rgb_array":
            self.screen = pygame.Surface((self.screen_height, self.screen_height))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        screen_height = self.screen_height
        screen_width = self.screen_height

        # Setup dimensions for 'x' and 'o' marks
        tile_size = int(screen_height / 12)
        tile_size_big = int(screen_height / 4)

        # Load and blit the board image for the game
        board_img = get_image(os.path.join("img", "board.jpg"))
        board_img = pygame.transform.scale(
            board_img, (int(screen_width), int(screen_height))
        )

        self.screen.blit(board_img, (0, 0))

        # Load and blit actions for the game
        def get_symbol(input):
            if input == 0:
                return None
            elif input == 1:
                return "cross"
            else:
                return "circle"

        squares = self.board.squares
        super_squares = self.board.super_squares
        board_state = list(map(get_symbol, squares))
        super_board_state = list(map(get_symbol, super_squares))

        mark_pos = 0
        for x in range(9):
            for y in range(9):
                mark = board_state[mark_pos]
                mark_pos += 1

                if mark is None:
                    continue

                mark_img = get_image(os.path.join("img", mark + ".png"))
                mark_img.set_alpha(100)
                mark_img = pygame.transform.scale(mark_img, (tile_size, tile_size))

                self.screen.blit(
                    mark_img,
                    (
                        (screen_width / 8.8) * x + (screen_width / 190),
                        (screen_width / 8.8) * y + (screen_height / 190),
                    ),
                )
        # big symbols
        mark_pos = 0
        for x in range(3):
            for y in range(3):
                mark = super_board_state[mark_pos]
                mark_pos += 1

                if mark is None:
                    continue

                mark_img = get_image(os.path.join("img", mark + ".png"))
                mark_img = pygame.transform.scale(mark_img, (tile_size_big, tile_size_big))


                self.screen.blit(
                    mark_img,
                    (
                        (screen_width / 2.9) * x + (screen_width / 30),
                        (screen_width / 2.9) * y + (screen_height / 30),
                    ),
                )

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )
