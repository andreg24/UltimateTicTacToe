"""
DOCSTRING TO DO 
"""
from __future__ import annotations

from enum import Enum

import numpy as np
from utils.board_utils import absolute_to_relative, relative_to_absolute

class Status(Enum):
    PLAYER1_WIN = 1
    PLAYER2_WIN = 2
    TIE = -1
    GAME_NOT_OVER = 0

WINNING_COMBINATIONS = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]

class TicTacToeBoard:

    def __init__(self):
        # self.squares holds a flat representation of the tic tac toe board.
        # an empty board is [0, 0, 0, 0, 0, 0, 0, 0, 0].
        # player 1's squares are marked 1, while player 2's are marked 2.
        # mapping of the flat indices to the 3x3 grid is as follows:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        self.squares = [0] * 9

    @property
    def _n_empty_squares(self):
        """The current number of empty squares on the board."""
        return self.squares.count(0)

    def reset(self):
        """Remove all marks from the board."""
        self.squares = [0] * 9

    def play_turn(self, agent, pos):
        """Place a mark by the agent in the spot given.

        The following are required for a move to be valid:
        * The agent must be a known agent ID (either 0 or 1).
        * The spot must be be empty.
        * The spot must be in the board (integer: 0 <= spot <= 8)

        If any of those are not true, an assertion will fail.
        """
        assert 0 <= pos <= 8, "Invalid move location"
        assert agent in [0, 1], "Invalid agent"
        assert self.squares[pos] == 0, "Location is not empty"

        # agent is [0, 1]. board values are stored as [1, 2].
        self.squares[pos] = agent + 1

    def game_status(self):
        """Return status (winner, TTT_TIE if no winner, or TTT_GAME_NOT_OVER)."""
        for indices in WINNING_COMBINATIONS:
            states = [self.squares[idx] for idx in indices]
            if states == [1, 1, 1]:
                return Status.PLAYER1_WIN
            if states == [2, 2, 2]:
                return Status.PLAYER2_WIN
        if self._n_empty_squares == 0:
            return Status.TIE
        return Status.GAME_NOT_OVER

    def __str__(self):
        return str(np.array(self.squares).reshape((3, 3)))

    def __repr__(self):
        return str(self)

    def legal_moves(self):
        """Return list of legal moves (as flat indices for spaces on the board)."""
        return [i for i, mark in enumerate(self.squares) if mark == 0]


class SuperTicTacToeBoard:


    def __init__(self):
        # self.squares holds a flat representation of the super tic tac toe board
        # an empty board is [0, 0, 0, 0, 0, 0, 0, 0, 0].
        # player 1's victories are marked 1, while player 2's victories are marked 2
        # ties are marked -1
        # mapping of the flat indices to the 3x3 grid is as follows:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # COULD START USING CELL AND GRID NOMENCLATURE INSTEAD
        self.super_squares = [0] * 9 # high level board
        self.squares = [0] * 81 # board accounting for all the possible squares
        self.sub_boards = [TicTacToeBoard() for _ in range(9)]
        self.current_pos = -1 # -1 indicates free choice

    @property
    def _n_empty_squares(self):
        """The current number of empty squares on the board."""
        return self.super_squares.count(0)

    def reset(self):
        """Remove all marks from the board."""
        self.super_squares = [0] * 9
        self.squares = [0] * 81
        for sub_board in self.sub_boards:
            sub_board.reset()
        self.current_pos = -1 # -1 indicates free choice

    def play_turn(self, agent, pos):
        """Place a mark by the agent in the spot given.

        The following are required for a move to be valid:
        * The agent must be a known agent ID (either 0 or 1).
        * The spot must be be empty.
        * The spot must be in the board (integer: 0 <= spot <= 8)

        If any of those are not true, an assertion will fail.
        """
        # agent is [0, 1]. board values are stored as [1, 2].
        self.squares[pos] = agent + 1
        super_pos, sub_pos = absolute_to_relative(pos)
        sub_board = self.sub_boards[super_pos]
        sub_board.play_turn(agent, sub_pos)
        sub_board_status = sub_board.game_status()
        self.super_squares[super_pos] = sub_board_status.value
        # update current pos
        self.current_pos = sub_pos if self.super_squares[sub_pos] == 0 else -1

    def game_status(self):
        """Return status (winner, TTT_TIE if no winner, or TTT_GAME_NOT_OVER)."""
        for indices in WINNING_COMBINATIONS:
            states = [self.super_squares[idx] for idx in indices]
            if states == [1, 1, 1]:
                return Status.PLAYER1_WIN
            if states == [2, 2, 2]:
                return Status.PLAYER2_WIN
        if self._n_empty_squares == 0:
            return Status.TIE
        return Status.GAME_NOT_OVER

    def __str__(self):
        return str(np.array(self.super_squares).reshape((3, 3)))

    def __repr__(self):
        return str(self)

    def legal_moves(self):
        """Return list of legal moves (as flat indices for spaces on the board)."""
        available_moves = []
        if self.current_pos == -1:
            for super_pos in range(9):
                if self.super_squares[super_pos] == 0:
                    sub_board = self.sub_boards[super_pos]
                    for sub_pos in sub_board.legal_moves():
                        pos = relative_to_absolute(super_pos, sub_pos)
                        available_moves.append(pos)
        else:
            super_pos = self.current_pos
            sub_board = self.sub_boards[super_pos]
            for sub_pos in sub_board.legal_moves():
                pos = relative_to_absolute(super_pos, sub_pos)
                available_moves.append(pos)
        return available_moves
