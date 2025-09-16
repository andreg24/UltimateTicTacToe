"""
DOCSTRING TO DO 
"""
from __future__ import annotations

from enum import Enum

import numpy as np
from .board_utils import absolute_to_relative, relative_to_absolute

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

class SubTicTacToeBoard:

    def __init__(self):
        # self.squares holds a flat representation of the tic tac toe board.
        # an empty board is [0, 0, 0, 0, 0, 0, 0, 0, 0].
        # player 1's squares are marked 1, while player 2's are marked 2.
        # mapping of the flat indices to the 3x3 grid is as follows:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        self.cells = [0] * 9

    @property
    def _n_empty_cells(self):
        """The current number of empty squares on the board."""
        return self.cells.count(0)

    def reset(self):
        """Remove all marks from the board."""
        self.cells = [0] * 9

    def play_turn(self, player, pos):
        """Place a mark by the player in the spot given.

        The following are required for a move to be valid:
        * The player must be a known player ID (either 0 or 1).
        * The spot must be be empty.
        * The spot must be in the board (integer: 0 <= spot <= 8)

        If any of those are not true, an assertion will fail.
        """
        assert 0 <= pos <= 8, "Invalid move location"
        assert player in [0, 1], "Invalid player"
        assert self.cells[pos] == 0, "Location is not empty"

        # player is [0, 1]. board values are stored as [1, 2].
        self.cells[pos] = player + 1

    def game_status(self):
        """Return status (winner, TTT_TIE if no winner, or TTT_GAME_NOT_OVER)."""
        for indices in WINNING_COMBINATIONS:
            states = [self.cells[idx] for idx in indices]
            if states == [1, 1, 1]:
                return Status.PLAYER1_WIN
            if states == [2, 2, 2]:
                return Status.PLAYER2_WIN
        if self._n_empty_cells == 0:
            return Status.TIE
        return Status.GAME_NOT_OVER

    def __str__(self):
        return str(np.array(self.cells).reshape((3, 3)))

    def __repr__(self):
        return str(self)

    def legal_moves(self):
        """Return list of legal moves (as flat indices for spaces on the board)."""
        return [i for i, mark in enumerate(self.cells) if mark == 0]


class UltimateTicTacToeBoard:

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
        self.super_cells = [0] * 9 # high level board
        self.cells = [0] * 81 # board accounting for all the possible squares
        self.sub_boards = [SubTicTacToeBoard() for _ in range(9)]
        self.current_pos = -1 # -1 indicates free choice
        self.current_player = 0

    @property
    def _n_empty_super_cells(self):
        """The current number of empty squares on the board."""
        return self.super_cells.count(0)

    def reset(self):
        """Remove all marks from the board."""
        self.super_cells = [0] * 9
        self.cells = [0] * 81
        for sub_board in self.sub_boards:
            sub_board.reset()
        self.current_pos = -1 # -1 indicates free choice
        self.current_player = 0

    def play_turn(self, player, pos):
        """Place a mark by the player in the spot given.

        The following are required for a move to be valid:
        * The player must be a known player ID (either 0 or 1).
        * The spot must be be empty.
        * The spot must be in the board (integer: 0 <= spot <= 8)

        If any of those are not true, an assertion will fail.
        """
        assert player == self.current_player
        assert 0 <= pos <= 80
        assert self.is_valid_move(player, pos), f"player={player}, current={self.current_player}, pos={pos}"


        # player is [0, 1]. board values are stored as [1, 2].
        self.cells[pos] = player + 1
        super_pos, sub_pos = absolute_to_relative(pos)
        sub_board = self.sub_boards[super_pos]
        sub_board.play_turn(player, sub_pos)
        sub_board_status = sub_board.game_status()
        self.super_cells[super_pos] = sub_board_status.value
        # update current pos
        self.current_pos = sub_pos if self.super_cells[sub_pos] == 0 else -1
        self.current_player = (self.current_player + 1) % 2 # 0 -> 1, 1->0

    def is_valid_move(self, player, pos):
        # correct player
        if player != self.current_player:
            return False
        # valid pos range
        if not (0 <= pos <= 80):
            return False
        super_pos, sub_pos = absolute_to_relative(pos)
        # wrong uper pos
        if self.current_pos != -1 and self.current_pos != super_pos:
            return False
        # super pos already occupied
        if self.super_cells[super_pos] != 0:
            return False
        # sub pos occupied
        if self.sub_boards[super_pos].cells[sub_pos] != 0:
            return False
        return True

    def game_status(self):
        """Return status (winner, TTT_TIE if no winner, or TTT_GAME_NOT_OVER)."""
        for indices in WINNING_COMBINATIONS:
            states = [self.super_cells[idx] for idx in indices]
            if states == [1, 1, 1]:
                return Status.PLAYER1_WIN
            if states == [2, 2, 2]:
                return Status.PLAYER2_WIN
        if self._n_empty_super_cells == 0:
            return Status.TIE
        return Status.GAME_NOT_OVER

    def __str__(self):
        return str(np.array(self.super_cells).reshape((3, 3)))

    def __repr__(self):
        return str(self)

    def legal_moves(self):
        """Return list of legal moves (as flat indices for spaces on the board)."""
        available_moves = []
        if self.current_pos == -1:
            for super_pos in range(9):
                if self.super_cells[super_pos] == 0:
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
