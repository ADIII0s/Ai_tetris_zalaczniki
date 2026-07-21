from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


BOARD_FEATURE_NAMES = [
    "aggregate_height",
    "max_height",
    "holes",
    "bumpiness",
    "hole_depth",
    "blocks_above_holes",
    "wells",
    "row_transitions",
    "column_transitions",
    "danger_zone",
    "filled_cells",
]

DECISION_WEIGHT_NAMES = BOARD_FEATURE_NAMES + ["line_clear_reward"]

N_BOARD_FEATURES = len(BOARD_FEATURE_NAMES)
N_WEIGHTS = len(DECISION_WEIGHT_NAMES)

IDX_MAX_HEIGHT = BOARD_FEATURE_NAMES.index("max_height")
IDX_FILLED_CELLS = BOARD_FEATURE_NAMES.index("filled_cells")


class ExtendedFeatureObservation(gym.ObservationWrapper):
    """Normalized board features for every projected placement."""

    def __init__(self, env):
        super().__init__(env)
        self.width = int(env.unwrapped.width)
        self.height = int(env.unwrapped.height)
        self.padding = int(env.unwrapped.padding)
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(N_BOARD_FEATURES,),
            dtype=np.float32,
        )

    def _crop_board(self, observation) -> np.ndarray:
        board = np.array(observation["board"], copy=True)
        active_mask = np.asarray(
            observation["active_tetromino_mask"],
            dtype=bool,
        )
        board[active_mask] = 0
        return board[
            0 : -self.padding,
            self.padding : -self.padding,
        ]

    @staticmethod
    def _column_heights(filled: np.ndarray) -> np.ndarray:
        height = filled.shape[0]
        first_filled = np.argmax(filled, axis=0)
        heights = height - first_filled
        heights[~np.any(filled, axis=0)] = 0
        return heights.astype(np.int32)

    @staticmethod
    def _hole_mask(filled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        filled_above_count = np.cumsum(filled, axis=0)
        holes = (~filled) & (filled_above_count > 0)
        return holes, filled_above_count

    def _well_sum(self, heights: np.ndarray) -> int:
        total = 0
        for column in range(self.width):
            left_height = (
                self.height if column == 0 else int(heights[column - 1])
            )
            right_height = (
                self.height
                if column == self.width - 1
                else int(heights[column + 1])
            )
            depth = max(
                0,
                min(left_height, right_height) - int(heights[column]),
            )
            total += depth * (depth + 1) // 2
        return int(total)

    def observation(self, observation) -> np.ndarray:
        board = self._crop_board(observation)
        filled = board != 0
        heights = self._column_heights(filled)
        holes, filled_above_count = self._hole_mask(filled)

        aggregate_height = int(np.sum(heights))
        max_height = int(np.max(heights))
        hole_count = int(np.sum(holes))
        bumpiness = int(np.sum(np.abs(np.diff(heights))))
        hole_depth = int(np.sum(filled_above_count[holes]))

        holes_below = np.cumsum(holes[::-1], axis=0)[::-1] > 0
        blocks_above_holes = int(np.sum(filled & holes_below))
        well_sum = self._well_sum(heights)

        horizontal = np.pad(
            filled,
            ((0, 0), (1, 1)),
            mode="constant",
            constant_values=True,
        )
        row_transitions = int(
            np.sum(horizontal[:, 1:] != horizontal[:, :-1])
        )

        top_boundary = np.zeros((1, self.width), dtype=bool)
        bottom_boundary = np.ones((1, self.width), dtype=bool)
        vertical = np.vstack((top_boundary, filled, bottom_boundary))
        column_transitions = int(
            np.sum(vertical[1:] != vertical[:-1])
        )

        danger_rows = min(5, self.height)
        danger_zone = int(np.sum(filled[:danger_rows]))
        filled_cells = int(np.sum(filled))

        board_area = self.width * self.height
        max_bumpiness = max(1, (self.width - 1) * self.height)
        max_triangular = max(
            1,
            self.width * self.height * (self.height + 1) // 2,
        )
        max_row_transitions = max(1, self.height * (self.width + 1))
        max_column_transitions = max(
            1,
            self.width * (self.height + 1),
        )
        max_danger_cells = max(1, danger_rows * self.width)

        features = np.array(
            [
                aggregate_height / board_area,
                max_height / self.height,
                hole_count / board_area,
                bumpiness / max_bumpiness,
                hole_depth / max_triangular,
                blocks_above_holes / board_area,
                well_sum / max_triangular,
                row_transitions / max_row_transitions,
                column_transitions / max_column_transitions,
                danger_zone / max_danger_cells,
                filled_cells / board_area,
            ],
            dtype=np.float32,
        )
        return np.clip(features, 0.0, 1.0)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-9)
    return weights / np.sum(weights)


def probable_game_over_projection(
    current_features: np.ndarray,
    candidate_features: np.ndarray,
) -> bool:
    current_max_height = float(current_features[IDX_MAX_HEIGHT])
    candidate_is_empty = bool(
        np.all(np.abs(candidate_features) < 1e-9)
    )
    return current_max_height >= 0.55 and candidate_is_empty


def infer_cleared_lines(
    current_features: np.ndarray,
    candidate_features: np.ndarray,
    width: int,
    height: int,
) -> int:
    board_area = width * height
    current_cells = int(
        round(float(current_features[IDX_FILLED_CELLS]) * board_area)
    )
    candidate_cells = int(
        round(float(candidate_features[IDX_FILLED_CELLS]) * board_area)
    )
    removed_cells = current_cells + 4 - candidate_cells

    if removed_cells < width:
        return 0

    lines = int(round(removed_cells / width))
    if lines < 1 or lines > 4:
        return 0
    if abs(removed_cells - lines * width) > 1:
        return 0
    return lines


def score_candidate(
    current_features: np.ndarray,
    candidate_features: np.ndarray,
    weights: np.ndarray,
    width: int,
    height: int,
) -> float:
    if probable_game_over_projection(current_features, candidate_features):
        return float("inf")

    board_penalty = float(
        np.dot(weights[:N_BOARD_FEATURES], candidate_features)
    )
    cleared_lines = infer_cleared_lines(
        current_features,
        candidate_features,
        width,
        height,
    )
    line_reward = float(weights[-1]) * (cleared_lines / 4.0)
    return board_penalty - line_reward
