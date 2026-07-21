import json
import time

import gymnasium as gym
import numpy as np
import tetris_gymnasium

from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation


WEIGHTS_FILE = "best_weights.json"
GAME_SEED = 42
ANIMATION_DELAY = 0.05


def create_environment():
    base_env = gym.make(
        "tetris_gymnasium/Tetris",
        render_mode="human",
        gravity=False,
    )

    feature_wrapper = FeatureVectorObservation(base_env)

    env = GroupedActionsObservations(
        base_env,
        observation_wrappers=[feature_wrapper],
        terminate_on_illegal_action=True,
    )

    return env


def load_weights():
    try:
        with open(WEIGHTS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        weights = np.array(
            [
                data["aggregate_height"],
                data["max_height"],
                data["holes"],
                data["bumpiness"],
            ],
            dtype=float,
        )

        return weights

    except FileNotFoundError:
        raise FileNotFoundError(
            f"File '{WEIGHTS_FILE}' was not found. "
            "Place it in the same directory as main.py."
        )

    except KeyError as error:
        raise ValueError(
            f"Missing weight in JSON file: {error}"
        )


def evaluate_position(
    features,
    board_width,
    weights,
):
    column_heights = features[:board_width]

    aggregate_height = float(np.sum(column_heights))
    max_height = float(features[board_width])
    holes = float(features[board_width + 1])
    bumpiness = float(features[board_width + 2])

    board_features = np.array(
        [
            aggregate_height,
            max_height,
            holes,
            bumpiness,
        ],
        dtype=float,
    )

    return float(np.dot(weights, board_features))


def choose_action(
    observations,
    action_mask,
    board_width,
    weights,
):
    legal_actions = np.flatnonzero(action_mask)

    if len(legal_actions) == 0:
        raise RuntimeError("No legal actions available.")

    best_action = int(legal_actions[0])
    best_score = float("inf")

    for action in legal_actions:
        features = observations[action]

        score = evaluate_position(
            features,
            board_width,
            weights,
        )

        if score < best_score:
            best_score = score
            best_action = int(action)

    return best_action


def main():
    weights = load_weights()
    env = create_environment()

    observation, info = env.reset(seed=GAME_SEED)

    board_width = env.unwrapped.width

    terminated = False
    truncated = False

    placed_pieces = 0
    total_lines = 0
    total_reward = 0.0

    print("Loaded weights:")
    print(f"Aggregate height: {weights[0]:.6f}")
    print(f"Max height:       {weights[1]:.6f}")
    print(f"Holes:            {weights[2]:.6f}")
    print(f"Bumpiness:        {weights[3]:.6f}")
    print()

    try:
        while not terminated and not truncated:
            action_mask = np.asarray(
                info["action_mask"],
                dtype=bool,
            )

            action = choose_action(
                observation,
                action_mask,
                board_width,
                weights,
            )

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            placed_pieces += 1
            total_lines += int(info.get("lines_cleared", 0))
            total_reward += float(reward)

            env.render()
            time.sleep(ANIMATION_DELAY)

    finally:
        env.close()

    print()
    print("--- GAME FINISHED ---")
    print(f"Placed pieces: {placed_pieces}")
    print(f"Cleared lines: {total_lines}")
    print(f"Total reward:  {total_reward:.2f}")


if __name__ == "__main__":
    main()