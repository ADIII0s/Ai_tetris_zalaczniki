import csv
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tetris_gymnasium

from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation


NUMBER_OF_GAMES = 100
MAX_PIECES_PER_GAME = 1000

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_FILE = BASE_DIR / "best_weights.json"
CSV_FILE = BASE_DIR / "agent_comparison.csv"
LINES_CHART_FILE = BASE_DIR / "comparison_lines.png"
PIECES_CHART_FILE = BASE_DIR / "comparison_pieces.png"

OLD_WEIGHTS = np.array(
    [
        0.45,
        1.00,
        7.50,
        0.35,
    ],
    dtype=float,
)


def create_environment():
    base_env = gym.make(
        "tetris_gymnasium/Tetris",
        gravity=False,
    )

    feature_wrapper = FeatureVectorObservation(base_env)

    return GroupedActionsObservations(
        base_env,
        observation_wrappers=[feature_wrapper],
        terminate_on_illegal_action=True,
    )


def load_new_weights():
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {WEIGHTS_FILE}"
        )

    with open(WEIGHTS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return np.array(
        [
            data["aggregate_height"],
            data["max_height"],
            data["holes"],
            data["bumpiness"],
        ],
        dtype=float,
    )


def evaluate_position(
    features,
    board_width,
    weights,
):
    column_heights = features[:board_width]

    values = np.array(
        [
            np.sum(column_heights),
            features[board_width],
            features[board_width + 1],
            features[board_width + 2],
        ],
        dtype=float,
    )

    return float(np.dot(weights, values))


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
        score = evaluate_position(
            observations[action],
            board_width,
            weights,
        )

        if score < best_score:
            best_score = score
            best_action = int(action)

    return best_action


def play_game(
    env,
    seed,
    weights,
):
    observation, info = env.reset(seed=seed)

    board_width = env.unwrapped.width

    terminated = False
    truncated = False

    pieces = 0
    lines = 0
    reward_sum = 0.0

    while (
        not terminated
        and not truncated
        and pieces < MAX_PIECES_PER_GAME
    ):
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

        pieces += 1
        lines += int(info.get("lines_cleared", 0))
        reward_sum += float(reward)

    return {
        "pieces": pieces,
        "lines": lines,
        "reward": reward_sum,
    }


def run_agent(
    agent_name,
    weights,
):
    env = create_environment()
    results = []

    try:
        for seed in range(NUMBER_OF_GAMES):
            game_result = play_game(
                env,
                seed,
                weights,
            )

            game_result["agent"] = agent_name
            game_result["game"] = seed + 1
            game_result["seed"] = seed

            results.append(game_result)

            if (seed + 1) % 10 == 0:
                print(
                    f"{agent_name}: "
                    f"{seed + 1}/{NUMBER_OF_GAMES} games"
                )

    finally:
        env.close()

    return results


def print_summary(
    agent_name,
    results,
):
    lines = np.array(
        [result["lines"] for result in results],
        dtype=float,
    )

    pieces = np.array(
        [result["pieces"] for result in results],
        dtype=float,
    )

    rewards = np.array(
        [result["reward"] for result in results],
        dtype=float,
    )

    print()
    print(f"--- {agent_name.upper()} ---")
    print(f"Mean lines:     {np.mean(lines):.2f}")
    print(f"Median lines:   {np.median(lines):.2f}")
    print(f"Std lines:      {np.std(lines):.2f}")
    print(f"Best lines:     {np.max(lines):.0f}")
    print(f"Worst lines:    {np.min(lines):.0f}")
    print(f"Mean pieces:    {np.mean(pieces):.2f}")
    print(f"Mean reward:    {np.mean(rewards):.2f}")


def print_comparison(
    old_results,
    new_results,
):
    old_lines = np.array(
        [result["lines"] for result in old_results],
        dtype=float,
    )

    new_lines = np.array(
        [result["lines"] for result in new_results],
        dtype=float,
    )

    difference = new_lines - old_lines

    wins = int(np.sum(difference > 0))
    ties = int(np.sum(difference == 0))
    losses = int(np.sum(difference < 0))

    old_mean = float(np.mean(old_lines))
    new_mean = float(np.mean(new_lines))

    improvement = (
        ((new_mean - old_mean) / old_mean) * 100.0
        if old_mean != 0
        else float("inf")
    )

    print()
    print("--- DIRECT COMPARISON ---")
    print(f"Old mean lines:       {old_mean:.2f}")
    print(f"New mean lines:       {new_mean:.2f}")
    print(f"Mean difference:      {np.mean(difference):.2f}")
    print(f"Improvement:          {improvement:.2f}%")
    print(f"New agent wins:       {wins}")
    print(f"Ties:                 {ties}")
    print(f"New agent losses:     {losses}")


def save_csv(
    old_results,
    new_results,
):
    all_results = old_results + new_results

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "agent",
                "game",
                "seed",
                "pieces",
                "lines",
                "reward",
            ],
        )

        writer.writeheader()
        writer.writerows(all_results)

    print(f"CSV saved: {CSV_FILE.name}")


def create_lines_chart(
    old_results,
    new_results,
):
    old_lines = np.array(
        [result["lines"] for result in old_results],
        dtype=float,
    )

    new_lines = np.array(
        [result["lines"] for result in new_results],
        dtype=float,
    )

    means = [
        np.mean(old_lines),
        np.mean(new_lines),
    ]

    deviations = [
        np.std(old_lines),
        np.std(new_lines),
    ]

    figure, axis = plt.subplots(figsize=(7, 5))

    axis.bar(
        ["Old weights", "Genetic weights"],
        means,
        yerr=deviations,
        capsize=8,
    )

    axis.set_title("Mean cleared lines")
    axis.set_ylabel("Cleared lines")
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        LINES_CHART_FILE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Chart saved: {LINES_CHART_FILE.name}")


def create_pieces_chart(
    old_results,
    new_results,
):
    old_pieces = np.array(
        [result["pieces"] for result in old_results],
        dtype=float,
    )

    new_pieces = np.array(
        [result["pieces"] for result in new_results],
        dtype=float,
    )

    means = [
        np.mean(old_pieces),
        np.mean(new_pieces),
    ]

    deviations = [
        np.std(old_pieces),
        np.std(new_pieces),
    ]

    figure, axis = plt.subplots(figsize=(7, 5))

    axis.bar(
        ["Old weights", "Genetic weights"],
        means,
        yerr=deviations,
        capsize=8,
    )

    axis.set_title("Mean number of placed pieces")
    axis.set_ylabel("Placed pieces")
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        PIECES_CHART_FILE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Chart saved: {PIECES_CHART_FILE.name}")


def main():
    new_weights = load_new_weights()

    print("Testing old weights...")
    old_results = run_agent(
        "old_weights",
        OLD_WEIGHTS,
    )

    print()
    print("Testing genetic weights...")
    new_results = run_agent(
        "genetic_weights",
        new_weights,
    )

    print_summary(
        "old weights",
        old_results,
    )

    print_summary(
        "genetic weights",
        new_results,
    )

    print_comparison(
        old_results,
        new_results,
    )

    save_csv(
        old_results,
        new_results,
    )

    create_lines_chart(
        old_results,
        new_results,
    )

    create_pieces_chart(
        old_results,
        new_results,
    )


if __name__ == "__main__":
    main()
