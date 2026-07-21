import csv

import gymnasium as gym
import numpy as np

from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation


NUMBER_OF_GAMES = 100
MAX_PIECES_PER_GAME = 5000


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


def evaluate_position(features: np.ndarray, board_width: int) -> float:
    column_heights = features[:board_width]

    max_height = float(features[board_width])
    holes = float(features[board_width + 1])
    bumpiness = float(features[board_width + 2])

    aggregate_height = float(np.sum(column_heights))

    # Mniejszy wynik oznacza lepszą planszę.
    return (
        0.45 * aggregate_height
        + 1.00 * max_height
        + 7.50 * holes
        + 0.35 * bumpiness
    )


def choose_heuristic_action(
    observations: np.ndarray,
    action_mask: np.ndarray,
    board_width: int,
) -> int:
    legal_actions = np.flatnonzero(action_mask)

    if len(legal_actions) == 0:
        raise RuntimeError("Brak legalnych ruchow.")

    best_action = int(legal_actions[0])
    best_score = float("inf")

    for action in legal_actions:
        score = evaluate_position(
            observations[action],
            board_width,
        )

        if score < best_score:
            best_score = score
            best_action = int(action)

    return best_action


def choose_random_action(
    action_mask: np.ndarray,
    rng: np.random.Generator,
) -> int:
    legal_actions = np.flatnonzero(action_mask)

    if len(legal_actions) == 0:
        raise RuntimeError("Brak legalnych ruchow.")

    return int(rng.choice(legal_actions))


def play_game(env, seed: int, agent_name: str) -> dict:
    observation, info = env.reset(seed=seed)

    rng = np.random.default_rng(seed + 10_000)
    board_width = env.unwrapped.width

    terminated = False
    truncated = False

    pieces = 0
    lines = 0
    total_reward = 0.0

    while (
        not terminated
        and not truncated
        and pieces < MAX_PIECES_PER_GAME
    ):
        action_mask = np.asarray(
            info["action_mask"],
            dtype=bool,
        )

        if agent_name == "heuristic":
            action = choose_heuristic_action(
                observation,
                action_mask,
                board_width,
            )
        elif agent_name == "random":
            action = choose_random_action(
                action_mask,
                rng,
            )
        else:
            raise ValueError(f"Nieznany agent: {agent_name}")

        observation, reward, terminated, truncated, info = env.step(action)

        pieces += 1
        lines += int(info.get("lines_cleared", 0))
        total_reward += float(reward)

    return {
        "agent": agent_name,
        "game": seed + 1,
        "seed": seed,
        "pieces": pieces,
        "lines": lines,
        "reward": total_reward,
    }


def run_benchmark(agent_name: str) -> list[dict]:
    env = create_environment()
    results = []

    try:
        for seed in range(NUMBER_OF_GAMES):
            result = play_game(env, seed, agent_name)
            results.append(result)

            if (seed + 1) % 10 == 0:
                print(
                    f"{agent_name}: "
                    f"ukonczono {seed + 1}/{NUMBER_OF_GAMES} gier"
                )
    finally:
        env.close()

    return results


def print_summary(agent_name: str, results: list[dict]):
    pieces = np.array([result["pieces"] for result in results])
    lines = np.array([result["lines"] for result in results])
    rewards = np.array([result["reward"] for result in results])

    print(f"\n--- AGENT: {agent_name.upper()} ---")
    print(f"Srednia liczba klockow: {np.mean(pieces):.2f}")
    print(f"Mediana klockow:        {np.median(pieces):.2f}")
    print(f"Srednia liczba linii:   {np.mean(lines):.2f}")
    print(f"Mediana linii:          {np.median(lines):.2f}")
    print(f"Najlepszy wynik linii:  {np.max(lines)}")
    print(f"Najgorszy wynik linii:  {np.min(lines)}")
    print(f"Srednia nagroda:        {np.mean(rewards):.2f}")
    print(f"Odchylenie linii:       {np.std(lines):.2f}")


def save_results(results: list[dict]):
    filename = "benchmark_results.csv"

    with open(filename, "w", newline="", encoding="utf-8") as file:
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
        writer.writerows(results)

    print(f"\nWyniki zapisano w: {filename}")


def main():
    print("Test agenta losowego...")
    random_results = run_benchmark("random")

    print("\nTest agenta heurystycznego...")
    heuristic_results = run_benchmark("heuristic")

    print_summary("random", random_results)
    print_summary("heuristic", heuristic_results)

    save_results(random_results + heuristic_results)


if __name__ == "__main__":
    main()