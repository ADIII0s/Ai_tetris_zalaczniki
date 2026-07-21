from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import tetris_gymnasium

from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations

from tetris_features_v3 import (
    BOARD_FEATURE_NAMES,
    DECISION_WEIGHT_NAMES,
    ExtendedFeatureObservation,
    N_BOARD_FEATURES,
    N_WEIGHTS,
    normalize_weights,
    score_candidate,
)


BASE_DIR = Path(__file__).resolve().parent

BEST_FILE = BASE_DIR / "best_weights_v3.json"
CHECKPOINT_FILE = BASE_DIR / "checkpoint_v3.pkl"
HISTORY_FILE = BASE_DIR / "training_history_v3.csv"
REPORT_FILE = BASE_DIR / "final_report_v3.txt"

POPULATION_SIZE = 40
GENERATIONS = 36

TRAINING_GAMES = 16
VALIDATION_GAMES = 60
FINAL_TEST_GAMES = 250

TOP_VALIDATION_CANDIDATES = 3
ELITE_COUNT = 6
PARENT_POOL_SIZE = 18
MAX_PIECES_PER_GAME = 1500

MUTATION_START = 0.18
MUTATION_END = 0.04
RANDOM_IMMIGRANT_RATE = 0.05
EARLY_STOPPING_PATIENCE = 100

MASTER_SEED = 20260619

rng = np.random.default_rng(MASTER_SEED)


def create_environment():
    base_env = gym.make(
        "tetris_gymnasium/Tetris",
        gravity=False,
    )
    feature_wrapper = ExtendedFeatureObservation(base_env)
    return GroupedActionsObservations(
        base_env,
        observation_wrappers=[feature_wrapper],
        terminate_on_illegal_action=True,
    )


def choose_action(
    observations,
    action_mask,
    current_features,
    weights,
    width,
    height,
):
    legal_actions = np.flatnonzero(action_mask)

    if len(legal_actions) == 0:
        raise RuntimeError("No legal actions available.")

    best_action = int(legal_actions[0])
    best_score = float("inf")

    for action in legal_actions:
        candidate = np.asarray(
            observations[action],
            dtype=float,
        )
        score = score_candidate(
            current_features,
            candidate,
            weights,
            width,
            height,
        )
        if score < best_score:
            best_score = score
            best_action = int(action)

    return best_action


def play_game(env, seed, weights):
    observations, info = env.reset(seed=int(seed))

    width = int(env.unwrapped.width)
    height = int(env.unwrapped.height)

    current_features = np.asarray(
        info["board"],
        dtype=float,
    )

    terminated = False
    truncated = False
    pieces = 0
    lines = 0

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
            observations,
            action_mask,
            current_features,
            weights,
            width,
            height,
        )

        (
            observations,
            _,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        pieces += 1
        lines += int(info.get("lines_cleared", 0))

        if "board" in info:
            current_features = np.asarray(
                info["board"],
                dtype=float,
            )

    return lines, pieces


def summarize(lines, pieces):
    lines = np.asarray(lines, dtype=float)
    pieces = np.asarray(pieces, dtype=float)

    mean_lines = float(np.mean(lines))
    median_lines = float(np.median(lines))
    std_lines = float(np.std(lines))
    mean_pieces = float(np.mean(pieces))
    zero_rate = float(np.mean(lines == 0))

    fitness = (
        mean_lines
        + 0.20 * median_lines
        + mean_pieces / 200.0
        - 0.06 * std_lines
        - 2.0 * zero_rate
    )

    return {
        "fitness": float(fitness),
        "mean_lines": mean_lines,
        "median_lines": median_lines,
        "std_lines": std_lines,
        "mean_pieces": mean_pieces,
        "best_lines": int(np.max(lines)),
        "worst_lines": int(np.min(lines)),
        "zero_rate": zero_rate,
    }


def evaluate_candidate(env, weights, seeds):
    lines_results = []
    pieces_results = []

    for seed in seeds:
        lines, pieces = play_game(
            env,
            int(seed),
            weights,
        )
        lines_results.append(lines)
        pieces_results.append(pieces)

    return summarize(lines_results, pieces_results)


def initial_templates():
    balanced = [
        0.13, 0.06, 0.13, 0.07,
        0.10, 0.10, 0.06, 0.07,
        0.06, 0.04, 0.02, 0.16,
    ]
    hole_focused = [
        0.09, 0.05, 0.16, 0.05,
        0.16, 0.15, 0.05, 0.06,
        0.05, 0.04, 0.02, 0.12,
    ]
    line_focused = [
        0.11, 0.05, 0.11, 0.07,
        0.08, 0.08, 0.06, 0.06,
        0.05, 0.04, 0.02, 0.27,
    ]
    survival_focused = [
        0.16, 0.10, 0.12, 0.08,
        0.09, 0.09, 0.05, 0.06,
        0.05, 0.08, 0.02, 0.10,
    ]

    return [
        normalize_weights(np.array(values, dtype=float))
        for values in [
            balanced,
            hole_focused,
            line_focused,
            survival_focused,
        ]
    ]


def create_initial_population():
    population = initial_templates()

    while len(population) < POPULATION_SIZE:
        population.append(
            rng.dirichlet(
                np.full(N_WEIGHTS, 1.4)
            )
        )

    return np.asarray(population, dtype=float)


def mutation_scale(generation):
    if GENERATIONS <= 1:
        return MUTATION_END

    progress = generation / (GENERATIONS - 1)

    return float(
        MUTATION_START
        + progress * (MUTATION_END - MUTATION_START)
    )


def create_child(parent_a, parent_b, scale):
    alpha = rng.random(N_WEIGHTS)
    child = (
        alpha * parent_a
        + (1.0 - alpha) * parent_b
    )
    child *= np.exp(
        rng.normal(
            0.0,
            scale,
            size=N_WEIGHTS,
        )
    )
    return normalize_weights(child)


def create_next_population(
    population,
    scores,
    global_best,
    scale,
):
    order = np.argsort(scores)[::-1]
    sorted_population = population[order]

    next_population = []

    if global_best is not None:
        next_population.append(global_best.copy())

    for elite in sorted_population[:ELITE_COUNT]:
        if len(next_population) >= POPULATION_SIZE:
            break
        next_population.append(elite.copy())

    parent_pool = sorted_population[:PARENT_POOL_SIZE]

    while len(next_population) < POPULATION_SIZE:
        if rng.random() < RANDOM_IMMIGRANT_RATE:
            next_population.append(
                rng.dirichlet(
                    np.full(N_WEIGHTS, 1.4)
                )
            )
            continue

        indexes = rng.choice(
            len(parent_pool),
            size=2,
            replace=False,
        )

        child = create_child(
            parent_pool[indexes[0]],
            parent_pool[indexes[1]],
            scale,
        )
        next_population.append(child)

    return np.asarray(next_population, dtype=float)


def save_best(weights, generation, validation):
    data = {
        "version": 3,
        "generation": int(generation),
        "board_feature_names": BOARD_FEATURE_NAMES,
        "decision_weight_names": DECISION_WEIGHT_NAMES,
        "weights": {
            name: float(value)
            for name, value in zip(
                DECISION_WEIGHT_NAMES,
                weights,
            )
        },
        "validation": {
            key: float(value)
            for key, value in validation.items()
        },
    }

    with open(BEST_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_history(history):
    fieldnames = [
        "generation",
        "mutation_scale",
        "training_fitness",
        "training_mean_lines",
        "training_std_lines",
        "validation_fitness",
        "validation_mean_lines",
        "validation_median_lines",
        "validation_std_lines",
        "validation_mean_pieces",
        "validation_zero_rate",
    ]

    with open(
        HISTORY_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    next_generation,
    population,
    global_best,
    global_best_generation,
    global_best_validation,
    best_validation_fitness,
    no_improvement_count,
    history,
):
    checkpoint = {
        "weight_names": DECISION_WEIGHT_NAMES,
        "next_generation": int(next_generation),
        "population": population,
        "global_best": global_best,
        "global_best_generation": int(global_best_generation),
        "global_best_validation": global_best_validation,
        "best_validation_fitness": float(best_validation_fitness),
        "no_improvement_count": int(no_improvement_count),
        "history": history,
        "rng_state": rng.bit_generator.state,
    }

    with open(CHECKPOINT_FILE, "wb") as file:
        pickle.dump(checkpoint, file)


def load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        return None

    with open(CHECKPOINT_FILE, "rb") as file:
        checkpoint = pickle.load(file)

    if checkpoint.get("weight_names") != DECISION_WEIGHT_NAMES:
        raise RuntimeError(
            "checkpoint_v3.pkl does not match this V3 code. "
            "Delete it to start a fresh training run."
        )

    rng.bit_generator.state = checkpoint["rng_state"]
    return checkpoint


def smoke_test():
    env = create_environment()

    try:
        observations, info = env.reset(seed=123)

        expected_shape = (
            env.unwrapped.width * 4,
            N_BOARD_FEATURES,
        )

        if observations.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected observation shape: {observations.shape}; "
                f"expected: {expected_shape}"
            )

        current_features = np.asarray(
            info["board"],
            dtype=float,
        )
        action_mask = np.asarray(
            info["action_mask"],
            dtype=bool,
        )

        action = choose_action(
            observations,
            action_mask,
            current_features,
            initial_templates()[0],
            int(env.unwrapped.width),
            int(env.unwrapped.height),
        )

        env.step(action)

    finally:
        env.close()

    print("V3 smoke test: OK")
    print(f"Observation shape: {observations.shape}")
    print(f"Board features: {N_BOARD_FEATURES}")
    print(f"Decision weights: {N_WEIGHTS}")


def print_weights(weights):
    for name, value in zip(
        DECISION_WEIGHT_NAMES,
        weights,
    ):
        print(f"  {name:<24} {value:.6f}")


def stats_text(title, stats):
    return (
        f"--- {title} ---\n"
        f"Fitness:          {stats['fitness']:.4f}\n"
        f"Mean lines:       {stats['mean_lines']:.4f}\n"
        f"Median lines:     {stats['median_lines']:.4f}\n"
        f"Std lines:        {stats['std_lines']:.4f}\n"
        f"Mean pieces:      {stats['mean_pieces']:.4f}\n"
        f"Best lines:       {stats['best_lines']}\n"
        f"Worst lines:      {stats['worst_lines']}\n"
        f"Zero result rate: {stats['zero_rate']:.4f}\n"
    )


def main():
    smoke_test()

    validation_seeds = np.arange(
        20_000_000,
        20_000_000 + VALIDATION_GAMES,
        dtype=int,
    )
    final_test_seeds = np.arange(
        30_000_000,
        30_000_000 + FINAL_TEST_GAMES,
        dtype=int,
    )

    checkpoint = load_checkpoint()

    if checkpoint is None:
        start_generation = 0
        population = create_initial_population()
        global_best = None
        global_best_generation = -1
        global_best_validation = None
        best_validation_fitness = float("-inf")
        no_improvement_count = 0
        history = []
        print("Starting a new V3 training run.")
    else:
        start_generation = int(checkpoint["next_generation"])
        population = np.asarray(
            checkpoint["population"],
            dtype=float,
        )
        global_best = checkpoint["global_best"]
        global_best_generation = int(
            checkpoint["global_best_generation"]
        )
        global_best_validation = checkpoint[
            "global_best_validation"
        ]
        best_validation_fitness = float(
            checkpoint["best_validation_fitness"]
        )
        no_improvement_count = int(
            checkpoint["no_improvement_count"]
        )
        history = list(checkpoint["history"])
        print(
            f"Resuming from generation {start_generation + 1}."
        )

    env = create_environment()
    stopped_early = False

    try:
        for generation in range(
            start_generation,
            GENERATIONS,
        ):
            generation_number = generation + 1
            scale = mutation_scale(generation)

            training_seeds = rng.integers(
                1_000_000,
                9_999_999,
                size=TRAINING_GAMES,
            )

            training_stats = []
            training_scores = np.empty(
                POPULATION_SIZE,
                dtype=float,
            )

            print()
            print(
                f"Generation {generation_number}/{GENERATIONS}"
            )

            for index, candidate in enumerate(population):
                stats = evaluate_candidate(
                    env,
                    candidate,
                    training_seeds,
                )
                training_stats.append(stats)
                training_scores[index] = stats["fitness"]

                print(
                    f"  Candidate {index + 1:02d}/"
                    f"{POPULATION_SIZE}: "
                    f"fitness={stats['fitness']:.3f}, "
                    f"lines={stats['mean_lines']:.2f}"
                )

            ranking = np.argsort(training_scores)[::-1]
            validation_indexes = ranking[
                :TOP_VALIDATION_CANDIDATES
            ]

            best_index = None
            best_validation = None

            for candidate_index in validation_indexes:
                candidate_validation = evaluate_candidate(
                    env,
                    population[candidate_index],
                    validation_seeds,
                )

                if (
                    best_validation is None
                    or candidate_validation["fitness"]
                    > best_validation["fitness"]
                ):
                    best_index = int(candidate_index)
                    best_validation = candidate_validation

            generation_weights = population[best_index].copy()
            generation_training = training_stats[best_index]

            improved = (
                best_validation["fitness"]
                > best_validation_fitness + 1e-9
            )

            if improved:
                best_validation_fitness = best_validation["fitness"]
                global_best = generation_weights.copy()
                global_best_generation = generation_number
                global_best_validation = dict(best_validation)
                no_improvement_count = 0

                save_best(
                    global_best,
                    global_best_generation,
                    global_best_validation,
                )
            else:
                no_improvement_count += 1

            history.append(
                {
                    "generation": generation_number,
                    "mutation_scale": scale,
                    "training_fitness": generation_training[
                        "fitness"
                    ],
                    "training_mean_lines": generation_training[
                        "mean_lines"
                    ],
                    "training_std_lines": generation_training[
                        "std_lines"
                    ],
                    "validation_fitness": best_validation[
                        "fitness"
                    ],
                    "validation_mean_lines": best_validation[
                        "mean_lines"
                    ],
                    "validation_median_lines": best_validation[
                        "median_lines"
                    ],
                    "validation_std_lines": best_validation[
                        "std_lines"
                    ],
                    "validation_mean_pieces": best_validation[
                        "mean_pieces"
                    ],
                    "validation_zero_rate": best_validation[
                        "zero_rate"
                    ],
                }
            )

            save_history(history)

            population = create_next_population(
                population,
                training_scores,
                global_best,
                scale,
            )

            save_checkpoint(
                generation + 1,
                population,
                global_best,
                global_best_generation,
                global_best_validation,
                best_validation_fitness,
                no_improvement_count,
                history,
            )

            print()
            print(
                f"Validation: fitness="
                f"{best_validation['fitness']:.3f}, "
                f"mean lines="
                f"{best_validation['mean_lines']:.2f}, "
                f"std="
                f"{best_validation['std_lines']:.2f}"
            )
            print(f"New global best: {improved}")
            print(
                f"No improvement: "
                f"{no_improvement_count}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

            if (
                no_improvement_count
                >= EARLY_STOPPING_PATIENCE
            ):
                stopped_early = True
                print("Early stopping activated.")
                break

    except KeyboardInterrupt:
        print()
        print(
            "Training interrupted. Run the script again "
            "to resume from checkpoint_v3.pkl."
        )
        return

    finally:
        env.close()

    if global_best is None:
        raise RuntimeError("No valid V3 candidate was found.")

    print()
    print("Training finished.")
    print(f"Stopped early: {stopped_early}")
    print(
        f"Best validation generation: "
        f"{global_best_generation}"
    )
    print("Best weights:")
    print_weights(global_best)

    final_env = create_environment()

    try:
        final_stats = evaluate_candidate(
            final_env,
            global_best,
            final_test_seeds,
        )
    finally:
        final_env.close()

    report = (
        "--- TETRIS AI V3 FINAL REPORT ---\n"
        f"Best generation: {global_best_generation}\n"
        f"Board features: {N_BOARD_FEATURES}\n"
        f"Decision weights: {N_WEIGHTS}\n"
        f"Stopped early: {stopped_early}\n\n"
        "Weights:\n"
    )

    for name, value in zip(
        DECISION_WEIGHT_NAMES,
        global_best,
    ):
        report += f"{name:<24} {value:.10f}\n"

    report += "\n"
    report += stats_text(
        "VALIDATION",
        global_best_validation,
    )
    report += "\n"
    report += stats_text(
        "FINAL TEST",
        final_stats,
    )

    print()
    print(report)

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write(report)

    save_best(
        global_best,
        global_best_generation,
        global_best_validation,
    )

    print(f"Saved: {BEST_FILE.name}")
    print(f"Saved: {HISTORY_FILE.name}")
    print(f"Saved: {REPORT_FILE.name}")
    print(f"Checkpoint: {CHECKPOINT_FILE.name}")


if __name__ == "__main__":
    main()
