import csv
import json
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import tetris_gymnasium

from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation


# ---------- CONFIG ----------

POPULATION_SIZE = 32
GENERATIONS = 25
GAMES_PER_CANDIDATE = 16
VALIDATION_GAMES = 60
TEST_GAMES = 200

ELITE_COUNT = 4
PARENT_POOL_SIZE = 12
START_MUTATION_SCALE = 0.35
END_MUTATION_SCALE = 0.10
MAX_PIECES_PER_GAME = 1500
RANDOM_SEED = 20260618

TRAIN_SEED_LIMIT = 900_000
VALIDATION_SEED_START = 1_000_000
TEST_SEED_START = 2_000_000

BASE_DIR = Path(__file__).resolve().parent
V1_WEIGHTS_FILE = BASE_DIR / "best_weights.json"
V2_WEIGHTS_FILE = BASE_DIR / "best_weights_v2.json"
CHECKPOINT_FILE = BASE_DIR / "checkpoint_v2.pkl"
HISTORY_FILE = BASE_DIR / "training_history_v2.csv"
REPORT_FILE = BASE_DIR / "final_report_v2.txt"

rng = np.random.default_rng(RANDOM_SEED)


# ---------- ENVIRONMENT ----------

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


# ---------- WEIGHTS ----------

def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.000001)
    total = float(np.sum(weights))

    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Invalid weights.")

    return weights / total


def load_v1_weights():
    fallback = np.array([0.45, 1.00, 7.50, 0.35], dtype=float)

    if not V1_WEIGHTS_FILE.exists():
        print("V1 weights not found. Using fallback weights.")
        return normalize_weights(fallback)

    with open(V1_WEIGHTS_FILE, "r", encoding="utf-8") as file:
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

    print(f"Loaded V1 weights from: {V1_WEIGHTS_FILE.name}")
    return normalize_weights(weights)


def save_v2_weights(weights, generation, metrics):
    data = {
        "aggregate_height": float(weights[0]),
        "max_height": float(weights[1]),
        "holes": float(weights[2]),
        "bumpiness": float(weights[3]),
        "generation": int(generation),
        "validation_fitness": float(metrics["fitness"]),
        "validation_mean_lines": float(metrics["mean_lines"]),
        "validation_median_lines": float(metrics["median_lines"]),
        "validation_std_lines": float(metrics["std_lines"]),
        "validation_mean_pieces": float(metrics["mean_pieces"]),
    }

    with open(V2_WEIGHTS_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def print_weights(weights):
    print(
        "Weights: "
        f"height={weights[0]:.6f}, "
        f"max_height={weights[1]:.6f}, "
        f"holes={weights[2]:.6f}, "
        f"bumpiness={weights[3]:.6f}"
    )


# ---------- AGENT ----------

def evaluate_position(features, board_width, weights):
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


def choose_action(observations, action_mask, board_width, weights):
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


def play_game(env, seed, weights):
    observation, info = env.reset(seed=int(seed))
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
        action_mask = np.asarray(info["action_mask"], dtype=bool)

        action = choose_action(
            observation,
            action_mask,
            board_width,
            weights,
        )

        observation, reward, terminated, truncated, info = env.step(action)

        pieces += 1
        lines += int(info.get("lines_cleared", 0))
        reward_sum += float(reward)

    return lines, pieces, reward_sum


# ---------- METRICS ----------

def calculate_fitness(mean_lines, median_lines, std_lines, mean_pieces, zero_rate):
    return float(
        mean_lines
        + 0.25 * median_lines
        - 0.08 * std_lines
        + 0.002 * mean_pieces
        - 2.00 * zero_rate
    )


def evaluate_candidate(env, weights, seeds):
    line_results = []
    piece_results = []
    reward_results = []

    for seed in seeds:
        lines, pieces, reward = play_game(env, seed, weights)
        line_results.append(lines)
        piece_results.append(pieces)
        reward_results.append(reward)

    lines_array = np.asarray(line_results, dtype=float)
    pieces_array = np.asarray(piece_results, dtype=float)
    rewards_array = np.asarray(reward_results, dtype=float)

    mean_lines = float(np.mean(lines_array))
    median_lines = float(np.median(lines_array))
    std_lines = float(np.std(lines_array))
    mean_pieces = float(np.mean(pieces_array))
    zero_rate = float(np.mean(lines_array == 0))

    return {
        "fitness": calculate_fitness(
            mean_lines,
            median_lines,
            std_lines,
            mean_pieces,
            zero_rate,
        ),
        "mean_lines": mean_lines,
        "median_lines": median_lines,
        "std_lines": std_lines,
        "mean_pieces": mean_pieces,
        "mean_reward": float(np.mean(rewards_array)),
        "zero_rate": zero_rate,
        "best_lines": int(np.max(lines_array)),
        "worst_lines": int(np.min(lines_array)),
    }


# ---------- GENETIC ALGORITHM ----------

def create_initial_population(base_weights):
    population = [normalize_weights(base_weights)]
    mutated_target = int(POPULATION_SIZE * 0.75)

    while len(population) < mutated_target:
        mutation = np.exp(rng.normal(0.0, 0.65, size=4))
        population.append(normalize_weights(base_weights * mutation))

    while len(population) < POPULATION_SIZE:
        population.append(rng.dirichlet(np.ones(4)))

    return np.asarray(population, dtype=float)


def mutation_scale_for_generation(generation):
    if GENERATIONS <= 1:
        return END_MUTATION_SCALE

    progress = generation / (GENERATIONS - 1)

    return float(
        START_MUTATION_SCALE
        + progress * (END_MUTATION_SCALE - START_MUTATION_SCALE)
    )


def select_parent(population, scores):
    tournament_size = min(4, len(population))
    indexes = rng.choice(
        len(population),
        size=tournament_size,
        replace=False,
    )

    return population[indexes[np.argmax(scores[indexes])]]


def create_child(parent_a, parent_b, mutation_scale):
    alpha = rng.random(4)
    child = alpha * parent_a + (1.0 - alpha) * parent_b
    child *= np.exp(rng.normal(0.0, mutation_scale, size=4))
    return normalize_weights(child)


def create_next_generation(population, scores, mutation_scale):
    order = np.argsort(scores)[::-1]
    elites = population[order[:ELITE_COUNT]]
    parent_pool = population[order[:PARENT_POOL_SIZE]]
    parent_scores = scores[order[:PARENT_POOL_SIZE]]

    next_population = [elite.copy() for elite in elites]

    while len(next_population) < POPULATION_SIZE:
        parent_a = select_parent(parent_pool, parent_scores)
        parent_b = select_parent(parent_pool, parent_scores)
        next_population.append(
            create_child(parent_a, parent_b, mutation_scale)
        )

    return np.asarray(next_population, dtype=float)


# ---------- CHECKPOINT ----------

def save_checkpoint(
    next_generation,
    population,
    best_weights,
    best_validation_fitness,
    best_generation,
    best_validation_metrics,
):
    checkpoint = {
        "next_generation": int(next_generation),
        "population": population,
        "best_weights": best_weights,
        "best_validation_fitness": float(best_validation_fitness),
        "best_generation": int(best_generation),
        "best_validation_metrics": best_validation_metrics,
        "rng_state": rng.bit_generator.state,
    }

    with open(CHECKPOINT_FILE, "wb") as file:
        pickle.dump(checkpoint, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        return None

    with open(CHECKPOINT_FILE, "rb") as file:
        checkpoint = pickle.load(file)

    rng.bit_generator.state = checkpoint["rng_state"]
    return checkpoint


# ---------- HISTORY ----------

def append_history(row):
    fieldnames = [
        "generation",
        "mutation_scale",
        "train_fitness",
        "train_mean_lines",
        "train_median_lines",
        "train_std_lines",
        "train_mean_pieces",
        "train_zero_rate",
        "validation_fitness",
        "validation_mean_lines",
        "validation_median_lines",
        "validation_std_lines",
        "validation_mean_pieces",
        "validation_zero_rate",
        "weight_aggregate_height",
        "weight_max_height",
        "weight_holes",
        "weight_bumpiness",
    ]

    file_exists = HISTORY_FILE.exists()

    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ---------- REPORT ----------

def save_final_report(
    best_weights,
    best_generation,
    validation_metrics,
    test_metrics,
):
    report = (
        "--- TETRIS AI V2 FINAL REPORT ---\n"
        f"Best generation:         {best_generation}\n"
        f"Aggregate height weight: {best_weights[0]:.10f}\n"
        f"Max height weight:       {best_weights[1]:.10f}\n"
        f"Holes weight:            {best_weights[2]:.10f}\n"
        f"Bumpiness weight:        {best_weights[3]:.10f}\n"
        "\n"
        "--- VALIDATION ---\n"
        f"Games:                   {VALIDATION_GAMES}\n"
        f"Fitness:                 {validation_metrics['fitness']:.4f}\n"
        f"Mean lines:              {validation_metrics['mean_lines']:.4f}\n"
        f"Median lines:            {validation_metrics['median_lines']:.4f}\n"
        f"Std lines:               {validation_metrics['std_lines']:.4f}\n"
        f"Mean pieces:             {validation_metrics['mean_pieces']:.4f}\n"
        f"Zero result rate:        {validation_metrics['zero_rate']:.4f}\n"
        "\n"
        "--- FINAL TEST ---\n"
        f"Games:                   {TEST_GAMES}\n"
        f"Fitness:                 {test_metrics['fitness']:.4f}\n"
        f"Mean lines:              {test_metrics['mean_lines']:.4f}\n"
        f"Median lines:            {test_metrics['median_lines']:.4f}\n"
        f"Std lines:               {test_metrics['std_lines']:.4f}\n"
        f"Mean pieces:             {test_metrics['mean_pieces']:.4f}\n"
        f"Best lines:              {test_metrics['best_lines']}\n"
        f"Worst lines:             {test_metrics['worst_lines']}\n"
        f"Zero result rate:        {test_metrics['zero_rate']:.4f}\n"
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write(report)

    print()
    print(report)


# ---------- MAIN ----------

def main():
    checkpoint = load_checkpoint()

    if checkpoint is None:
        population = create_initial_population(load_v1_weights())
        start_generation = 0
        best_weights = None
        best_validation_fitness = float("-inf")
        best_generation = -1
        best_validation_metrics = None
        print("Starting a new V2 training.")
    else:
        population = checkpoint["population"]
        start_generation = checkpoint["next_generation"]
        best_weights = checkpoint["best_weights"]
        best_validation_fitness = checkpoint["best_validation_fitness"]
        best_generation = checkpoint["best_generation"]
        best_validation_metrics = checkpoint["best_validation_metrics"]
        print(
            "Resuming V2 training from generation "
            f"{start_generation + 1}."
        )

    validation_seeds = list(
        range(
            VALIDATION_SEED_START,
            VALIDATION_SEED_START + VALIDATION_GAMES,
        )
    )

    test_seeds = list(
        range(TEST_SEED_START, TEST_SEED_START + TEST_GAMES)
    )

    env = create_environment()

    try:
        for generation in range(start_generation, GENERATIONS):
            generation_number = generation + 1
            mutation_scale = mutation_scale_for_generation(generation)

            # Save before drawing seeds, so an interrupted generation can repeat exactly.
            save_checkpoint(
                generation,
                population,
                best_weights,
                best_validation_fitness,
                best_generation,
                best_validation_metrics,
            )

            training_seeds = rng.choice(
                TRAIN_SEED_LIMIT,
                size=GAMES_PER_CANDIDATE,
                replace=False,
            ).tolist()

            scores = np.empty(POPULATION_SIZE, dtype=float)
            candidate_metrics = []

            print()
            print(f"=== GENERATION {generation_number}/{GENERATIONS} ===")
            print(f"Mutation scale: {mutation_scale:.4f}")

            for index, candidate in enumerate(population):
                metrics = evaluate_candidate(
                    env,
                    candidate,
                    training_seeds,
                )

                scores[index] = metrics["fitness"]
                candidate_metrics.append(metrics)

                print(
                    f"Candidate {index + 1:02d}/{POPULATION_SIZE}: "
                    f"fitness={metrics['fitness']:.3f}, "
                    f"lines={metrics['mean_lines']:.2f}, "
                    f"median={metrics['median_lines']:.2f}, "
                    f"std={metrics['std_lines']:.2f}"
                )

            best_index = int(np.argmax(scores))
            generation_best_weights = population[best_index].copy()
            train_metrics = candidate_metrics[best_index]

            validation_metrics = evaluate_candidate(
                env,
                generation_best_weights,
                validation_seeds,
            )

            print()
            print(f"Generation {generation_number} best:")
            print_weights(generation_best_weights)
            print(
                "Training:   "
                f"fitness={train_metrics['fitness']:.3f}, "
                f"mean lines={train_metrics['mean_lines']:.2f}"
            )
            print(
                "Validation: "
                f"fitness={validation_metrics['fitness']:.3f}, "
                f"mean lines={validation_metrics['mean_lines']:.2f}, "
                f"std={validation_metrics['std_lines']:.2f}"
            )

            if validation_metrics["fitness"] > best_validation_fitness:
                best_validation_fitness = validation_metrics["fitness"]
                best_weights = generation_best_weights.copy()
                best_generation = generation_number
                best_validation_metrics = validation_metrics.copy()

                save_v2_weights(
                    best_weights,
                    best_generation,
                    best_validation_metrics,
                )

                print(
                    "New global V2 best saved to "
                    f"{V2_WEIGHTS_FILE.name}"
                )

            append_history(
                {
                    "generation": generation_number,
                    "mutation_scale": mutation_scale,
                    "train_fitness": train_metrics["fitness"],
                    "train_mean_lines": train_metrics["mean_lines"],
                    "train_median_lines": train_metrics["median_lines"],
                    "train_std_lines": train_metrics["std_lines"],
                    "train_mean_pieces": train_metrics["mean_pieces"],
                    "train_zero_rate": train_metrics["zero_rate"],
                    "validation_fitness": validation_metrics["fitness"],
                    "validation_mean_lines": validation_metrics["mean_lines"],
                    "validation_median_lines": validation_metrics["median_lines"],
                    "validation_std_lines": validation_metrics["std_lines"],
                    "validation_mean_pieces": validation_metrics["mean_pieces"],
                    "validation_zero_rate": validation_metrics["zero_rate"],
                    "weight_aggregate_height": generation_best_weights[0],
                    "weight_max_height": generation_best_weights[1],
                    "weight_holes": generation_best_weights[2],
                    "weight_bumpiness": generation_best_weights[3],
                }
            )

            population = create_next_generation(
                population,
                scores,
                mutation_scale,
            )

            save_checkpoint(
                generation + 1,
                population,
                best_weights,
                best_validation_fitness,
                best_generation,
                best_validation_metrics,
            )

        if best_weights is None or best_validation_metrics is None:
            raise RuntimeError("Training finished without valid weights.")

        print()
        print("Training finished.")
        print(f"Best validation generation: {best_generation}")
        print_weights(best_weights)

        print()
        print(f"Running final test on {TEST_GAMES} untouched seeds...")

        test_metrics = evaluate_candidate(
            env,
            best_weights,
            test_seeds,
        )

        save_v2_weights(
            best_weights,
            best_generation,
            best_validation_metrics,
        )

        save_final_report(
            best_weights,
            best_generation,
            best_validation_metrics,
            test_metrics,
        )

        print(f"Saved: {V2_WEIGHTS_FILE.name}")
        print(f"Saved: {HISTORY_FILE.name}")
        print(f"Saved: {REPORT_FILE.name}")
        print(f"Checkpoint: {CHECKPOINT_FILE.name}")

    except KeyboardInterrupt:
        print()
        print("Training interrupted. The last checkpoint is safe.")
        print("Run the same command again to resume.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
