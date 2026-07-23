from __future__ import annotations

import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import tetris_gymnasium

from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations

from tetris_features_v3 import (
    DECISION_WEIGHT_NAMES,
    ExtendedFeatureObservation,
    N_WEIGHTS,
    normalize_weights,
    score_candidate,
)


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_FILE = BASE_DIR / "best_weights_v3.json"

GAME_SEED = 42
ANIMATION_DELAY = 0.05
MAX_PIECES = 1500


def create_environment():
    base_env = gym.make(
        "tetris_gymnasium/Tetris",
        render_mode="human",
        gravity=False,
    )

    feature_wrapper = ExtendedFeatureObservation(base_env)

    return GroupedActionsObservations(
        base_env,
        observation_wrappers=[feature_wrapper],
        terminate_on_illegal_action=True,
    )


def load_weights() -> np.ndarray:
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {WEIGHTS_FILE}\n"
            "Umieść best_weights_v3.json w tym samym katalogu co main_v3.py."
        )

    with WEIGHTS_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    weights_data = data.get("weights")
    if not isinstance(weights_data, dict):
        raise ValueError(
            "Plik best_weights_v3.json nie zawiera słownika 'weights'."
        )

    missing = [
        name for name in DECISION_WEIGHT_NAMES
        if name not in weights_data
    ]
    if missing:
        raise ValueError(
            "W pliku brakuje wag: " + ", ".join(missing)
        )

    weights = np.array(
        [weights_data[name] for name in DECISION_WEIGHT_NAMES],
        dtype=float,
    )

    if weights.shape != (N_WEIGHTS,):
        raise ValueError(
            f"Oczekiwano {N_WEIGHTS} wag, otrzymano {weights.size}."
        )

    return normalize_weights(weights)


def choose_action(
    observations,
    action_mask,
    current_features,
    weights,
    width,
    height,
) -> int:
    legal_actions = np.flatnonzero(action_mask)

    if len(legal_actions) == 0:
        raise RuntimeError("Brak legalnych akcji.")

    best_action = int(legal_actions[0])
    best_score = float("inf")

    for action in legal_actions:
        candidate_features = np.asarray(
            observations[action],
            dtype=float,
        )

        score = score_candidate(
            current_features=current_features,
            candidate_features=candidate_features,
            weights=weights,
            width=width,
            height=height,
        )

        if score < best_score:
            best_score = score
            best_action = int(action)

    return best_action


def main() -> None:
    weights = load_weights()
    env = create_environment()

    observations, info = env.reset(seed=GAME_SEED)

    width = int(env.unwrapped.width)
    height = int(env.unwrapped.height)

    current_features = np.asarray(
        info["board"],
        dtype=float,
    )

    terminated = False
    truncated = False

    placed_pieces = 0
    total_lines = 0
    total_reward = 0.0

    print("--- FINALNY AGENT TETRIS V3 ---")
    print(f"Seed gry: {GAME_SEED}")
    print(f"Liczba cech planszy: {len(current_features)}")
    print(f"Liczba wag decyzyjnych: {len(weights)}")
    print()

    print("Wczytane wagi:")
    for name, value in zip(DECISION_WEIGHT_NAMES, weights):
        print(f"{name:24s} {value:.10f}")

    print()

    try:
        while (
            not terminated
            and not truncated
            and placed_pieces < MAX_PIECES
        ):
            action_mask = np.asarray(
                info["action_mask"],
                dtype=bool,
            )

            action = choose_action(
                observations=observations,
                action_mask=action_mask,
                current_features=current_features,
                weights=weights,
                width=width,
                height=height,
            )

            (
                observations,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            placed_pieces += 1
            total_lines += int(info.get("lines_cleared", 0))
            total_reward += float(reward)

            if "board" in info:
                current_features = np.asarray(
                    info["board"],
                    dtype=float,
                )

            env.render()
            time.sleep(ANIMATION_DELAY)

    finally:
        env.close()

    print()
    print("--- KONIEC GRY ---")
    print(f"Ustawione klocki: {placed_pieces}")
    print(f"Usunięte linie:   {total_lines}")
    print(f"Łączna nagroda:   {total_reward:.2f}")


if __name__ == "__main__":
    main()
