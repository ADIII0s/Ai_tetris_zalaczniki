import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, wilcoxon


BASE_DIR = Path(__file__).resolve().parent

CSV_FILE = BASE_DIR / "agent_comparison.csv"
BOXPLOT_FILE = BASE_DIR / "results_boxplot.png"
DIFFERENCE_FILE = BASE_DIR / "paired_differences.png"
REPORT_FILE = BASE_DIR / "statistical_report.txt"

OLD_AGENT_NAME = "old_weights"
NEW_AGENT_NAME = "genetic_weights"

BOOTSTRAP_SAMPLES = 10_000
RANDOM_SEED = 12345


def load_results():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {CSV_FILE}"
        )

    old_results = {}
    new_results = {}

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            agent = row["agent"]
            seed = int(row["seed"])
            lines = int(row["lines"])

            if agent == OLD_AGENT_NAME:
                old_results[seed] = lines

            elif agent == NEW_AGENT_NAME:
                new_results[seed] = lines

    common_seeds = sorted(
        set(old_results.keys())
        & set(new_results.keys())
    )

    if not common_seeds:
        raise RuntimeError(
            "No matching seeds were found."
        )

    old_lines = np.array(
        [old_results[seed] for seed in common_seeds],
        dtype=float,
    )

    new_lines = np.array(
        [new_results[seed] for seed in common_seeds],
        dtype=float,
    )

    seeds = np.array(
        common_seeds,
        dtype=int,
    )

    return seeds, old_lines, new_lines


def calculate_rank_biserial(differences):
    nonzero_differences = differences[
        differences != 0
    ]

    if len(nonzero_differences) == 0:
        return 0.0

    ranks = rankdata(
        np.abs(nonzero_differences)
    )

    positive_rank_sum = np.sum(
        ranks[nonzero_differences > 0]
    )

    negative_rank_sum = np.sum(
        ranks[nonzero_differences < 0]
    )

    total_rank_sum = (
        positive_rank_sum
        + negative_rank_sum
    )

    return float(
        (
            positive_rank_sum
            - negative_rank_sum
        )
        / total_rank_sum
    )


def bootstrap_confidence_interval(
    differences,
):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    sample_size = len(differences)

    bootstrap_means = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=float,
    )

    for index in range(BOOTSTRAP_SAMPLES):
        sample = rng.choice(
            differences,
            size=sample_size,
            replace=True,
        )

        bootstrap_means[index] = np.mean(
            sample
        )

    lower_limit = np.percentile(
        bootstrap_means,
        2.5,
    )

    upper_limit = np.percentile(
        bootstrap_means,
        97.5,
    )

    return (
        float(lower_limit),
        float(upper_limit),
    )


def run_statistical_analysis(
    old_lines,
    new_lines,
):
    differences = new_lines - old_lines

    test_result = wilcoxon(
        new_lines,
        old_lines,
        alternative="greater",
        zero_method="wilcox",
        method="auto",
    )

    effect_size = calculate_rank_biserial(
        differences
    )

    confidence_interval = (
        bootstrap_confidence_interval(
            differences
        )
    )

    wins = int(
        np.sum(differences > 0)
    )

    ties = int(
        np.sum(differences == 0)
    )

    losses = int(
        np.sum(differences < 0)
    )

    return {
        "old_mean": float(np.mean(old_lines)),
        "new_mean": float(np.mean(new_lines)),
        "mean_difference": float(
            np.mean(differences)
        ),
        "median_difference": float(
            np.median(differences)
        ),
        "wilcoxon_statistic": float(
            test_result.statistic
        ),
        "p_value": float(
            test_result.pvalue
        ),
        "effect_size": effect_size,
        "ci_lower": confidence_interval[0],
        "ci_upper": confidence_interval[1],
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def interpret_p_value(p_value):
    if p_value < 0.001:
        return (
            "Very strong evidence that "
            "the genetic agent is better."
        )

    if p_value < 0.01:
        return (
            "Strong evidence that "
            "the genetic agent is better."
        )

    if p_value < 0.05:
        return (
            "Statistically significant evidence "
            "that the genetic agent is better."
        )

    return (
        "No statistically significant "
        "difference was detected."
    )


def interpret_effect_size(effect_size):
    absolute_value = abs(effect_size)

    if absolute_value < 0.1:
        strength = "negligible"
    elif absolute_value < 0.3:
        strength = "small"
    elif absolute_value < 0.5:
        strength = "medium"
    else:
        strength = "large"

    direction = (
        "genetic agent"
        if effect_size > 0
        else "old agent"
    )

    return (
        f"{strength} effect in favour "
        f"of the {direction}"
    )


def create_boxplot(
    old_lines,
    new_lines,
):
    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.boxplot(
        [
            old_lines,
            new_lines,
        ],
        tick_labels=[
            "Old weights",
            "Genetic weights",
        ],
        showmeans=True,
    )

    axis.set_title(
        "Distribution of cleared lines"
    )

    axis.set_ylabel(
        "Cleared lines"
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        BOXPLOT_FILE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_difference_plot(
    seeds,
    old_lines,
    new_lines,
):
    differences = new_lines - old_lines

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.scatter(
        seeds,
        differences,
        s=28,
    )

    axis.axhline(
        0,
        linewidth=1,
    )

    axis.set_title(
        "Paired result differences"
    )

    axis.set_xlabel(
        "Seed"
    )

    axis.set_ylabel(
        "Genetic lines - old lines"
    )

    axis.grid(
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        DIFFERENCE_FILE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_report(results):
    report = (
        "--- STATISTICAL ANALYSIS ---\n"
        f"Games:                 "
        f"{results['wins'] + results['ties'] + results['losses']}\n"
        f"Old mean lines:        "
        f"{results['old_mean']:.2f}\n"
        f"Genetic mean lines:    "
        f"{results['new_mean']:.2f}\n"
        f"Mean paired difference:"
        f" {results['mean_difference']:.2f}\n"
        f"Median difference:     "
        f"{results['median_difference']:.2f}\n"
        f"95% bootstrap CI:      "
        f"[{results['ci_lower']:.2f}, "
        f"{results['ci_upper']:.2f}]\n"
        f"Wins / ties / losses:  "
        f"{results['wins']} / "
        f"{results['ties']} / "
        f"{results['losses']}\n"
        f"Wilcoxon statistic:    "
        f"{results['wilcoxon_statistic']:.2f}\n"
        f"Wilcoxon p-value:      "
        f"{results['p_value']:.10g}\n"
        f"Rank-biserial effect:  "
        f"{results['effect_size']:.4f}\n"
        f"Effect interpretation: "
        f"{interpret_effect_size(results['effect_size'])}\n"
        f"Conclusion:            "
        f"{interpret_p_value(results['p_value'])}\n"
    )

    print()
    print(report)

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)


def main():
    seeds, old_lines, new_lines = (
        load_results()
    )

    results = run_statistical_analysis(
        old_lines,
        new_lines,
    )

    create_boxplot(
        old_lines,
        new_lines,
    )

    create_difference_plot(
        seeds,
        old_lines,
        new_lines,
    )

    create_report(results)

    print(f"Saved: {BOXPLOT_FILE.name}")
    print(f"Saved: {DIFFERENCE_FILE.name}")
    print(f"Saved: {REPORT_FILE.name}")


if __name__ == "__main__":
    main()
