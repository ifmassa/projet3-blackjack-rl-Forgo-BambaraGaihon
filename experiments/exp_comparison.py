import os
import numpy as np
import matplotlib.pyplot as plt

from src.td_prediction import td_prediction


# ============================================================
# PARAMÈTRES DE L'EXPÉRIENCE
# ============================================================

REFERENCE_FILE = "results/data/V_mc_500000.npy"

EPISODES_LIST = [
    100,
    1000,
    10000,
    100000
]

ALPHAS = [
    0.05,
    0.10,
    0.20
]

SEEDS = list(range(10))

GAMMA = 1.0


# ============================================================
# CHARGEMENT DE LA RÉFÉRENCE MC
# ============================================================

def load_reference():

    print("=" * 70)
    print("CHARGEMENT DE LA RÉFÉRENCE MONTE-CARLO")
    print("=" * 70)

    if not os.path.exists(REFERENCE_FILE):

        raise FileNotFoundError(
            f"Fichier introuvable : {REFERENCE_FILE}\n"
            "Lance d'abord :\n"
            "python -m src.mc_prediction --episodes 500000"
        )

    reference = np.load(
        REFERENCE_FILE,
        allow_pickle=True
    ).item()

    print(
        f"Référence chargée : {REFERENCE_FILE}"
    )

    print(
        f"Nombre d'états de référence : {len(reference)}"
    )

    return reference


# ============================================================
# CALCUL DE LA MSE
# ============================================================

def calculate_mse(values, reference):

    common_states = set(values.keys()) & set(reference.keys())

    if len(common_states) == 0:
        return np.nan, 0

    errors = []

    for state in common_states:

        error = (
            values[state]
            - reference[state]
        )

        errors.append(error ** 2)

    mse = np.mean(errors)

    return mse, len(common_states)


# ============================================================
# EXPÉRIENCE TD(0)
# ============================================================

def run_experiment(reference):

    results = []

    total_runs = (
        len(EPISODES_LIST)
        * len(ALPHAS)
        * len(SEEDS)
    )

    current_run = 0

    print()
    print("=" * 70)
    print("EXPÉRIENCE TD(0) — CONVERGENCE ET SENSIBILITÉ À ALPHA")
    print("=" * 70)

    print(
        f"Épisodes testés : {EPISODES_LIST}"
    )

    print(
        f"Alpha testés : {ALPHAS}"
    )

    print(
        f"Nombre de graines : {len(SEEDS)}"
    )

    print(
        f"Gamma : {GAMMA}"
    )

    print(
        f"Nombre total d'exécutions : {total_runs}"
    )

    print("=" * 70)

    for episodes in EPISODES_LIST:

        for alpha in ALPHAS:

            print()
            print("-" * 70)
            print(
                f"Épisodes = {episodes} | "
                f"Alpha = {alpha}"
            )
            print("-" * 70)

            mse_values = []
            state_counts = []

            for seed in SEEDS:

                current_run += 1

                print(
                    f"[{current_run}/{total_runs}] "
                    f"Seed = {seed}"
                )

                values = td_prediction(
                    num_episodes=episodes,
                    alpha=alpha,
                    gamma=GAMMA,
                    seed=seed
                )

                mse, common_states = calculate_mse(
                    values,
                    reference
                )

                mse_values.append(mse)
                state_counts.append(common_states)

                print(
                    f"    États communs : {common_states}"
                )

                print(
                    f"    MSE : {mse:.6f}"
                )

            mean_mse = np.mean(
                mse_values
            )

            std_mse = np.std(
                mse_values
            )

            mean_states = np.mean(
                state_counts
            )

            print()
            print(
                f"MSE moyenne : {mean_mse:.6f}"
            )

            print(
                f"Écart-type : {std_mse:.6f}"
            )

            print(
                f"États communs moyens : "
                f"{mean_states:.1f}"
            )

            results.append(
                {
                    "episodes": episodes,
                    "alpha": alpha,
                    "mse_mean": mean_mse,
                    "mse_std": std_mse,
                    "states_mean": mean_states
                }
            )

    return results


# ============================================================
# AFFICHAGE DU TABLEAU
# ============================================================

def print_summary(results):

    print()
    print("=" * 80)
    print("RÉSUMÉ FINAL")
    print("=" * 80)

    print(
        f"{'Episodes':<12}"
        f"{'Alpha':<10}"
        f"{'MSE moyenne':<18}"
        f"{'Écart-type':<18}"
        f"{'États moyens':<15}"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result['episodes']:<12}"
            f"{result['alpha']:<10.2f}"
            f"{result['mse_mean']:<18.6f}"
            f"{result['mse_std']:<18.6f}"
            f"{result['states_mean']:<15.1f}"
        )

    print("=" * 80)


# ============================================================
# SAUVEGARDE DES RÉSULTATS
# ============================================================

def save_results(results):

    os.makedirs(
        "results/data",
        exist_ok=True
    )

    output_file = (
        "results/data/"
        "comparison_td_convergence.npy"
    )

    np.save(
        output_file,
        results,
        allow_pickle=True
    )

    print()
    print(
        f"Résultats sauvegardés dans : "
        f"{output_file}"
    )


# ============================================================
# COURBES
# ============================================================

def plot_results(results):

    os.makedirs(
        "results/figures",
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 6)
    )

    for alpha in ALPHAS:

        alpha_results = [
            result
            for result in results
            if result["alpha"] == alpha
        ]

        episodes = [
            result["episodes"]
            for result in alpha_results
        ]

        mse_mean = [
            result["mse_mean"]
            for result in alpha_results
        ]

        mse_std = [
            result["mse_std"]
            for result in alpha_results
        ]

        episodes = np.array(
            episodes
        )

        mse_mean = np.array(
            mse_mean
        )

        mse_std = np.array(
            mse_std
        )

        plt.plot(
            episodes,
            mse_mean,
            marker="o",
            label=f"α = {alpha}"
        )

        plt.fill_between(
            episodes,
            mse_mean - mse_std,
            mse_mean + mse_std,
            alpha=0.2
        )

    plt.xscale("log")

    plt.xlabel(
        "Nombre d'épisodes"
    )

    plt.ylabel(
        "Erreur quadratique moyenne (MSE)"
    )

    plt.title(
        "Convergence de TD(0) vers la référence Monte-Carlo"
    )

    plt.legend()

    plt.grid(
        True,
        which="both",
        alpha=0.3
    )

    plt.tight_layout()

    output_file = (
        "results/figures/"
        "td_convergence_mse.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        f"Figure sauvegardée : {output_file}"
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    # 1. Charger V_ref
    reference = load_reference()

    # 2. Exécuter toutes les expériences
    results = run_experiment(
        reference
    )

    # 3. Afficher le résumé
    print_summary(
        results
    )

    # 4. Sauvegarder les résultats
    save_results(
        results
    )

    # 5. Générer la courbe
    plot_results(
        results
    )

    print()
    print("=" * 70)
    print("EXPÉRIENCE TERMINÉE")
    print("=" * 70)


if __name__ == "__main__":
    main()