import argparse
import numpy as np

from src.environment import create_environment
from src.policies import stick_on_20_policy


# ============================================================
# TD(0) — PRÉDICTION
# ============================================================

def td_prediction(
    num_episodes=100,
    alpha=0.1,
    gamma=1.0,
    seed=None
):
    """
    Prédiction TD(0) pour une politique fixe.

    Politique utilisée :
        rester à partir de 20

    Paramètres :
        num_episodes : nombre d'épisodes
        alpha        : taux d'apprentissage
        gamma        : facteur d'actualisation
        seed          : graine aléatoire

    Retour :
        V : dictionnaire contenant V(s)
    """

    env = create_environment()

    V = {}

    # --------------------------------------------------------
    # Initialisation de V(s)
    # --------------------------------------------------------

    def get_value(state):
        """
        Retourne V(s).
        Les états non encore rencontrés ont une valeur 0.
        """
        return V.get(state, 0.0)

    # --------------------------------------------------------
    # Boucle principale
    # --------------------------------------------------------

    for episode in range(num_episodes):

        # Utilisation d'une graine différente par épisode
        # lorsque seed est fourni.
        if seed is not None:
            episode_seed = seed + episode
        else:
            episode_seed = None

        state, info = env.reset(
            seed=episode_seed
        )

        terminated = False
        truncated = False

        while not (terminated or truncated):

            # ------------------------------------------------
            # Choix de l'action selon la politique fixe
            # ------------------------------------------------

            action = stick_on_20_policy(
                state
            )

            # ------------------------------------------------
            # Exécution de l'action
            # ------------------------------------------------

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            # ------------------------------------------------
            # Initialisation des états rencontrés
            # ------------------------------------------------

            if state not in V:
                V[state] = 0.0

            if next_state not in V and not (
                terminated or truncated
            ):
                V[next_state] = 0.0

            # ------------------------------------------------
            # Cible TD(0)
            # ------------------------------------------------

            if terminated or truncated:

                target = reward

            else:

                target = (
                    reward
                    + gamma * get_value(next_state)
                )

            # ------------------------------------------------
            # Mise à jour TD(0)
            # ------------------------------------------------

            V[state] = (
                V[state]
                + alpha
                * (
                    target
                    - V[state]
                )
            )

            # Passage à l'état suivant
            state = next_state

    env.close()

    return V


# ============================================================
# SAUVEGARDE
# ============================================================

def save_values(values, filename):
    """
    Sauvegarde V(s) au format .npy.
    """

    np.save(
        filename,
        values,
        allow_pickle=True
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prédiction TD(0) sur Blackjack-v1"
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Nombre d'épisodes"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Taux d'apprentissage alpha"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Facteur d'actualisation gamma"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Graine aléatoire"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Exécution TD(0)
    # --------------------------------------------------------

    print(
        "=== TD(0) — PRÉDICTION ==="
    )

    print(
        f"Nombre d'épisodes : {args.episodes}"
    )

    print(
        f"Alpha : {args.alpha}"
    )

    print(
        f"Gamma : {args.gamma}"
    )

    if args.seed is not None:

        print(
            f"Seed : {args.seed}"
        )

    V = td_prediction(
        num_episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed
    )

    # --------------------------------------------------------
    # Quelques résultats
    # --------------------------------------------------------

    print()
    print(
        "Nombre d'états visités :",
        len(V)
    )

    print()
    print(
        "=== QUELQUES VALEURS V(s) ==="
    )

    displayed = 0

    for state, value in V.items():

        print(
            f"État : {state} | "
            f"V(s) = {value:.4f}"
        )

        displayed += 1

        if displayed >= 10:
            break

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    filename = (
        f"results/data/"
        f"V_td_{args.episodes}_"
        f"alpha_{args.alpha}.npy"
    )

    save_values(
        V,
        filename
    )

    print()
    print(
        f"Résultats sauvegardés dans : {filename}"
    )


if __name__ == "__main__":
    main()