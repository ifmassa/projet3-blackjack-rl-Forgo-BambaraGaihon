import argparse
import numpy as np
import gymnasium as gym

from src.policies import stick_on_20_policy


def generate_episode(env):
    """
    Génère un épisode complet en suivant la politique fixe
    « rester à partir de 20 ».

    Retour
    ------
    list
        Liste des transitions sous la forme :
        (state, action, reward)
    """

    episode = []

    state, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Choix de l'action selon la politique fixe
        action = stick_on_20_policy(state)

        # Exécution de l'action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Enregistrement de la transition
        episode.append((state, action, reward))

        # Passage à l'état suivant
        state = next_state

    return episode


def calculate_returns(episode, gamma=1.0):
    """
    Calcule le retour G_t pour chaque état de l'épisode.

    Paramètres
    ----------
    episode : list
        Liste des transitions (state, action, reward).

    gamma : float
        Facteur d'actualisation.

    Retour
    ------
    list
        Liste de tuples (state, G_t).
    """

    returns = []

    # Retour initial
    G = 0.0

    # Parcours de l'épisode de la fin vers le début
    for state, action, reward in reversed(episode):

        G = reward + gamma * G

        # On replace le résultat au début
        returns.insert(0, (state, G))

    return returns


def first_visit_mc_prediction(env, num_episodes, gamma=1.0):
    """
    Implémente la prédiction Monte-Carlo First-Visit.

    Pour chaque épisode :
    1. Génère un épisode selon la politique fixe.
    2. Calcule les retours G_t.
    3. Ne prend en compte que la première visite de chaque état.
    4. Moyenne les retours observés pour obtenir V(s).

    Paramètres
    ----------
    env : environnement Gymnasium
    num_episodes : int
        Nombre d'épisodes à générer.

    gamma : float
        Facteur d'actualisation.

    Retour
    ------
    dict
        Fonction de valeur estimée V(s).
    """

    # Somme des retours pour chaque état
    returns_sum = {}

    # Nombre de premières visites de chaque état
    returns_count = {}

    # Fonction de valeur V(s)
    values = {}

    # Génération des épisodes
    for episode_number in range(num_episodes):

        episode = generate_episode(env)

        # Calcul des retours
        episode_returns = calculate_returns(
            episode,
            gamma=gamma
        )

        # États déjà rencontrés dans cet épisode
        visited_states = set()

        # Parcours des états et de leurs retours
        for state, G in episode_returns:

            # Si l'état a déjà été rencontré,
            # on l'ignore : First-Visit
            if state in visited_states:
                continue

            # Première visite de cet état
            visited_states.add(state)

            # Initialisation si nécessaire
            if state not in returns_sum:
                returns_sum[state] = 0.0
                returns_count[state] = 0

            # Accumulation du retour
            returns_sum[state] += G

            # Nombre de visites
            returns_count[state] += 1

            # Moyenne des retours
            values[state] = (
                returns_sum[state]
                / returns_count[state]
            )

    return values


def save_values(values, filename):
    """
    Sauvegarde la fonction de valeur V(s) dans un fichier .npy.
    """

    np.save(
        filename,
        values,
        allow_pickle=True
    )


def main():
    """
    Point d'entrée du programme.
    """

    # Création du paramètre --episodes
    parser = argparse.ArgumentParser(
        description=(
            "Prédiction First-Visit Monte-Carlo "
            "pour Blackjack-v1"
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Nombre d'épisodes à générer"
    )

    args = parser.parse_args()

    # Création de l'environnement
    env = gym.make(
        "Blackjack-v1",
        sab=True
    )

    print("=== FIRST-VISIT MONTE-CARLO ===")
    print(
        f"Nombre d'épisodes : {args.episodes}"
    )

    # Exécution de l'algorithme
    values = first_visit_mc_prediction(
        env,
        num_episodes=args.episodes,
        gamma=1.0
    )

    # Nombre d'états effectivement visités
    print(
        "Nombre d'états visités :",
        len(values)
    )

    # Nom du fichier de sortie
    output_file = (
        f"results/data/V_mc_{args.episodes}.npy"
    )

    # Sauvegarde
    save_values(
        values,
        output_file
    )

    print(
        f"Résultats sauvegardés dans : "
        f"{output_file}"
    )

    # Fermeture de l'environnement
    env.close()


if __name__ == "__main__":
    main()