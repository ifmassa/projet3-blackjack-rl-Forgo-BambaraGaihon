import argparse
import random

import gymnasium as gym
import numpy as np


STICK = 0
HIT = 1


def epsilon_soft_action(q_values, epsilon):
    """
    Choisit une action selon une politique epsilon-douce.

    Avec une probabilité epsilon :
        exploration aléatoire.

    Sinon :
        exploitation de la meilleure action.

    Paramètres
    ----------
    q_values : numpy.ndarray
        Valeurs Q(s,a) pour les actions disponibles.

    epsilon : float
        Probabilité d'exploration.

    Retour
    ------
    int
        Action choisie.
    """

    if random.random() < epsilon:
        return random.randrange(len(q_values))

    # En cas d'égalité, np.random.choice permet
    # de choisir aléatoirement parmi les meilleures actions.
    max_value = np.max(q_values)

    best_actions = np.flatnonzero(
        q_values == max_value
    )

    return int(
        np.random.choice(best_actions)
    )


def generate_episode(env, q, epsilon):
    """
    Génère un épisode en suivant une politique
    epsilon-douce dérivée de Q.

    Retour
    ------
    list
        Liste des transitions :
        (state, action, reward)
    """

    episode = []

    state, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Si l'état n'existe pas encore dans Q,
        # on initialise ses deux actions à 0.
        if state not in q:
            q[state] = np.zeros(2)

        # Choix de l'action epsilon-douce
        action = epsilon_soft_action(
            q[state],
            epsilon
        )

        # Exécution de l'action
        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        # Enregistrement de la transition
        episode.append(
            (state, action, reward)
        )

        state = next_state

    return episode


def calculate_returns(episode, gamma=1.0):
    """
    Calcule les retours G_t de l'épisode.

    G_t = R_{t+1} + gamma R_{t+2}
          + gamma^2 R_{t+3} + ...

    Dans Blackjack, gamma = 1.
    """

    returns = []

    G = 0.0

    for state, action, reward in reversed(episode):

        G = reward + gamma * G

        returns.insert(
            0,
            (state, action, G)
        )

    return returns


def mc_control(
    env,
    num_episodes,
    epsilon=0.1,
    gamma=1.0
):
    """
    Contrôle Monte-Carlo First-Visit avec
    politique epsilon-douce et moyennage incrémental.

    Paramètres
    ----------
    env : Gymnasium environment

    num_episodes : int
        Nombre d'épisodes.

    epsilon : float
        Taux d'exploration.

    gamma : float
        Facteur d'actualisation.

    Retour
    ------
    q : dict
        Fonction de valeur Q(s,a).

    counts : dict
        Nombre de premières visites de chaque paire (s,a).
    """

    # Fonction de valeur action
    q = {}

    # Nombre de visites de chaque paire (s,a)
    counts = {}

    for episode_number in range(num_episodes):

        # Génération d'un épisode
        episode = generate_episode(
            env,
            q,
            epsilon
        )

        # Calcul des retours
        episode_returns = calculate_returns(
            episode,
            gamma
        )

        # Pour First-Visit :
        # une paire (état, action) ne doit être
        # traitée qu'à sa première occurrence.
        visited_pairs = set()

        for state, action, G in episode_returns:

            pair = (state, action)

            if pair in visited_pairs:
                continue

            visited_pairs.add(pair)

            # Initialisation
            if state not in q:
                q[state] = np.zeros(2)

            if pair not in counts:
                counts[pair] = 0

            # Nombre de visites
            counts[pair] += 1

            # Moyenne incrémentale
            alpha = 1.0 / counts[pair]

            q[state][action] += (
                alpha
                * (
                    G
                    - q[state][action]
                )
            )

    return q, counts


def extract_policy(q):
    """
    Extrait la politique gloutonne à partir de Q.

    Retour
    ------
    dict
        policy[state] = action optimale selon Q.
    """

    policy = {}

    for state, q_values in q.items():

        policy[state] = int(
            np.argmax(q_values)
        )

    return policy


def save_q(q, filename):
    """
    Sauvegarde Q(s,a) au format .npy.
    """

    np.save(
        filename,
        q,
        allow_pickle=True
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Contrôle Monte-Carlo First-Visit "
            "avec politique epsilon-douce"
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Nombre d'épisodes"
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Probabilité d'exploration"
    )

    args = parser.parse_args()

    # Création de l'environnement
    env = gym.make(
        "Blackjack-v1",
        sab=True
    )

    print("=== MONTE-CARLO CONTROL ===")
    print(
        f"Nombre d'épisodes : {args.episodes}"
    )
    print(
        f"Epsilon : {args.epsilon}"
    )
    print(
        "Gamma : 1.0"
    )

    # Apprentissage
    q, counts = mc_control(
        env,
        num_episodes=args.episodes,
        epsilon=args.epsilon,
        gamma=1.0
    )

    # Extraction de la politique apprise
    policy = extract_policy(q)

    print(
        "Nombre d'états appris :",
        len(q)
    )

    print(
        "Nombre de couples (s,a) visités :",
        len(counts)
    )

    # Quelques exemples
    print("\n=== QUELQUES ACTIONS APPRISES ===")

    displayed = 0

    for state in sorted(policy):

        action = policy[state]

        action_name = (
            "STICK"
            if action == STICK
            else "HIT"
        )

        print(
            f"État : {state} -> "
            f"Action : {action_name} "
            f"({action})"
        )

        displayed += 1

        if displayed >= 10:
            break

    # Sauvegarde
    output_file = (
        f"results/data/"
        f"Q_mc_{args.episodes}.npy"
    )

    save_q(
        q,
        output_file
    )

    print(
        f"\nQ(s,a) sauvegardé dans : "
        f"{output_file}"
    )

    env.close()


if __name__ == "__main__":
    main()