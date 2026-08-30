import argparse
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PARTIE 1 — ÉVALUATION DE V(s) : MONTE-CARLO PREDICTION
# ============================================================

def load_values(filename):
    """
    Charge une fonction de valeur V(s) sauvegardée
    au format .npy.
    """
    values = np.load(
        filename,
        allow_pickle=True
    ).item()

    return values


def create_value_matrix(values, usable_ace):
    """
    Transforme le dictionnaire V(s) en matrice
    pour une carte de chaleur.

    État :
        (player_sum, dealer_card, usable_ace)
    """

    player_sums = range(12, 22)
    dealer_cards = range(1, 11)

    matrix = np.full(
        (10, 10),
        np.nan
    )

    for i, player_sum in enumerate(player_sums):

        for j, dealer_card in enumerate(dealer_cards):

            state = (
                player_sum,
                dealer_card,
                usable_ace
            )

            if state in values:
                matrix[i, j] = values[state]

    return matrix


def plot_value_function(values, usable_ace, episodes):
    """
    Génère une carte de chaleur de V(s).
    """

    matrix = create_value_matrix(
        values,
        usable_ace
    )

    if usable_ace == 1:

        title = (
            f"Monte-Carlo First-Visit - {episodes:,} épisodes\n"
            "Avec as utilisable"
        )

        filename = (
            f"results/figures/"
            f"mc_{episodes}_usable_ace.png"
        )

    else:

        title = (
            f"Monte-Carlo First-Visit - {episodes:,} épisodes\n"
            "Sans as utilisable"
        )

        filename = (
            f"results/figures/"
            f"mc_{episodes}_no_usable_ace.png"
        )

    plt.figure(
        figsize=(10, 7)
    )

    plt.imshow(
        matrix,
        origin="lower",
        aspect="auto"
    )

    plt.colorbar(
        label="V(s)"
    )

    plt.xticks(
        range(10),
        range(1, 11)
    )

    plt.yticks(
        range(10),
        range(12, 22)
    )

    plt.xlabel(
        "Carte visible du croupier"
    )

    plt.ylabel(
        "Somme du joueur"
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()

    print(
        f"Figure sauvegardée : {filename}"
    )


# ============================================================
# PARTIE 2 — ÉVALUATION DU CONTRÔLE MONTE-CARLO
# ============================================================

def load_q(filename):
    """
    Charge Q(s,a) depuis un fichier .npy.
    """

    q = np.load(
        filename,
        allow_pickle=True
    ).item()

    return q


def extract_policy(q):
    """
    Extrait la politique gloutonne à partir de Q.

    0 = Stick
    1 = Hit
    """

    policy = {}

    for state, q_values in q.items():

        policy[state] = int(
            np.argmax(q_values)
        )

    return policy


def action_symbol(action):
    """
    Transforme une action numérique en symbole.

    S = Stick
    H = Hit
    """

    if action == 0:
        return "S"

    return "H"


def create_policy_matrix(policy, usable_ace):
    """
    Crée une matrice représentant la politique apprise.
    """

    player_sums = range(12, 22)
    dealer_cards = range(1, 11)

    matrix = np.full(
        (10, 10),
        "",
        dtype="<U1"
    )

    for i, player_sum in enumerate(player_sums):

        for j, dealer_card in enumerate(dealer_cards):

            state = (
                player_sum,
                dealer_card,
                usable_ace
            )

            if state in policy:

                matrix[i, j] = action_symbol(
                    policy[state]
                )

    return matrix


def display_policy_matrix(matrix, title):
    """
    Affiche une politique sous forme de tableau.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        "        "
        + " ".join(
            f"{card:>2}"
            for card in range(1, 11)
        )
    )

    for i, player_sum in enumerate(
        range(12, 22)
    ):

        row = " ".join(
            f"{matrix[i, j]:>2}"
            for j in range(10)
        )

        print(
            f"{player_sum:>3}     {row}"
        )


# ============================================================
# PARTIE 3 — POLITIQUE OPTIMALE DE RÉFÉRENCE
# ============================================================

def reference_policy(state):
    """
    Politique optimale de référence du Blackjack.

    État :
        (somme du joueur, carte visible du croupier,
         as utilisable)

    Actions :
        0 = Stick
        1 = Hit

    La politique utilise les règles classiques de la
    politique optimale tabulaire du Blackjack de
    Sutton & Barto pour les états étudiés.
    """

    player_sum, dealer_card, usable_ace = state

    # ========================================================
    # SANS AS UTILISABLE
    # ========================================================

    if usable_ace == 0:

        # 20 ou 21 : Stick
        if player_sum >= 20:
            return 0

        # 17, 18, 19 :
        # Stick contre 2 à 6,
        # Hit contre 7 à As.
        if player_sum >= 17:

            if dealer_card in [2, 3, 4, 5, 6]:
                return 0

            return 1

        # 13 à 16 :
        # Stick contre 2 à 6,
        # Hit contre 7 à As.
        if player_sum >= 13:

            if dealer_card in [2, 3, 4, 5, 6]:
                return 0

            return 1

        # 12 :
        # Stick contre 4, 5 et 6.
        if player_sum == 12:

            if dealer_card in [4, 5, 6]:
                return 0

            return 1

        # 11 ou moins : Hit
        return 1

    # ========================================================
    # AVEC AS UTILISABLE
    # ========================================================

    # 19, 20, 21 : Stick
    if player_sum >= 19:
        return 0

    # 18 :
    # Stick contre 2, 7 et 8.
    # Hit contre 9, 10 et As.
    if player_sum == 18:

        if dealer_card in [2, 7, 8]:
            return 0

        return 1

    # 17 ou moins : Hit
    return 1


def create_reference_matrix(usable_ace):
    """
    Crée la matrice de la politique optimale
    de référence.
    """

    matrix = np.full(
        (10, 10),
        "",
        dtype="<U1"
    )

    for i, player_sum in enumerate(
        range(12, 22)
    ):

        for j, dealer_card in enumerate(
            range(1, 11)
        ):

            state = (
                player_sum,
                dealer_card,
                usable_ace
            )

            action = reference_policy(
                state
            )

            matrix[i, j] = action_symbol(
                action
            )

    return matrix


# ============================================================
# PARTIE 4 — COMPARAISON DES POLITIQUES
# ============================================================

def compare_policies(
    learned_policy,
    reference,
    usable_ace
):
    """
    Compare la politique apprise avec la politique
    de référence.

    Retourne :
        total     : nombre d'états comparés
        correct   : nombre d'actions identiques
        accuracy  : taux d'accord
    """

    total = 0
    correct = 0

    for player_sum in range(12, 22):

        for dealer_card in range(1, 11):

            state = (
                player_sum,
                dealer_card,
                usable_ace
            )

            # L'état doit avoir été appris
            if state not in learned_policy:
                continue

            learned_action = (
                learned_policy[state]
            )

            reference_action = (
                reference(state)
            )

            total += 1

            if learned_action == reference_action:
                correct += 1

    if total == 0:

        accuracy = 0.0

    else:

        accuracy = (
            100.0
            * correct
            / total
        )

    return total, correct, accuracy


def plot_policy_comparison(
    learned_policy,
    usable_ace,
    episodes
):
    """
    Génère une figure montrant les différences
    entre la politique apprise et la référence.

    0 = même action
    1 = action différente
    """

    matrix = np.full(
        (10, 10),
        np.nan
    )

    for i, player_sum in enumerate(
        range(12, 22)
    ):

        for j, dealer_card in enumerate(
            range(1, 11)
        ):

            state = (
                player_sum,
                dealer_card,
                usable_ace
            )

            if state not in learned_policy:
                continue

            learned_action = (
                learned_policy[state]
            )

            reference_action = (
                reference_policy(state)
            )

            if learned_action == reference_action:

                matrix[i, j] = 0

            else:

                matrix[i, j] = 1

    if usable_ace == 1:

        ace_label = "usable_ace"
        title_ace = "Avec as utilisable"

    else:

        ace_label = "no_usable_ace"
        title_ace = "Sans as utilisable"

    filename = (
        f"results/figures/"
        f"mc_control_{episodes}_"
        f"{ace_label}_comparison.png"
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.imshow(
        matrix,
        origin="lower",
        aspect="auto"
    )

    plt.colorbar(
        label="0 = accord | 1 = différence"
    )

    plt.xticks(
        range(10),
        range(1, 11)
    )

    plt.yticks(
        range(10),
        range(12, 22)
    )

    plt.xlabel(
        "Carte visible du croupier"
    )

    plt.ylabel(
        "Somme du joueur"
    )

    plt.title(
        "MC Control — comparaison avec la référence\n"
        f"{title_ace} — {episodes:,} épisodes"
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()

    print(
        f"Figure sauvegardée : {filename}"
    )


# ============================================================
# PARTIE 5 — PROGRAMME PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Évaluation des expériences "
            "Monte-Carlo Blackjack"
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        required=True,
        help="Nombre d'épisodes"
    )

    parser.add_argument(
        "--mc-control",
        action="store_true",
        help="Évaluer le contrôle Monte-Carlo"
    )

    args = parser.parse_args()

    episodes = args.episodes

    # ========================================================
    # MC CONTROL
    # ========================================================

    if args.mc_control:

        input_file = (
            f"results/data/"
            f"Q_mc_{episodes}.npy"
        )

        print(
            "=== ÉVALUATION MC CONTROL ==="
        )

        print(
            f"Chargement : {input_file}"
        )

        q = load_q(
            input_file
        )

        print(
            "Nombre d'états dans Q :",
            len(q)
        )

        policy = extract_policy(
            q
        )

        print(
            "Nombre d'états dans la politique :",
            len(policy)
        )

        # ----------------------------------------------------
        # Politique apprise
        # ----------------------------------------------------

        for usable_ace in [0, 1]:

            matrix = create_policy_matrix(
                policy,
                usable_ace
            )

            if usable_ace == 0:

                title = (
                    "Politique MC apprise — "
                    "sans as utilisable"
                )

            else:

                title = (
                    "Politique MC apprise — "
                    "avec as utilisable"
                )

            display_policy_matrix(
                matrix,
                title
            )

        # ----------------------------------------------------
        # Politique de référence
        # ----------------------------------------------------

        for usable_ace in [0, 1]:

            reference_matrix = (
                create_reference_matrix(
                    usable_ace
                )
            )

            if usable_ace == 0:

                title = (
                    "Politique de référence — "
                    "sans as utilisable"
                )

            else:

                title = (
                    "Politique de référence — "
                    "avec as utilisable"
                )

            display_policy_matrix(
                reference_matrix,
                title
            )

        # ----------------------------------------------------
        # Comparaison
        # ----------------------------------------------------

        print()
        print(
            "=" * 70
        )

        print(
            "COMPARAISON MC CONTROL / RÉFÉRENCE"
        )

        print(
            "=" * 70
        )

        total_global = 0
        correct_global = 0

        for usable_ace in [0, 1]:

            total, correct, accuracy = (
                compare_policies(
                    policy,
                    reference_policy,
                    usable_ace
                )
            )

            if usable_ace == 1:

                label = (
                    "Avec as utilisable"
                )

            else:

                label = (
                    "Sans as utilisable"
                )

            print(
                f"{label} :"
            )

            print(
                f"  États comparés : {total}"
            )

            print(
                f"  Actions identiques : {correct}"
            )

            print(
                f"  Taux d'accord : "
                f"{accuracy:.2f}%"
            )

            total_global += total
            correct_global += correct

        if total_global > 0:

            global_accuracy = (
                100.0
                * correct_global
                / total_global
            )

        else:

            global_accuracy = 0.0

        print()

        print(
            f"Taux d'accord global : "
            f"{global_accuracy:.2f}%"
        )

        # ----------------------------------------------------
        # Figures
        # ----------------------------------------------------

        plot_policy_comparison(
            policy,
            usable_ace=0,
            episodes=episodes
        )

        plot_policy_comparison(
            policy,
            usable_ace=1,
            episodes=episodes
        )

        print()
        print(
            "=== ÉVALUATION TERMINÉE ==="
        )

        return

    # ========================================================
    # MC PREDICTION
    # ========================================================

    input_file = (
        f"results/data/"
        f"V_mc_{episodes}.npy"
    )

    print(
        "=== ÉVALUATION MONTE-CARLO ==="
    )

    print(
        f"Chargement : {input_file}"
    )

    values = load_values(
        input_file
    )

    print(
        "Nombre d'états chargés :",
        len(values)
    )

    plot_value_function(
        values,
        usable_ace=0,
        episodes=episodes
    )

    plot_value_function(
        values,
        usable_ace=1,
        episodes=episodes
    )

    print(
        "\n=== TERMINÉ ==="
    )


if __name__ == "__main__":
    main()