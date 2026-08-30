STICK = 0
HIT = 1


def stick_on_20_policy(state):
    """
    Politique fixe :
    - Stick si la somme du joueur est >= 20
    - Hit sinon

    Paramètre
    ---------
    state : tuple
        État Blackjack sous la forme
        (player_sum, dealer_card, usable_ace)

    Retour
    ------
    int
        0 pour Stick
        1 pour Hit
    """
    player_sum, dealer_card, usable_ace = state

    if player_sum >= 20:
        return STICK

    return HIT