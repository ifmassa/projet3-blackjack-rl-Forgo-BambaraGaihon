import gymnasium as gym


def create_environment():
    """
    Crée l'environnement Blackjack-v1 avec sab=True.
    """
    return gym.make("Blackjack-v1", sab=True)


def main():
    env = create_environment()

    print("=== ENVIRONNEMENT ===")
    print("Nom : Blackjack-v1")
    print("Règles Sutton & Barto : sab=True")

    print("\n=== ESPACES ===")
    print("Observation :", env.observation_space)
    print("Action :", env.action_space)

    print("\n=== CONVENTION DES ACTIONS ===")
    print("0 = Stick")
    print("1 = Hit")

    state, info = env.reset()

    print("\n=== ÉTAT INITIAL ===")
    print("État :", state)
    print("Info :", info)

    env.close()


if __name__ == "__main__":
    main()