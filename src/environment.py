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

    print("\n=== PARTIE TEST ===")

    state, info = env.reset()

    print("État initial :", state)

    terminated = False
    truncated = False
    step_number = 0

    while not (terminated or truncated):

        action = env.action_space.sample()

        next_state, reward, terminated, truncated, info = env.step(action)

        step_number += 1

        print(
            f"Étape {step_number} | "
            f"État={state} | "
            f"Action={action} | "
            f"Récompense={reward} | "
            f"Nouvel état={next_state} | "
            f"Terminé={terminated}"
        )

        state = next_state

    print("\n=== FIN DE LA PARTIE ===")
    print("Récompense finale :", reward)
    print("Terminated :", terminated)
    print("Truncated :", truncated)

    env.close()


if __name__ == "__main__":
    main()