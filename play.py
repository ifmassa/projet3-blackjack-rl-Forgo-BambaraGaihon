import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

def get_action(state, q_table):
    if state not in q_table:
        return 0  # Stick par défaut
    actions = q_table[state]
    return max(actions, key=actions.get) if isinstance(actions, dict) else int(np.argmax(actions))

def play_episode(env, policy=None, is_random=False):
    state, info = env.reset()
    terminated, truncated = False, False
    print(f"Départ : Somme={state[0]}, Croupier={state[1]}, As={bool(state[2])}")
    
    while not (terminated or truncated):
        action = env.action_space.sample() if is_random else get_action(state, policy)
        action_name = "HIT (1)" if action == 1 else "STICK (0)"
        print(f"  -> Action : {action_name}")
        time.sleep(0.8)  # Ralenti pour que la vidéo soit lisible
        state, reward, terminated, truncated, info = env.step(action)
        
    resultat = 'Gagné' if reward == 1 else 'Perdu' if reward == -1 else 'Égalité'
    print(f"  -> Résultat : {resultat} (Récompense: {reward})\n")

def main():
    print("=== DÉMONSTRATION BLACKJACK RL ===\n")
    env = gym.make("Blackjack-v1", sab=True)
    
    print("--- 1. COMPORTEMENT INITIAL (Agent Aléatoire) ---")
    play_episode(env, is_random=True)
    
    print("--- 2. COMPORTEMENT FINAL (Agent Entraîné MC 500k) ---")
    q_table = np.load("results/data/Q_mc_500000.npy", allow_pickle=True).item()
    print(f"Modèle chargé : {len(q_table)} états appris.\n")
    for i in range(3):
        print(f"Partie {i+1}/3 :")
        play_episode(env, policy=q_table)
        
    print("--- 3. COURBES D'APPRENTISSAGE (à filmer maintenant) ---")
    img = plt.imread("results/figures/td_convergence_mse.png")
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Courbe de convergence MSE (Travail n°5)")
    plt.show()
    
    env.close()
    print("Démonstration terminée.")

if __name__ == "__main__":
    main()