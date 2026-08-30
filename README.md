# Projet 3 : Monte-Carlo et Différences Temporelles sur Blackjack

**Module :** Apprentissage par renforcement  
**Année universitaire :** 2025-2026  
**Auteurs :** [FORGO Issouf] & [BAMBARA Gaihon Cheick Aboubacar]

## 1. Description

Ce projet implémente et compare des algorithmes d'apprentissage par renforcement sans modèle (Monte-Carlo et TD(0)) pour résoudre le problème du Blackjack. L'environnement utilisé est `Blackjack-v1` de Gymnasium avec les règles de Sutton & Barto (`sab=True`).

## 2. Installation

### Prérequis

- Python 3.10 ou supérieur
- Git

### Environnement virtuel et dépendances

```bash
# Création de l'environnement virtuel
python -m venv .venv

# Activation (Windows)
.venv\Scripts\activate

# Activation (Linux/Mac)
source .venv/bin/activate

# Installation des dépendances
pip install -r requirements.txt


### 3. Arborescence du projet
### projet3-blackjack-rl-Forgo-BambaraGaihon/
── README.md
├── requirements.txt
── .gitignore
├── play.py
├── src/
│   ├── __init__.py
│   ├── environment.py
│   ├── policies.py
│   ├── mc_prediction.py
│   ├── mc_control.py
│   ├── td_prediction.py
│   └── evaluation.py
├── experiments/
│   ├── exp_mc_prediction.py
│   ├── exp_mc_control.py
│   ├── exp_td_prediction.py
│   └── exp_comparison.py
├── tests/
│   ├── __init__.py
│   └── test_policy.py
└── results/
    ├── figures/
    └── data/


src/ : Contient les algorithmes fondamentaux (MC, TD(0), politiques).
experiments/ : Contient les scripts pour reproduire les expériences du rapport.
tests/ : Contient les tests unitaires pour valider les politiques.
results/figures/ : Contient les figures (.png) générées pour le rapport.
results/data/ : Contient les données et modèles sauvegardés (.npy).
play.py : Script de démonstration de l'agent entraîné.


### 4. Commandes pour reproduire les expériences
### Toutes les commandes sont à exécuter à la racine du projet, avec l'environnement virtuel activé.

### Prédiction Monte-Carlo (First-Visit)
# 100 épisodes (test rapide)
python -m src.mc_prediction --episodes 100

# 10 000 épisodes
python -m src.mc_prediction --episodes 10000

# 500 000 épisodes (référence pour la comparaison)
python -m src.mc_prediction --episodes 500000

Contrôle Monte-Carlo (ε-douce)
# 500 000 épisodes
python -m src.mc_control --episodes 500000

Prédiction TD(0)
# 100 épisodes avec alpha = 0.1
python -m src.td_prediction --episodes 100 --alpha 0.1

# 1000 épisodes avec alpha = 0.05
python -m src.td_prediction --episodes 1000 --alpha 0.05

Visualisation et Évaluation
# Générer les cartes de chaleur pour MC (10 000 épisodes)
python -m src.evaluation --episodes 10000

# Générer les cartes de chaleur pour MC (500 000 épisodes)
python -m src.evaluation --episodes 500000

# Comparer la politique MC apprise avec la référence
python -m src.evaluation --episodes 500000 --mc-control

Courbes de convergence et comparaison MC vs TD(0) (10 graines)

python -m experiments.exp_comparison

Démonstration de l'agent (Play)
python play.py

### 5. Résultats
Les figures générées pour le rapport se trouvent dans le dossier results/figures/. Les modèles (Q-tables et V-values) sont sauvegardés dans results/data/.
Figures principales
mc_10000_no_usable_ace.png / mc_10000_usable_ace.png : Fonction de valeur après 10 000 épisodes
mc_500000_no_usable_ace.png / mc_500000_usable_ace.png : Fonction de valeur après 500 000 épisodes
mc_control_500000_no_usable_ace_comparison.png / mc_control_500000_usable_ace_comparison.png : Comparaison politique MC vs référence
td_convergence_mse.png : Courbes de convergence TD(0) vers la référence MC (10 graines)
Données sauvegardées
V_mc_100.npy, V_mc_10000.npy, V_mc_500000.npy : Fonctions de valeur MC
V_td_100_alpha_0.1.npy : Fonction de valeur TD(0)
Q_mc_100.npy, Q_mc_500000.npy : Tables Q du contrôle MC
comparison_mc_td_test.npy, comparison_td_convergence.npy : Résultats de comparaison


### 6. Livrables
Rapport : rapport.pdf et rapport.docx
Code source : Dépôt Git complet avec historique des commits
Vidéo de démonstration : demo.mp4 (2-3 minutes)
Modèles sauvegardés : Fichiers .npy dans results/data/


### 7. Références
Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine Learning, 3, 9–44.
Singh, S. P., & Sutton, R. S. (1996). Reinforcement learning with replacing eligibility traces. Machine Learning, 22, 123–158.