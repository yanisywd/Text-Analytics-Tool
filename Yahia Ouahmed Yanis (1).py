from xcs import XCSAlgorithm
from xcs.scenarios import Scenario
import random

# Définir la classe GridWorldScenario avant son utilisation
class GridWorldScenario(Scenario):
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.position = (0, 0)  # Position initiale du robot (en haut à gauche)
        self.grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]  # Grille vide
        # Ajout d'obstacles (1 = obstacle, 0 = cellule libre)
        self.grid[1][2] = 1  # Exemple d'obstacle
        self.grid[2][2] = 1  # Exemple d'obstacle
        self.grid[3][1] = 1  # Exemple d'obstacle

    def get_possible_actions(self):
        # Définir les actions possibles: haut, bas, gauche, droite
        return ['↑', '↓', '←', '→']

    def sense(self):
        # Sensibilité : retourner l'état actuel de l'environnement
        x, y = self.position
        return [self.grid[x][y], x, y]  # État de la cellule du robot (0 ou 1 pour obstacle)

    def execute(self, action):
        # Exécuter l'action et mettre à jour la position du robot
        new_position = self.get_new_position(action)
        if self.is_out_of_bounds(new_position) or self.grid[new_position[0]][new_position[1]] == 1:
            return -1  # Pénalité pour avoir heurté un obstacle ou être sorti de la grille
        self.position = new_position
        return 1  # Récompense positive pour une action valide

    def is_dynamic(self):
        # L'environnement n'est pas dynamique (il ne change pas durant l'exécution)
        return False

    def reset(self):
        # Réinitialiser la position du robot et l'environnement
        self.position = (0, 0)  # Réinitialiser la position
        return self.sense()  # Retourner l'état initial

    def more(self):
        # Définir si d'autres actions doivent être effectuées ou non
        return False  # On ne poursuit pas d'autres actions pour cet exemple

    def get_new_position(self, action):
        # Calculer la nouvelle position après avoir effectué une action
        x, y = self.position
        if action == '↑':  # Haut
            x -= 1
        elif action == '↓':  # Bas  
            x += 1
        elif action == '←':  # Gauche
            y -= 1
        elif action == '→':  # Droite
            y += 1
        return (x, y)

    def is_out_of_bounds(self, position):
        # Vérifier si la position est hors de la grille
        x, y = position
        return x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size

# Le reste du code pour le modèle XCS reste le même.

class GridWorldModel:
    def __init__(self):
        self.scenario = GridWorldScenario()
        self.algorithm = XCSAlgorithm()
        self.model = self.algorithm.new_model(self.scenario)
        self.model.exploration_probability = 0.1  # Exploration vs exploitation
        self.model.discount_factor = 0  # Aucun effet du futur sur la récompense actuelle
        self.model.do_ga_subsumption = True  # Subsumption GA pour simplifier le modèle
        self.model.do_action_set_subsumption = True  # Subsumption pour l'ensemble des actions

    def train(self, cycles=1000):
        # Entraîner le modèle (en utilisant `cycles` directement)
        for cycle in range(cycles):
            self.model.run(self.scenario, learn=True)
            print(f"Cycle {cycle+1}/{cycles} terminé")

    def test(self):
        # Tester le modèle et simuler le déplacement du robot
        self.scenario.position = (0, 0)  # Réinitialiser la position initiale
        total_reward = 0
        for _ in range(20):  # Effectuer 20 mouvements
            action = random.choice(self.scenario.get_possible_actions())  # Choisir une action aléatoire
            print(f"Action choisie: {action}")
            reward = self.scenario.execute(action)
            total_reward += reward
            print(f"Récompense: {reward}")
            print(f"Position du robot: {self.scenario.position}")
            print(f"Grille: {self.scenario.grid}")
        print(f"Récompense totale: {total_reward}")

# Utilisation du modèle
if __name__ == "__main__":
    model = GridWorldModel()
    model.train(cycles=1000)  # Entraînement sur 1000 cycles
    model.test()  # Tester le robot après l'entraînement
