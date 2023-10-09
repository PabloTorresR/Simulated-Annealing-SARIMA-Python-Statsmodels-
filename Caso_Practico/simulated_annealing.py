import random
import math
from typing import List, Any
from dataclasses import dataclass


def print_solution_history(history_list):
    for entry in history_list:
        print(f"Solution: {entry.solution}, Cost: {entry.cost}")


@dataclass
class SolutionHistoryEntry:
    solution: Any
    cost: float


class SimulatedAnnealing:
    def __init__(self, initial_solution, temperature, cooling_rate, max_iterations):
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.solution_history: List[SolutionHistoryEntry] = []

    def __get_acceptance_probability__(self, old_cost, new_cost, temperature):
        if new_cost < old_cost:
            return 1.0
        return math.exp((old_cost - new_cost) / temperature)

    def __update_history__(self, new_entry: SolutionHistoryEntry):
        self.solution_history.append(new_entry)

    def __get_cost_from_cache__(self, solution):
        for entry in self.solution_history:
            if entry.solution == solution:
                return entry.cost

    def iteration(self, cost_function):
        # Generar un nuevo vecino solución
        neighbor_solution = self.current_solution.generate_neighbor()
        # Calcular el coste de la solución actual y la del vecino, intentar siempre mirar el cache
        current_cost = self.__get_cost_from_cache__(
            self.current_solution
        ) or cost_function(self.current_solution)

        neighbor_cost = self.__get_cost_from_cache__(
            neighbor_solution
        ) or cost_function(neighbor_solution)

        # Si la solución del vecino es mejor o se acepta con cierta probabilidad, actualizar la solución actual
        if (
            neighbor_cost < current_cost
            or random.random()
            < self.__get_acceptance_probability__(
                current_cost, neighbor_cost, self.temperature
            )
        ):
            self.current_solution = neighbor_solution

        # Actualizar la mejor solución con la actual si la actual es la mejor
        self.__update_history__(SolutionHistoryEntry(neighbor_solution, neighbor_cost))
        if neighbor_cost < cost_function(self.best_solution):
            self.best_solution = neighbor_solution
        # Reducir temperatura
        self.temperature *= self.cooling_rate

    def optimize(self, cost_function):
        iteration = 0
        while iteration < self.max_iterations:
            self.iteration(cost_function)
            iteration += 1

        return self.best_solution

    def get_history(self) -> List[SolutionHistoryEntry]:
        return self.solution_history
