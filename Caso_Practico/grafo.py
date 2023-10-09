import networkx as nx
from networkx.classes import Graph
from dataclasses import dataclass
import random
from typing import Tuple


@dataclass
class Coeficients:
    p: int
    d: int
    q: int
    P: int
    D: int
    Q: int
    m: int

    def to_tuple(self) -> Tuple[int, int, int, int, int, int, int]:
        return (self.p, self.d, self.q, self.P, self.D, self.Q, self.m)

    @classmethod
    def from_tuple(cls, coef_tuple: Tuple[int, int, int, int, int, int, int]):
        """Crea el objeto Coeficients desde un tuple."""
        return cls(*coef_tuple)


class SolutionGraph:
    def __init__(self, max_coefs: Coeficients, m: int):
        self.max_coefs = max_coefs
        self.m = m
        self.graph = self.__generate_graph__()

    def __generate_graph__(self) -> Graph:
        G = nx.Graph()
        for p in range(self.max_coefs.p):
            for q in range(self.max_coefs.q):
                for d in range(
                    max(self.max_coefs.d - 1, 0), self.max_coefs.d + 2
                ):  # Solo queremos uno por arriba y abajo (sin negativos)
                    for P in range(self.max_coefs.P):
                        for Q in range(self.max_coefs.Q):
                            for D in range(
                                max(self.max_coefs.D - 1, 0),
                                self.max_coefs.D
                                + 2,  # Solo queremos uno por arriba y abajo (sin negativos)
                            ):
                                # Cada nodo se representa como una tupla de parámetros
                                node = (p, d, q, P, D, Q, self.m)
                                G.add_node(node)
        for node in G.nodes():
            for i in range(len(node)):
                for diff in [
                    -1,
                    1,
                ]:  # va a sumar y restar uno a cada parámetro para calcular quienes son los vecinos
                    new_params = list(node)
                    new_params[i] += diff
                    neighbor = tuple(new_params)
                    if neighbor in G.nodes() and not G.has_edge(node, neighbor):
                        G.add_edge(node, neighbor)

        return G

    def get_random_node(self) -> Coeficients:
        """Devuelve un nodo aleatorio de la red"""
        random_node = random.choice(list(self.graph.nodes()))
        return Coeficients.from_tuple(random_node)

    def get_random_neighbor(self, node: Coeficients) -> Coeficients:
        """Devuelve un vecino aleatorio de la red a un nodo de entrada"""
        neighbors = list(self.graph.neighbors(node.to_tuple()))
        if neighbors:
            random_neighbor = random.choice(neighbors)
            return Coeficients.from_tuple(random_neighbor)
        else:
            # Si el nodo no tiene vecinos, se devuelve el propio nodo como resultado
            return node


@dataclass
class Solution(Coeficients):
    def __init__(self, coeficients: Coeficients, graph: SolutionGraph):
        super().__init__(
            p=coeficients.p,
            q=coeficients.q,
            d=coeficients.d,
            P=coeficients.P,
            Q=coeficients.Q,
            D=coeficients.D,
            m=coeficients.m,
        )
        self.graph = graph

    def generate_neighbor(self) -> Coeficients:
        """Devuelve un vecino dentro de un grafo"""
        return Solution(self.graph.get_random_neighbor(self), self.graph)
