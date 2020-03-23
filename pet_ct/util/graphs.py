"""
A minimal import of source from the pgmpy library.

Source: https://github.com/pgmpy/pgmpy/
"""
import itertools
from collections import defaultdict

import networkx as nx
import numpy as np


class TriangleGraph(nx.Graph):
    """
    Class derived from the MarkovModel class in pgmpy.

    Used primarily for its triangulate function. Updated in order to work with
    the most recent version of networkx.

    Parameters
    ----------
    data : input graph
        Data to initialize graph.  If data=None (default) an empty
        graph is created.  The data can be an edge list, or any
        NetworkX graph object.
    """

    def __init__(self, ebunch=None, cardinalities={}):
        super(TriangleGraph, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.cardinalities = cardinalities

    def get_cardinality(self, node=None):
        """
        New function that obviates Factor class used in original function.
        """
        if node:
            return self.cardinalities[node]
        else:
            return self.cardinalities

    def is_triangulated(self):
        """
        """
        return nx.is_chordal(self)

    def triangulate(self, heuristic='H6', order=None, inplace=False):
        """
        Triangulate the graph.
        If order of deletion is given heuristic algorithm will not be used.
        Parameters
        ----------
        heuristic: H1 | H2 | H3 | H4 | H5 | H6
            The heuristic algorithm to use to decide the deletion order of
            the variables to compute the triangulated graph.
            Let X be the set of variables and X(i) denotes the i-th variable.
            * S(i) - The size of the clique created by deleting the variable.
            * E(i) - Cardinality of variable X(i).
            * M(i) - Maximum size of cliques given by X(i) and its adjacent nodes.
            * C(i) - Sum of size of cliques given by X(i) and its adjacent nodes.
            The heuristic algorithm decide the deletion order if this way:
            * H1 - Delete the variable with minimal S(i).
            * H2 - Delete the variable with minimal S(i)/E(i).
            * H3 - Delete the variable with minimal S(i) - M(i).
            * H4 - Delete the variable with minimal S(i) - C(i).
            * H5 - Delete the variable with minimal S(i)/M(i).
            * H6 - Delete the variable with minimal S(i)/C(i).
        order: list, tuple (array-like)
            The order of deletion of the variables to compute the triagulated
            graph. If order is given heuristic algorithm will not be used.
        inplace: True | False
            if inplace is true then adds the edges to the object from
            which it is called else returns a new object.
        Reference
        ---------
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.56.3607
        """
        if self.is_triangulated():
            if inplace:
                return
            else:
                return self

        graph_copy = nx.Graph(self.edges)
        edge_set = set()

        def _find_common_cliques(cliques_list):
            """
            Finds the common cliques among the given set of cliques for
            corresponding node.
            """
            common = set([tuple(x) for x in cliques_list[0]])
            for i in range(1, len(cliques_list)):
                common = common & set([tuple(x) for x in cliques_list[i]])
            return list(common)

        def _find_size_of_clique(clique, cardinalities):
            """
            Computes the size of a clique.
            Size of a clique is defined as product of cardinalities of all the
            nodes present in the clique.
            """
            return list(map(lambda x: np.prod([cardinalities[node] for node in x]),
                            clique))

        def _get_cliques_dict(node):
            """
            Returns a dictionary in the form of {node: cliques_formed} of the
            node along with its neighboring nodes.
            clique_dict_removed would be containing the cliques created
            after deletion of the node
            clique_dict_node would be containing the cliques created before
            deletion of the node
            """
            graph_working_copy = nx.Graph(graph_copy.edges)
            neighbors = list(graph_working_copy.neighbors(node))
            graph_working_copy.add_edges_from(itertools.combinations(neighbors, 2))
            clique_dict = nx.cliques_containing_node(graph_working_copy,
                                                     nodes=([node] + neighbors))
            graph_working_copy.remove_node(node)
            clique_dict_removed = nx.cliques_containing_node(graph_working_copy,
                                                             nodes=neighbors)
            return clique_dict, clique_dict_removed

        if not order:
            order = []

            cardinalities = self.get_cardinality()
            for index in range(self.number_of_nodes()):
                # S represents the size of clique created by deleting the
                # node from the graph
                S = {}
                # M represents the size of maximum size of cliques given by
                # the node and its adjacent node
                M = {}
                # C represents the sum of size of the cliques created by the
                # node and its adjacent node
                C = {}
                for node in set(graph_copy.nodes()) - set(order):
                    clique_dict, clique_dict_removed = _get_cliques_dict(node)
                    S[node] = _find_size_of_clique(
                        _find_common_cliques(list(clique_dict_removed.values())),
                        cardinalities
                    )[0]
                    common_clique_size = _find_size_of_clique(
                        _find_common_cliques(list(clique_dict.values())),
                        cardinalities
                    )
                    M[node] = np.max(common_clique_size)
                    C[node] = np.sum(common_clique_size)

                if heuristic == 'H1':
                    node_to_delete = min(S, key=S.get)

                elif heuristic == 'H2':
                    S_by_E = {key: S[key] / cardinalities[key] for key in S}
                    node_to_delete = min(S_by_E, key=S_by_E.get)

                elif heuristic == 'H3':
                    S_minus_M = {key: S[key] - M[key] for key in S}
                    node_to_delete = min(S_minus_M, key=S_minus_M.get)

                elif heuristic == 'H4':
                    S_minus_C = {key: S[key] - C[key] for key in S}
                    node_to_delete = min(S_minus_C, key=S_minus_C.get)

                elif heuristic == 'H5':
                    S_by_M = {key: S[key] / M[key] for key in S}
                    node_to_delete = min(S_by_M, key=S_by_M.get)

                else:
                    S_by_C = {key: S[key] / C[key] for key in S}
                    node_to_delete = min(S_by_C, key=S_by_C.get)

                order.append(node_to_delete)

        graph_copy = nx.Graph(self.edges())
        for node in order:
            for edge in itertools.combinations(graph_copy.neighbors(node), 2):
                graph_copy.add_edge(edge[0], edge[1])
                edge_set.add(edge)
            graph_copy.remove_node(node)

        if inplace:
            for edge in edge_set:
                self.add_edge(edge[0], edge[1])
            return self

        else:
            graph_copy = TriangleGraph(self.edges())
            for edge in edge_set:
                graph_copy.add_edge(edge[0], edge[1])
            return graph_copy