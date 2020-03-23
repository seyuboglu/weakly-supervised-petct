"""
"""
import itertools
from collections import defaultdict

import numpy as np
import networkx as nx

from metal.multitask.task_graph import TaskHierarchy, TaskGraph


class TaskHierarchyFlex(TaskHierarchy):
    """A TaskHierarchy graph that allows for toggling of mutex relationships.

    In addition to edges, this task graph allows the caller to specify the
    tasks that must be considered mutually exclusive.

    Args:
        edges: A list of (a,b) tuples meaning a is a parent of b in a tree.
        cardinalities: A t-length list of integers corresponding to the
            cardinalities of each task.
        mutex_tasks: A list of integers corresponding to those tasks with a
            mutex lock. A task with a mutex may only choose one class at a time.
    """

    def __init__(self, cardinalities=[2], edges=[], mutex_tasks=[]):
        if len(cardinalities) == 1:
            self.mutex_tasks = [0]
        else:
            self.mutex_tasks = mutex_tasks
        super().__init__(cardinalities=cardinalities, edges=edges)

    def error_check(self, root):
        for task, k in enumerate(self.K):
            if task == root:
                if root not in self.mutex_tasks:
                    assert k == 1, "non-mutex root must have k=1"
            else:
                if task not in self.mutex_tasks and task not in self.leaf_nodes:
                    assert k == 2, "non-root, non-mutex, non-leaf task must have k=2"


    def get_local_assignments(self, node, children, child_selected=None):
        if len(children) == 0:
            return []
        if child_selected:
            default = [self.K[child] for child in children]
            assignments = []
            for i in range(1, self.K[child_selected]):
                assignment = list(default)
                assignment[children.index(child_selected)] = i
                assignments.append(assignment)
        else:
            children_k = [range(1, self.K[node]) for node in children]
            assignments = list(itertools.product(*children_k))
        res = []
        for assignment in assignments:
            assignment_dict = {}
            for i, child in enumerate(children):
                assignment_dict[child] = assignment[i]
            res.append(assignment_dict)
        return res


    def feasible_set(self):
        """Finds all feasible assignments for a mutex variant tree.

        A feasible set is determined by first generating all possible
        permutations as constrained by parent node type. A BFS tree traversal
        is then used in order to generate all assignments from these
        permutations.

        Due to the recursive nature of the algorithm, all computations must be
        frontloaded, precluding the use of lazy evaluation.

        Algorithm for mutex-aware feasible set construction created in
        collaboration with Sophia Kivelson (skivelso).
        """
        root = [i for i in self.G.nodes() if self.G.in_degree(i) == 0][0]
        self.error_check(root)
        # pre-load all valid local assignments
        local_assignments = defaultdict(dict)
        for node in self.G.nodes():
            children = self.children[node]
            if node in self.mutex_tasks:
                for child_selected in children:
                    local_assignments[node][child_selected] = (
                        self.get_local_assignments(node, children, child_selected))
            else:
                local_assignments[node][1] = (
                    self.get_local_assignments(node, children))

        fs = []
        layers = nx.bfs_tree(self.G, root)
        traversal_order = [0] + [dst for src, dst in layers.edges()]
        traversal_map = {node: idx for idx, node in enumerate(traversal_order)}
        def recurse(vec, traversal_idx):
            """Recursively constructs an assignment vector.

            Begins with some class assignment of the root node. Leverages order
            of BFS traversal in order to assign values to node children.

            Args:
                vec (list)  partial assignment. Assumes the nodes are in BFS
                    order. Mapped to final nodes in base case.
                traversal_idx (int) tracks of node of interest in traversal
                    order. Nodes may be assigned faster than this increments.
            """
            node = traversal_order[traversal_idx]
            # Base case 1: our vector is fully assigned -> add to assignments
            if len(vec) == self.t:
                final = np.zeros(self.t, dtype=int)
                for i in range(self.t):
                    final[traversal_order[i]] = vec[i]
                fs.append(final)
            # Base case 2: we are currently at a leaf node -> move on
            elif node in self.leaf_nodes:
                recurse(vec, traversal_idx + 1)
            # Recursive case 1: we are currently mutexed -> make children mutexed as well)
            elif node != root and vec[traversal_idx] == self.K[node]:
                children = self.children[node]
                vec_copy = list(vec)
                for child in children:
                    vec_copy.append(self.K[child])
                recurse(vec_copy, traversal_idx + 1)
            # Recursive case 2: we are not mutexed -> let all children flourish
            else:
                if node in self.mutex_tasks:
                    child_idx = vec[traversal_idx] - 1 # assignment of parent
                    child_selected = self.children[node][child_idx]
                else:
                    child_selected = 1
                for segment in local_assignments[node][child_selected]:
                    vec_copy = list(vec)
                    for child, local_assignment in sorted(segment.items()):
                        vec_copy.append(local_assignment)
                    recurse(vec_copy, traversal_idx + 1)

        for root_assignment in range(1, self.K[root]+1):
            recurse([root_assignment], 0)
        for vec in fs:
            yield vec