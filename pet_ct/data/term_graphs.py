"""
"""
import logging
import os
import re
import json
from collections import OrderedDict, Counter, deque
from itertools import combinations

import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph
from networkx.algorithms.cycles import find_cycle
from networkx.readwrite.json_graph.cytoscape import cytoscape_data


import pet_ct.data.match_fns as match_fns


class TermGraph(object):

    def __init__(self, graph_dir):
        """
        A directed acyclic graph representing a hierarchy of terms. Useful for finding
        matches with
        """
        self.graph_dir = graph_dir
        terms = json.load(open(os.path.join(graph_dir, 'terms.json'), 'r'),
                               object_pairs_hook=OrderedDict)["terms"]
        self.name_to_term = {term["name"]: term for term in terms}
        self.term_names = list(self.name_to_term.keys())
        self.build_graph()

    def build_graph(self):
        """
        Calibrates the term graph:
        For each term:
            1. Removes all nonexistent children and parents
            2. Adds the term as a child for all parents
            3. Adds the term as a parent for all children
        """
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(self.name_to_term.keys())

        for name, term in self.name_to_term.items():
            for child_name in term.setdefault("children", []):
                if child_name in self.name_to_term:
                    self.dag.add_edge(name, child_name)
                else:
                    # remove nonexistent children
                    logging.info(f"Removing child {child_name} from {name}.")
                    children = term.setdefault("children", [])
                    children.remove(child_name)
        if not is_directed_acyclic_graph(self.dag):
            cycle = find_cycle(self.dag)
            raise ValueError(f"Found cycle: {cycle}")

    def save(self, path=None):
        """
        If path is none, saves to "terms.json" in graph_dir.
        """
        if path is None:
            path = os.path.join(self.graph_dir, 'terms.json')
        json.dump({"terms": list(self.name_to_term.values())},
                  open(path, 'w'), indent=4)

    def get_descendants(self, term_name):
        """
        """
        return nx.descendants(self.dag, term_name)

    def get_ancestors(self, term_name):
        """
        """
        return nx.ancestors(self.dag, term_name)

    def add(self, name, regexes=[], children=[], parents=[]):
        """
        """
        if name in self.name_to_term:
            raise ValueError(f"Name '{name}' already exists.")

        term = {
            "name": name,
            "match_res": regexes,
            "children": children,
        }
        for parent in parents:
            parents["children"].add(name)

        self.name_to_term[name] = term
        self.build_graph()

    def remove(self, name):
        """
        """
        del self.name_to_term[name]
        self.build_graph()

    def rename(self, old_name, new_name):
        """
        """
        if new_name in self.name_to_term:
            raise ValueError(f"Name '{new_name}' already exists.")

        term = self.name_to_term[old_name]

        term["name"] = new_name
        self.name_to_term[new_name] = term
        del self.name_to_term[old_name]

        self.build_graph()

    def subdivide(self, term_name, subdivisions):
        term = self.name_to_term[term_name]
        regexes = []
        for subdivision in subdivisions:
            regexes = [f"{subdivision}[^.]*{regex}"
                       for regex in term["regexes"]]
            self.add(**{
                "name": subdivision + "_" + term_name,
                "match_res": regexes,
                "children": [],
                "parents": [term_name]
            })
        self.build_graph()

    def __getitem__(self, name):
        """
        """
        return self.name_to_term[name]

    def match_string(self, string,
                     remove_submentions=True,
                     remove_descendant_submentions=True):
        """
        Finds all
        Removes all sub mentions.
        """
        matches = []
        for term_name, term in self.name_to_term.items():
            for match_fn_dict in term.get("match_fns", []):
                fn = match_fn_dict['fn']
                fn_args = match_fn_dict['args']
                rem_string = string[:]
                offset = 0
                while True:
                    match = getattr(match_fns, fn)(rem_string, **fn_args)
                    if match:
                        start_idx = match['start']
                        end_idx = match['end']
                        matches.append({
                            "start": offset + start_idx,
                            "end": offset + end_idx,
                            "term_name": term_name,
                            "text": string,
                            "pattern": f'{fn}({fn_args})',
                        })
                        offset = offset + end_idx
                        rem_string = string[offset:]
                    else:
                        break

            for match_re in term.get("match_res", []):
                match_re = re.compile(r"\b" + match_re + r"\b")
                rem_string = string[:]
                offset = 0
                while True:
                    match = re.search(match_re, rem_string)
                    if match:
                        start_idx = match.start()
                        end_idx = match.end()
                        matches.append({
                            "start": offset + start_idx,
                            "end": offset + end_idx,
                            "term_name": term_name,
                            "text": string,
                            "pattern": match_re,
                        })
                        offset = offset + end_idx
                        rem_string = string[offset:]
                    else:
                        break

        if remove_submentions:
            # remove all sub mentions
            def is_submention(mention, submention):
                return (submention["start"] >= mention["start"] and
                        submention["end"] <= mention["end"] and
                        (mention["term_name"] == submention["term_name"] or
                         (mention["term_name"] in nx.descendants(self.dag,
                                                                 submention["term_name"]) and
                          remove_descendant_submentions)))

            to_remove = []
            for match_a, match_b in combinations(matches, 2):

                if is_submention(match_a, match_b):
                    to_remove.append(match_b)
                elif is_submention(match_b, match_a):
                    to_remove.append(match_a)

            for match in to_remove:
                if match in matches:
                    matches.remove(match)

        return matches

    def to_cytoscape(self, path=None):
        """
        """
        if path is None:
            path = os.path.join(self.graph_dir, "cytoscape.json")
        data = cytoscape_data(self.dag)

        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def from_cytoscape(cls, cytoscape_path, graph_dir):
        """
        Loads network from cytoscape but other term data from graph_dir
        """
        attrs = dict(name='name', ident='id')
        with open(cytoscape_path, "r") as f:
            data = json.load(f)

        term_graph = cls(graph_dir)

        name = attrs["name"]
        ident = attrs["ident"]
        graph = nx.DiGraph()
        #if not data.get('directed'):
        #    raise ValueError("Cytoscape graph must be directed!")

        # add nodes
        id_to_name = {}
        for d in data["elements"]["nodes"]:
            node_name = d["data"]["value"]
            node_id = d["data"]["id"]
            graph.add_node(node_name)
            id_to_name[node_id] = node_name

        # add edges
        for d in data["elements"]["edges"]:
            source = id_to_name[d["data"].pop("source")]
            target = id_to_name[d["data"].pop("target")]
            graph.add_edge(source, target)

        if not is_directed_acyclic_graph(graph):
            cycle = find_cycle(graph)
            raise ValueError(f"Found cycle: {cycle}")

        name_to_term = {}
        for name in graph.nodes():
            if name in term_graph.name_to_term:
                term = term_graph.name_to_term[name].copy()
            else:
                term = {"match_res": [],
                        "match_fns": [],
                        "children": [],
                        "name": name}
            term["children"] = list(graph.successors(name))
            name_to_term[name] = term

        term_graph.name_to_term = name_to_term
        term_graph.dag = graph

        return term_graph





