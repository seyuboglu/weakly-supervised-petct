"""
noisy labeling functions to be used for weak supervision tasks.
"""
import re

import pet_ct.data.term_graphs as term_graphs
import pet_ct.data.match_fns as match_fns


class Labeler(object):
    """
    Labeler must always return a tuple of `labels` and a list of `edges`,
    denoting each label by its index in `labels` list.
    """
    def __init__(self, term_graph_class="", term_graph_args={}):
        """
        """
        self.term_graph = getattr(term_graphs, term_graph_class)(**term_graph_args)

    def _contains_term(self, c, name="", neg_regexes=[],
                       hit_code=2, miss_code=1, is_root=True):
        """returns hit_code if name is in s, else miss_code

        Uses the term regexes specified in term_graph to find positions. If
        term is the parent of another term, it will continue down the term
        graph until it finds a term or reaches a leaf. If there is a hit, the
        all node and all of its ancestors return hit_code, while the rest of the
        graph returns abstentions. If there is no hit, the entire tree returns
        the miss_code.

        We use strings in order to allow for the use of more elaborate phrasal
        regular expressions. Uses BFS when iterating through children.

        Args:
            s   (string) lowercase string containing all tokens of interest
            name    (string)    name of term of interest in term_graph.

        Returns:
            an integer reflecting metal's paradigm, where 0 is abstain, 1 is
            a negative classification, and 2 is a positive classification.
        """
        assert name in self.term_graph.name_to_term.keys(), (
            f"term graph does not contain {name}"
        )

        term = self.term_graph.name_to_term[name]
        labels = []
        names = []

        children = term.get("children", [])
        for i, child in enumerate(children):
            child_labels, child_names = self._contains_term(c, name=child, neg_regexes=neg_regexes,
                                                            miss_code=miss_code, hit_code=hit_code,
                                                            is_root=False)
            labels += child_labels
            names += child_names

        search_results = Labeler.search(c,
                                        regexes=term.get('match_res', []),
                                        neg_regexes=neg_regexes,
                                        hit_code=hit_code, miss_code=miss_code)

        for match_fn_dict in term.get("match_fns", []):
            fn = match_fn_dict['fn']
            fn_args = match_fn_dict['args']

            match = getattr(match_fns, fn)(c, **fn_args)
            if match:
                negated = False
                pos = match['start']
                start = max(c[:pos].rfind('.'), 0)
                end = pos + c[pos:].find('.') # returns -1 if not found
                for neg in neg_regexes:
                    pattern = re.compile(r"\b" + neg + r"\b")
                    # search for negations within the same sentence
                    if re.search(pattern, c[start:pos]) or re.search(pattern, c[pos:end]):
                        negated = True
                        break
                if not negated:
                    search_results.append(hit_code)
                else:
                    search_results.append(miss_code)
            else:
                search_results.append(miss_code)

        if len(search_results) > 0:
            labels += [max(search_results)]
            names.append(name)

        # no regexes matched in any of the nodes, set the whole tree to negative
        if is_root:
            names = {n:i for i, n in enumerate(names)}
            if hit_code not in labels:
                labels = [miss_code for label in labels]

        return labels, names

    def contains_term(self, c, names=[], neg_regexes=["no", "not", "without"],
                      hit_code=2, miss_code=1, aggregate=False):
        """
        """
        if type(names) != list:
            names = [names]

        res = []
        for name in names:
            labels, _ = self._contains_term(c, name=name, neg_regexes=neg_regexes,
                                            hit_code=hit_code, miss_code=miss_code,
                                            is_root=True)
            res += labels

        if aggregate:
            if res:
                res = [max(res)]
            else:
                res = [miss_code]


        # TODO: incorporate edges of G_{source}
        return res, []


    @staticmethod
    def contains_evidence(c, neg_regexes=["no", "not", "without"],
                          hit_code=2, miss_code=1, aggregate=False):
        """
        """
        regexes = [
            "evidence of metabolically active disease",
            "evidence of metabolically active",
            "evidence.*?malignancy",
            "evidence.*?disease",
            "significant evidence",
            "significant.*?activity",
            "significant.*?uptake",
            "significant",
            "definite evidence",
            "definite scintigraphic evidence"
            "scintigraphic evidence",
            "sign.*?malignancy",
            "abnormal.*?uptake",
            "hypermetabolic activity",
            "evidence",
            "disease",
            "activity",
            "uptake",
            "malignancy",
        ]
        labels = Labeler.search(c, regexes, neg_regexes=neg_regexes,
                                hit_code=hit_code, miss_code=miss_code)
        if aggregate:
            labels = [Labeler.aggregate_labels(labels, agg_type='max')]

        return labels, [] # always independent

    @staticmethod
    def search(c, regexes, neg_regexes=[], hit_code=2, miss_code=1):
        """
        """
        labels = []
        for regex in regexes:
            pattern = re.compile(r"\b" + regex + r"\b")
            res = re.search(pattern, c)
            if res:
                pos = res.start()
                start = max(c[:pos].rfind('.'), 0)
                end = pos + c[pos:].find('.') # returns -1 if not found
                negated = False
                for neg in neg_regexes:
                    pattern = re.compile(r"\b" + neg + r"\b")
                    # search for negations within the same sentence
                    if re.search(pattern, c[start:pos]) or re.search(pattern, c[pos:end]):
                        negated = True
                        labels.append(miss_code)
                        break
                if not negated:
                    labels.append(hit_code)
            else:
                labels.append(0)

        return labels

    @staticmethod
    def extract_from_sources_df(series, cols=[], agg_type=None):
        """
        Extracts labels directly from a dataframe series (row).
        """
        labels = []
        for col in cols:
            labels.append(series[col])
        if agg_type:
            labels = [Labeler.aggregate_labels(labels, agg_type=agg_type)]
        return labels, []

    @staticmethod
    def aggregate_labels(labels, agg_type='max'):
        """
        Returns a single (representative) value for a set of labels.

        Args:
            labels (list) list of prospective labels for given sources
            agg_type (string)
        """
        if agg_type == 'max':
            return max(labels)
        elif agg_type == 'majority':
            value, count = Counter(labels).most_common()
            return value
        else:
            raise ValueError(f"agg_type {agg_type} not recognized.")



