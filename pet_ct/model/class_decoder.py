"""
"""
import os
from collections import deque
import warnings
import json
import math
import logging

import torch
import torch.nn as nn
import networkx as nx

from pet_ct.data import task_graphs
from pet_ct.util.util import expand_to_list
from pet_ct.model.i3d import Unit3Dpy, Mixed
import pet_ct.model.modules as modules


class ClassDecoder(nn.Module):
    """
    """
    def __init__(self, num_classes, encoding_size=1024, dropout_prob=0):
        """
        """
        super().__init__()

        self.classifier = nn.Linear(in_features=encoding_size,
                                    out_features=num_classes)
        self.use_dropout = dropout_prob > 0
        self.dropout = nn.Dropout(dropout_prob)

    def aggregate(self, encoding):
        """
        """
        raise NotImplementedError("Not implemented, use LinearAttClassDecoder instead.")

    def classify(self, emb):
        """
        """
        return self.classifier(emb)

    def encode(self, encoding):
        """
        Additional convolutional layers.
        """
        return encoding

    def forward(self, encoding):
        """
        """
        encoding = self.encode(encoding)
        emb = self.aggregate(encoding)
        if self.use_dropout:
            emb = self.dropout(emb)

        out = self.classify(emb)
        return out


class AvgClassDecoder(ClassDecoder):
    """
    """
    def __init__(self, num_classes, encoding_size=1024, num_heads=None):
        """
        """
        super().__init__(num_classes, encoding_size=encoding_size)

        self.avg_pool = torch.nn.AvgPool3d((1, 7, 7), (1, 1, 1))

    def aggregate(self, encoding):
        """
        """
        encoding = self.avg_pool(encoding)
        encoding = encoding.squeeze(-1)
        encoding = encoding.squeeze(-1)

        return encoding.mean(dim=-1)

class AttClassDecoder(ClassDecoder):
    """
    """
    def __init__(self, num_classes, encoding_size=1024,
                 encoder_configs=[],region_aware=False):
        """
        """
        super().__init__(num_classes, encoding_size)

        self.score_projection = nn.Linear(in_features=encoding_size,
                                          out_features=1)
        self.key_projection = nn.Linear(in_features=encoding_size,
                                        out_features=encoding_size)
        self.value_projection = nn.Linear(in_features=encoding_size,
                                          out_features=encoding_size)

        encoder_modules = []
        for encoder_config in encoder_configs:
            module = getattr(modules,
                             encoder_config["class"])(**encoder_config["args"])
            encoder_modules.append(module)

        self.task_encoder = nn.Sequential(*encoder_modules)

        self.region_aware = region_aware

    def encode(self, encoding):
        return self.task_encoder(encoding)

    def aggregate(self, encoding):
        """
        """
        # flatten encoding
        batch_size, encoding_size, length, height, width = encoding.shape
        encoding = encoding.view(batch_size, encoding_size, -1).permute(0, 2, 1)

        values = self.value_projection(encoding)
        keys = self.key_projection(encoding)
        scores = self.score_projection(keys)
        scores = torch.nn.functional.softmax(scores, dim=1)
        scores = scores.permute(0, 2, 1)  # batch, heads, dim

        # contexts = torch.matmul(scores.unsqueeze(2), values)
        contexts = torch.matmul(scores, values)
        contexts_flat = contexts.squeeze(2).view(batch_size, -1)
        if self.region_aware:
            scores = scores.view(batch_size,
                                 length,
                                 height,
                                 width)
            return {
                'out': contexts_flat,
                'attn_scores': scores
            }

        else:
            return contexts_flat

    def classify(self, emb):
        """
        """
        if type(emb) == dict:
            return {
                'out': self.classifier(emb['out']),
                'attn_scores': emb['attn_scores']
            }
        else:
            return self.classifier(emb)


class MultiAttClassDecoder(ClassDecoder):
    """
    """
    def __init__(self, num_classes, encoding_size=1024,
                 num_heads=8, num_mixed_conv=0, dropout_prob=0, 
                 region_aware=False, collect_embeddings=False):
        """
        """
        super().__init__(num_classes, encoding_size, dropout_prob=dropout_prob)

        if encoding_size % num_heads != 0:
            raise ValueError("The encoding size is not a multiple of num heads.")
        self.num_heads = num_heads
        self.head_size = int(encoding_size / num_heads)
        self.score_projection = nn.Linear(in_features=encoding_size,
                                          out_features=num_heads)
        self.key_projection = nn.Linear(in_features=encoding_size,
                                        out_features=encoding_size)
        self.value_projection = nn.Linear(in_features=encoding_size,
                                          out_features=self.head_size * num_heads)
        self.region_aware = region_aware
        self.collect_embeddings = collect_embeddings

    def aggregate(self, encoding):
        """
        """
        # flatten encoding
        batch_size, encoding_size, length, height, width = encoding.shape
        encoding = encoding.view(
            batch_size, encoding_size, -1).permute(0, 2, 1)

        mixed_values = self.value_projection(encoding)
        new_shape = mixed_values.size()[:-1] + (self.num_heads,
                                                self.head_size)
        values = mixed_values.view(*new_shape).permute(0, 2, 1, 3)

        keys = self.key_projection(encoding)
        scores = self.score_projection(keys)
        scores = scores / math.sqrt(self.head_size)
        scores = torch.nn.functional.softmax(scores, dim=1)
        scores = scores.permute(0, 2, 1)  # batch, heads, dim

        contexts = torch.matmul(scores.unsqueeze(2), values)
        contexts_flat = contexts.squeeze(2).view(batch_size, -1)

        if self.region_aware:
            scores = scores.view(batch_size,
                                 self.num_heads,
                                 length,
                                 height,
                                 width)
            return {
                'out': contexts_flat,
                'attn_scores': scores
            }
        elif self.collect_embeddings:
            scores = scores.view(batch_size,
                                 self.num_heads,
                                 length,
                                 height,
                                 width)
            encoding = encoding.contiguous().view(batch_size, 
                                                  encoding_size,
                                                  length,
                                                  height, 
                                                  width)
            return {
                "out": contexts_flat,
                "attn_scores": scores,
                "encoding": encoding
            }
        else:
            return contexts_flat

    def classify(self, emb):
        """
        """
        if self.region_aware:
            return {
                'out': self.classifier(emb['out']),
                'attn_scores': emb['attn_scores']
            }
        elif self.collect_embeddings:
            return {
                'out': self.classifier(emb['out']),
                'attn_scores': emb['attn_scores'],
                'encoding': emb["encoding"]
            }
        else:
            return self.classifier(emb)


class LinearAttClassDecoder(ClassDecoder):
    """
    """
    def __init__(self, num_classes, encoding_size=1024, region_aware=False):
        """
        """
        super().__init__(num_classes, encoding_size=1024, region_aware=region_aware)

        self.att_projection = nn.Linear(in_features=encoding_size,
                                        out_features=1,
                                        bias=True)
        # for passing attention weights forward
        self.keep_attention = False

    def aggregate(self, encoding):
        """
        """
        batch_size, encoding_size, length, height, width = encoding.shape
        encoding = encoding.view(
            batch_size, encoding_size, -1).permute(0, 2, 1)
        encoding_proj = self.att_projection(encoding).squeeze(2)
        alpha = torch.nn.functional.softmax(encoding_proj, dim=-1)

        out = torch.bmm(alpha.unsqueeze(1), encoding).squeeze(1)
        if self.keep_attention:
            return

        return out


class ConvAttClassDecoder(ClassDecoder):
    """
    """
    def __init__(self, num_classes, encoding_size=1024, region_aware=False):
        """
        """
        super().__init__(num_classes, encoding_size, region_aware=region_aware)

        self.att_conv = nn.Sequential(
            Unit3Dpy(encoding_size, encoding_size, (3, 3, 3)),
            Unit3Dpy(encoding_size, 1, (1, 1, 1)),
        )
        # for saving attentions
        self.keep_attention = False
        self.attention_probs = []

    def aggregate(self, encoding):
        """
        """
        batch_size, encoding_size, length, height, width = encoding.shape
        encoding_proj = self.att_conv(encoding)
        encoding_proj = encoding_proj.view(batch_size, -1)
        alpha = torch.nn.functional.softmax(encoding_proj, dim=-1)
        if self.keep_attention:
            self.attention_probs.append(alpha.view(batch_size,
                                                   length,
                                                   height,
                                                   width))
        encoding = encoding.view(
            batch_size, encoding_size, -1).permute(0, 2, 1)
        emb = torch.bmm(alpha.unsqueeze(1), encoding).squeeze(1)
        return emb


class FullAttentionConvClassDecoder(ClassDecoder):
    """
    """
    def __init__(self, num_classes, encoding_size=1024, out_channels=64, region_aware=False):
        """
        """
        super().__init__(num_classes, encoding_size, region_aware=region_aware)

        self.att_conv = nn.Sequential(
            Unit3Dpy(encoding_size, out_channels, (1, 1, 1)),
            Unit3Dpy(out_channels, out_channels, (3, 3, 3)),
            Unit3Dpy(out_channels, encoding_size, (1, 1, 1)),
        )
        self.classifier = ConvClassDecoder(num_classes, encoding_size, 64)

    def preprocess(self, encoding):
        """
        """
        batch_size, encoding_size, length, height, width = encoding.shape
        encoding_proj = self.att_conv(encoding)
        encoding_proj = encoding_proj.view(batch_size, encoding_size, -1)
        alpha = torch.nn.functional.softmax(encoding_proj, dim=1)

        if self.keep_attention:
            self.attention_probs.append(alpha.view(batch_size,
                                                   length,
                                                   height,
                                                   width))

        encoding_flat = encoding.view(batch_size, encoding_size, -1)
        emb = torch.bmm(torch.bmm(encoding_flat, alpha.permute(0, 2, 1)), encoding_flat)
        return emb.view(encoding.shape)


class MTDecoder(nn.Module):
    """
    """
    def __init__(self,
                 task_configs=[]):
        """Superclass for all multitask decoders.

        Args:
            task_graph_dir (str) specifies the location of task_graph spec.
            task_head_configs (list or dict) list of decoders to use, mapped to
                task index. If string, assumed that the same class will be used
                on all tasks.
            decoder_args (list or dict) list of decoder arguments to use. If
                dict, assumed that the same args will be used on all tasks.
        """
        super().__init__()

        self.task_configs = task_configs

        self.task_to_config = {}
        # add custom task_configs
        for task_config in self.task_configs:
            task = task_config['task']
            self.task_to_config[task] = task_config

        self._build_task_heads()

    def _build_task_heads(self):
        # add num classes to configs
        self.task_heads = nn.ModuleDict()
        for task, task_config in self.task_to_config.items():
            head_class = globals()[task_config["decoder_class"]]
            args = task_config["decoder_args"]
            num_classes = task_config["num_classes"]
            self.task_heads[task] = head_class(num_classes=num_classes, **args)

    def forward(self, encodings):
        """
        """
        task_outputs = {}
        for task, task_head in self.task_heads.items():
            task_outputs[task] = task_head(encodings)
        return task_outputs


class MTInTreeDecoder(MTDecoder):
    """
    This class implements an in-tree decoder, which allows the user to specify
    a task graph that leads to one final task. This decoder concatenates
    pre-classification embeddings to parent.
    """
    def __init__(self, task_graph_path, task_head_configs=[]):
        super().__init__(task_graph_path,
                         task_head_configs)

        assert nx.is_arborescence(self.tg.G.reverse()), \
            'MTTreeDecoder requires in-tree as task graph.'

        self.root = [i for i in self.tg.G.nodes()
                        if self.tg.G.out_degree(i) == 0][0]

        self.task_order = self._compute_task_order()

    def _build_task_heads(self):
        """
        Adjusts encoding sizes to accommodate for concatenation.
        """
        new_encoding_sizes = {}
        for task_head_config in self.task_head_configs:
            task = task_head_config['task']
            task_idx = self.task_to_idx[task]
            if len(self.tg.parents[task_idx]) > 0:
                encoding_size = task_head_config['args']['encoding_size']
                for parent_idx in self.tg.parents[task_idx]:
                    parent_task = self.idx_to_task[parent_idx]
                    encoding_size += self.task_to_config[parent_task]['args']['encoding_size']
                new_encoding_sizes[task] = encoding_size

        super()._build_task_heads()
         # change them all at the same time
        for task, new_encoding_size in new_encoding_sizes.items():
            num_classes = self.tg.K[self.task_to_idx[task]]
            self.task_heads[task].classifier = nn.Linear(in_features=new_encoding_size,
                                                         out_features=num_classes)

    def _compute_task_order(self):
        """
        """
        G_rev = self.tg.G.reverse()
        edges = nx.bfs_edges(G_rev, self.root)
        order = list(reversed([self.root] + [v for u, v in edges]))
        return [(task_idx, self.idx_to_task[task_idx]) for task_idx in order]

    def forward(self, encodings):
        """
        """
        outputs = {}
        history = {}
        for task_idx, task in self.task_order:
            emb = self.task_heads[task].aggregate(encodings)
            if len(self.tg.parents[task_idx]) > 0:
                for parent_idx in self.tg.parents[task_idx]:
                    parent_emb = history[self.idx_to_task[parent_idx]]
                    if type(parent_emb) == dict:
                        emb['out'] = torch.cat((emb['out'], parent_emb['out']), dim=1)
                    else:
                        emb = torch.cat((emb, parent_emb), dim=1)

            outputs[task] = self.task_heads[task].classify(emb)
            children = self.tg.children[task_idx]
            if len(children) > 0:
                history[task] = emb

        del history
        return outputs

class MTOutTreeDecoder(MTDecoder):
    """
    """
    def __init__(self,
                 task_graph_path,
                 task_head_configs=[],
                 use_residual=False):
        """Acts as a decoder to multitask settings with tree-like dependencies.

        Specifically, assumes there exists a root and each task has one
        in-degree, making it an out-tree (or arborescence as defined by nx).
        With guarantees about structure, we can safely pass embeddings from
        parent tasks to children tasks.

        Args:
             (bool) whether or not to concatenate parent
                embeddings to current task.
        """
        super().__init__(task_graph_path,
                         task_head_configs)

        assert nx.is_arborescence(self.tg.G), \
            'MTTreeDecoder requires out-tree as task graph.'

        self.root = [i for i in self.tg.G.nodes()
                     if self.tg.G.in_degree(i) == 0][0]

        self.use_residual = use_residual

    def forward(self, encoding):
        """Forward pass. Takes encoding and makes classification on all tasks.

        Relies on tree structure to perform BFS from root to wire together
        decoders without weird errors.
        """
        outputs = {}
        history = {}  # max number of elements is max tree width
        tasks = deque([self.root])  # small performance boost
        while len(tasks) > 0:
            idx = tasks.popleft()
            task = self.idx_to_task[idx]

            if idx == self.root:
                emb = self.task_heads[task].aggregate(encoding)
            else:
                p_idx = self.tg.parents[idx][0]
                p_emb = history[p_idx]

                emb = self.task_heads[task].aggregate(p_emb)
                if self.use_residual:
                    emb += encoding # residual layer

                # memory optimization
                gp_idx = self.tg.parents.get(p_idx, None)
                if gp_idx and gp_idx[0] in history.keys():
                    del history[gp_idx]

            out = self.task_heads[task].classify(emb)
            outputs[task] = out

            children = self.tg.children[idx]
            if len(children) > 0:
                history[idx] = emb
                tasks += children

        return outputs

