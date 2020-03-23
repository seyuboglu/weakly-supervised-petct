"""
"""

import torch
import torch.nn as nn
import pandas as pd


def load_embeddings(path):
    """
    """
    word2embed = {}
    with open(path) as f:
        for line in f:
            row = line.split(" ")
            word = row[0]
            embedding = torch.FloatTensor(list(map(float, row[1:])))
            word2embed[word] = embedding
    return word2embed


class ReportEmbeddings(nn.Embedding):
    """
    """
    def __init__(self, vocab, embedding_dim, pretrained_path=None, 
                 freeze=False, sparse=False):
        """
        """
        num_embeddings = len(vocab)
        weight = torch.Tensor(num_embeddings, embedding_dim)
        nn.init.normal_(weight)

        if pretrained_path is not None:
            word2embed = load_embeddings(pretrained_path)
            for word, idx in vocab.word2idx.items():
                if word in word2embed:
                    weight[idx, :] = word2embed[word]

        super().__init__(num_embeddings=num_embeddings,
                         embedding_dim=embedding_dim,
                         sparse=sparse, _weight=weight)

        self.weight.requires_grad = not freeze
