
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
import logging
import os
import re
import json
from collections import OrderedDict, Counter
from itertools import chain
import json
from typing import List

import pandas as pd
import sentencepiece as spm
from nltk import sent_tokenize
from tqdm import tqdm
import torch
import numpy as np
from transformers import WordpieceTokenizer

from pet_ct.util.util import Process, place_on_cpu

     
class WordPieceVocab(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_path, do_lower_case=True, max_len=None, freq_path=None):
        """Constructs a BertTokenizer.

        Args:
          vocab_file: Path to a one-wordpiece-per-line vocabulary file
          max_len: An artificial maximum length to truncate tokenized sequences to;
                         Effective maximum length is always the minimum of this
                         value (if specified) and the underlying BERT model's
                         sequence length.
        """
        self.token_to_idx = json.load(open(vocab_path, 'r'),
                               object_pairs_hook=OrderedDict)
        self.idx_to_token = OrderedDict([(idx, tok) for tok, idx in self.token_to_idx.items()])
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.token_to_idx)
        self.max_len = max_len if max_len is not None else int(1e12)

        if freq_path is not None:
            self.token_to_freq = json.load(open(freq_path, 'r'), object_pairs_hook=OrderedDict)

    def tokenize(self, text):
        split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def detokenize(self, tokens):
        text = ' '.join(tokens)
        return text.replace(' ##', '')

    def to_input_tensor(self, sents: List[List[str]], device) -> torch.Tensor:
        """ Convert list of tokens into tensor with necessary padding for
        shorter sentences.

        @param sents (List[List[str]]): list of sentences (words)
        @param device: device on which to load the tesnor, i.e. CPU or GPU

        @returns sents_var: tensor of (max_sentence_length, batch_size)
        """
        sents = [self.convert_tokens_to_idx(sent) for sent in sents]
        sents, mask = self.pad_sentences(sents)
        sents_var = torch.tensor(sents, dtype=torch.long, device=device)
        mask_var = torch.tensor(mask, dtype=torch.long, device=device)
        return sents_var, mask_var

    def from_output_tensor(self, batch_output):
        """ Places batch output on cpu and converts it to tokens ignoring -1's and padding.
        args:
            batch_output    (tensor)   (batch_size, max_len)
        """
        place_on_cpu(batch_output)
        sents = []
        for output in batch_output:
            sent = []
            for idx in output:
                idx = idx.item()
                if idx == -1:
                    continue

                token = self.idx_to_token[idx]

                if token == "[PAD]":
                    continue

                sent.append(token)
            sents.append(sent)
        return sents

    def pad_sentences(self, sents):
        """
        args:
            sents   (list(list(str)))
        """
        sents_padded = []
        mask_padded = []

        max_len = max(map(len, sents))
        for sent in sents:
            sents_padded.append(sent[:] + [self.token_to_idx['[PAD]']] * (max_len - len(sent)))

        mask = [[int(token != self.token_to_idx['[PAD]']) for token in sent]for sent in sents_padded]

        return sents_padded, mask

    def wrap_sentence(self, sent):
        """ Wrap sentences with start and stop tokens.
        args:
            sent (list[str]])
        """
        sent = ['[CLS]'] + sent + ['[SEP]']

        return sent
    
    def unwrap_sentence(self, tokens):
        new_tokens = [token for token in tokens 
                      if token != '[CLS]' and token != '[SEP]']
        return new_tokens

    def convert_tokens_to_idx(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.token_to_idx[token])
        if len(ids) > self.max_len:
            logging.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_idxs_to_token(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.idx_to_token[i])
        return tokens

    def get_tokens_in_range(self, tokens, text, start, end):
        """
        Get all of the tokens in the range (start, end) in original string.
        """
        token_idxs = []
        find_start = 0
        
        for idx, token in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
                
            if token.startswith("##"):
                # remove pounds
                token = token[2:]

            token_start = text.find(token, find_start)
            token_end = token_start + len(token)
            find_start = token_end

            if ((token_start >= start and token_start < end) or
                (token_end >= start and token_end < end)):
                token_idxs.append(idx)
        return token_idxs

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.token_to_idx)


class MergeVocabs(Process):

    def __init__(self, dir, primary_vocab_path, secondary_vocab_path,
                 overwrite_regexes=['\[unused[0-9]+\]', '[^\x00-\x7F]'],
                 overflow=False):
        """
        Args:
            overwrite_regexes   (list(str)) words in primary_vocab that match
                any one of these will be overwritten by words in secondary_vocab.
                Defaults to regexes for unused and non-english characters.
            overflow    (bool)  if true and there are fewer overwriteable words in primary
                        vocab than words in secondary vocab, overflow will be added to
                        merged vocab at indices >len(primary_vocab)
        """
        super().__init__(dir)
        self.overflow = overflow
        self.primary_vocab_path = primary_vocab_path
        self.secondary_vocab_path = secondary_vocab_path
        self.primary_vocab = json.load(open(primary_vocab_path),
                                       object_pairs_hook=OrderedDict)
        self.secondary_vocab = json.load(open(secondary_vocab_path),
                                         object_pairs_hook=OrderedDict)
        self.merged_vocab = self.primary_vocab.copy()

        # find overwriteable tokens
        self.overwrite_tokens = []
        for token, idx in self.primary_vocab.items():
            for regex in overwrite_regexes:
                if re.match(regex, token) is not None:
                    self.overwrite_tokens.append(token)
                    break

    def _run(self, overwrite=False):
        """
        """
        self._merge()
        self._write()


    def _write(self):
        """
        """
        with open(os.path.join(self.dir, 'vocab.json'), 'w') as f:
            json.dump(self.merged_vocab, f, indent=4)

    def _merge(self):
        """
        """
        overwrite_count = 0
        overflow_count = 0
        union_count = 0
        for token in tqdm(self.secondary_vocab.keys()):
            if token in self.merged_vocab:
                union_count += 1
                continue

            if not self.overwrite_tokens:
                if self.overflow:
                    overflow_count += 1
                    idx = len(self.merged_vocab)
                    self.merged_vocab[token] = idx
                else:
                    break
            else:
                overwrite_token = self.overwrite_tokens.pop(0)
                # assign overwrite's idx to token's idx
                idx =  self.primary_vocab[overwrite_token]
                del self.merged_vocab[overwrite_token]
                self.merged_vocab[token] = idx
                overwrite_count += 1
        logging.info(f"Merged {overwrite_count + overflow_count} tokens from " +
                     f"{self.secondary_vocab_path} to {self.primary_vocab_path}.")
        logging.info(f"Overwrote {overwrite_count}/" +
                     f"{overwrite_count + len(self.overwrite_tokens)} tokens.")
        logging.info(f"Added {overflow_count} tokens as overflow.")
        logging.info(f"{union_count} tokens from {self.secondary_vocab_path} " +
                     f"were already in {self.primary_vocab_path}.")
        leftover = len(self.secondary_vocab) - (union_count + overwrite_count + overflow_count)
        logging.info(f"{leftover} tokens from " +
                     f"{self.secondary_vocab_path} were not included.")



class GenerateWordPieceVocab(Process):

    def __init__(self, dir, reports_path, vocab_size=8000,
                 exams_to_exclude_paths=[], min_sent_char_length=10,
                 character_coverage=1.0):
        super().__init__(dir)
        self.reports_path = reports_path
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage

        # load exam_ids to exclude from the
        self.exams_to_exclude = set()
        for exams_to_exlcude_path in exams_to_exclude_paths:
            exams_df = pd.read_csv(exams_to_exlcude_path)
            self.exams_to_exclude.update(list(exams_df['exam_id']))

        self.reports_df = pd.read_csv(reports_path, error_bad_lines=False, index_col=0)

        # build file for SentencePieceTrainer
        logging.info("Building raw input file for spm")
        self.raw_reports_path = os.path.join(self.dir, "spm.txt")
        self.exams_to_include = set()
        with open(self.raw_reports_path, 'w') as f:
            for exam_id, row in tqdm(self.reports_df.iterrows()):
                break
                if exam_id in self.exams_to_exclude:
                    continue
                self.exams_to_include.add(exam_id)

                report_text = row["report_txt"]
                sents = sent_tokenize(report_text)
                for sent in sents:
                    if len(sent) < min_sent_char_length:
                        continue
                    sent = sent.lower()
                    f.write(sent + "\n")

    def _run(self, overwrite=False):
        #spm.SentencePieceTrainer.Train(f'--input={self.raw_reports_path} \
        #                                 --model_prefix={self.dir}/spm \
        #                                 --vocab_size={self.vocab_size} \
        #                                 --character_coverage={self.character_coverage}')
        vocab = OrderedDict()
        with open(os.path.join(self.dir, 'spm.vocab')) as f:
            for idx, line in enumerate(f):
                word_piece, score = line.split("\t")

                # convert from SentencePiece to BERT format
                if word_piece[0] == '\u2581':
                    word_piece = word_piece[1:]
                else:
                    word_piece = f"##{word_piece}"

                vocab[word_piece] = idx

        with open(os.path.join(self.dir, 'vocab.json'), 'w') as f:
            json.dump(vocab, f, indent=4)



class WordVocab(object):
    """
    """
    def __init__(self, vocab_path=None, freq_cutoff=None):
        """ Init VocabEntry Instance.
        @param word2idx (dict): dictionary mapping words 2 indices
        """
        self.word2idx = dict()
        self.word2idx['<pad>'] = 0   # Pad Token
        self.word2idx['<s>'] = 1 # Start Token
        self.word2idx['</s>'] = 2    # End Token
        self.word2idx['<unk>'] = 3   # Unknown Token
        self.unk_id = self.word2idx['<unk>']
        self.id2word = {v: k for k, v in self.word2idx.items()}

        # load vocab
        if vocab_path is not None:
            with open(vocab_path) as f:
                self.word2freq = json.load(f)

            for word, freq in self.word2freq.items():
                if freq_cutoff is None or freq >= freq_cutoff:
                    self.add(word)

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word
        """
        return self.word2idx.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained
        """
        return word in self.word2idx

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2idx)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def get_word(self, wid):
        """ Return mapping of index to word.
        @param wid (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        @param word (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.word2idx[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) in words
        @return word_ids (list[int] or list[list[int]]): sentence(s) in indices
        """

        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for
        shorter sentences.

        @param sents (List[List[str]]): list of sentences (words)
        @param device: device on which to load the tesnor, i.e. CPU or GPU

        @returns sents_var: tensor of (max_sentence_length, batch_size)
        """
        sents = self.words2indices(sents)
        sents = self._wrap_sentences(sents)
        sents = self._pad_sentences(sents)
        sents_var = torch.tensor(sents, dtype=torch.long, device=device)
        return sents_var

    def _wrap_sentences(self, sents):
        """ Wrap sentences with start and stop tokens.
        args:
            sents (list[list[str]])
        """
        sents = [[self['<s>']] + sent + [self['</s>']] for sent in sents]
        return sents

    def _pad_sentences(self, sents):
        """
        """
        sents_padded = []

        max_len = max(map(len, sents))
        for sent in sents:
            sents_padded.append(sent[:] + [self['<pad>']] * (max_len - len(sent)))

        return sents_padded

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        """ Given a corpus construct a Vocab Entry.
        @param corpus (list[str]): corpus of text produced by read_corpus function
        @param size (int): # of words in vocabulary
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab_entry (VocabEntry): VocabEntry instance produced from provided corpus
        """
        vocab_entry = WordVocab()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry
