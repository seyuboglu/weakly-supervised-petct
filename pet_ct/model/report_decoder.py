"""
"""
from collections import namedtuple

import torch
import torch.nn as nn

import pet_ct.model.embeddings as embeddings
from pet_ct.model.attention import AttentionAggregator

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class ReportDecoder(nn.Module):

    def __init__(self, vocab, embeddings_class, embeddings_args):
        """
        """
        super().__init__()
        self.embeddings = getattr(embeddings, embeddings_class)(vocab, **embeddings_args)
        self.vocab = vocab

    def forward(self, encoding, targets):
        raise NotImplementedError
    
    def beam_search(self, encoding):
        raise NotImplementedError


class LSTMDecoder(ReportDecoder):

    def __init__(self, vocab, embedding_class, embedding_args, 
                 encoding_size=1024, hidden_size=1024, dropout_rate=0.5,
                 beam_size=5, max_decoding_time_step=70,
                 diverse_decoding=False, diverse_lambda=0.1):
        """
        """
        super().__init__(vocab, embedding_class, embedding_args)
        self.hidden_size = hidden_size
        self.encoding_size = encoding_size
        self.embed_size = self.embeddings.embedding_dim 

        self.beam_size = beam_size
        self.max_decoding_time_step = max_decoding_time_step
        self.diverse_decoding = diverse_decoding
        self.diverse_lambda = diverse_lambda

        self.hidden_encoding_aggregator = AttentionAggregator(encoding_size=encoding_size)
        self.encoding_to_hidden = nn.Linear(in_features=encoding_size,
                                            out_features=hidden_size,
                                            bias=True)
        self.cell_encoding_aggregator = AttentionAggregator(encoding_size=encoding_size)
        self.encoding_to_cell = nn.Linear(in_features=encoding_size,
                                          out_features=hidden_size,
                                          bias=True)

        self.decoder = nn.LSTMCell(input_size=self.embed_size + hidden_size, 
                                   hidden_size=hidden_size,
                                   bias=True)
        self.att_projection = nn.Linear(in_features=encoding_size, 
                                        out_features=hidden_size, 
                                        bias=False)
        self.combined_output_projection = nn.Linear(in_features=encoding_size + hidden_size, 
                                                    out_features=hidden_size, 
                                                    bias=False)
        self.vocab_projection = nn.Linear(in_features=hidden_size, 
                                          out_features=len(vocab),
                                          bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, encodings, targets):
        """
        args:
            encoding (torch.tensor) encoding of the shape (batch_size, dimensions, length, height, width)
            target  (torch.tensor) Padded target sentences of shape (batch_size, tgt_len)
        returns: 
        """
        batch_size, encoding_size, length, height, width = encodings.shape

        # chop of the <END> token for max length sentences
        targets = targets[:, :-1]

        # flatten 3D encodings, into 1D sequence
        encodings = encodings.view(batch_size, encoding_size, -1).permute(0, 2, 1)

        # Initialize the decoder state (hidden and cell)
        hidden_aggregated_encoding = self.hidden_encoding_aggregator(encodings)
        hidden_init_state =  self.encoding_to_hidden(hidden_aggregated_encoding)

        cell_aggregated_encoding = self.cell_encoding_aggregator(encodings)
        cell_init_state =  self.encoding_to_cell(cell_aggregated_encoding)
        
        dec_state = (hidden_init_state, cell_init_state)

        # Initialize previous combined output vector o_{t-1} as zero
        device =  next(self.parameters()).device
        o_prev = torch.zeros(batch_size, self.hidden_size, device=device)

        encoding_proj = self.att_projection(encodings)

        Y = self.embeddings(targets)
        
        combined_outputs = []
        for Y_t in torch.split(Y, split_size_or_sections=1, dim=1):
            Y_t = Y_t.squeeze(1)
            Ybar_t = torch.cat([Y_t, o_prev], dim=-1)
            dec_state, o_t = self.step(Ybar_t, dec_state, encodings, encoding_proj)
            combined_outputs.append(o_t)
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs, dim=1)

        return self.vocab_projection(combined_outputs)
    
    def step(self, Ybar_t,
             dec_state,
             encoding,
             encoding_proj):
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """
        dec_state = self.decoder(Ybar_t, dec_state)
        (dec_hidden, dec_cell) = dec_state
        e_t = torch.bmm(encoding_proj, dec_hidden.unsqueeze(2)).squeeze(2)
        alpha_t = torch.nn.functional.softmax(e_t, dim=-1)
        a_t = torch.bmm(alpha_t.unsqueeze(1), encoding).squeeze(1)
        U_t = torch.cat([dec_hidden, a_t], dim=1)
        V_t = self.combined_output_projection(U_t)
        O_t = self.dropout(torch.tanh(V_t))

        return dec_state, O_t
    
    def beam_search(self, encodings):
        """
        """
        batch_size, encoding_size, length, height, width = encodings.shape
        encodings = encodings.view(batch_size, encoding_size, -1).permute(0, 2, 1)
        encoding_proj = self.att_projection(encodings)

        hidden_aggregated_encoding = self.hidden_encoding_aggregator(encodings)
        hidden_init_state =  self.encoding_to_hidden(hidden_aggregated_encoding)

        cell_aggregated_encoding = self.cell_encoding_aggregator(encodings)
        cell_init_state =  self.encoding_to_cell(cell_aggregated_encoding)
        
        h_tm1 = (hidden_init_state, cell_init_state)
        
        device = next(self.parameters()).device

        att_tm1 = torch.zeros(1, self.hidden_size, device=device)

        eos_id = self.vocab['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < self.beam_size and t < self.max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            # expand to match number of hypotheses
            exp_encodings = encodings.expand(hyp_num, -1, -1)
            exp_encodings_proj = encoding_proj.expand(hyp_num, -1, -1)

            y_tm1 = torch.tensor([self.vocab[hyp[-1]] for hyp in hypotheses], 
                                 dtype=torch.long, device=device)
            y_tm1_embed = self.embeddings(y_tm1)

            Y_bar_t = torch.cat([y_tm1_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t = self.step(Y_bar_t, h_tm1, exp_encodings, exp_encodings_proj)

            # log probabilities over target words
            log_p_t = nn.functional.log_softmax(self.vocab_projection(att_t), dim=-1)  # (hyp_num, vocab_size)


            if self.diverse_decoding:
                argsort_log_p_t = torch.argsort(-log_p_t, dim=1)
                ranks_log_p_t = torch.zeros_like(log_p_t, device=device)
                arange = torch.arange(argsort_log_p_t.shape[1], 
                                      device=device, dtype=torch.float)
                for idx in range(ranks_log_p_t.shape[0]):
                    ranks_log_p_t[idx, argsort_log_p_t[idx, :]] = arange
                
                log_p_t -= self.diverse_lambda * (ranks_log_p_t) 

            live_hyp_num = self.beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1) # (hyp_num, vocab_size)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []
            print("---Next Step----")
            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.id2word[hyp_word_id]

                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                print(" ".join(new_hyp_sent))
                print(cand_new_hyp_score)
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == self.beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        return completed_hypotheses



