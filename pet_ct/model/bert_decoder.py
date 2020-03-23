"""
"""
import copy
import math
from collections import namedtuple

import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import (BertModel, BertEncoder, BertAttention, 
                                              BertIntermediate, BertLayer, BertOutput,
                                              BertPreTrainedModel, BertEmbeddings, BertPooler,
                                              BertConfig, BertSelfOutput)

class BertTokenClassifierHead(nn.Module):
    
    def __init__(self, config):
        super(BertTokenClassifierHead, self).__init__()
        self.classifier = nn.Linear(in_features=config.hidden_size,
                                    out_features=config.num_classes)

    def forward(self, sequence_output):
        prediction_scores = self.classifier(sequence_output)
        return prediction_scores
    
    
class BertScanDecoder(BertPreTrainedModel):

    def __init__(self, config):
        """
        """
        config = BertConfig.from_dict(config)

        super().__init__(config)
        #self.embeddings = BertEmbeddings(config)
        self.encoder = BertScanEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, scan_encodings, report_encodings, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(report_encodings)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0

        encoded_layers = self.encoder(scan_encodings,
                                      report_encodings,
                                      attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output
    

class BertScanEncoder(nn.Module):
    #TODO
    def __init__(self, config):
        super(BertScanEncoder, self).__init__()
        layer = BertScanLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, scan_encodings, hidden_states, 
                attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(scan_encodings, hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertScanLayer(nn.Module):
    def __init__(self, config):
        super(BertScanLayer, self).__init__()
        self.attention = BertAttention(config)
        self.scan_attention = BertScanAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, scan_encodings, hidden_states, attention_mask):
        # self attention
        attention_output = self.attention(hidden_states, attention_mask)

        scan_attention_output = self.scan_attention(scan_encodings, hidden_states, 
                                                    attention_mask)

        # feed-forward
        intermediate_output = self.intermediate(scan_attention_output)
        layer_output = self.output(intermediate_output, scan_attention_output)
        return layer_output


class BertScanAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertScanAttentionHeads(config)
        self.output = BertSelfOutput(config)

    def forward(self, scan_encodings, input_tensor, attention_mask):
        self_output = self.self(scan_encodings, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertScanAttentionHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.scan_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.scan_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.keep_attention = False
        self.attention_probs = []

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, scan_encodings, hidden_states, attention_mask):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(scan_encodings)
        mixed_value_layer = self.value(scan_encodings)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        scan_key_layer = self.transpose_for_scores(mixed_key_layer)
        scan_value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, scan_key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        #attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.keep_attention:
            batch_size, num_heads, len_seq, _ = attention_probs.shape
            reshaped_attention_probs = attention_probs.view(batch_size, num_heads, len_seq, -1, 7, 7)
            self.attention_probs.append(reshaped_attention_probs.detach().cpu())
            
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, scan_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
