import torch
from torch._C import dtype
import torch.nn as nn
from model.embedding_layer import Embeddings
from model.multi_headed_attn import MultiHeadedAttention
from model.position_ffn import PositionwiseFeedForward
from model.position_ffn import ActivationFunction
from model.utils import sequence_mask_

class SITLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 pos_ffn_activation_fn=ActivationFunction.relu):
        super(SITLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout,
                                                    pos_ffn_activation_fn)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``
        Returns:
            (FloatTensor):
            * outputs ``(batch_size, src_len, model_dim)``
        """
        assert mask is not None
        assert inputs.size(0) == mask.size(0)
        assert inputs.size(1) == mask.size(2)

        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class SIT(nn.Module):
    """
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings: embeddings to use, should have positional encodings
        pos_ffn_activation_fn (ActivationFunction): activation function choice for PositionwiseFeedForward layer

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 attention_dropout, num_layers, embedding,
                 pos_ffn_activation_fn=ActivationFunction.relu):
        super(SIT, self).__init__()
        self.embedding = embedding

        self.padding_idx = embedding.word_padding_idx

        self.sit_layers = nn.ModuleList(
            [SITLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                pos_ffn_activation_fn=pos_ffn_activation_fn)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self,sentence_phrases, phrases_len):
        '''
        Args:
            sentence_phrases [batch size X max phrase count X max phrase length]
            phrases_len  [batch size X max phrase count]
        Returns:
            out [max phrase count X batch size X model dim]
            lengths [batch size]

        '''
        #embed nodes and leaves and generate L,N
        batch_size = sentence_phrases.size(0)
        assert batch_size == phrases_len.size(0)
        max_phrase_count = phrases_len.size(1)
        assert max_phrase_count == sentence_phrases.size(1)

        out = self.embed_phrases(sentence_phrases, phrases_len)

        lengths = (phrases_len!=0).sum(dim=1)
        mask = ~sequence_mask_(lengths).unsqueeze(1)
        # Run the forward pass of every layer.
        for layer in self.pie_layers:
            out = layer(out, mask)
        out = self.layer_norm(out)
        assert out.shape[1] == lengths.max()

        return out.transpose(0, 1).contiguous(), lengths

    def embed_phrases(self, sentence_phrases, phrases_len):
        '''
        Args:
            sentence_phrases [batch size X max phrase count X max phrase length]
            phrases_len  [batch size X max phrase count]
        Returns:
            [batch size X max phrase count X model dim]
        '''
        emb = self.embedding(sentence_phrases.unsqueeze(-1).contiguous())
        emb = emb.sum(dim=2)/(phrases_len.unsqueeze(-1) + 1e-10)
        return emb

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)