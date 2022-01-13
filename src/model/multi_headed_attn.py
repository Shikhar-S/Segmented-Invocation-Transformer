""" Multi-Head Attention module - Self Attention in Decoder """
import math
import torch
import torch.nn as nn
from main_utils import get_logger
logger = get_logger()


def aeq(x,y):
    assert x==y, f'{x} != {y}'

class MultiHeadedAttention(nn.Module):
    """
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, padding_idx = 1, attn_type = 'self'):
        try:
            assert model_dim % head_count == 0
        except AssertionError as err:
            logger.error('Model dimension not divisible by head count',err)
            raise err
        super(MultiHeadedAttention, self).__init__()

        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count
        self.attn_type = attn_type

        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)
        self.padding_idx = padding_idx

    def shape(self,x):
        batch_size = x.size(0)
        """Projection: Break the last dimension into (heads,dim per head) then bring head forward to second dim."""
        return x.view(batch_size, -1, self.head_count, self.dim_per_head) \
            .transpose(1, 2)
        # -1 is for sequence length

    def unshape(self,x):
        batch_size = x.size(0)
        """Compute inverse of shape"""
        return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, self.head_count * self.dim_per_head)

    def process_qkv(self,layer_cache,query,key,value):
        # 1) Project key, value, and query.
        if layer_cache is not None:
            if self.attn_type == 'self':
                #decoder in decoding
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = self.shape(key)
                value = self.shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif self.attn_type == 'context':
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    #first time
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = self.shape(key)
                    value = self.shape(value)
                else:
                    # use cached keys and values in context attention
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            #decoder in training
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query) # batch x seq_len x model_dimension
            key = self.shape(key)
            value = self.shape(value) # batch x heads x seq_len x dim_per_head

        return (query,key,value)

    def forward(self, key, value, query, mask=None,
                layer_cache=None):
        """
        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """
        # # TRAIN CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # if mask is not None:
        #     batch_, q_len_, k_len_ = mask.size()
        #     aeq(batch_, batch)
        #     aeq(k_len_, k_len)
        # #    aeq(q_len_, q_len)
        # # END TRAIN CHECKS
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        query,key,value = self.process_qkv(layer_cache,query,key,value)

        query = self.shape(query)
        key_len = key.size(2)
        query_len = query.size(2)
        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3)) # batch x heads x qlen x klen
        scores = query_key
        scores = scores.float()
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)
        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)
        context_original = torch.matmul(drop_attn, value) #batch x heads x len x dim_per_head
        context = self.unshape(context_original) #batch x len x model_dim
        output = self.final_linear(context)
        # # TRAIN CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return multi-head attn
        # re-assure that shape matches
        attns = attn \
            .view(batch_size, head_count,
                  query_len, key_len)
        return output, attns

    def update_dropout(self, dropout):
        self.dropout.p = dropout