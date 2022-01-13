import torch
import torch.nn as nn

from model.multi_headed_attn import MultiHeadedAttention
from model.position_ffn import PositionwiseFeedForward, ActivationFunction
from model.utils import sequence_mask_

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                d_model,
                heads,
                d_ff,
                dropout,
                attention_dropout,
                padding_idx=1,
                pos_ffn_activation_fn = ActivationFunction.relu):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads,
            d_model,
            dropout=attention_dropout,
            padding_idx = padding_idx,
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout,
                                            pos_ffn_activation_fn
                                            )
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.context_attn = MultiHeadedAttention(
                heads, d_model, dropout=attention_dropout, padding_idx = padding_idx,
                attn_type='context'
        )
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

    def update_dropout(self, dropout, attention_dropout):
        self.context_attn.update_dropout(attention_dropout)
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout

    def forward(
        self,
        inputs,
        memory_bank,
        src_pad_mask,
        tgt_pad_mask,
        layer_cache=None,
        future=False,
    ):
        """A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            inputs (FloatTensor): ``(batch_size, T, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (bool): ``(batch_size, 1, src_len)``
            tgt_pad_mask (bool): ``(batch_size, 1, T)``
            layer_cache (dict or None): cached layer info when stepwise decode
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, T, model_dim)``
            * attns ``(batch_size, head, T, src_len)``

        Returns:
            (FloatTensor, FloatTensor, FloatTensor or None):

            * output ``(batch_size, T, model_dim)``
            * top_attn ``(batch_size, T, src_len)``
            * attn_align ``(batch_size, T, src_len)`` or None

        """
        assert memory_bank.shape[1] == src_pad_mask.shape[-1]
        assert memory_bank.shape[0] == src_pad_mask.shape[0]
        dec_mask = None
        if inputs.size(1) > 1:
            # masking is necessary when sequence length is greater than one
            dec_mask = self._compute_dec_mask(tgt_pad_mask, future)
        inputs_norm = self.layer_norm_1(inputs)
        #Self attention on decoder target input
        query, _ = self.self_attn(
                inputs_norm,
                inputs_norm,
                inputs_norm,
                mask=dec_mask,
                layer_cache=layer_cache,
            )
        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)
        #Cross attention on encoder output
        mid, _ = self.context_attn(
            memory_bank,
            memory_bank,
            query_norm,
            mask=src_pad_mask,
            layer_cache=layer_cache,
        )
        # residual connection from query
        output = self.feed_forward(self.drop(mid) + query)
        # output: batch x q_len x model_dim
        return output

    def _compute_dec_mask(self, tgt_pad_mask, future):
        tgt_len = tgt_pad_mask.size(-1)
        if not future:  # apply future_mask, result mask in (B, T, T)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8,
            )
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            # BoolTensor was introduced in pytorch 1.2
            try:
                future_mask = future_mask.bool()
            except AttributeError:
                pass
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
        else:  # only mask padding, result mask in (B, 1, T)
            dec_mask = tgt_pad_mask
        return dec_mask


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        embeddings,
        padding_idx,
        pos_ffn_activation_fn=ActivationFunction.relu,
    ):
        super(TransformerDecoder, self).__init__()

        self.embeddings = embeddings
        # Decoder State
        self.state = {}
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.padding_idx = padding_idx
        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    padding_idx = padding_idx,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                )
                for _ in range(num_layers)
            ]
        )

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)
        # Add map to self state src
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)

    def forward(self, sequence, memory_bank=None, step=None, **kwargs):
        """Decode, possibly stepwise.
            sequence :  len X batch X 1
            tgt : len X batch X features
            memory_bank : len X batch X features
            memory_lengths : batch
        """
        if step == 0:
            self._init_cache()

        emb = self.embeddings(sequence,step=step) # T, B, dim
        #######################################################

        assert emb.dim() == 3  # len x batch x embedding_dim

        #Prepare masks
        pad_idx = self.embeddings.word_padding_idx

        src_lens = kwargs["memory_lengths"] # (batch,)
        src_pad_mask = ~sequence_mask_(src_lens, max_len = memory_bank.shape[0]).unsqueeze(1) # [B, 1, T_max_src]
        # It is necessary to construct a source padding mask from max_len stored in the cache
        # because during beam search decoding, some hypothesis get removed from the beams.
        tgt_pad_mask = sequence.transpose(0,1).squeeze(-1).data.eq(pad_idx).unsqueeze(1) # [B, 1, T_tgt]

        #Prepare inputs for decoder stack
        output = emb.transpose(0, 1).contiguous() # batch x len x embedding_dim
        src_memory_bank = memory_bank.transpose(0, 1).contiguous() # batch x len x embedding_dim
        for i, layer in enumerate(self.transformer_layers):
            layer_cache = (
                self.state["cache"]["layer_{}".format(i)]
                if step is not None
                else None
            )

            output = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache = layer_cache,
            )

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        return dec_outs

    def init_state(self, src):
        """Initialize decoder state.
            src : max_len X src X features
        """
        self.state["src"] = src
        self.state["cache"] = None

    def _init_cache(self):
        self.state["cache"] = {}
        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys":None, "memory_values": None}
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache