"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn

from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 max_relative_positions=0):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=dropout,
                max_relative_positions=max_relative_positions)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                code_type="enc", layer_cache=None, step=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``

        """
        dec_mask = None
        if step is None:
            if code_type == 'dec':
                tgt_len = tgt_pad_mask.size(-1)
                future_mask = torch.ones(
                    [tgt_len, tgt_len],
                    device=tgt_pad_mask.device,
                    dtype=torch.uint8)
                future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
                dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
            else:
                dec_mask = tgt_pad_mask

        input_norm = self.layer_norm_1(inputs)

        query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         type="self")

        query = self.drop(query) + inputs

        if code_type == 'dec':
            query_norm = self.layer_norm_2(query)
            mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                          mask=src_pad_mask,
                                          layer_cache=layer_cache,
                                          type="context")
            mid = self.drop(mid) + query
        else:
            mid = query

        output = self.feed_forward(mid)

        # batch_size, head_count, query_len, key_len = attn.size()

        # file=open('txt','a')
        # import numpy
        # for b in range(batch_size):
        #  for h in range(head_count):
        #    for q in range(query_len):
        #      file.write(str(list(numpy.array(attn.cpu()[b,h,q,:])))+'\n')
        #    file.write('\n')

        return output

    def update_dropout(self, dropout):
        self.self_attn.update_dropout(dropout)
        self.context_attn.update_dropout(dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout


class TransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       self_attn_type (str): type of self-attention scaled-dot, average
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, dropout, embeddings,
                 max_relative_positions):
        super(TransformerDecoder, self).__init__()

        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout,
             max_relative_positions=max_relative_positions)
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.max_relative_positions)

    def init_state(self, src):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, memory_bank=None, code_type = 'dec', step=None, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)

        pad_idx = self.embeddings.word_padding_idx


        src_memory_bank = None
        src_pad_mask = None
        if code_type == 'dec':
            src = self.state["src"]
            src_words = src[:, :, 0].transpose(0, 1)
            src_memory_bank = memory_bank.transpose(0, 1).contiguous()
            src_pad_mask = src_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_src]

        tgt_words = tgt[:, :, 0].transpose(0, 1)
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim
        output = emb.transpose(0, 1).contiguous()

        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                code_type=code_type,
                layer_cache=layer_cache,
                step=step)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs

    def _init_cache(self):
        self.state["cache"] = {}

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache

    def update_dropout(self, dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout)
