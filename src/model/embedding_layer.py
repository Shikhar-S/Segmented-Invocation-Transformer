""" Embeddings module """
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import Elementwise
from main_utils import get_logger
logger = get_logger()

class SequenceTooLongError(Exception):
    pass

class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding,self).__init__()
        if dim % 2 != 0:
            error_msg = "Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(dim)
            logger.error(error_msg)
            raise ValueError(error_msg)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        step = step or 0
        if self.pe.size(0) < step + emb.size(0):
            error_msg = f"Sequence is {emb.size(0) + step} but PositionalEncoding is limited to {self.pe.size(0)}. See max_len argument."
            logger.error(error_msg)
            raise SequenceTooLongError(error_msg)
        emb = emb + self.pe[step:emb.size(0)+step]
        emb = self.dropout(emb)
        return emb


class Embeddings(nn.Module):
    """
    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feat_padding_idx (List[int]): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes (List[int], optional): list of size of dictionary
            of embeddings for each feature.
        position_encoding (bool): True for transformer
        feat_merge (string): merge action for the features embeddings:
            concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
            embedding size is N^feat_dim_exponent, where N is the
            number of values the feature takes.
        feat_vec_size (int): embedding dimension for features when using
            `-feat_merge mlp`
        dropout (float): dropout probability.
        freeze_word_vecs (bool): freeze weights of word vectors.
    """

    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 position_encoding=True,
                 feat_merge="concat",
                 feat_vec_exponent=0.7,
                 feat_vec_size=-1,
                 feat_padding_idx=[],
                 feat_vocab_sizes=[],
                 dropout=0,
                 sparse=False,
                 freeze_word_vecs=False,
                 word_vectors=None):
        self._validate_args(feat_merge, feat_vocab_sizes, feat_vec_exponent,
                            feat_vec_size, feat_padding_idx)

        if feat_padding_idx is None:
            feat_padding_idx = []
        self.word_padding_idx = word_padding_idx

        self.word_vec_size = word_vec_size

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse)
                      for vocab, dim, pad in emb_params]
        embeddings[0] = self.load_pretrained_vectors(embeddings[0],word_vectors)
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

        if feat_merge == 'mlp' and len(feat_vocab_sizes) > 0:
            in_dim = sum(emb_dims)
            mlp = nn.Sequential(nn.Linear(in_dim, word_vec_size), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)

        self.position_encoding = position_encoding

        if self.position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)


        if freeze_word_vecs:
            self.word_lut.weight.requires_grad = False

    def _validate_args(self, feat_merge, feat_vocab_sizes, feat_vec_exponent,
                       feat_vec_size, feat_padding_idx):
        if feat_merge == "sum":
            # features must use word_vec_size
            if feat_vec_exponent != 0.7:
                logger.warning("Merging with sum, but got non-default "
                              "feat_vec_exponent. It will be unused.")
            if feat_vec_size != -1:
                logger.warning("Merging with sum, but got non-default "
                              "feat_vec_size. It will be unused.")
        elif feat_vec_size > 0:
            # features will use feat_vec_size
            if feat_vec_exponent != -1:
                logger.warning("Not merging with sum and positive "
                              "feat_vec_size, but got non-default "
                              "feat_vec_exponent. It will be unused.")
        else:
            if feat_vec_exponent <= 0:
                error_msg = "Using feat_vec_exponent to determine feature vec size, but got feat_vec_exponent less than or equal to 0."
                logger.error(error_msg)
                raise ValueError(error_msg)
        n_feats = len(feat_vocab_sizes)
        if n_feats != len(feat_padding_idx):
            error_msg = "Got unequal number of feat_vocab_sizes and feat_padding_idx ({:d} != {:d})".format(n_feats, len(feat_padding_idx))
            logger.error(error_msg)
            raise ValueError(error_msg)

    @property
    def word_lut(self):
        """Word look-up table."""
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        """Embedding look-up table."""
        return self.make_embedding[0]

    def load_pretrained_vectors(self, word_lut, word_vectors = None):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
        """
        if word_vectors is not None:
            word_vectors = torch.Tensor(word_vectors)
            logger.info(f'Supplied word vectors size is {word_vectors.size()} while placeholder is {word_lut.weight.data.size()}')
            pretrained_vec_size = word_vectors.size(1)
            if self.word_vec_size > pretrained_vec_size:
                word_lut.weight.data[:, :pretrained_vec_size] = word_vectors
            elif self.word_vec_size < pretrained_vec_size:
                word_lut.weight.data.copy_(word_vectors[:, :self.word_vec_size])
            else:
                word_lut.weight.data.copy_(word_vectors)
        return word_lut

    def forward(self, source, step=None):
        """Computes the embeddings for words and features.

        Args:
            source (LongTensor): index tensor ``(len, batch, nfeat)``

        Returns:
            FloatTensor: Word embeddings ``(len, batch, embedding_size)``
        """
        if self.position_encoding:
            for i, module in enumerate(self.make_embedding._modules.values()):
                if i == len(self.make_embedding._modules.values()) - 1:
                    source = module(source, step=step)
                else:
                    source = module(source)
        else:
            source = self.make_embedding(source)

        return source

    def update_dropout(self, dropout):
        if self.position_encoding:
            self._modules['make_embedding'][1].dropout.p = dropout


class SegmentPositionEncoding(nn.Module):
    """
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """
    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            error_msg = "Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(dim)
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.max_len = max_len
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(SegmentPositionEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, position_mask):
        """Embed inputs.

        Args:
            emb : [segment count, segment maxlen, batch, dim]
            position_mask : [segment count, segment maxlen, batch]
        """
        emb = emb * math.sqrt(self.dim)
        assert position_mask is not None
        seq_len = position_mask.sum(dim=0).sum(dim=0)
        assert seq_len.size(0) == emb.size(2)
        if seq_len.max() > self.pe.size(0):
            error_msg = "Sequence length {:d} exceeds max_len {:d}".format(seq_len.max(), self.pe.size(0))
            logger.error(error_msg)
            raise SequenceTooLongError(error_msg)
        position_embedding = []
        for b in range(seq_len.size(0)):
            position_embedding.append(self.pe[:seq_len[b]].squeeze(1))
        position_embedding = torch.cat(position_embedding).type_as(emb)
        assert position_embedding.size(0) == position_mask.sum()
        emb[position_mask] = emb[position_mask] + position_embedding
        emb = self.dropout(emb)
        return emb


class SegmentEmbeddings(nn.Module):
    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 position_encoding=True,
                 feat_merge="concat",
                 feat_vec_exponent=0.7,
                 feat_vec_size=-1,
                 feat_padding_idx=[],
                 feat_vocab_sizes=[],
                 dropout=0,
                 max_norm=None,
                 sparse=False,
                 freeze_word_vecs=False,
                 word_vectors=None):
        self._validate_args(feat_merge, feat_vocab_sizes, feat_vec_exponent,
                            feat_vec_size, feat_padding_idx)

        if feat_padding_idx is None:
            feat_padding_idx = []
        self.word_padding_idx = word_padding_idx

        self.word_vec_size = word_vec_size

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse, max_norm=max_norm)
                      for vocab, dim, pad in emb_params]
        embeddings[0] = self.load_pretrained_vectors(embeddings[0],word_vectors)
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(SegmentEmbeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

        if feat_merge == 'mlp' and len(feat_vocab_sizes) > 0:
            in_dim = sum(emb_dims)
            mlp = nn.Sequential(nn.Linear(in_dim, word_vec_size), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)

        self.segment_position_encoding = position_encoding

        if self.segment_position_encoding:
            pe = SegmentPositionEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)

        if freeze_word_vecs:
            self.word_lut.weight.requires_grad = False

    def _validate_args(self, feat_merge, feat_vocab_sizes, feat_vec_exponent,
                       feat_vec_size, feat_padding_idx):
        if feat_merge == "sum":
            # features must use word_vec_size
            if feat_vec_exponent != 0.7:
                logger.warning("Merging with sum, but got non-default "
                              "feat_vec_exponent. It will be unused.")
            if feat_vec_size != -1:
                logger.warning("Merging with sum, but got non-default "
                              "feat_vec_size. It will be unused.")
        elif feat_vec_size > 0:
            # features will use feat_vec_size
            if feat_vec_exponent != -1:
                logger.warning("Not merging with sum and positive "
                              "feat_vec_size, but got non-default "
                              "feat_vec_exponent. It will be unused.")
        else:
            if feat_vec_exponent <= 0:
                error_msg = "Using feat_vec_exponent to determine feature vec size, but got feat_vec_exponent less than or equal to 0."
                logger.error(error_msg)
                raise ValueError(error_msg)
        n_feats = len(feat_vocab_sizes)
        if n_feats != len(feat_padding_idx):
            error_msg = "Got unequal number of feat_vocab_sizes and feat_padding_idx ({:d} != {:d})".format(n_feats, len(feat_padding_idx))
            logger.error(error_msg)
            raise ValueError(error_msg)

    @property
    def word_lut(self):
        """Word look-up table."""
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        """Embedding look-up table."""
        return self.make_embedding[0]

    def load_pretrained_vectors(self, word_lut, word_vectors = None):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
        """
        if word_vectors is not None:
            word_vectors = torch.Tensor(word_vectors)
            logger.info(f'Supplied word vectors size is {word_vectors.size()} while placeholder is {word_lut.weight.data.size()}')
            pretrained_vec_size = word_vectors.size(1)
            if self.word_vec_size > pretrained_vec_size:
                # Smaller pretrained vectors than model requires; init remaining vectors dimensions
                nn.init.kaiming_normal_(word_lut.weight.data,nonlinearity='leaky_relu')
                word_lut.weight.data[:, :pretrained_vec_size] = word_vectors
            elif self.word_vec_size < pretrained_vec_size:
                # Larger pretrained vectors than model requires; truncate
                word_lut.weight.data.copy_(word_vectors[:, :self.word_vec_size])
            else:
                # Exact same vectors as model requires
                word_lut.weight.data.copy_(word_vectors)
        # F.normalize(word_lut.weight.data, dim=1, out= word_lut.weight.data) Hurts performance!
        return word_lut

    def forward(self, source, step=None):
        """
        Args:
            source (LongTensor): index tensor ``(batch, segment_count, segment_max_len, nfeat)``

        Returns:
            FloatTensor: Word embeddings ``(batch, segment_count, segment_max_len, embedding_size)``
        """
        if self.segment_position_encoding:
            #important to create position mask on entry because we overwrite source
            position_mask = (source.permute(1,2,0,3)[:,:,:,0]!=self.word_padding_idx) #assuming first coordinate is word
            for i, module in enumerate(self.make_embedding._modules.values()):
                if i == len(self.make_embedding._modules.values()) - 1:
                    # The last module is the segment position encoding
                    source = source.permute(1,2,0,3)
                    source = module(source, position_mask = position_mask)
                    source = source.permute(2,0,1,3)
                else:
                    source = module(source)
                    # Performs Elementwise module on source,
                    #each last dim feature is expanded using embedding lookup and result merged
        else:
            source = self.make_embedding(source)

        return source

    def update_dropout(self, dropout):
        if self.segment_position_encoding:
            self._modules['make_embedding'][1].dropout.p = dropout
