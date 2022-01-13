""" Translator Class """
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import tile

from decoder.decoding_strategy import BeamSearch

from main_utils import get_logger
logger = get_logger()


class Translator(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (onmt.modules.NMTModel): NMT model to use for translation
        datamodule (pytorch_lightning.datamodule) : DataModule to use for translation
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        random_sampling_temp (float): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        dump_beam (bool): Debugging option.
        block_ngram_repeat (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        ignore_when_blocking (set or frozenset): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        tgt_prefix (bool): Force the predictions begin with provided -tgt.
        data_type (str): Source data type.
        verbose (bool): Print/log every translation.
        report_time (bool): Print/log total time/frequency.
        global_scorer (onmt.translate.GNMTGlobalScorer): Translation
            scoring/reranking object.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
        self,
        model,
        datamodule,
        decoding_strategy,
        n_best=1,
        min_length=2,
        max_length=128,
        ratio=0.0,
        beam_size=1,
        random_sampling_topk=0,
        random_sampling_topp=0,
        random_sampling_temp=1.0,
        stepwise_penalty=False,
        dump_beam='',
        block_ngram_repeat=0,
        ignore_when_blocking=[],
        ban_unk_token=False,
        tgt_prefix=False,
        data_type="text",
        verbose=False,
        report_time=False,
        global_scorer=None,
        report_score=True,
        logger=None,
    ):
        self.model = model
        self._src_vocab = datamodule.src_tokenizer
        self._tgt_vocab = datamodule.trg_tokenizer
        self._tgt_eos_idx = datamodule.trg_tokenizer.get_id('[EOS]') #ET node
        self._tgt_bos_idx = datamodule.trg_tokenizer.get_id('[SOS]') #ROOT node
        self._tgt_unk_idx = datamodule.trg_tokenizer.unknown_id

        self.n_best = n_best
        self.min_length = min_length
        self.max_length = max_length

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk
        self.sample_from_topp = random_sampling_topp

        self.ban_unk_token = ban_unk_token
        self.ratio = ratio
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {
            self._tgt_vocab.get_id(t) for t in self.ignore_when_blocking
        }
        self.tgt_prefix = tgt_prefix
        self.data_type = data_type
        self.global_scorer = global_scorer
        self.decoding_strategy = decoding_strategy

        self.verbose = verbose
        self.report_time = report_time
        self.report_score = report_score

        self.use_filter_pred = False
        self._filter_pred = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": [],
            }

    def translate(
        self,
        phrase_batch, phrase_len_batch,
        attn_debug=False,
        has_tgt=False
    ):
        """
        Returns:
            (`list`, `list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        # Statistics

        translations = self._translate_batch_with_strategy(phrase_batch, phrase_len_batch)
        return translations

    def _decode_and_generate(
        self,
        decoder_in,
        memory_bank,
        memory_lengths,
        step=None,
    ):
        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam x batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out = self.model.decoder(
            decoder_in, memory_bank = memory_bank, memory_lengths=memory_lengths, step=step,\
            max_dec_len = self.max_length
        )
        logits = self.model.generator_head(dec_out)
        if logits.shape[0]==1:
            logits = logits.squeeze(0)
        # returns [(batch_size x beam_size) , vocab ] when 1 step
        # or [ tgt_len, batch_size, vocab ] when full sentence


        log_probs = F.log_softmax(logits, dim=-1)
        # print(log_probs)
        # print(decoder_in,parents_idx,tree_coordinates)
        return log_probs

    def get_strategy(self,batch_size,attn_debug=False):
        # if self.decoding_strategy == 'greedy':
        #     decode_strategy = GreedySearch(
        #             bos=self._tgt_bos_idx,
        #             eos=self._tgt_eos_idx,
        #             eoc=self._tgt_eoc_idx,
        #             unk=self._tgt_unk_idx,
        #             batch_size = batch_size,
        #             global_scorer=self.global_scorer,
        #             min_length=self.min_length,
        #             max_length=self.max_length,
        #             block_ngram_repeat=self.block_ngram_repeat,
        #             exclusion_tokens=self._exclusion_idxs,
        #             return_attention = attn_debug,
        #             sampling_temp=self.random_sampling_temp,
        #             keep_topk=self.sample_from_topk,
        #             keep_topp=self.sample_from_topp,
        #             beam_size=1,
        #             ban_unk_token=self.ban_unk_token,
        #         )
        if self.decoding_strategy=='beam':
            decode_strategy = BeamSearch(
                    beam_size = self.beam_size,
                    batch_size= batch_size,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    n_best=self.n_best,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    return_attention=attn_debug,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio,
                    ban_unk_token=self.ban_unk_token,
                )
        else:
            raise Exception
        return decode_strategy

    def _translate_batch_with_strategy(self, sentence_phrases, phrases_len):
        """Translate a batch of sentences step by step using cache.
        Args:
            sentence_phrases [batch size X max phrase count X max phrase length]
            phrases_len  [batch size X max phrase count]
            target [max target length X batch X 1]
        Returns:
            results (dict): The translation results.
        """

        batch_size = sentence_phrases.shape[0]
        decode_strategy = self.get_strategy(batch_size)
        # (0) Prep the components of the search.
        parallel_paths = decode_strategy.parallel_paths  # beam_size

        # (1) Run the encoder on the src.
        encodings, lengths = self.model.invocation_encoder(sentence_phrases, phrases_len)
        # encodings [max phrase count X batch size X model dim]
        # lengths [batch size]
        assert encodings.shape[0] == lengths.max()

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = None
        (
            _,
            memory_bank,
            memory_lengths,
            src_map
        ) = decode_strategy.initialize(encodings, lengths, src_map)
        select_indices = None
        assert memory_bank.shape[0] == memory_lengths.max()
        assert memory_lengths.max() == lengths.max()
        assert memory_bank.shape[1] == encodings.shape[1] * parallel_paths

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            # if step<TargetClone.shape[0]:
            #     decoder_input = TargetClone[step,:,:].unsqueeze(0) #Previous gold target
            # else:
            #     decoder_input = decode_strategy.current_predictions.view(1, -1, 1) #1, batch_size * parallel_beams, 1
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1) #1, batch_size * parallel_beams, 1

            logits = self._decode_and_generate(
                decoder_input,
                memory_bank,
                memory_lengths=memory_lengths,
                step=step
            )

            # # GOLD INPUT!
            # logits = torch.randn_like(logits)
            # logits_idx = TargetClone[step,:,0]
            # # print(logits_idx.shape,logits.shape)
            # logits[torch.arange(logits.shape[0]),logits_idx] = 1000
            # logits = F.log_softmax(logits,dim=-1)

            decode_strategy.advance(logits, None)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices
            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(
                        B.index_select(1, select_indices) for B in memory_bank
                    )
                    memory_lengths = tuple(
                        M.index_select(0, select_indices) for M in memory_lengths
                    )
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)
                    memory_lengths = memory_lengths.index_select(0, select_indices)



            if parallel_paths > 1 or any_finished:
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices)
                )

        invocation = sentence_phrases.transpose(1,0)
        translation_builder = TranslationBuilder(self._src_vocab,self._tgt_vocab)
        translations = translation_builder.build_batch(decode_strategy.predictions,\
                decode_strategy.scores,\
                invocation,\
                phrases_len)
        return translations


class TranslationBuilder:
    def __init__(self,src_tokenizer,trg_tokenizer):
        self.src_tokenizer =  src_tokenizer
        self.trg_tokenizer = trg_tokenizer

    def tokens_to_text(self,toklist,vocab,maxlen=10000):
        tokens = []
        for i,tok in enumerate(toklist):
            if i==maxlen:
                break
            tokid = tok.item()
            token = vocab.get_token(tokid)
            tokens.append(token)
            if token == '[EOS]':
                break
        return tokens

    def build_batch(self,pred,score,invocation,invocation_length):
        translations = []
        for b in range(len(pred)):
            #invocation : [max phrase count, batch, max phrase length]
            #invocation_length : [batch, max phrase count]

            # Slice by tensor entries
            # inv_mask = invocation[:,b,:]!=self.src_tokenizer.pad
            # inv = invocation[:,b,:][inv_mask]
            # assert inv.shape[0]==invocation_length[b].sum().item()

            # text = [self.src_tokenizer.get_token(tokid.item()) for tokid in inv]
            # phrase_start_idx = 0
            # phrases = []
            # for phrase_len in invocation_length[b]:
            #     phrases.append(text[phrase_start_idx:phrase_start_idx+phrase_len.item()])
            #     phrase_start_idx += phrase_len.item()
            translation = Translation("", "")
            assert len(score[b]) == len(pred[b]) and len(pred[b])>=1
            for pred_score,cur_pred in zip(score[b],pred[b]):
                p = self.tokens_to_text(cur_pred,self.trg_tokenizer)[:-1] #remove 1 from start
                translation.append("",p,pred_score.item())
            translations.append(translation)
        return translations

    def build_golden_batch(self,pred,truth, truth_len, invocation, invocation_length):
        translations = []
        for b in range(len(pred)):
            inv = invocation[b,:invocation_length[b]]
            text = [self.src_tokenizer.get_token(tokid.item()) for tokid in inv]
            translation = Translation(text)
            cur_truth = truth[:truth_len[b],b,0]
            cur_pred = pred[b,:truth_len[b]]
            p = self.tokens_to_text(cur_pred,self.trg_tokenizer)
            t = self.tokens_to_text(cur_truth,self.trg_tokenizer)[1:-1] #remove sos and eos
            translation.append(t,p,0)
            translations.append(translation)
        return translations


class Translation:
    def __init__(self,text, text_phrase):
        self.truth = []
        self.pred = []
        self.pred_score = []
        self.inv = text
        self.inv_phrasing = text_phrase

    def append(self,truth,pred,pred_score):
        self.truth.append(truth)
        self.pred.append(pred)
        self.pred_score.append(pred_score)